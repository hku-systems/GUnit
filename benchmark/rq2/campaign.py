#!/usr/bin/env python3
"""Run the RQ2 coverage-growth campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq1.artifacts import timeout_output as _timeout_text
from benchmark.rq2.verify_results import (
    CfgMetricContract,
    SIMT_MEMORY_METRIC_VERSION,
    MemoryMetricContract,
    SampleContext,
    parse_cfg_metric_contract,
    parse_memory_metric_contract,
    validate_and_enrich_samples,
)
from benchmark.workloads.schema import sha256_file
from benchmark.rq2.campaign_model import (
    BACKENDS,
    CONFIGURATIONS,
    RQ2_ROOT,
    VCONFIG_UNSUPPORTED_REASONS,
    BackendArtifact,
    CampaignInput,
    Configuration,
    _read_object,
    load_campaign_input,
    load_campaign_inputs,
    select_campaign_inputs,
    validate_cfg_presence_contracts,
)
from benchmark.rq2.campaign_storage import (
    _append_jsonl,
    _load_jsonl,
    _remove_orphan_samples,
    _terminal_base,
    append_unique_samples,
    load_attempted_trials,
)


def seed_for_repetition(base_seed: int, repetition: int, seed_mode: str) -> int:
    if seed_mode == "fixed":
        return base_seed
    if seed_mode == "per-rep":
        # Consecutive offsets keep (base seed, repetition) reproducible and distinct.
        return base_seed + repetition
    raise ValueError(f"unknown seed mode: {seed_mode}")


def campaign_command(
    *,
    configuration: Configuration,
    fuzzer: Path,
    fuzzer_async: Path,
    library: Path,
    manifest: Path,
    coverage_seconds: int,
    coverage_log: Path,
) -> list[str]:
    executable = fuzzer_async if configuration.asynchronous else fuzzer
    command = [
        str(executable),
        str(library),
        "--manifest",
        str(manifest),
        "--coverage-seconds",
        str(coverage_seconds),
        "--coverage-log",
        str(coverage_log),
        "--vconfig",
        configuration.vconfig,
    ]
    if configuration.backend in ("rapid", "rapid2"):
        command.extend(["--window-size", str(configuration.window)])
    return command

def _relative(path: Path, output: Path) -> str:
    return str(path.resolve().relative_to(output.resolve()))


def _configuration_row(configuration: Configuration) -> dict[str, Any]:
    return {"schema_version": 1, **asdict(configuration)}


def _write_or_verify_configurations(output: Path) -> None:
    path = output / "configurations.jsonl"
    expected = [_configuration_row(configuration) for configuration in CONFIGURATIONS]
    if path.exists():
        if _load_jsonl(path) != expected:
            raise RuntimeError("existing configurations.jsonl differs from campaign matrix")
        return
    for row in expected:
        _append_jsonl(path, row)

def _capture(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        return {"command": list(command), "returncode": None, "output": str(error)}
    output = completed.stdout.strip()
    if completed.stderr.strip():
        output = "\n".join(part for part in (output, completed.stderr.strip()) if part)
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "output": output,
    }


def _snapshot_fuzzer_binaries(
    args: argparse.Namespace,
    output: Path,
) -> dict[str, dict[str, str]]:
    environment_path = output / "environment.json"
    snapshot_root = output / "binaries"
    expected_paths = {
        "fuzzer": (snapshot_root / "fuzzer").resolve(),
        "fuzzer_async": (snapshot_root / "fuzzer_async").resolve(),
    }
    if environment_path.exists():
        environment = _read_object(environment_path, "campaign environment")
        identities: dict[str, dict[str, str]] = {}
        for name, expected_path in expected_paths.items():
            identity = environment.get(name)
            if not isinstance(identity, dict):
                raise RuntimeError(f"existing environment {name} identity is invalid")
            if identity.get("path") != str(expected_path):
                raise RuntimeError(
                    f"existing environment {name} does not use the result snapshot"
                )
            recorded_sha256 = identity.get("sha256")
            if (
                not isinstance(recorded_sha256, str)
                or f"sha256:{sha256_file(expected_path)}" != recorded_sha256
            ):
                raise RuntimeError(f"{name} snapshot hash mismatch")
            identities[name] = {
                "path": str(expected_path),
                "sha256": recorded_sha256,
            }
        return identities

    snapshot_root.mkdir(exist_ok=True)
    identities = {}
    for name, expected_path in expected_paths.items():
        source_path = Path(getattr(args, name)).resolve()
        if not source_path.is_file():
            raise RuntimeError(f"{name} executable does not exist: {source_path}")
        if expected_path.exists():
            if sha256_file(expected_path) != sha256_file(source_path):
                raise RuntimeError(
                    f"existing {name} snapshot differs from requested executable"
                )
        else:
            shutil.copy2(source_path, expected_path)
        identities[name] = {
            "path": str(expected_path),
            "sha256": f"sha256:{sha256_file(expected_path)}",
        }
    return identities


def _write_or_verify_environment(
    args: argparse.Namespace,
    output: Path,
    inputs: Sequence[CampaignInput],
    fuzzer_identities: Mapping[str, Mapping[str, str]],
) -> None:
    path = output / "environment.json"
    immutable = {
        "schema_version": 1,
        "gpu_device": str(args.gpu_device),
        "fixed_seed": int(args.fixed_seed),
        "seed_mode": args.seed_mode,
        "coverage_seconds": int(args.coverage_seconds),
        "coverage_recording_mode": "completion-change-v1",
        "repetitions": int(args.repetitions),
        "timeout_seconds": int(args.timeout),
        "workloads": [
            {
                "workload_id": item.workload_id,
                "build_report_path": str(item.build_report_path),
                "build_report_sha256": item.build_report_sha256,
                "vconfig_effective": item.vconfig_effective,
                "vconfig_disabled_reason": item.vconfig_disabled_reason,
            }
            for item in inputs
        ],
        "configurations": [item.configuration_id for item in CONFIGURATIONS],
        "fuzzer": dict(fuzzer_identities["fuzzer"]),
        "fuzzer_async": dict(fuzzer_identities["fuzzer_async"]),
    }
    if path.exists():
        existing = _read_object(path, "campaign environment")
        existing_seed_mode = existing.get("seed_mode", "fixed")
        required_snapshot_fields = (
            "captured_at",
            "hostname",
            "platform",
            "python",
            "nvidia_smi",
            "cuda_path",
            "nvcc",
            "clang",
            "rustc",
            "rapid_head",
            "rapid_status",
        )
        immutable_without_seed_mode = {
            key: value for key, value in immutable.items() if key != "seed_mode"
        }
        if (
            any(field not in existing for field in required_snapshot_fields)
            or existing_seed_mode != immutable["seed_mode"]
            or any(
                existing.get(key) != value
                for key, value in immutable_without_seed_mode.items()
            )
        ):
            raise RuntimeError("existing environment.json differs from campaign arguments")
        return
    snapshot = {
        **immutable,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": " ".join(
            (platform.system(), platform.release(), platform.machine())
        ),
        "python": platform.python_version(),
        "nvidia_smi": _capture(
            (
                "nvidia-smi",
                f"--id={args.gpu_device}",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            )
        ),
        "cuda_path": os.environ.get("CUDA_PATH", "/usr/local/cuda"),
        "nvcc": _capture(("nvcc", "--version")),
        "clang": _capture(("clang++", "--version")),
        "rustc": _capture(("rustc", "--version")),
        "rapid_head": _capture(("git", "rev-parse", "HEAD")),
        "rapid_status": _capture(("git", "status", "--short")),
    }
    path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )



def run_campaign(
    args: argparse.Namespace,
    *,
    campaign_inputs: Sequence[CampaignInput] | None = None,
) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    raw_root = output / "raw"
    logs.mkdir(exist_ok=True)
    raw_root.mkdir(exist_ok=True)
    for ledger in ("samples.jsonl", "trials.jsonl", "failures.jsonl"):
        (output / ledger).touch(exist_ok=True)
    _write_or_verify_configurations(output)
    available_inputs = (
        tuple(campaign_inputs)
        if campaign_inputs is not None
        else load_campaign_inputs(args.build_root)
    )
    inputs = select_campaign_inputs(
        available_inputs,
        getattr(args, "workload", None),
    )
    validate_cfg_presence_contracts(inputs)
    fuzzer_identities = _snapshot_fuzzer_binaries(args, output)
    _write_or_verify_environment(args, output, inputs, fuzzer_identities)
    _remove_orphan_samples(output)
    attempted = load_attempted_trials(output)
    subprocess_env = os.environ.copy()
    subprocess_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    trials_path = output / "trials.jsonl"
    failures_path = output / "failures.jsonl"
    samples_path = output / "samples.jsonl"

    plan_index = 0
    for repetition in range(args.repetitions):
        repetition_seed = seed_for_repetition(
            int(args.fixed_seed), repetition, args.seed_mode
        )
        repetition_env = subprocess_env.copy()
        repetition_env["RAPID_FIXED_SEED"] = str(repetition_seed)
        for campaign_input in inputs:
            for configuration in CONFIGURATIONS:
                key = (
                    campaign_input.workload_id,
                    configuration.configuration_id,
                    repetition,
                )
                current_plan_index = plan_index
                plan_index += 1
                if key in attempted:
                    continue
                base = _terminal_base(
                    campaign_input,
                    configuration,
                    repetition,
                    repetition_seed,
                    current_plan_index,
                )
                if configuration.vconfig == "on" and not campaign_input.vconfig_effective:
                    _append_jsonl(
                        trials_path,
                        {
                            **base,
                            "status": "skipped",
                            "vconfig_effective": "unsupported",
                            "vconfig_disabled_reason": campaign_input.vconfig_disabled_reason,
                        },
                    )
                    attempted.add(key)
                    continue

                trial_id = base["trial_id"]
                raw_path = raw_root / f"{trial_id}.jsonl"
                raw_path.unlink(missing_ok=True)
                artifact = campaign_input.backends[configuration.backend]
                command = campaign_command(
                    configuration=configuration,
                    fuzzer=Path(fuzzer_identities["fuzzer"]["path"]),
                    fuzzer_async=Path(fuzzer_identities["fuzzer_async"]["path"]),
                    library=artifact.library_path,
                    manifest=campaign_input.manifest_path,
                    coverage_seconds=args.coverage_seconds,
                    coverage_log=raw_path,
                )
                stdout_path = logs / f"{trial_id}.stdout.log"
                stderr_path = logs / f"{trial_id}.stderr.log"
                try:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=repetition_env,
                        text=True,
                        capture_output=True,
                        timeout=args.timeout,
                        check=False,
                    )
                    stdout = completed.stdout
                    stderr = completed.stderr
                except subprocess.TimeoutExpired as error:
                    stdout = _timeout_text(error.stdout)
                    stderr = _timeout_text(error.stderr)
                    stdout_path.write_text(stdout, encoding="utf-8")
                    stderr_path.write_text(stderr, encoding="utf-8")
                    _append_jsonl(
                        failures_path,
                        {
                            **base,
                            "status": "failed",
                            "failure": "timeout",
                            "timeout_seconds": args.timeout,
                            "command": command,
                            "vconfig_effective": configuration.vconfig,
                            "vconfig_disabled_reason": None,
                            "stdout_log": _relative(stdout_path, output),
                            "stdout_sha256": f"sha256:{sha256_file(stdout_path)}",
                            "stderr_log": _relative(stderr_path, output),
                            "stderr_sha256": f"sha256:{sha256_file(stderr_path)}",
                            "raw_telemetry_path": _relative(raw_path, output),
                            "raw_telemetry_sha256": f"sha256:{sha256_file(raw_path)}"
                            if raw_path.is_file()
                            else None,
                        },
                    )
                    attempted.add(key)
                    continue
                stdout_path.write_text(stdout, encoding="utf-8")
                stderr_path.write_text(stderr, encoding="utf-8")
                common_terminal = {
                    **base,
                    "command": command,
                    "vconfig_effective": configuration.vconfig,
                    "vconfig_disabled_reason": None,
                    "stdout_log": _relative(stdout_path, output),
                    "stdout_sha256": f"sha256:{sha256_file(stdout_path)}",
                    "stderr_log": _relative(stderr_path, output),
                    "stderr_sha256": f"sha256:{sha256_file(stderr_path)}",
                    "raw_telemetry_path": _relative(raw_path, output),
                    "raw_telemetry_sha256": f"sha256:{sha256_file(raw_path)}"
                    if raw_path.is_file()
                    else None,
                }
                if completed.returncode != 0:
                    _append_jsonl(
                        failures_path,
                        {
                            **common_terminal,
                            "status": "failed",
                            "failure": "nonzero",
                            "returncode": completed.returncode,
                        },
                    )
                    attempted.add(key)
                    continue
                if not raw_path.is_file():
                    _append_jsonl(
                        failures_path,
                        {
                            **common_terminal,
                            "status": "failed",
                            "failure": "missing_raw_telemetry",
                        },
                    )
                    attempted.add(key)
                    continue
                raw_sha256 = f"sha256:{sha256_file(raw_path)}"
                context = SampleContext(
                    trial_id=trial_id,
                    workload_id=campaign_input.workload_id,
                    configuration_id=configuration.configuration_id,
                    paper_label=configuration.paper_label,
                    backend=configuration.backend,
                    window=configuration.window,
                    repetition=repetition,
                    seed=repetition_seed,
                    vconfig_requested=configuration.vconfig,
                    vconfig_effective=configuration.vconfig,
                    feedback_enabled=configuration.feedback_enabled,
                    build_report_sha256=campaign_input.build_report_sha256,
                    manifest_sha256=campaign_input.manifest_sha256,
                    backend_sha256=artifact.library_sha256,
                    instrumentation_metadata_sha256=(
                        artifact.instrumentation_metadata_sha256
                    ),
                    instrumented_cfg_sites=artifact.instrumented_cfg_sites,
                    memory_metric_version=artifact.memory_metric_version,
                    memory_map_bits=artifact.memory_map_bits,
                    memory_sector_bytes=artifact.memory_sector_bytes,
                    thread_activity_map_bits=artifact.thread_activity_map_bits,
                    memory_hash_contract_version=(
                        artifact.memory_hash_contract_version
                    ),
                    cfg_metric_version=artifact.cfg_metric_version,
                    raw_telemetry_path=_relative(raw_path, output),
                    raw_telemetry_sha256=raw_sha256,
                )
                try:
                    samples = validate_and_enrich_samples(raw_path, context)
                except ValueError as error:
                    _append_jsonl(
                        failures_path,
                        {
                            **common_terminal,
                            "raw_telemetry_sha256": raw_sha256,
                            "status": "failed",
                            "failure": "invalid_raw_telemetry",
                            "error": str(error),
                        },
                    )
                    attempted.add(key)
                    continue
                append_unique_samples(samples_path, samples)
                _append_jsonl(
                    trials_path,
                    {
                        **common_terminal,
                        "raw_telemetry_sha256": raw_sha256,
                        "status": "completed",
                        "sample_count": len(samples),
                        "final_timestamp_s": samples[-1]["timestamp_s"],
                        "final_executions_completed": samples[-1][
                            "executions_completed"
                        ],
                    },
                )
                attempted.add(key)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--coverage-seconds", type=int, default=10)
    parser.add_argument(
        "--workload",
        action="append",
        help="run only this workload; repeat to select multiple workloads",
    )
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--fixed-seed", type=int, default=1)
    parser.add_argument(
        "--seed-mode",
        choices=("fixed", "per-rep"),
        default="per-rep",
        help="reuse --fixed-seed or add the repetition index",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--fuzzer",
        type=Path,
        default=REPO_ROOT / "cuda-fuzzer" / "target" / "release" / "fuzzer",
    )
    parser.add_argument(
        "--fuzzer-async",
        type=Path,
        default=REPO_ROOT
        / "cuda-fuzzer"
        / "target"
        / "release"
        / "fuzzer_async",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in ("repetitions", "coverage_seconds", "timeout"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    run_campaign(args)


if __name__ == "__main__":
    main()
