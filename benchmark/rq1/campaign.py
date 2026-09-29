#!/usr/bin/env python3
"""Run and aggregate the fixed-input RQ1 throughput campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark.rq1.artifacts import dump_seed, timeout_output as _timeout_output
from benchmark.workloads.schema import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKENDS = (
    "cufuzz",
    "origin-no-feedback",
    "origin",
    "rapid-no-feedback",
    "rapid",
    "rapid2-no-feedback",
    "rapid2",
)
RESULT_PREFIX = "RAPID_BENCHMARK_RESULT "
logger = logging.getLogger(__name__)


def configurations(window_size: int) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "cufuzz",
                "origin-no-feedback",
                "origin",
                "rapid-w1",
                f"rapid-w{window_size}",
                "rapid-no-feedback-w1",
                f"rapid-no-feedback-w{window_size}",
                "rapid2",
                "rapid2-no-feedback",
            )
        )
    )


CONFIGURATIONS = configurations(2)


@dataclass(frozen=True)
class BenchmarkRecord:
    workload_id: str
    backend: str
    repetition: int
    executions_per_second: float


def trial_order(
    repetition: int, configured: tuple[str, ...] = CONFIGURATIONS
) -> tuple[str, ...]:
    offset = repetition % len(configured)
    return configured[offset:] + configured[:offset]


def artifact_backend(configuration: str) -> str:
    if configuration.startswith("rapid-no-feedback-w"):
        return "rapid-no-feedback"
    if configuration.startswith("rapid-w"):
        return "rapid"
    return configuration


def _window_size(configuration: str, configured_size: int) -> int | None:
    if configuration.startswith(("rapid-w", "rapid-no-feedback-w")):
        return int(configuration.rsplit("w", maxsplit=1)[1])
    if configuration in ("rapid2", "rapid2-no-feedback"):
        return configured_size
    return None


def benchmark_command(
    *,
    executable: str | Path,
    library: str | Path,
    manifest: str | Path,
    warmup_runs: int,
    benchmark_seconds: int,
    window_size: int | None,
) -> list[str]:
    command = [
        str(executable),
        str(library),
        "--manifest",
        str(manifest),
        "--benchmark",
        "--warmup-runs",
        str(warmup_runs),
        "--benchmark-seconds",
        str(benchmark_seconds),
    ]
    if window_size is not None:
        command.extend(["--window-size", str(window_size)])
    return command


def parse_benchmark_result(output: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one benchmark result, got {len(lines)}")
    result = json.loads(lines[0][len(RESULT_PREFIX) :])
    required = (
        "requested_min_duration_ns",
        "measured_iterations",
        "completed",
        "elapsed_ns",
        "corpus_size",
        "solutions",
        "pending",
    )
    if not isinstance(result, dict) or any(
        not isinstance(result.get(key), int) for key in required
    ):
        raise RuntimeError("benchmark result has missing or non-integer fields")
    if result["completed"] <= 0:
        raise RuntimeError("benchmark completed no executions")
    if result["elapsed_ns"] <= 0:
        raise RuntimeError("benchmark elapsed_ns must be positive")
    if result["elapsed_ns"] < result["requested_min_duration_ns"]:
        raise RuntimeError("benchmark ended before its minimum duration")
    if result["pending"] != 0:
        raise RuntimeError(f"benchmark exited with pending={result['pending']}")
    if result.get("mutation_enabled") is not False:
        raise RuntimeError("benchmark result must report mutation_enabled=false")
    return result


def aggregate_records(records: Iterable[BenchmarkRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        grouped.setdefault((record.workload_id, record.backend), []).append(
            record.executions_per_second
        )
    rows: list[dict[str, Any]] = []
    for (workload_id, backend), values in sorted(grouped.items()):
        ordered = sorted(values)
        quartiles = (
            statistics.quantiles(ordered, n=4, method="inclusive")
            if len(ordered) > 1
            else [ordered[0], ordered[0], ordered[0]]
        )
        rows.append(
            {
                "workload_id": workload_id,
                "backend": backend,
                "repetitions": len(ordered),
                "median_exec_per_sec": statistics.median(ordered),
                "q1_exec_per_sec": quartiles[0],
                "q3_exec_per_sec": quartiles[2],
            }
        )
    return rows


def _capture(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        return f"unavailable: {error}"
    return completed.stdout.strip()


def _environment(gpu_device: str, fuzzer: Path, fuzzer_async: Path) -> dict[str, Any]:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_device": gpu_device,
        "nvidia_smi": _capture(
            [
                "nvidia-smi",
                f"--id={gpu_device}",
                "--query-gpu=name,uuid,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "clang": _capture(["clang++", "--version"]),
        "rustc": _capture(["rustc", "--version"]),
        "rapid_head": _capture(["git", "rev-parse", "HEAD"]),
        "rapid_status": _capture(["git", "status", "--short"]),
        "fuzzer": {"path": str(fuzzer), "sha256": sha256_file(fuzzer)},
        "fuzzer_async": {"path": str(fuzzer_async), "sha256": sha256_file(fuzzer_async)},
    }


def _load_build_reports(build_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    summary_path = build_root / "build_summary.json"
    runtime_path = build_root / "runtime_verification_summary.json"
    if not summary_path.is_file() or not runtime_path.is_file():
        raise RuntimeError("build root must contain build and runtime verification summaries")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_paths = summary.get("reports")
    if not isinstance(report_paths, list) or not report_paths:
        raise RuntimeError("build summary must contain at least one report")
    if len(report_paths) != len(set(report_paths)):
        raise RuntimeError("build summary must contain distinct reports")
    reports = []
    workload_ids: set[str] = set()
    for raw_path in report_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        report = json.loads(path.read_text(encoding="utf-8"))
        if set(report.get("backends", {})) != set(BACKENDS):
            raise RuntimeError(f"unexpected backend set in {path}")
        workload_id = report.get("workload_id")
        if not isinstance(workload_id, str) or not workload_id:
            raise RuntimeError(f"missing workload ID in {path}")
        if workload_id in workload_ids:
            raise RuntimeError(f"duplicate workload ID in build summary: {workload_id}")
        workload_ids.add(workload_id)
        reports.append((path.resolve(), report))
    return reports


def _backend_library(report_path: Path, report: dict[str, Any], backend: str) -> Path:
    relative = report["backends"][backend]["artifacts"]["shared_library"]["path"]
    return (report_path.parent / "backends" / backend / relative).resolve()


def _manifest(report: dict[str, Any]) -> Path:
    return (Path(report["shared_phase2"]["phase2_dir"]).parent / "manifest.json").resolve()


def _seed_sha256(fuzzer: Path, manifest: Path) -> str:
    return hashlib.sha256(dump_seed(fuzzer, manifest)).hexdigest()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "workload_id",
        "backend",
        "repetitions",
        "median_exec_per_sec",
        "q1_exec_per_sec",
        "q3_exec_per_sec",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_attempted_trials(
    output: Path,
) -> tuple[set[tuple[str, str, int]], list[BenchmarkRecord]]:
    attempted: set[tuple[str, str, int]] = set()
    records: list[BenchmarkRecord] = []
    trials_path = output / "trials.jsonl"
    if trials_path.exists():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            trial = json.loads(line)
            key = (trial["workload_id"], trial["backend"], trial["repetition"])
            attempted.add(key)
            records.append(BenchmarkRecord(*key, trial["executions_per_second"]))

    failures_path = output / "failures.jsonl"
    if failures_path.exists():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            failure = json.loads(line)
            attempted.add(
                (failure["workload_id"], failure["backend"], failure["repetition"])
            )
    return attempted, records


def run_campaign(args: argparse.Namespace) -> None:
    build_root = args.build_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    fuzzer = args.fuzzer.resolve()
    fuzzer_async = args.fuzzer_async.resolve()
    reports = _load_build_reports(build_root)

    environment_path = output / "environment.json"
    if not environment_path.exists():
        environment_path.write_text(
            json.dumps(
                _environment(args.gpu_device, fuzzer, fuzzer_async),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    trials_path = output / "trials.jsonl"
    failures_path = output / "failures.jsonl"
    existing, records = load_attempted_trials(output)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    configured = configurations(args.window_size)
    for repetition in range(args.repetitions):
        for report_path, report in reports:
            workload_id = report["workload_id"]
            manifest = _manifest(report)
            input_sha256: str | None = None
            for configuration in trial_order(repetition, configured):
                key = (workload_id, configuration, repetition)
                if key in existing:
                    continue
                if input_sha256 is None:
                    input_sha256 = _seed_sha256(fuzzer, manifest)
                backend = artifact_backend(configuration)
                library = _backend_library(report_path, report, backend)
                executable = (
                    fuzzer_async if configuration.startswith("rapid2") else fuzzer
                )
                command = benchmark_command(
                    executable=executable,
                    library=library,
                    manifest=manifest,
                    warmup_runs=args.warmup_runs,
                    benchmark_seconds=args.benchmark_seconds,
                    window_size=_window_size(configuration, args.window_size),
                )
                stem = f"{repetition:02d}-{workload_id}-{configuration}"
                try:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=args.timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    (logs / f"{stem}.stdout.log").write_text(
                        _timeout_output(error.stdout), encoding="utf-8"
                    )
                    (logs / f"{stem}.stderr.log").write_text(
                        _timeout_output(error.stderr), encoding="utf-8"
                    )
                    failure = {
                        "schema_version": 1,
                        "workload_id": workload_id,
                        "backend": configuration,
                        "artifact_backend": backend,
                        "window_size": _window_size(configuration, args.window_size),
                        "repetition": repetition,
                        "command": command,
                        "failure": "timeout",
                        "timeout_seconds": args.timeout,
                    }
                    with failures_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(failure, sort_keys=True) + "\n")
                    existing.add(key)
                    print(
                        f"{workload_id} {configuration} rep={repetition}: TIMEOUT after {args.timeout}s",
                        flush=True,
                    )
                    continue
                (logs / f"{stem}.stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (logs / f"{stem}.stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                if completed.returncode != 0:
                    failure = {
                        "schema_version": 1,
                        "workload_id": workload_id,
                        "backend": configuration,
                        "artifact_backend": backend,
                        "window_size": _window_size(configuration, args.window_size),
                        "repetition": repetition,
                        "command": command,
                        "failure": "nonzero_exit",
                        "exit_code": completed.returncode,
                    }
                    with failures_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(failure, sort_keys=True) + "\n")
                    existing.add(key)
                    print(
                        f"{workload_id} {configuration} rep={repetition}: "
                        f"FAILED with exit code {completed.returncode}",
                        flush=True,
                    )
                    continue
                result = parse_benchmark_result(completed.stdout)
                exec_per_sec = result["completed"] * 1_000_000_000 / result["elapsed_ns"]
                trial = {
                    "schema_version": 1,
                    "workload_id": workload_id,
                    "backend": configuration,
                    "artifact_backend": backend,
                    "window_size": _window_size(configuration, args.window_size),
                    "repetition": repetition,
                    "command": command,
                    "canonical_input_sha256": input_sha256,
                    "manifest_sha256": report["shared_phase2"]["artifacts"]["manifest.json"]["sha256"],
                    "backend_sha256": report["backends"][backend]["artifacts"]["shared_library"]["sha256"],
                    "source_provenance": report["source_provenance"],
                    "benchmark": result,
                    "executions_per_second": exec_per_sec,
                }
                with trials_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(trial, sort_keys=True) + "\n")
                records.append(BenchmarkRecord(*key, exec_per_sec))
                existing.add(key)
                print(
                    f"{workload_id} {configuration} rep={repetition}: {exec_per_sec:.2f} exec/s",
                    flush=True,
                )

    completed_repetitions: dict[tuple[str, str], set[int]] = {}
    for record in records:
        completed_repetitions.setdefault(
            (record.workload_id, record.backend), set()
        ).add(record.repetition)
    for _, report in reports:
        workload_id = report["workload_id"]
        for configuration in dict.fromkeys(configured):
            completed = len(
                completed_repetitions.get((workload_id, configuration), set())
            )
            if completed < args.repetitions:
                logger.warning(
                    "%s %s has %d/%d completed repetitions; summary.csv is incomplete",
                    workload_id,
                    configuration,
                    completed,
                    args.repetitions,
                )

    _write_summary(output / "summary.csv", aggregate_records(records))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--benchmark-seconds", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--fuzzer",
        type=Path,
        default=REPO_ROOT / "cuda-fuzzer/target/release/fuzzer",
    )
    parser.add_argument(
        "--fuzzer-async",
        type=Path,
        default=REPO_ROOT / "cuda-fuzzer/target/release/fuzzer_async",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in ("repetitions", "warmup_runs", "benchmark_seconds"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.window_size <= 32:
        raise SystemExit("--window-size must be in 1..=32")
    run_campaign(args)


if __name__ == "__main__":
    main()
