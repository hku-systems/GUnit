#!/usr/bin/env python3
"""Helpers for RQ4 CUDA API and persistent-kernel timing profiles."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from benchmark.rq1 import campaign as rq1
from benchmark.rq4.campaign import _load_reports, benchmark_command
from benchmark.rq4.schema import (
    DEFAULT_WINDOW_SIZE,
    configurations,
)
from benchmark.rq4.profile_capture import (
    CPU_FEEDBACK_SEGMENTS,
    DEVICE_SEGMENTS,
    MAIN_LOOP_SEGMENTS,
    PROFILING_RECORD_FIELDS,
    load_profiling_records,
    profiling_summary,
    summarize_trace_evidence,
    validate_profiling_records,
)
from benchmark.rq4.profile_model import (
    API_CATEGORIES,
    MUTATING_QUEUE_FIELDS,
    MUTATING_RESULT_FIELDS,
    MUTATING_RESULT_PREFIX,
    load_completed_profiles,
    mutating_command,
    nsys_export_command,
    nsys_profile_command,
    parse_mutating_result,
    profile_environment,
    profile_identity,
    profile_stem,
    timing_libraries as _timing_libraries,
    validate_mutating_result,
    validate_mutating_result_fields,
)
from benchmark.rq4 import profile_process
from benchmark.rq4.profile_process import CellResidualError, CrashLoopTimeout
from benchmark.rq4.trace_extract import (
    TraceEvidence,
    write_compact_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MUTATING_BROKER_ADDRESS = profile_process.MUTATING_BROKER_ADDRESS
DEFAULT_MUTATING_SECONDS = 30
DEFAULT_CELL_TIMEOUT = 180
PROCESS_CLEANUP_TIMEOUT = profile_process.PROCESS_CLEANUP_TIMEOUT
LOCK_HOLDER_TERM_TIMEOUT = profile_process.LOCK_HOLDER_TERM_TIMEOUT
LOCK_HOLDER_POLL_INTERVAL = profile_process.LOCK_HOLDER_POLL_INTERVAL
LOCK_HOLDER_RESCAN_INTERVAL = profile_process.LOCK_HOLDER_RESCAN_INTERVAL
LOCK_HOLDER_TOTAL_TIMEOUT = profile_process.LOCK_HOLDER_TOTAL_TIMEOUT
LOGGER = logging.getLogger(__name__)


def _run_profile_command(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return profile_process.run_profile_command(
        command, cwd=cwd, env=env, timeout=timeout
    )


def _mutating_broker_port_is_available() -> bool:
    return profile_process.mutating_broker_port_is_available()


def _gpu_client_lock_path(gpu_device: str) -> Path:
    return profile_process.gpu_client_lock_path(gpu_device)


def _gpu_client_lock_is_available(gpu_device: str) -> bool:
    return profile_process.gpu_client_lock_is_available(gpu_device)


def _find_cell_residual_pids(pgid: int) -> list[int]:
    return profile_process.find_cell_residual_pids(pgid)


def _find_lock_fd_holders(gpu_device: str) -> list[int]:
    return profile_process.find_lock_fd_holders(gpu_device)


def _verify_cell_boundary(
    *, gpu_device: str, pgid: int | None = None
) -> str | None:
    return profile_process.verify_cell_boundary(
        gpu_device=gpu_device,
        pgid=pgid,
        lock_holder_total_timeout=LOCK_HOLDER_TOTAL_TIMEOUT,
        lock_holder_term_timeout=LOCK_HOLDER_TERM_TIMEOUT,
    )


def _teardown_cell(pgid: int | None, *, gpu_device: str) -> str | None:
    return profile_process.teardown_cell(
        pgid,
        gpu_device=gpu_device,
        lock_holder_total_timeout=LOCK_HOLDER_TOTAL_TIMEOUT,
        lock_holder_term_timeout=LOCK_HOLDER_TERM_TIMEOUT,
    )


def mutating_cell_timeout(mutate_seconds: int, *, base_timeout: int) -> float:
    return profile_process.mutating_cell_timeout(
        mutate_seconds, base_timeout=base_timeout
    )


def _run_mutating_profile(
    command: list[str],
    *,
    broker_command: list[str],
    cwd: Path,
    env: dict[str, str],
    broker_log: Path,
    timeout: float,
    budget_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], int]:
    return profile_process.run_mutating_profile(
        command,
        broker_command=broker_command,
        cwd=cwd,
        env=env,
        broker_log=broker_log,
        timeout=timeout,
        budget_seconds=budget_seconds,
    )


class CapturePostprocessError(RuntimeError):
    def __init__(self, stage: str, error: Exception) -> None:
        self.stage = stage
        self.detail = str(error)
        super().__init__(f"{stage}: {error}")


def _export_trace(
    *,
    nsys: Path,
    profile: Path,
    sqlite_path: Path,
    evidence_path: Path,
    keep_sqlite: bool,
) -> tuple[dict[str, Any], str, str | None]:
    try:
        export_result = subprocess.run(
            nsys_export_command(nsys=nsys, report=profile, sqlite_path=sqlite_path),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        if export_result.returncode != 0:
            detail = export_result.stderr.strip() or export_result.stdout.strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"nsys export failed ({export_result.returncode}){suffix}"
            )
    except Exception as error:
        raise CapturePostprocessError("nsys_export", error) from error

    try:
        evidence = write_compact_evidence(
            sqlite_path, evidence_path, keep_sqlite=keep_sqlite
        )
        trace_summary = summarize_trace_evidence(evidence)
    except Exception as error:
        raise CapturePostprocessError("trace_extract", error) from error
    return trace_summary, str(evidence_path), (
        str(sqlite_path) if keep_sqlite else None
    )


def _postprocess_capture(
    *,
    nsys: Path,
    profile: Path,
    sqlite_path: Path,
    evidence_path: Path,
    profiling_path: Path,
    keep_sqlite: bool,
) -> tuple[dict[str, Any], str, str | None, dict[str, Any]]:
    trace_summary, evidence_str, sqlite_str = _export_trace(
        nsys=nsys,
        profile=profile,
        sqlite_path=sqlite_path,
        evidence_path=evidence_path,
        keep_sqlite=keep_sqlite,
    )
    try:
        profiling = profiling_summary(load_profiling_records(profiling_path))
    except Exception as error:
        raise CapturePostprocessError("profiling_stats", error) from error
    return trace_summary, evidence_str, sqlite_str, profiling


def _try_extract_trace(
    *,
    nsys: Path,
    profile: Path,
    sqlite_path: Path,
    evidence_path: Path,
    keep_sqlite: bool,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not profile.exists():
        return None, None, None
    try:
        return _export_trace(
            nsys=nsys,
            profile=profile,
            sqlite_path=sqlite_path,
            evidence_path=evidence_path,
            keep_sqlite=keep_sqlite,
        )
    except CapturePostprocessError:
        if sqlite_path.exists() and not keep_sqlite:
            sqlite_path.unlink(missing_ok=True)
        return None, None, None


def _profile_row_base(
    *,
    mode: str,
    workload_id: str,
    config: Any,
    repetition: int,
    command: list[str],
    library: Path,
    manifest: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "mode": mode,
        "workload_id": workload_id,
        "configuration": config.name,
        "repetition": repetition,
        "label": config.label,
        "artifact_backend": config.artifact_backend,
        "window_size": config.window_size,
        "command": command,
        "library": str(library),
        "manifest": str(manifest),
    }


def _profile_mode_fields(
    *,
    mode: str,
    mutate_seconds: int,
    profile_seconds: int,
    application: list[str],
    broker_log: Path,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    if mode == "mutating":
        return {
            "mutate_seconds": mutate_seconds,
            "fixed_seed": 1,
            "mutating": result,
            "broker_command": application,
            "broker_log": str(broker_log),
        }
    return {"profile_seconds": profile_seconds, "benchmark": result}


def _capture_incomplete_row(
    *,
    workload_id: str,
    config: Any,
    repetition: int,
    stem: Path,
    command: list[str],
    library: Path,
    manifest: Path,
    mutate_seconds: int,
    application: list[str],
    failure: dict[str, Any],
    nsys: Path,
    keep_sqlite: bool,
    mode: str,
    profile_seconds: int,
    extract_partial_trace: bool = True,
) -> dict[str, Any]:
    profiling_path = stem.with_suffix(".profiling.jsonl")
    profile = stem.with_suffix(".nsys-rep")
    sqlite_path = stem.with_suffix(".sqlite")
    evidence_path = stem.with_suffix(".evidence.json")

    if extract_partial_trace:
        trace_summary, evidence_str, sqlite_str = _try_extract_trace(
            nsys=nsys,
            profile=profile,
            sqlite_path=sqlite_path,
            evidence_path=evidence_path,
            keep_sqlite=keep_sqlite,
        )
    else:
        trace_summary = None
        evidence_str = str(evidence_path) if evidence_path.exists() else None
        sqlite_str = (
            str(sqlite_path) if keep_sqlite and sqlite_path.exists() else None
        )

    profiling_record = None
    if profiling_path.exists():
        try:
            profiling_summary(load_profiling_records(profiling_path))
            profiling_record = str(profiling_path)
        except Exception:
            pass

    row: dict[str, Any] = {
        **_profile_row_base(
            mode=mode,
            workload_id=workload_id,
            config=config,
            repetition=repetition,
            command=command,
            library=library,
            manifest=manifest,
        ),
        "capture_incomplete": True,
        **failure,
        "cuda_api": (
            trace_summary["cuda_api"] if trace_summary else None
        ),
        "gpu_kernel_time_ns": (
            trace_summary["gpu_kernel_time_ns"] if trace_summary else None
        ),
        "gpu_mem_time_ns": (
            trace_summary["gpu_mem_time_ns"] if trace_summary else None
        ),
        "profiling_record": profiling_record,
        "nsys_report": str(profile) if profile.exists() else None,
        "trace_evidence": evidence_str,
        "sqlite_trace": sqlite_str,
        **_profile_mode_fields(
            mode=mode,
            mutate_seconds=mutate_seconds,
            profile_seconds=profile_seconds,
            application=application,
            broker_log=stem.with_suffix(".broker.log"),
            result=None,
        ),
    }
    return row


def run_profiles(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    rows_path = output / "profiles.jsonl"
    mutate = args.mutate
    mutate_seconds = args.mutate_seconds
    mode = "mutating" if mutate else "fixed"
    completed_keys = load_completed_profiles(rows_path, mode=mode)

    fuzzer = args.fuzzer.resolve()
    fuzzer_async = args.fuzzer_async.resolve()
    timing_libraries = _timing_libraries(
        args.timing_build_root.resolve() if args.timing_build_root else None
    )
    selected = set(args.workload or ())
    reports = _load_reports(args.build_root)
    for report_path, report in reports:
        workload_id = report["workload_id"]
        if selected and workload_id not in selected:
            continue
        manifest = rq1._manifest(report)
        for config in configurations(args.window_size):
            for repetition in range(1, args.repetitions + 1):
                key = profile_identity(workload_id, config.name, repetition)
                if key in completed_keys:
                    continue
                library = timing_libraries.get(
                    (workload_id, config.artifact_backend),
                    rq1._backend_library(
                        report_path, report, config.artifact_backend
                    ),
                )
                executable = fuzzer_async if config.async_frontend else fuzzer
                if mutate:
                    application = mutating_command(
                        executable=executable,
                        library=library,
                        manifest=manifest,
                        mutate_seconds=mutate_seconds,
                        window_size=config.window_size,
                    )
                else:
                    application = benchmark_command(
                        executable=executable,
                        library=library,
                        manifest=manifest,
                        warmup_runs=args.warmup_runs,
                        benchmark_seconds=args.profile_seconds,
                        window_size=config.window_size,
                    )
                stem = profile_stem(raw, *key)
                command = nsys_profile_command(
                    nsys=args.nsys.resolve(),
                    output=stem,
                    application=application,
                    finalize_on_stop=mutate,
                )
                profiling_path = stem.with_suffix(".profiling.jsonl")
                env = profile_environment(
                    os.environ.copy(),
                    gpu_device=args.gpu_device,
                    profiling_output=profiling_path,
                )
                if mutate:
                    env["RAPID_FIXED_SEED"] = "1"
                broker_pgid: int | None = None
                try:
                    if mutate:
                        _verify_cell_boundary(gpu_device=args.gpu_device)
                        workdir = stem.with_suffix(".workdir")
                        workdir.mkdir(exist_ok=True)
                        cell_timeout = mutating_cell_timeout(
                            mutate_seconds, base_timeout=args.timeout
                        )
                        process, broker_pgid = _run_mutating_profile(
                            command,
                            broker_command=application,
                            cwd=workdir,
                            env=env,
                            broker_log=stem.with_suffix(".broker.log"),
                            timeout=cell_timeout,
                            budget_seconds=mutate_seconds,
                        )
                    else:
                        process = _run_profile_command(
                            command, cwd=REPO_ROOT, env=env, timeout=args.timeout
                        )
                except CrashLoopTimeout as loop_err:
                    _teardown_cell(broker_pgid, gpu_device=args.gpu_device)
                    stem.with_suffix(".stdout.log").write_text(
                        loop_err.stdout, encoding="utf-8"
                    )
                    stem.with_suffix(".stderr.log").write_text(
                        loop_err.stderr, encoding="utf-8"
                    )
                    row = _capture_incomplete_row(
                        workload_id=workload_id,
                        config=config,
                        repetition=repetition,
                        stem=stem,
                        command=command,
                        library=library,
                        manifest=manifest,
                        mutate_seconds=mutate_seconds,
                        application=application,
                        failure={
                            "crash_loop": True,
                            "crash_loop_elapsed": loop_err.elapsed,
                        },
                        nsys=args.nsys.resolve(),
                        keep_sqlite=args.keep_sqlite,
                        mode=mode,
                        profile_seconds=args.profile_seconds,
                    )
                    with rows_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    completed_keys.add(key)
                    LOGGER.warning(
                        "crash loop detected: %s %s repetition=%d "
                        "(elapsed=%.1fs, budget=%.1fs)",
                        workload_id,
                        config.name,
                        repetition,
                        loop_err.elapsed,
                        loop_err.budget,
                    )
                    print(
                        f"profiled {workload_id} {config.name} "
                        f"repetition={repetition} [capture_incomplete]",
                        flush=True,
                    )
                    continue
                except subprocess.TimeoutExpired as error:
                    if mutate:
                        _teardown_cell(broker_pgid, gpu_device=args.gpu_device)
                    stem.with_suffix(".stdout.log").write_text(
                        rq1._timeout_output(error.stdout), encoding="utf-8"
                    )
                    stem.with_suffix(".stderr.log").write_text(
                        rq1._timeout_output(error.stderr), encoding="utf-8"
                    )
                    if not mutate:
                        raise
                    row = _capture_incomplete_row(
                        workload_id=workload_id,
                        config=config,
                        repetition=repetition,
                        stem=stem,
                        command=command,
                        library=library,
                        manifest=manifest,
                        mutate_seconds=mutate_seconds,
                        application=application,
                        failure={
                            "timeout_expired": True,
                            "timeout_budget": error.timeout,
                        },
                        nsys=args.nsys.resolve(),
                        keep_sqlite=args.keep_sqlite,
                        mode=mode,
                        profile_seconds=args.profile_seconds,
                    )
                    with rows_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    completed_keys.add(key)
                    LOGGER.warning(
                        "cell timeout: %s %s repetition=%d "
                        "(budget=%.1fs)",
                        workload_id,
                        config.name,
                        repetition,
                        error.timeout,
                    )
                    print(
                        f"profiled {workload_id} {config.name} "
                        f"repetition={repetition} [capture_incomplete:timeout]",
                        flush=True,
                    )
                    continue
                stem.with_suffix(".stdout.log").write_text(
                    process.stdout, encoding="utf-8"
                )
                stem.with_suffix(".stderr.log").write_text(
                    process.stderr, encoding="utf-8"
                )
                if process.returncode != 0:
                    raise RuntimeError(
                        f"nsys profile failed ({process.returncode}): {' '.join(command)}"
                    )
                run_result = (
                    parse_mutating_result(
                        process.stdout,
                        expected_seconds=mutate_seconds,
                        require_feedback_activity=config.feedback_enabled,
                    )
                    if mutate
                    else rq1.parse_benchmark_result(process.stdout)
                )
                profile = stem.with_suffix(".nsys-rep")
                sqlite_path = stem.with_suffix(".sqlite")
                evidence_path = stem.with_suffix(".evidence.json")
                try:
                    trace_summary, evidence_str, sqlite_str, profiling = (
                        _postprocess_capture(
                            nsys=args.nsys.resolve(),
                            profile=profile,
                            sqlite_path=sqlite_path,
                            evidence_path=evidence_path,
                            profiling_path=profiling_path,
                            keep_sqlite=args.keep_sqlite,
                        )
                    )
                except CapturePostprocessError as error:
                    if mutate and broker_pgid is not None:
                        _teardown_cell(broker_pgid, gpu_device=args.gpu_device)
                    row = _capture_incomplete_row(
                        workload_id=workload_id,
                        config=config,
                        repetition=repetition,
                        stem=stem,
                        command=command,
                        library=library,
                        manifest=manifest,
                        mutate_seconds=mutate_seconds,
                        application=application,
                        failure={
                            "postprocess_failed": True,
                            "postprocess_stage": error.stage,
                            "postprocess_error": error.detail,
                        },
                        nsys=args.nsys.resolve(),
                        keep_sqlite=args.keep_sqlite,
                        mode=mode,
                        profile_seconds=args.profile_seconds,
                        extract_partial_trace=False,
                    )
                    with rows_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    completed_keys.add(key)
                    LOGGER.warning(
                        "capture postprocessing failed: %s %s repetition=%d "
                        "stage=%s: %s",
                        workload_id,
                        config.name,
                        repetition,
                        error.stage,
                        error.detail,
                    )
                    print(
                        f"profiled {workload_id} {config.name} "
                        f"repetition={repetition} "
                        f"[capture_incomplete:{error.stage}]",
                        flush=True,
                    )
                    continue
                row: dict[str, Any] = {
                    **_profile_row_base(
                        mode=mode,
                        workload_id=workload_id,
                        config=config,
                        repetition=repetition,
                        command=command,
                        library=library,
                        manifest=manifest,
                    ),
                    "cuda_api": trace_summary["cuda_api"],
                    "gpu_kernel_time_ns": trace_summary["gpu_kernel_time_ns"],
                    "gpu_mem_time_ns": trace_summary["gpu_mem_time_ns"],
                    "profiling_record": str(profiling_path),
                    "nsys_report": str(profile),
                    "trace_evidence": evidence_str,
                    "sqlite_trace": sqlite_str,
                    **_profile_mode_fields(
                        mode=mode,
                        mutate_seconds=mutate_seconds,
                        profile_seconds=args.profile_seconds,
                        application=application,
                        broker_log=stem.with_suffix(".broker.log"),
                        result=run_result,
                    ),
                }
                if (
                    config.artifact_backend != "cufuzz"
                    and profiling["device_timing_cycles"] is None
                ):
                    raise RuntimeError(
                        f"missing device timing in {profiling_path}: "
                        f"{workload_id} {config.name}"
                    )
                with rows_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                completed_keys.add(key)
                if mutate and broker_pgid is not None:
                    _teardown_cell(broker_pgid, gpu_device=args.gpu_device)
                print(
                    f"profiled {workload_id} {config.name} repetition={repetition}",
                    flush=True,
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, action="append", required=True)
    parser.add_argument("--timing-build-root", type=Path)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-seconds", type=int, default=10)
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument(
        "--mutate-seconds", type=int, default=DEFAULT_MUTATING_SECONDS
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--keep-sqlite", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--timeout", type=int, default=DEFAULT_CELL_TIMEOUT)
    parser.add_argument("--nsys", type=Path, default=Path("/usr/local/bin/nsys"))
    parser.add_argument(
        "--fuzzer", type=Path, default=REPO_ROOT / "cuda-fuzzer/target/release/fuzzer"
    )
    parser.add_argument(
        "--fuzzer-async",
        type=Path,
        default=REPO_ROOT / "cuda-fuzzer/target/release/fuzzer_async",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in (
        "profile_seconds",
        "mutate_seconds",
        "repetitions",
        "warmup_runs",
        "timeout",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.window_size <= 32:
        raise SystemExit("--window-size must be in 1..=32")
    run_profiles(args)


if __name__ == "__main__":
    main()
