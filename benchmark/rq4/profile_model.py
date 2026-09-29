"""Stable identities, commands, and result contracts for RQ4 profiles."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


API_CATEGORIES = (
    "allocation",
    "free",
    "memcpy",
    "launch",
    "synchronization",
    "other",
)
MUTATING_RESULT_PREFIX = "RAPID_MUTATING_RESULT "
MUTATING_RESULT_FIELDS = frozenset(
    {
        "requested_seconds",
        "executions",
        "corpus_size",
        "solutions",
        "mutation_calls",
        "coverage_nonzero_bytes",
        "simt_memcov_nonzero_bits",
        "pending",
        "completed",
        "outstanding",
        "in_flight",
        "queued_submissions",
    }
)
MUTATING_QUEUE_FIELDS = (
    "pending",
    "completed",
    "outstanding",
    "in_flight",
    "queued_submissions",
)
LOGGER = logging.getLogger("benchmark.rq4.profile")


def nsys_profile_command(
    *,
    nsys: Path,
    output: Path,
    application: list[str],
    capture_range: bool = True,
    finalize_on_stop: bool = False,
) -> list[str]:
    command = [
        str(nsys),
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
    ]
    if capture_range:
        capture_end = "stop-shutdown" if finalize_on_stop else "stop"
        command.extend(
            ["--capture-range=cudaProfilerApi", f"--capture-range-end={capture_end}"]
        )
        if finalize_on_stop:
            command.append("--kill=none")
    return [*command, "--output", str(output), *application]


def nsys_export_command(
    *, nsys: Path, report: Path, sqlite_path: Path
) -> list[str]:
    return [
        str(nsys),
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(report),
    ]


def profile_identity(
    workload_id: str, configuration: str, repetition: int
) -> tuple[str, str, int]:
    return workload_id, configuration, repetition


def validate_mutating_result_fields(
    result: Any, *, context: str
) -> dict[str, int]:
    actual = set(result) if isinstance(result, dict) else set()
    if not isinstance(result, dict) or actual != MUTATING_RESULT_FIELDS:
        missing = sorted(MUTATING_RESULT_FIELDS - actual)
        extra = sorted(actual - MUTATING_RESULT_FIELDS)
        raise RuntimeError(
            f"incomplete {context}: missing={missing} extra={extra}"
        )
    return {name: int(result[name]) for name in MUTATING_RESULT_FIELDS}


def validate_mutating_result(
    result: Any,
    *,
    expected_seconds: int,
    require_feedback_activity: bool,
    allow_no_mutations: bool = False,
    context: str,
    logger: logging.Logger = LOGGER,
) -> dict[str, int]:
    parsed = validate_mutating_result_fields(result, context=context)
    if parsed["requested_seconds"] != expected_seconds:
        raise RuntimeError(
            f"{context} requested_seconds mismatch: expected={expected_seconds} "
            f"actual={parsed['requested_seconds']}"
        )
    if parsed["executions"] <= 0:
        raise RuntimeError(f"{context} completed no executions")
    if parsed["mutation_calls"] <= 0 and not allow_no_mutations:
        raise RuntimeError(f"{context} performed no mutations")
    dirty = {name: parsed[name] for name in MUTATING_QUEUE_FIELDS if parsed[name] != 0}
    if dirty:
        raise RuntimeError(f"{context} did not drain cleanly: {dirty}")
    coverage_activity = (
        parsed["coverage_nonzero_bytes"] + parsed["simt_memcov_nonzero_bits"]
    )
    if require_feedback_activity and coverage_activity <= 0:
        raise RuntimeError(f"{context} feedback produced no coverage")
    if require_feedback_activity and parsed["corpus_size"] <= 1:
        logger.warning("%s corpus did not grow beyond the initial seed", context)
    return parsed


def profile_stem(
    raw: Path, workload_id: str, configuration: str, repetition: int
) -> Path:
    return raw / f"{workload_id}-{configuration}-r{repetition}"


def profile_environment(
    base: dict[str, str], *, gpu_device: str, profiling_output: Path
) -> dict[str, str]:
    child = base.copy()
    child["CUDA_VISIBLE_DEVICES"] = gpu_device
    child["RAPID_PROFILE"] = "1"
    child["RAPID_PROFILE_OUTPUT"] = str(profiling_output)
    return child


def mutating_command(
    *,
    executable: str | Path,
    library: str | Path,
    manifest: str | Path,
    mutate_seconds: int,
    window_size: int | None,
) -> list[str]:
    command = [
        str(executable),
        str(library),
        "--manifest",
        str(manifest),
        "--mutate-seconds",
        str(mutate_seconds),
    ]
    if window_size is not None:
        command.extend(["--window-size", str(window_size)])
    return command


def load_completed_profiles(
    path: Path, *, mode: str
) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") != mode:
            raise RuntimeError(
                f"profile output mode mismatch: expected={mode} "
                f"actual={row.get('mode', '<missing>')}"
            )
        if mode == "mutating" and not row.get("capture_incomplete"):
            validate_mutating_result_fields(
                row.get("mutating"),
                context=f"mutating profile {row['workload_id']} {row['configuration']}",
            )
        completed.add(
            profile_identity(
                row["workload_id"],
                row["configuration"],
                int(row.get("repetition", 1)),
            )
        )
    return completed


def parse_mutating_result(
    output: str, *, expected_seconds: int, require_feedback_activity: bool = True
) -> dict[str, int]:
    lines = [line for line in output.splitlines() if line.startswith(MUTATING_RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one mutating result, got {len(lines)}")
    result = json.loads(lines[0][len(MUTATING_RESULT_PREFIX) :])
    solutions = int(result.get("solutions", 0)) if isinstance(result, dict) else 0
    return validate_mutating_result(
        result,
        expected_seconds=expected_seconds,
        require_feedback_activity=require_feedback_activity,
        allow_no_mutations=solutions > 0,
        context="mutating run",
    )


def timing_libraries(root: Path | None) -> dict[tuple[str, str], Path]:
    if root is None:
        return {}
    summary = json.loads((root / "build_summary.json").read_text(encoding="utf-8"))
    libraries: dict[tuple[str, str], Path] = {}
    for workload in summary["workloads"]:
        for backend, record in workload["backends"].items():
            libraries[(workload["workload_id"], backend)] = Path(
                record["shared_library"]
            ).resolve()
    return libraries
