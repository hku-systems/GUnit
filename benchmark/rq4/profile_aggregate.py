#!/usr/bin/env python3
"""Validate RQ4 profiles and export lane-specific breakdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from benchmark.rq4.profile import (
    DEVICE_SEGMENTS,
    load_profiling_records,
    profiling_summary,
    validate_mutating_result,
)
from benchmark.rq4.schema import DEFAULT_WINDOW_SIZE, configurations as rq4_configurations


RQ4_ROOT = Path(__file__).resolve().parent
TIMED_CONFIGS = ("libafl", "libafl-plus", "gunit-sync", "gunit")
PRIMARY_CONFIGS = ("cufuzz", "libafl", "libafl-plus", "gunit-sync", "gunit")
LANE_ORDER = (
    "CuFuzz",
    "LibAFL",
    "LibAFL+",
    "GUnit-s-GPU",
    "GUnit-s-CPU",
    "GUnit-GPU",
    "GUnit-Coll",
    "GUnit-Disp",
)
LANES_BY_CONFIG = {
    "cufuzz": ("CuFuzz",),
    "libafl": ("LibAFL",),
    "libafl-plus": ("LibAFL+",),
    "gunit-sync": ("GUnit-s-GPU", "GUnit-s-CPU"),
    "gunit": ("GUnit-GPU", "GUnit-Coll", "GUnit-Disp"),
}
CATEGORIES = (
    "kernel_exec",
    "feedback",
    "gpu_overhead",
    "memcpy",
    "free",
    "launch",
    "idle",
    "allocation",
)
DEVICE_BUCKETS = DEVICE_SEGMENTS
PROFILE_WINDOW_RELATIVE_TOLERANCE = 0.05
LOGGER = logging.getLogger(__name__)


def _profiling(profile: dict[str, Any]) -> dict[str, Any]:
    inline = profile.get("profiling_records_data")
    if isinstance(inline, list):
        return profiling_summary(inline)
    path = profile.get("profiling_record")
    if not path:
        raise RuntimeError(
            f"missing unified profiling record: {profile['workload_id']} "
            f"{profile['configuration']}"
        )
    return profiling_summary(load_profiling_records(Path(path)))


def _elapsed_ns(profile: dict[str, Any]) -> int:
    if profile["mode"] == "fixed":
        benchmark = profile.get("benchmark")
        if not isinstance(benchmark, dict):
            raise RuntimeError(
                f"fixed profile is missing benchmark elapsed time: "
                f"{profile['workload_id']} {profile['configuration']}"
            )
        return int(benchmark["elapsed_ns"])
    return int(_profiling(profile)["wall_time_ns"])


def _profile_markers(profile: dict[str, Any]) -> dict[str, int | bool | str]:
    result = profile.get("mutating")
    solutions = int(result.get("solutions", 0)) if isinstance(result, dict) else 0
    capture_incomplete = bool(profile.get("capture_incomplete"))
    has_crashes = solutions > 0
    if capture_incomplete:
        status = "capture_incomplete"
    elif has_crashes:
        status = "crash"
    else:
        status = "complete"
    return {
        "capture_incomplete": capture_incomplete,
        "solutions": solutions,
        "has_crashes": has_crashes,
        "status": status,
    }


def _profile_is_unavailable(profile: dict[str, Any]) -> bool:
    return _profile_markers(profile)["status"] != "complete"


def _validate_mutating_result(
    profile: dict[str, Any], *, require_feedback_activity: bool
) -> None:
    if profile.get("fixed_seed") != 1:
        raise RuntimeError(
            f"mutating profiles require RAPID_FIXED_SEED=1: "
            f"{profile['workload_id']} {profile['configuration']}"
        )
    result = profile.get("mutating")
    if not isinstance(result, dict):
        raise RuntimeError(
            f"mutating profile is missing run statistics: "
            f"{profile['workload_id']} {profile['configuration']}"
        )
    requested_seconds = profile.get("mutate_seconds")
    if not isinstance(requested_seconds, int) or requested_seconds <= 0:
        raise RuntimeError(
            f"mutating profile duration mismatch: "
            f"{profile['workload_id']} {profile['configuration']}"
        )
    validate_mutating_result(
        result,
        expected_seconds=requested_seconds,
        require_feedback_activity=require_feedback_activity,
        allow_no_mutations=bool(_profile_markers(profile)["has_crashes"]),
        context=(
            f"mutating profile {profile['workload_id']} "
            f"{profile['configuration']}"
        ),
        logger=LOGGER,
    )


def validate_profiles(profiles: list[dict[str, Any]], workloads: list[str]) -> None:
    modes = {row.get("mode") for row in profiles}
    if None in modes or not modes <= {"fixed", "mutating"}:
        raise RuntimeError(f"unsupported or missing profile mode: {sorted(map(str, modes))}")
    if len(modes) != 1:
        raise RuntimeError(f"profile modes must not be mixed: {sorted(modes)}")
    configurations_by_name = {
        config.name: config for config in rq4_configurations(DEFAULT_WINDOW_SIZE)
    }
    configured = list(configurations_by_name)
    repetitions = sorted({int(row.get("repetition", 1)) for row in profiles})
    if repetitions != list(range(1, len(repetitions) + 1)):
        raise RuntimeError(f"profile repetitions must be contiguous from 1: {repetitions}")
    expected = {
        (workload, config, repetition)
        for workload in workloads
        for config in configured
        for repetition in repetitions
    }
    actual = {
        (
            row["workload_id"],
            row["configuration"],
            int(row.get("repetition", 1)),
        )
        for row in profiles
        if row["configuration"] in configured
    }
    unexpected = sorted(
        {
            row["configuration"]
            for row in profiles
            if row["configuration"] not in configured
        }
    )
    primary_count = sum(row["configuration"] in configured for row in profiles)
    if actual != expected or len(actual) != primary_count or unexpected:
        raise RuntimeError(
            f"incomplete profile matrix: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)} unexpected={unexpected}"
    )
    for row in profiles:
        configuration = configurations_by_name[row["configuration"]]
        if (
            "window_size" not in row
            or row["window_size"] != configuration.window_size
        ):
            raise RuntimeError(
                f"profile window_size mismatch: {row['workload_id']} "
                f"{row['configuration']} expected={configuration.window_size} "
                f"actual={row.get('window_size', '<missing>')}"
            )
        if _profile_is_unavailable(row):
            continue
        if row["mode"] == "mutating":
            _validate_mutating_result(
                row, require_feedback_activity=configuration.feedback_enabled
            )
        evidence = _trace_evidence(row)
        trace_window_ns = int(evidence["capture_end_ns"]) - int(
            evidence["capture_start_ns"]
        )
        elapsed_ns = _elapsed_ns(row)
        if row["mode"] == "mutating":
            requested_ns = int(row["mutate_seconds"]) * 1_000_000_000
            minimum_ns = int(
                requested_ns * (1.0 - PROFILE_WINDOW_RELATIVE_TOLERANCE)
            )
            if elapsed_ns < minimum_ns:
                raise RuntimeError(
                    f"mutating profile ended before requested duration: "
                    f"{row['workload_id']} {row['configuration']} "
                    f"elapsed_ns={elapsed_ns} requested_ns={requested_ns}"
                )
        if elapsed_ns <= 0:
            raise RuntimeError(
                f"non-positive profile wall time: {row['workload_id']} "
                f"{row['configuration']} elapsed_ns={elapsed_ns}"
            )
        if trace_window_ns <= 0:
            raise RuntimeError(
                f"non-positive trace window: {row['workload_id']} "
                f"{row['configuration']} trace_window_ns={trace_window_ns}"
            )
        relative_error = abs(elapsed_ns - trace_window_ns) / trace_window_ns
        if relative_error > PROFILE_WINDOW_RELATIVE_TOLERANCE:
            raise RuntimeError(
                f"profile window mismatch: {row['workload_id']} "
                f"{row['configuration']} elapsed_ns={elapsed_ns} "
                f"trace_window_ns={trace_window_ns} "
                f"relative_error={relative_error:.6f}"
            )
        measurements = _profiling(row)
        cpu_feedback = measurements["cpu_feedback_timing_ns"]
        if not isinstance(cpu_feedback, dict) or int(cpu_feedback.get("total_ns", -1)) < 0:
            raise RuntimeError(
                f"missing CPU feedback timing: {row['workload_id']} "
                f"{row['configuration']}"
            )
        if row["configuration"] in TIMED_CONFIGS:
            timing = measurements["device_timing_cycles"]
            if not isinstance(timing, dict) or int(timing.get("iterations", 0)) <= 0:
                raise RuntimeError(
                    f"missing device timing: {row['workload_id']} "
                    f"{row['configuration']}"
                )


def _trace_evidence(profile: dict[str, Any]) -> dict[str, Any]:
    inline = profile.get("trace_evidence_data")
    if isinstance(inline, dict):
        return inline
    path = profile.get("trace_evidence")
    if not path:
        raise RuntimeError(
            f"missing trace evidence: {profile['workload_id']} "
            f"{profile['configuration']}"
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _kernel_time_ns(evidence: dict[str, Any]) -> int:
    total = 0
    for kernel in evidence.get("kernels", []):
        start = int(kernel["start_ns"])
        end = int(kernel["end_ns"])
        if end < start:
            raise RuntimeError(f"invalid GPU kernel interval: {kernel}")
        total += end - start
    return total


def _thread_name(evidence: dict[str, Any], prefix: str) -> str:
    names = sorted(
        name
        for name in evidence.get("cuda_api_time_ns", {})
        if name == prefix or name.startswith(prefix)
    )
    if len(names) != 1:
        raise RuntimeError(
            f"expected exactly one CUDA API thread matching {prefix!r}, got {names}"
        )
    return names[0]


def _lane_rows(
    profile: dict[str, Any],
    *,
    lane: str,
    active: dict[str, int],
    denominator: int,
    unit: str,
    measurement_domain: str,
    measured_idle: int | None = None,
) -> list[dict[str, Any]]:
    if denominator <= 0:
        raise RuntimeError(f"non-positive denominator for {lane}: {denominator}")
    active_total = sum(int(active.get(category, 0)) for category in CATEGORIES)
    if measured_idle is None:
        idle = denominator - active_total
    else:
        idle = int(measured_idle)
    if idle < 0:
        raise RuntimeError(
            f"negative residual Idle for {lane}: active={active_total}, "
            f"denominator={denominator}"
        )
    values = {category: int(active.get(category, 0)) for category in CATEGORIES}
    values["idle"] = idle
    rows = [
        {
            "mode": profile["mode"],
            "workload_id": profile["workload_id"],
            "configuration": profile["configuration"],
            "lane": lane,
            "repetition": int(profile.get("repetition", 1)),
            **_profile_markers(profile),
            "category": category,
            "value": values[category],
            "unit": unit,
            "share": values[category] / denominator,
            "measurement_domain": measurement_domain,
        }
        for category in CATEGORIES
    ]
    if abs(sum(row["share"] for row in rows) - 1.0) > 1e-9:
        raise RuntimeError(f"lane shares do not sum to one: {lane}")
    return rows


def _serial_lane(profile: dict[str, Any], lane: str) -> list[dict[str, Any]]:
    evidence = _trace_evidence(profile)
    api_totals = {category: 0 for category in CATEGORIES}
    for thread in evidence.get("cuda_api_time_ns", {}).values():
        for category in ("allocation", "free", "memcpy", "launch"):
            api_totals[category] += int(thread.get(category, 0))
    kernel_time = _kernel_time_ns(evidence)
    if kernel_time <= 0:
        raise RuntimeError(
            f"missing target kernel activity: {profile['workload_id']} "
            f"{profile['configuration']}"
        )
    measurements = _profiling(profile)
    cpu_feedback = int(measurements["cpu_feedback_timing_ns"]["total_ns"])
    timing = measurements["device_timing_cycles"]
    if isinstance(timing, dict) and int(timing.get("iterations", 0)) > 0:
        timed_total = sum(
            int(timing.get(bucket, 0))
            for bucket in DEVICE_BUCKETS
            if bucket != "idle"
        )
        if timed_total <= 0:
            raise RuntimeError(
                f"empty serial device timing: {profile['workload_id']} "
                f"{profile['configuration']}"
            )
        def scaled(*buckets: str) -> int:
            cycles = sum(int(timing.get(bucket, 0)) for bucket in buckets)
            return round(kernel_time * cycles / timed_total)

        api_totals["kernel_exec"] = scaled("target_execution")
        api_totals["feedback"] = cpu_feedback + scaled(
            "feedback_init", "feedback_prepare", "feedback_merge"
        )
        api_totals["gpu_overhead"] = scaled(
            "input_decode", "signal", "bookkeeping"
        )
        rounding = kernel_time - (
            api_totals["kernel_exec"]
            + api_totals["feedback"]
            - cpu_feedback
            + api_totals["gpu_overhead"]
        )
        api_totals["gpu_overhead"] += rounding
    else:
        api_totals["kernel_exec"] = kernel_time
        api_totals["feedback"] = cpu_feedback
    return _lane_rows(
        profile,
        lane=lane,
        active=api_totals,
        denominator=_elapsed_ns(profile),
        unit="ns",
        measurement_domain="serial_critical_path",
    )


def _persistent_device_lane(
    profile: dict[str, Any], lane: str
) -> list[dict[str, Any]]:
    timing = _profiling(profile)["device_timing_cycles"]
    if not isinstance(timing, dict) or int(timing.get("iterations", 0)) <= 0:
        raise RuntimeError(
            f"missing device timing: {profile['workload_id']} "
            f"{profile['configuration']}"
        )
    kernel = int(timing.get("target_execution", 0))
    feedback = sum(
        int(timing.get(bucket, 0))
        for bucket in ("feedback_init", "feedback_prepare", "feedback_merge")
    )
    overhead = sum(
        int(timing.get(bucket, 0))
        for bucket in ("input_decode", "signal", "bookkeeping")
    )
    idle = int(timing.get("idle", 0))
    denominator = kernel + feedback + overhead + idle
    return _lane_rows(
        profile,
        lane=lane,
        active={
            "kernel_exec": kernel,
            "feedback": feedback,
            "gpu_overhead": overhead,
        },
        denominator=denominator,
        unit="cycles",
        measurement_domain="persistent_device_cycles",
        measured_idle=idle,
    )


def _host_thread_lane(
    profile: dict[str, Any], lane: str, prefix: str
) -> list[dict[str, Any]]:
    evidence = _trace_evidence(profile)
    thread = _thread_name(evidence, prefix)
    memcpy = int(evidence["cuda_api_time_ns"][thread].get("memcpy", 0))
    if memcpy <= 0:
        raise RuntimeError(f"missing DMemcpy activity for {lane} ({thread})")
    return _lane_rows(
        profile,
        lane=lane,
        active={"memcpy": memcpy},
        denominator=_elapsed_ns(profile),
        unit="ns",
        measurement_domain="host_thread_time",
    )


def overhead_lanes(profile: dict[str, Any]) -> list[dict[str, Any]]:
    configuration = profile["configuration"]
    if configuration == "cufuzz":
        return _serial_lane(profile, "CuFuzz")
    if configuration == "libafl":
        return _serial_lane(profile, "LibAFL")
    if configuration == "libafl-plus":
        return _serial_lane(profile, "LibAFL+")
    if configuration == "gunit-sync":
        return [
            *_persistent_device_lane(profile, "GUnit-s-GPU"),
            *_host_thread_lane(profile, "GUnit-s-CPU", "RAPID-Sync"),
        ]
    if configuration == "gunit":
        return [
            *_persistent_device_lane(profile, "GUnit-GPU"),
            *_host_thread_lane(profile, "GUnit-Coll", "RAPID2-Coll-"),
            *_host_thread_lane(profile, "GUnit-Disp", "RAPID2-Disp-"),
        ]
    raise RuntimeError(f"unsupported profile configuration: {configuration}")


def _overhead_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    if not _profile_is_unavailable(profile):
        return overhead_lanes(profile)
    return [
        {
            "mode": profile["mode"],
            "workload_id": profile["workload_id"],
            "configuration": profile["configuration"],
            "lane": lane,
            "repetition": int(profile.get("repetition", 1)),
            **_profile_markers(profile),
            "category": "",
            "value": "",
            "unit": "",
            "share": "",
            "measurement_domain": "",
        }
        for lane in LANES_BY_CONFIG[profile["configuration"]]
    ]


def validate_pilot(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    workloads = sorted({profile["workload_id"] for profile in profiles})
    if len(workloads) != 1:
        raise RuntimeError(f"pilot requires exactly one workload: {workloads}")
    validate_profiles(profiles, workloads)
    for profile in profiles:
        if profile["configuration"] in ("gunit-sync", "gunit"):
            timing = _profiling(profile)["device_timing_cycles"]
            has_target_activity = int(timing.get("target_execution", 0)) > 0
        else:
            has_target_activity = _kernel_time_ns(_trace_evidence(profile)) > 0
        if not has_target_activity:
            raise RuntimeError(
                f"pilot has no target kernel activity: {profile['configuration']}"
            )
    lane_rows = [row for profile in profiles for row in overhead_lanes(profile)]
    lanes = {row["lane"] for row in lane_rows}
    if lanes != set(LANE_ORDER):
        raise RuntimeError(
            f"pilot does not contain exactly eight lanes: {sorted(lanes)}"
        )
    for lane in lanes:
        rows = [row for row in lane_rows if row["lane"] == lane]
        if abs(sum(float(row["share"]) for row in rows) - 1.0) > 1e-9:
            raise RuntimeError(f"pilot lane shares do not sum to one: {lane}")

    gunit = next(
        profile for profile in profiles if profile["configuration"] == "gunit"
    )
    thread_names = _trace_evidence(gunit).get("thread_names", {})
    coll_ids = [
        thread_id
        for thread_id, name in thread_names.items()
        if str(name).startswith("RAPID2-Coll-")
    ]
    disp_ids = [
        thread_id
        for thread_id, name in thread_names.items()
        if str(name).startswith("RAPID2-Disp-")
    ]
    if len(coll_ids) != 1 or len(disp_ids) != 1 or coll_ids[0] == disp_ids[0]:
        raise RuntimeError(
            f"pilot Coll/Disp thread identity is invalid: "
            f"coll={coll_ids}, disp={disp_ids}"
        )
    return {
        "workload_id": workloads[0],
        "profiles": sum(
            profile["configuration"] in PRIMARY_CONFIGS for profile in profiles
        ),
        "lanes": len(lanes),
        "coll_thread_id": coll_ids[0],
        "disp_thread_id": disp_ids[0],
    }


def persistent_breakdown(profile: dict[str, Any]) -> list[dict[str, Any]]:
    timing = _profiling(profile)["device_timing_cycles"]
    if not isinstance(timing, dict):
        return []
    total = sum(int(timing.get(bucket, 0)) for bucket in DEVICE_BUCKETS)
    if total <= 0:
        raise RuntimeError(
            f"empty device timing: {profile['workload_id']} {profile['configuration']}"
        )
    return [
        {
            "mode": profile["mode"],
            "workload_id": profile["workload_id"],
            "configuration": profile["configuration"],
            "repetition": int(profile.get("repetition", 1)),
            **_profile_markers(profile),
            "iterations": int(timing["iterations"]),
            "bucket": bucket,
            "cycles": int(timing.get(bucket, 0)),
            "share": int(timing.get(bucket, 0)) / total,
        }
        for bucket in DEVICE_BUCKETS
    ]


def cuda_api_breakdown(profile: dict[str, Any]) -> list[dict[str, Any]]:
    categories = profile["cuda_api"]
    total = sum(int(value["time_ns"]) for value in categories.values())
    return [
        {
            "mode": profile["mode"],
            "workload_id": profile["workload_id"],
            "configuration": profile["configuration"],
            "repetition": int(profile.get("repetition", 1)),
            **_profile_markers(profile),
            "category": category,
            "time_ns": int(value["time_ns"]),
            "calls": int(value["calls"]),
            "share_of_cuda_api_time": (
                int(value["time_ns"]) / total if total > 0 else 0.0
            ),
        }
        for category, value in categories.items()
    ]


def gpu_activity(profile: dict[str, Any]) -> dict[str, Any]:
    elapsed = _elapsed_ns(profile)
    kernels = int(profile["gpu_kernel_time_ns"])
    memory = int(profile["gpu_mem_time_ns"])
    residual = elapsed - kernels - memory
    if residual < 0:
        raise RuntimeError(
            f"negative residual GPU activity: {profile['workload_id']} "
            f"{profile['configuration']} residual={residual}"
        )
    return {
        "mode": profile["mode"],
        "workload_id": profile["workload_id"],
        "configuration": profile["configuration"],
        "repetition": int(profile.get("repetition", 1)),
        **_profile_markers(profile),
        "elapsed_ns": elapsed,
        "gpu_kernel_time_ns": kernels,
        "gpu_mem_time_ns": memory,
        "untraced_wall_time_ns": residual,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty profile table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(input_dir: Path, output: Path, suite_path: Path) -> None:
    workloads = json.loads(suite_path.read_text(encoding="utf-8"))["workloads"]
    profiles = [
        json.loads(line)
        for line in (input_dir / "profiles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_profiles(profiles, workloads)
    output.mkdir(parents=True, exist_ok=True)
    valid_profiles = [
        profile for profile in profiles if not _profile_is_unavailable(profile)
    ]
    _write(
        output / "rq4_cuda_api.csv",
        [row for profile in valid_profiles for row in cuda_api_breakdown(profile)],
    )
    _write(
        output / "rq4_gpu_activity.csv",
        [gpu_activity(profile) for profile in valid_profiles],
    )
    _write(
        output / "rq4_persistent_device.csv",
        [row for profile in valid_profiles for row in persistent_breakdown(profile)],
    )
    overhead_rows = [
        row
        for profile in profiles
        for row in _overhead_rows(profile)
        if profile["configuration"] in PRIMARY_CONFIGS
    ]
    repetitions = {int(profile.get("repetition", 1)) for profile in profiles}
    expected_lanes = {
        (workload, repetition, lane)
        for workload in workloads
        for repetition in repetitions
        for lane in LANE_ORDER
    }
    actual_lanes = {
        (row["workload_id"], int(row["repetition"]), row["lane"])
        for row in overhead_rows
    }
    if actual_lanes != expected_lanes:
        raise RuntimeError(
            f"eight-lane matrix is incomplete: "
            f"missing={sorted(expected_lanes - actual_lanes)} "
            f"extra={sorted(actual_lanes - expected_lanes)}"
        )
    _write(output / "rq4_overhead_breakdown.csv", overhead_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=RQ4_ROOT / "suite.json")
    return parser


def main() -> None:
    args = _parser().parse_args()
    aggregate(args.input, args.output, args.suite)


if __name__ == "__main__":
    main()
