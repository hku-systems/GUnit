#!/usr/bin/env python3
"""Recompute the RQ4 GUnit idle-attribution table from profile JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGETS = (
    "apex_maybe_cast",
    "shoc_reduction",
    "shoc_radix_sort",
    "shoc_scan",
    "cutlass_gemm",
    "flashattention_device1xn",
    "apex_index_mul_2d_vgeo",
    "cuda_samples_inverse_cnd",
)
CONTROLS = (
    "synth_complex",
    "pytorch_batchnorm",
    "llama_upscale_f32_bilinear",
)
DEVICE_SEGMENTS = (
    "idle",
    "feedback_init",
    "input_decode",
    "feedback_prepare",
    "target_execution",
    "feedback_merge",
    "signal",
    "bookkeeping",
)
FEEDBACK_SEGMENTS = ("feedback_init", "feedback_prepare", "feedback_merge")
DISPATCH_SEGMENTS = ("input_decode", "signal", "bookkeeping")
MAIN_LOOP_SEGMENTS = (
    "submit",
    "poll",
    "coverage",
    "evaluate",
    "release",
    "scheduler_stage",
    "other_idle",
)
ATTRIBUTION = {
    "apex_maybe_cast": "gpu-dispatch-bound; essential-light",
    "shoc_reduction": "gpu-dispatch-bound; essential-light",
    "shoc_radix_sort": "gpu-dispatch-bound; essential-light",
    "shoc_scan": "gpu-dispatch-bound; essential-light",
    "cutlass_gemm": "gpu-dispatch-bound + device feedback-init hotspot",
    "flashattention_device1xn": "gpu-dispatch-bound; essential-light",
    "apex_index_mul_2d_vgeo": "gpu-dispatch-bound; essential-light",
    "cuda_samples_inverse_cnd": "gpu-dispatch-bound; essential-light",
    "synth_complex": "true-device-bound control",
    "pytorch_batchnorm": "true-device-bound control",
    "llama_upscale_f32_bilinear": "balanced/device-bound control",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_record_path(result_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    fallback = result_dir / "raw" / path.name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(path)


def compute(profile: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    records = load_jsonl(resolve_record_path(result_dir, profile["profiling_record"]))
    by_key = {(row["domain"], row["segment"]): row for row in records}
    missing = [
        key
        for key in (
            *(("device_kernel", segment) for segment in DEVICE_SEGMENTS),
            *(("main_loop", segment) for segment in MAIN_LOOP_SEGMENTS),
            ("cpu_feedback", "predicate"),
            ("cpu_feedback", "metadata"),
        )
        if key not in by_key
    ]
    if missing:
        raise RuntimeError(f"{profile['workload_id']}: missing records {missing}")

    iterations = int(by_key[("device_kernel", "idle")]["count"])
    if iterations <= 0:
        raise RuntimeError(f"{profile['workload_id']}: non-positive iterations")
    device_counts = {
        int(by_key[("device_kernel", segment)]["count"])
        for segment in DEVICE_SEGMENTS
    }
    if device_counts != {iterations}:
        raise RuntimeError(
            f"{profile['workload_id']}: inconsistent device counts {device_counts}"
        )
    device_cycles = {
        segment: int(by_key[("device_kernel", segment)]["total"])
        for segment in DEVICE_SEGMENTS
    }
    total_cycles = sum(device_cycles.values())
    wall_ns = sum(
        int(row["total"])
        for row in records
        if row["domain"] in ("main_loop", "cpu_feedback")
    )
    wall_us_iter = wall_ns / iterations / 1_000

    def device_share(*segments: str) -> float:
        return sum(device_cycles[segment] for segment in segments) / total_cycles

    def device_us(*segments: str) -> float:
        return wall_us_iter * device_share(*segments)

    host = {
        segment: {
            "count": int(by_key[("main_loop", segment)]["count"]),
            "total_ns": int(by_key[("main_loop", segment)]["total"]),
        }
        for segment in MAIN_LOOP_SEGMENTS
    }
    for segment in ("coverage", "evaluate", "scheduler_stage"):
        if host[segment]["count"] != iterations:
            raise RuntimeError(
                f"{profile['workload_id']}: {segment} count "
                f"{host[segment]['count']} != {iterations}"
            )

    predicate = by_key[("cpu_feedback", "predicate")]
    metadata = by_key[("cpu_feedback", "metadata")]
    if int(predicate["count"]) != int(metadata["count"]):
        raise RuntimeError(f"{profile['workload_id']}: CPU feedback count mismatch")
    cpu_feedback_calls = int(predicate["count"])
    cpu_feedback_total_ns = int(predicate["total"]) + int(metadata["total"])
    cuda_api = profile["cuda_api"]

    return {
        "workload": profile["workload_id"],
        "iterations": iterations,
        "wall_us_iter": wall_us_iter,
        "idle_us": device_us("idle"),
        "idle_pct": 100 * device_share("idle"),
        "target_us": device_us("target_execution"),
        "target_pct": 100 * device_share("target_execution"),
        "feedback_us": device_us(*FEEDBACK_SEGMENTS),
        "feedback_pct": 100 * device_share(*FEEDBACK_SEGMENTS),
        "feedback_init_us": device_us("feedback_init"),
        "feedback_init_pct": 100 * device_share("feedback_init"),
        "feedback_prepare_us": device_us("feedback_prepare"),
        "feedback_prepare_pct": 100 * device_share("feedback_prepare"),
        "feedback_merge_us": device_us("feedback_merge"),
        "feedback_merge_pct": 100 * device_share("feedback_merge"),
        "dispatch_us": device_us(*DISPATCH_SEGMENTS),
        "dispatch_pct": 100 * device_share(*DISPATCH_SEGMENTS),
        "host": host,
        "cpu_feedback_calls": cpu_feedback_calls,
        "cpu_feedback_ns_call": (
            cpu_feedback_total_ns / cpu_feedback_calls if cpu_feedback_calls else None
        ),
        "cpu_feedback_ns_iter": cpu_feedback_total_ns / iterations,
        "memcpy_sync_us_iter": (
            int(cuda_api["memcpy"]["time_ns"])
            + int(cuda_api["synchronization"]["time_ns"])
        )
        / iterations
        / 1_000,
        "memcpy_calls_iter": int(cuda_api["memcpy"]["calls"]) / iterations,
        "sync_calls_iter": int(cuda_api["synchronization"]["calls"]) / iterations,
        "launch_calls": int(cuda_api["launch"]["calls"]),
    }


def host_ns_call(row: dict[str, Any], segment: str) -> str:
    value = row["host"][segment]
    if value["count"] == 0:
        return "n/a"
    return f"{value['total_ns'] / value['count']:.0f}"


def fmt_device(us: float, pct: float) -> str:
    return f"{us:.2f} ({pct:.1f}%)"


def print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "| kernel | device idle us/iter | target us/iter | device feedback "
        "us/iter | device dispatch us/iter | host sub/poll/rel calls | host "
        "cov/eval/sched ns/call | host other_idle ns/call [ns/iter] | CPU "
        "feedback ns/call [ns/iter] | CUDA memcpy+sync us/iter | attribution |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        host = row["host"]
        other_per_iter = host["other_idle"]["total_ns"] / row["iterations"]
        cpu_call = (
            f"{row['cpu_feedback_ns_call']:.0f}"
            if row["cpu_feedback_ns_call"] is not None
            else "n/a"
        )
        print(
            f"| {row['workload']} "
            f"| {fmt_device(row['idle_us'], row['idle_pct'])} "
            f"| {fmt_device(row['target_us'], row['target_pct'])} "
            f"| {fmt_device(row['feedback_us'], row['feedback_pct'])} "
            f"| {fmt_device(row['dispatch_us'], row['dispatch_pct'])} "
            f"| {host['submit']['count']}/{host['poll']['count']}/"
            f"{host['release']['count']} "
            f"| {host_ns_call(row, 'coverage')}/{host_ns_call(row, 'evaluate')}/"
            f"{host_ns_call(row, 'scheduler_stage')} "
            f"| {host_ns_call(row, 'other_idle')} [{other_per_iter:.0f}] "
            f"| {cpu_call} [{row['cpu_feedback_ns_iter']:.0f}] "
            f"| {row['memcpy_sync_us_iter']:.2f} "
            f"| {ATTRIBUTION[row['workload']]} |"
        )


def value_range(rows: list[dict[str, Any]], key: str) -> str:
    values = [float(row[key]) for row in rows]
    return f"{min(values):.2f}-{max(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(
            "benchmark/rq4/results/"
            "profile-mutating-30s-seed1-r1-w2-default22-20260811"
        ),
    )
    args = parser.parse_args()
    profiles = load_jsonl(args.input / "profiles.jsonl")
    wanted = TARGETS + CONTROLS
    selected = {
        profile["workload_id"]: profile
        for profile in profiles
        if profile["configuration"] == "gunit"
        and profile["workload_id"] in wanted
    }
    if set(selected) != set(wanted):
        raise RuntimeError(f"missing GUnit profiles: {sorted(set(wanted) - set(selected))}")
    rows = [compute(selected[workload], args.input) for workload in wanted]
    print_table(rows)

    targets = rows[: len(TARGETS)]
    print()
    print(
        "target ranges: device_idle_us="
        f"{value_range(targets, 'idle_us')}; target_us="
        f"{value_range(targets, 'target_us')}; memcpy_sync_us="
        f"{value_range(targets, 'memcpy_sync_us_iter')}"
    )
    scheduler_ns = [
        row["host"]["scheduler_stage"]["total_ns"] / row["iterations"]
        for row in targets
    ]
    feedback_ns = [
        row["host"]["coverage"]["total_ns"] / row["iterations"]
        + row["host"]["evaluate"]["total_ns"] / row["iterations"]
        + row["cpu_feedback_ns_iter"]
        for row in targets
    ]
    root_ns = [
        row["host"]["other_idle"]["total_ns"] / row["iterations"]
        for row in targets
    ]
    print(
        f"target host ranges: scheduler_ns={min(scheduler_ns):.0f}-"
        f"{max(scheduler_ns):.0f}; coverage+evaluate+cpu_feedback_ns="
        f"{min(feedback_ns):.0f}-{max(feedback_ns):.0f}; "
        f"other_idle_amortized_ns={min(root_ns):.0f}-{max(root_ns):.0f}"
    )
    print(
        "target CUDA call ratios: memcpy_calls/iter="
        f"{min(row['memcpy_calls_iter'] for row in targets):.5f}-"
        f"{max(row['memcpy_calls_iter'] for row in targets):.5f}; "
        "sync_calls/iter="
        f"{min(row['sync_calls_iter'] for row in targets):.5f}-"
        f"{max(row['sync_calls_iter'] for row in targets):.5f}; "
        f"launch_calls={sorted({row['launch_calls'] for row in targets})}"
    )
    cutlass = next(row for row in rows if row["workload"] == "cutlass_gemm")
    print(
        "cutlass device feedback split: "
        f"init={cutlass['feedback_init_us']:.2f}us/"
        f"{cutlass['feedback_init_pct']:.1f}%; "
        f"prepare={cutlass['feedback_prepare_us']:.2f}us/"
        f"{cutlass['feedback_prepare_pct']:.1f}%; "
        f"merge={cutlass['feedback_merge_us']:.3f}us/"
        f"{cutlass['feedback_merge_pct']:.3f}%"
    )


if __name__ == "__main__":
    main()
