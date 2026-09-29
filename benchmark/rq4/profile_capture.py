"""Validate and summarize RQ4 profiling records and trace evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.rq4.profile_model import API_CATEGORIES
from benchmark.rq4.trace_extract import TraceEvidence


MAIN_LOOP_SEGMENTS = (
    "submit",
    "poll",
    "coverage",
    "evaluate",
    "release",
    "scheduler_stage",
    "other_idle",
)
CPU_FEEDBACK_SEGMENTS = ("predicate", "metadata")
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
PROFILING_RECORD_FIELDS = {
    "schema_version",
    "record_type",
    "domain",
    "segment",
    "unit",
    "count",
    "total",
    "max",
}


def summarize_trace_evidence(evidence: TraceEvidence) -> dict[str, Any]:
    cuda_api = {
        category: {"time_ns": 0, "calls": 0} for category in API_CATEGORIES
    }
    for thread_times in evidence.cuda_api_time_ns.values():
        for category, duration_ns in thread_times.items():
            cuda_api[category]["time_ns"] += duration_ns
    for thread_calls in evidence.cuda_api_calls.values():
        for category, calls in thread_calls.items():
            cuda_api[category]["calls"] += calls
    return {
        "cuda_api": cuda_api,
        "gpu_kernel_time_ns": sum(kernel.duration_ns for kernel in evidence.kernels),
        "gpu_mem_time_ns": evidence.gpu_memcpy_time_ns,
    }


def load_profiling_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_profiling_records(records)
    return records


def validate_profiling_records(records: list[dict[str, Any]]) -> None:
    expected = {
        "main_loop": ("ns", set(MAIN_LOOP_SEGMENTS)),
        "cpu_feedback": ("ns", set(CPU_FEEDBACK_SEGMENTS)),
    }
    if any(record.get("domain") == "device_kernel" for record in records):
        expected["device_kernel"] = ("cycles", set(DEVICE_SEGMENTS))

    actual: dict[str, set[str]] = {domain: set() for domain in expected}
    for record in records:
        if set(record) != PROFILING_RECORD_FIELDS:
            raise RuntimeError(
                "invalid profiling record fields: "
                f"expected={sorted(PROFILING_RECORD_FIELDS)} actual={sorted(record)}"
            )
        if record["schema_version"] != 2:
            raise RuntimeError(
                f"profiling schema_version must be 2, got {record['schema_version']!r}"
            )
        if record["record_type"] != "segment":
            raise RuntimeError(
                "profiling record_type must be 'segment', "
                f"got {record['record_type']!r}"
            )
        domain = str(record["domain"])
        if domain not in expected:
            raise RuntimeError(f"unsupported profiling domain: {domain!r}")
        unit, segments = expected[domain]
        segment = str(record["segment"])
        if segment not in segments:
            raise RuntimeError(f"unsupported {domain} segment: {segment!r}")
        if segment in actual[domain]:
            raise RuntimeError(f"duplicate profiling segment: {domain}/{segment}")
        if record["unit"] != unit:
            raise RuntimeError(
                f"invalid unit for {domain}/{segment}: "
                f"expected={unit} actual={record['unit']}"
            )
        count = int(record["count"])
        total = int(record["total"])
        maximum = record["max"]
        if count < 0 or total < 0 or (maximum is not None and int(maximum) < 0):
            raise RuntimeError(f"negative profiling value: {record}")
        if domain == "device_kernel" and maximum is not None:
            raise RuntimeError(f"device profiling max must be null: {record}")
        if domain != "device_kernel" and maximum is None:
            raise RuntimeError(f"host profiling max is missing: {record}")
        if maximum is not None and int(maximum) > total:
            raise RuntimeError(f"profiling max exceeds total: {record}")
        actual[domain].add(segment)

    for domain, (_, segments) in expected.items():
        if actual[domain] != segments:
            raise RuntimeError(
                f"incomplete {domain} segments: "
                f"missing={sorted(segments - actual[domain])} "
                f"extra={sorted(actual[domain] - segments)}"
            )

    device_counts = {
        int(record["count"])
        for record in records
        if record["domain"] == "device_kernel"
    }
    if len(device_counts) > 1:
        raise RuntimeError(
            f"device profiling segments disagree on iterations: {sorted(device_counts)}"
        )


def profiling_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    validate_profiling_records(records)
    by_key = {(record["domain"], record["segment"]): record for record in records}
    predicate = by_key[("cpu_feedback", "predicate")]
    metadata = by_key[("cpu_feedback", "metadata")]
    device = [record for record in records if record["domain"] == "device_kernel"]
    return {
        "wall_time_ns": sum(
            int(record["total"])
            for record in records
            if record["domain"] in ("main_loop", "cpu_feedback")
        ),
        "cpu_feedback_timing_ns": {
            "calls": int(predicate["count"]),
            "predicate_ns": int(predicate["total"]),
            "metadata_ns": int(metadata["total"]),
            "total_ns": int(predicate["total"]) + int(metadata["total"]),
        },
        "device_timing_cycles": (
            {
                "iterations": int(device[0]["count"]),
                **{record["segment"]: int(record["total"]) for record in device},
            }
            if device
            else None
        ),
    }
