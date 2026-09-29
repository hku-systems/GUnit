"""Validation for compact RQ2 coverage-delta telemetry records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


_COVERAGE_FIELDS = (
    "cfg_sites",
    "memory_features",
    "thread_activity_features",
)
_DELTA_FIELDS = tuple(f"{field}_delta" for field in _COVERAGE_FIELDS)
_COUNTER_FIELDS = (
    "executions_submitted",
    "executions_completed",
    *_COVERAGE_FIELDS,
    "feedback_features_total",
)


def _integer(record: Mapping[str, Any], field: str, sequence: int) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"sample {sequence} {field} must be an integer")
    if value < 0:
        raise ValueError(f"sample {sequence} {field} must be nonnegative")
    return value


def _timestamp(record: Mapping[str, Any], sequence: int) -> float:
    value = record.get("timestamp_s")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"sample {sequence} timestamp must be numeric")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"sample {sequence} timestamp must be finite and nonnegative")
    return timestamp


def validate_delta_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Validate one completion-event schema-v1 baseline/change/final trace."""

    if not records:
        raise ValueError("delta telemetry must be nonempty")

    final_flags = [record.get("final_sample") for record in records]
    for sequence, final_sample in enumerate(final_flags):
        if not isinstance(final_sample, bool):
            raise ValueError(f"sample {sequence} final_sample must be boolean")
    if any(final_flags[:-1]):
        raise ValueError("final sample must be last")
    if not final_flags[-1]:
        raise ValueError("delta telemetry must contain exactly one final sample")

    previous_timestamp = -1.0
    previous_counters = {field: 0 for field in _COUNTER_FIELDS}
    for sequence, record in enumerate(records):
        if _integer(record, "schema_version", sequence) != 1:
            raise ValueError(f"sample {sequence} schema_version must be 1")
        if _integer(record, "sequence", sequence) != sequence:
            raise ValueError(f"sample {sequence} sequence must equal its zero-based index")
        if "memory_map_hex" in record:
            raise ValueError(f"sample {sequence} memory_map_hex is forbidden")

        timestamp = _timestamp(record, sequence)
        if timestamp < previous_timestamp:
            raise ValueError(f"sample {sequence} timestamp regressed")
        previous_timestamp = timestamp

        final_sample = final_flags[sequence]

        counters = {
            field: _integer(record, field, sequence) for field in _COUNTER_FIELDS
        }
        deltas = {field: _integer(record, field, sequence) for field in _DELTA_FIELDS}
        total_delta = _integer(record, "feedback_features_total_delta", sequence)

        if sequence == 0:
            if final_sample or any(counters[field] != 0 for field in _COVERAGE_FIELDS):
                raise ValueError("baseline coverage counters must be zero and nonfinal")
            if (
                counters["executions_submitted"] != 0
                or counters["executions_completed"] != 0
            ):
                raise ValueError("baseline execution counters must be zero")
            if any(deltas.values()) or total_delta != 0:
                raise ValueError("baseline coverage deltas must be zero")

        for field, value in counters.items():
            if value < previous_counters[field]:
                raise ValueError(f"sample {sequence} {field} regressed")
        if counters["executions_completed"] > counters["executions_submitted"]:
            raise ValueError(
                f"sample {sequence} completed executions exceed submitted executions"
            )

        expected_total = sum(counters[field] for field in _COVERAGE_FIELDS)
        if counters["feedback_features_total"] != expected_total:
            raise ValueError(
                f"sample {sequence} feedback_features_total must equal cumulative features"
            )
        expected_deltas = {
            f"{field}_delta": counters[field] - previous_counters[field]
            for field in _COVERAGE_FIELDS
        }
        for field, expected in expected_deltas.items():
            if deltas[field] != expected:
                raise ValueError(f"sample {sequence} {field} must equal {expected}")
        expected_total_delta = sum(expected_deltas.values())
        if total_delta != expected_total_delta:
            raise ValueError(
                f"sample {sequence} feedback_features_total_delta must equal "
                f"{expected_total_delta}"
            )
        if sequence > 0 and not final_sample and expected_total_delta == 0:
            raise ValueError(f"sample {sequence} is an unchanged nonfinal record")
        if (
            sequence > 0
            and not final_sample
            and counters["executions_completed"]
            <= previous_counters["executions_completed"]
        ):
            raise ValueError(
                f"sample {sequence} completion event must advance "
                "executions_completed"
            )

        previous_counters = counters

    final = records[-1]
    if final["executions_submitted"] != final["executions_completed"]:
        raise ValueError("final sample pending executions must be zero")
