#!/usr/bin/env python3
"""Load and validate RQ2 coverage-growth results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from benchmark.rq2.verify_results import (
    CFG_PRESENCE_METRIC_VERSION,
    parse_cfg_metric_contract,
)


WORKLOAD_TITLES = {
    "shoc_reduction": "SHOC Reduction",
    "shoc_radix_sort": "SHOC Radix Sort",
    "shoc_scan": "SHOC Scan",
    "cutlass_gemm": "CUTLASS GEMM",
    "flashattention_device1xn": "FlashAttention",
    "pytorch_batchnorm": "PyTorch BatchNorm",
}
VCONFIG_LINESTYLES = frozenset({"off", "on"})


@dataclass(frozen=True)
class CoveragePoint:
    sequence: int
    timestamp_ms: float
    executions_completed: int
    cfg_sites: int | None
    memory_features: int | None
    executions_pending: int
    final_sample: bool


@dataclass(frozen=True)
class CoverageCurve:
    trial_id: str
    workload_id: str
    workload_label: str
    configuration_id: str
    paper_label: str
    vconfig: str
    feedback_enabled: bool
    instrumented_cfg_sites: int | None
    memory_metric_version: str | None
    memory_map_bits: int | None
    points: tuple[CoveragePoint, ...]
    cfg_metric_version: str | None = None


@dataclass(frozen=True)
class SkippedCoverageCell:
    trial_id: str
    workload_id: str
    configuration_id: str
    paper_label: str
    vconfig: str
    vconfig_disabled_reason: str


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        stream = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read JSONL file {path}: {error}") from error
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"empty JSONL row in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL row in {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object in {path}:{line_number}"
                )
            yield row


def _required_string(value: object, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {field} must be a nonempty string")
    return value


def _required_int(value: object, field: str, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be an integer")
    return value


def _optional_count(value: object, field: str, context: str) -> int | None:
    if value is None:
        return None
    count = _required_int(value, field, context)
    if count < 0:
        raise ValueError(f"{context}: {field} must be nonnegative")
    return count


def load_coverage_data(
    result_root: Path,
) -> tuple[tuple[CoverageCurve, ...], tuple[SkippedCoverageCell, ...]]:
    """Load one complete RQ2 run while discarding large serialized bitmaps."""

    result_root = result_root.resolve()
    environment = _read_object(result_root / "environment.json", "environment")
    if environment.get("coverage_recording_mode") != "completion-change-v1":
        raise ValueError(
            "environment: coverage_recording_mode must be completion-change-v1"
        )
    repetitions = _required_int(
        environment.get("repetitions"), "repetitions", "environment"
    )
    if repetitions <= 0:
        raise ValueError("environment: repetitions must be positive")

    raw_workloads = environment.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ValueError("environment: workloads must be a nonempty array")
    workload_ids: list[str] = []
    workload_labels: dict[str, str] = {}
    for index, raw_workload in enumerate(raw_workloads):
        if not isinstance(raw_workload, dict):
            raise ValueError(f"environment workload {index} must be an object")
        workload_id = _required_string(
            raw_workload.get("workload_id"), "workload_id", f"workload {index}"
        )
        if workload_id in workload_labels:
            raise ValueError(f"environment: duplicate workload {workload_id}")
        workload_ids.append(workload_id)
        workload_labels[workload_id] = str(
            raw_workload.get("label") or WORKLOAD_TITLES.get(workload_id, workload_id)
        )

    raw_configuration_ids = environment.get("configurations")
    if not isinstance(raw_configuration_ids, list) or not raw_configuration_ids:
        raise ValueError("environment: configurations must be a nonempty array")
    configuration_ids = [
        _required_string(value, "configuration_id", "environment configurations")
        for value in raw_configuration_ids
    ]
    if len(configuration_ids) != len(set(configuration_ids)):
        raise ValueError("environment: configuration IDs must be unique")

    configurations: dict[str, dict[str, Any]] = {}
    configuration_order: list[str] = []
    for row in _iter_jsonl(result_root / "configurations.jsonl"):
        configuration_id = _required_string(
            row.get("configuration_id"), "configuration_id", "configuration"
        )
        if configuration_id in configurations:
            raise ValueError(f"duplicate configuration {configuration_id}")
        paper_label = _required_string(
            row.get("paper_label"), "paper_label", configuration_id
        )
        vconfig = row.get("vconfig")
        if vconfig not in VCONFIG_LINESTYLES:
            raise ValueError(f"{configuration_id}: vconfig must be off or on")
        feedback_enabled = row.get("feedback_enabled")
        if not isinstance(feedback_enabled, bool):
            raise ValueError(f"{configuration_id}: feedback_enabled must be boolean")
        configurations[configuration_id] = {
            "paper_label": paper_label,
            "vconfig": vconfig,
            "feedback_enabled": feedback_enabled,
        }
        configuration_order.append(configuration_id)
    if configuration_order != configuration_ids:
        raise ValueError("configuration order differs from environment")

    expected_keys = [
        (repetition, workload_id, configuration_id)
        for repetition in range(repetitions)
        for workload_id in workload_ids
        for configuration_id in configuration_ids
    ]
    trials: dict[str, dict[str, Any]] = {}
    skipped_cells: list[SkippedCoverageCell] = []
    seen_trial_ids: set[str] = set()
    actual_keys: list[tuple[int, str, str]] = []
    for row in _iter_jsonl(result_root / "trials.jsonl"):
        trial_id = _required_string(row.get("trial_id"), "trial_id", "trial")
        if trial_id in seen_trial_ids:
            raise ValueError(f"duplicate trial {trial_id}")
        seen_trial_ids.add(trial_id)
        repetition = _required_int(row.get("repetition"), "repetition", trial_id)
        if repetition < 0 or repetition >= repetitions:
            raise ValueError(f"{trial_id}: repetition is outside the campaign range")
        workload_id = _required_string(row.get("workload_id"), "workload_id", trial_id)
        configuration_id = _required_string(
            row.get("configuration_id"), "configuration_id", trial_id
        )
        actual_keys.append((repetition, workload_id, configuration_id))
        status = row.get("status")
        if status == "skipped":
            if row.get("vconfig_effective") != "unsupported":
                raise ValueError(f"{trial_id}: skipped VConfig must be unsupported")
            reason = _required_string(
                row.get("vconfig_disabled_reason"),
                "vconfig_disabled_reason",
                trial_id,
            )
            configuration = configurations.get(configuration_id)
            if configuration is None:
                raise ValueError(f"{trial_id}: unknown configuration {configuration_id}")
            if configuration["vconfig"] != "on":
                raise ValueError(f"{trial_id}: only VConfig-on cells may be skipped")
            skipped_cells.append(
                SkippedCoverageCell(
                    trial_id=trial_id,
                    workload_id=workload_id,
                    configuration_id=configuration_id,
                    paper_label=str(configuration["paper_label"]),
                    vconfig=str(configuration["vconfig"]),
                    vconfig_disabled_reason=reason,
                )
            )
            continue
        if status != "completed":
            raise ValueError(f"{trial_id}: trial status must be completed or skipped")
        trials[trial_id] = row
    if actual_keys != expected_keys:
        raise ValueError(
            "terminal trials do not cover the complete workload/configuration matrix"
        )

    samples_by_trial: dict[str, list[dict[str, Any]]] = {
        trial_id: [] for trial_id in trials
    }
    for sample in _iter_jsonl(result_root / "samples.jsonl"):
        trial_id = _required_string(sample.get("trial_id"), "trial_id", "sample")
        if trial_id not in samples_by_trial:
            raise ValueError(f"sample references unknown trial {trial_id}")
        samples_by_trial[trial_id].append(sample)

    curves: list[CoverageCurve] = []
    for trial_id, trial in trials.items():
        workload_id = str(trial["workload_id"])
        configuration_id = str(trial["configuration_id"])
        configuration = configurations[configuration_id]
        feedback_enabled = bool(configuration["feedback_enabled"])
        raw_points = samples_by_trial[trial_id]
        if not raw_points:
            raise ValueError(f"{trial_id}: missing samples")

        points: list[CoveragePoint] = []
        instrumented_cfg_sites: int | None = None
        memory_metric_version: str | None = None
        memory_map_bits: int | None = None
        cfg_metric_version: str | None = None
        previous_sequence: int | None = None
        previous_timestamp_ms: float | None = None
        previous_executions_completed: int | None = None
        previous_cfg: int | None = None
        previous_memory: int | None = None
        final_count = 0
        for sample in raw_points:
            if sample.get("workload_id") != workload_id or sample.get(
                "configuration_id"
            ) != configuration_id:
                raise ValueError(f"{trial_id}: sample identity differs from trial")
            if sample.get("paper_label") != configuration["paper_label"]:
                raise ValueError(f"{trial_id}: sample paper label differs from configuration")
            if sample.get("vconfig_effective") != configuration["vconfig"]:
                raise ValueError(f"{trial_id}: sample VConfig differs from configuration")
            if sample.get("feedback_enabled") != feedback_enabled:
                raise ValueError(f"{trial_id}: sample feedback flag differs from configuration")

            sequence = _required_int(sample.get("sequence"), "sequence", trial_id)
            if previous_sequence is not None and sequence <= previous_sequence:
                raise ValueError(f"{trial_id}: sequence is not strictly increasing")
            timestamp = sample.get("timestamp_s")
            if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
                raise ValueError(f"{trial_id}: timestamp must be numeric")
            timestamp_ms = float(timestamp) * 1000.0
            if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
                raise ValueError(f"{trial_id}: timestamp must be finite and nonnegative")
            if previous_timestamp_ms is not None and timestamp_ms < previous_timestamp_ms:
                raise ValueError(f"{trial_id}: timestamp regressed")

            executions_completed = _required_int(
                sample.get("executions_completed"),
                "executions_completed",
                trial_id,
            )
            if executions_completed < 0:
                raise ValueError(
                    f"{trial_id}: executions_completed must be nonnegative"
                )
            if (
                previous_executions_completed is not None
                and executions_completed < previous_executions_completed
            ):
                raise ValueError(f"{trial_id}: executions_completed regressed")

            cfg_sites = _optional_count(sample.get("cfg_sites"), "cfg_sites", trial_id)
            memory_features = _optional_count(
                sample.get("memory_features"), "memory_features", trial_id
            )
            sample_instrumented = _optional_count(
                sample.get("instrumented_cfg_sites"),
                "instrumented_cfg_sites",
                trial_id,
            )
            if feedback_enabled:
                if cfg_sites is None or memory_features is None:
                    raise ValueError(f"{trial_id}: feedback coverage metrics are unavailable")
                if sample_instrumented is None or sample_instrumented <= 0:
                    raise ValueError(f"{trial_id}: instrumented CFG site count is invalid")
                if (
                    instrumented_cfg_sites is not None
                    and sample_instrumented != instrumented_cfg_sites
                ):
                    raise ValueError(f"{trial_id}: instrumented CFG site count changed")
                if previous_cfg is not None and cfg_sites < previous_cfg:
                    raise ValueError(f"{trial_id}: cfg_sites coverage regressed")
                if previous_memory is not None and memory_features < previous_memory:
                    raise ValueError(f"{trial_id}: memory_features coverage regressed")
                instrumented_cfg_sites = sample_instrumented
                sample_metric_version = _required_string(
                    sample.get("memory_metric_version"),
                    "memory_metric_version",
                    trial_id,
                )
                if sample_metric_version not in {
                    "rapid-memory-bitset-v1",
                    "rapid-simt-memcov-v1",
                }:
                    raise ValueError(
                        f"{trial_id}: unsupported memory metric version "
                        f"{sample_metric_version}"
                    )
                sample_memory_map_bits = _required_int(
                    sample.get("memory_map_bits"),
                    "memory_map_bits",
                    trial_id,
                )
                if sample_memory_map_bits <= 0:
                    raise ValueError(f"{trial_id}: memory_map_bits must be positive")
                if (
                    memory_metric_version is not None
                    and sample_metric_version != memory_metric_version
                ):
                    raise ValueError(f"{trial_id}: memory metric version changed")
                if (
                    memory_map_bits is not None
                    and sample_memory_map_bits != memory_map_bits
                ):
                    raise ValueError(f"{trial_id}: memory map size changed")
                memory_metric_version = sample_metric_version
                memory_map_bits = sample_memory_map_bits
                cfg_contract = parse_cfg_metric_contract(sample)
                if (
                    cfg_metric_version is not None
                    and cfg_contract.cfg_metric_version != cfg_metric_version
                ):
                    raise ValueError(f"{trial_id}: CFG metric version changed")
                cfg_metric_version = cfg_contract.cfg_metric_version
                previous_cfg = cfg_sites
                previous_memory = memory_features
            elif any(
                value is not None
                for value in (
                    cfg_sites,
                    memory_features,
                    sample_instrumented,
                    sample.get("memory_metric_version"),
                    sample.get("memory_map_bits"),
                    sample.get("cfg_metric_version"),
                )
            ):
                raise ValueError(
                    f"{trial_id}: CuFuzz internal coverage must remain unavailable"
                )

            pending = _required_int(
                sample.get("executions_pending"), "executions_pending", trial_id
            )
            if pending < 0:
                raise ValueError(f"{trial_id}: executions_pending must be nonnegative")
            final_sample = sample.get("final_sample")
            if not isinstance(final_sample, bool):
                raise ValueError(f"{trial_id}: final_sample must be boolean")
            final_count += int(final_sample)
            points.append(
                CoveragePoint(
                    sequence=sequence,
                    timestamp_ms=timestamp_ms,
                    executions_completed=executions_completed,
                    cfg_sites=cfg_sites,
                    memory_features=memory_features,
                    executions_pending=pending,
                    final_sample=final_sample,
                )
            )
            previous_sequence = sequence
            previous_timestamp_ms = timestamp_ms
            previous_executions_completed = executions_completed

        if final_count != 1 or not points[-1].final_sample:
            raise ValueError(f"{trial_id}: exactly one final sample must terminate the trace")
        if points[-1].executions_pending != 0:
            raise ValueError(f"{trial_id}: final pending executions must be zero")
        curves.append(
            CoverageCurve(
                trial_id=trial_id,
                workload_id=workload_id,
                workload_label=workload_labels[workload_id],
                configuration_id=configuration_id,
                paper_label=str(configuration["paper_label"]),
                vconfig=str(configuration["vconfig"]),
                feedback_enabled=feedback_enabled,
                instrumented_cfg_sites=instrumented_cfg_sites,
                memory_metric_version=memory_metric_version,
                memory_map_bits=memory_map_bits,
                points=tuple(points),
                cfg_metric_version=(
                    cfg_metric_version
                    if cfg_metric_version is not None
                    else (
                        CFG_PRESENCE_METRIC_VERSION
                        if feedback_enabled
                        else None
                    )
                ),
            )
        )
    metric_versions = {
        curve.memory_metric_version
        for curve in curves
        if curve.memory_metric_version is not None
    }
    if len(metric_versions) != 1:
        raise ValueError(
            f"mixed memory metric versions are not comparable: "
            f"{sorted(metric_versions)!r}"
        )
    memory_map_sizes = {
        curve.memory_map_bits for curve in curves if curve.memory_map_bits is not None
    }
    if len(memory_map_sizes) != 1:
        raise ValueError(
            f"mixed memory map sizes are not comparable: {sorted(memory_map_sizes)!r}"
        )
    cfg_metric_versions = {
        curve.cfg_metric_version
        for curve in curves
        if curve.cfg_metric_version is not None
    }
    if len(cfg_metric_versions) != 1:
        raise ValueError(
            f"mixed CFG metric versions are not comparable: "
            f"{sorted(cfg_metric_versions)!r}"
        )
    return tuple(curves), tuple(skipped_cells)


def load_coverage_curves(result_root: Path) -> tuple[CoverageCurve, ...]:
    """Load completed curves while excluding documented unsupported cells."""

    curves, _skipped_cells = load_coverage_data(result_root)
    return curves

