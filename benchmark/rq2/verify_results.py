#!/usr/bin/env python3
"""Validate and enrich RQ2 raw coverage telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq2.coverage_delta import validate_delta_records
from benchmark.workloads.schema import sha256_file


LEGACY_MEMORY_METRIC_VERSION = "rapid-memory-bitset-v1"
SIMT_MEMORY_METRIC_VERSION = "rapid-simt-memcov-v1"
SIMT_MEMORY_HASH_CONTRACT_VERSION = "rapid-simt-memcov-hash-v1"
MEMORY_MAP_BITS = 61_440
THREAD_ACTIVITY_MAP_BITS = 4_096
SIMT_MEMORY_SECTOR_BYTES = 32
SIMT_MEMORY_PATTERN_ENCODINGS = {
    "single": 0,
    "full_broadcast": 1,
    "full_contiguous": 2,
    "full_other": 3,
    "partial_broadcast": 4,
    "partial_contiguous": 5,
    "partial_other": 6,
}
CFG_PRESENCE_METRIC_VERSION = "rapid-cfg-presence-v1"


@dataclass(frozen=True)
class CfgMetricContract:
    cfg_metric_version: str


def parse_cfg_metric_contract(metadata: Mapping[str, Any]) -> CfgMetricContract:
    raw_version = metadata.get("cfg_metric_version")
    version = (
        CFG_PRESENCE_METRIC_VERSION if raw_version is None else raw_version
    )
    if version != CFG_PRESENCE_METRIC_VERSION:
        raise ValueError(f"unsupported CFG metric version: {version!r}")
    if "cfg_counter_saturation" in metadata:
        raise ValueError("feedback metadata must not contain cfg_counter_saturation")
    return CfgMetricContract(version)


@dataclass(frozen=True)
class MemoryMetricContract:
    memory_metric_version: str
    memory_map_bits: int
    memory_sector_bytes: int | None
    thread_activity_map_bits: int
    memory_hash_contract_version: str | None


def parse_memory_metric_contract(
    metadata: Mapping[str, Any], *, allow_legacy: bool
) -> MemoryMetricContract:
    schema_version = metadata.get("schema_version")
    metric_version = metadata.get("memory_metric_version")
    if schema_version == 1 and metric_version is None:
        if not allow_legacy:
            raise ValueError("feedback metadata lacks an explicit memory metric version")
        legacy_scalars = {
            "feedback_abi_version": 1,
            "edge_map_size": 65_536,
            "simt_memcov_buckets": MEMORY_MAP_BITS + THREAD_ACTIVITY_MAP_BITS,
            "simt_memcov_data_buckets": MEMORY_MAP_BITS,
        }
        for field, expected_value in legacy_scalars.items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"legacy feedback metadata {field} must equal {expected_value!r}"
                )
        for field in (
            "instrumented_cfg_sites",
            "instrumented_memory_sites",
            "unknown_memory_sites",
        ):
            value = metadata.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"legacy feedback metadata {field} must be a nonnegative integer"
                )
        for field in ("entry_symbol", "kernel_id"):
            value = metadata.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"legacy feedback metadata {field} must be a nonempty string"
                )
        for field, count_field in (
            ("cfg_sites", "instrumented_cfg_sites"),
            ("memory_sites", "instrumented_memory_sites"),
        ):
            sites = metadata.get(field)
            if not isinstance(sites, list) or len(sites) != metadata[count_field]:
                raise ValueError(
                    f"legacy feedback metadata {field} must match {count_field}"
                )
        return MemoryMetricContract(
            memory_metric_version=LEGACY_MEMORY_METRIC_VERSION,
            memory_map_bits=MEMORY_MAP_BITS,
            memory_sector_bytes=None,
            thread_activity_map_bits=THREAD_ACTIVITY_MAP_BITS,
            memory_hash_contract_version=None,
        )

    if schema_version != 2:
        raise ValueError("feedback metadata schema_version must be 2")
    expected = {
        "memory_metric_version": SIMT_MEMORY_METRIC_VERSION,
        "memory_map_bits": MEMORY_MAP_BITS,
        "memory_sector_bytes": SIMT_MEMORY_SECTOR_BYTES,
        "thread_activity_map_bits": THREAD_ACTIVITY_MAP_BITS,
        "memory_hash_contract_version": SIMT_MEMORY_HASH_CONTRACT_VERSION,
        "memory_pattern_encodings": SIMT_MEMORY_PATTERN_ENCODINGS,
    }
    for field, expected_value in expected.items():
        if metadata.get(field) != expected_value:
            raise ValueError(
                f"feedback metadata {field} must equal {expected_value!r}"
            )
    return MemoryMetricContract(
        memory_metric_version=SIMT_MEMORY_METRIC_VERSION,
        memory_map_bits=MEMORY_MAP_BITS,
        memory_sector_bytes=SIMT_MEMORY_SECTOR_BYTES,
        thread_activity_map_bits=THREAD_ACTIVITY_MAP_BITS,
        memory_hash_contract_version=SIMT_MEMORY_HASH_CONTRACT_VERSION,
    )


@dataclass(frozen=True)
class SampleContext:
    trial_id: str
    workload_id: str
    configuration_id: str
    paper_label: str
    backend: str
    window: int
    repetition: int
    seed: int
    vconfig_requested: str
    vconfig_effective: str
    feedback_enabled: bool
    build_report_sha256: str
    manifest_sha256: str
    backend_sha256: str
    instrumentation_metadata_sha256: str | None
    instrumented_cfg_sites: int | None
    memory_metric_version: str | None
    memory_map_bits: int | None
    memory_sector_bytes: int | None
    thread_activity_map_bits: int | None
    memory_hash_contract_version: str | None
    raw_telemetry_path: str
    raw_telemetry_sha256: str
    include_extended_memory_contract: bool = True
    cfg_metric_version: str | None = CFG_PRESENCE_METRIC_VERSION
    include_cfg_metric_contract: bool = True


_COUNTERS = (
    "executions_submitted",
    "executions_completed",
    "cfg_sites",
    "memory_features",
    "thread_activity_features",
    "feedback_features_total",
)
_RAW_DELTA_FIELDS = (
    "cfg_sites_delta",
    "memory_features_delta",
    "thread_activity_features_delta",
    "feedback_features_total_delta",
)


def load_raw_telemetry(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read raw telemetry {path}: {error}") from error
    if not lines:
        raise ValueError("raw telemetry must be nonempty")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"raw telemetry line {line_number} is empty")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"raw telemetry line {line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"raw telemetry line {line_number} must be an object")
        records.append(record)
    return records


def validate_and_enrich_samples(
    path: Path, context: SampleContext
) -> list[dict[str, Any]]:
    records = load_raw_telemetry(path)
    validate_delta_records(records)
    if context.feedback_enabled:
        coverage_limits = {
            "cfg_sites": context.instrumented_cfg_sites,
            "memory_features": context.memory_map_bits,
            "thread_activity_features": context.thread_activity_map_bits,
        }
        for field, limit in coverage_limits.items():
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError(f"feedback context {field} capacity must be nonnegative")
            for sequence, record in enumerate(records):
                if record[field] > limit:
                    raise ValueError(
                        f"sample {sequence} {field} exceeds declared capacity {limit}"
                    )
    else:
        for sequence, record in enumerate(records):
            if any(record[field] != 0 for field in _COUNTERS[2:]):
                raise ValueError(
                    f"CuFuzz sample {sequence} internal feedback counters must be zero"
                )

    context_fields = asdict(context)
    for field in (
        "memory_metric_version",
        "memory_map_bits",
        "memory_sector_bytes",
        "thread_activity_map_bits",
        "memory_hash_contract_version",
        "include_extended_memory_contract",
        "cfg_metric_version",
        "include_cfg_metric_contract",
    ):
        context_fields.pop(field)
    samples: list[dict[str, Any]] = []
    for record in records:
        cumulative_record = {
            field: value
            for field, value in record.items()
            if field != "schema_version" and field not in _RAW_DELTA_FIELDS
        }
        sample = {
            "schema_version": 1,
            **context_fields,
            **cumulative_record,
            "executions_pending": (
                record["executions_submitted"] - record["executions_completed"]
            ),
            "external_features": None,
        }
        if context.feedback_enabled:
            sample.update(
                {
                    "instrumented_cfg_sites": context.instrumented_cfg_sites,
                    "memory_metric_version": context.memory_metric_version,
                    "memory_map_bits": context.memory_map_bits,
                    "thread_activity_map_bits": context.thread_activity_map_bits,
                }
            )
            if context.include_extended_memory_contract:
                sample.update(
                    {
                        "memory_sector_bytes": context.memory_sector_bytes,
                        "memory_hash_contract_version": (
                            context.memory_hash_contract_version
                        ),
                    }
                )
            if context.include_cfg_metric_contract:
                sample["cfg_metric_version"] = context.cfg_metric_version
        else:
            sample.update(
                {
                    "cfg_sites": None,
                    "instrumented_cfg_sites": None,
                    "memory_features": None,
                    "memory_metric_version": None,
                    "memory_map_bits": None,
                    "thread_activity_features": None,
                    "thread_activity_map_bits": None,
                    "feedback_features_total": None,
                    "instrumentation_metadata_sha256": None,
                }
            )
            if context.include_extended_memory_contract:
                sample.update(
                    {
                        "memory_sector_bytes": None,
                        "memory_hash_contract_version": None,
                    }
                )
            if context.include_cfg_metric_contract:
                sample["cfg_metric_version"] = None
        samples.append(sample)
    return samples


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"required result ledger does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise RuntimeError(f"empty JSONL row in {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid JSONL row in {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row must be an object in {path}:{line_number}")
        rows.append(row)
    return rows


def _required_file(path: Path, recorded_sha256: object, description: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{description} does not exist: {path}")
    if (
        not isinstance(recorded_sha256, str)
        or f"sha256:{sha256_file(path)}" != recorded_sha256
    ):
        raise RuntimeError(f"{description} hash mismatch: {path}")


def _result_file(
    output: Path,
    row: Mapping[str, Any],
    path_field: str,
    hash_field: str,
    *,
    required: bool = True,
) -> Path | None:
    raw_path = row.get(path_field)
    recorded_sha256 = row.get(hash_field)
    if raw_path is None and recorded_sha256 is None and not required:
        return None
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise RuntimeError(f"{path_field} must be a confined relative path")
    path = (output / raw_path).resolve()
    if not path.is_relative_to(output):
        raise RuntimeError(f"{path_field} escapes the result directory")
    if not required and recorded_sha256 is None:
        if path.exists():
            raise RuntimeError(f"{path_field} exists without a recorded hash")
        return None
    _required_file(path, recorded_sha256, path_field)
    return path


def _artifact_file(
    row: Mapping[str, Any],
    path_field: str,
    hash_field: str,
    *,
    required: bool = True,
) -> Path | None:
    raw_path = row.get(path_field)
    recorded_sha256 = row.get(hash_field)
    if raw_path is None and recorded_sha256 is None and not required:
        return None
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"terminal row missing {path_field}")
    path = Path(raw_path)
    _required_file(path, recorded_sha256, path_field)
    return path


def _context_from_trial(
    row: Mapping[str, Any],
    metric_contract: MemoryMetricContract | None,
    cfg_contract: CfgMetricContract | None,
    *,
    include_extended_memory_contract: bool,
    include_cfg_metric_contract: bool,
) -> SampleContext:
    try:
        return SampleContext(
            trial_id=row["trial_id"],
            workload_id=row["workload_id"],
            configuration_id=row["configuration_id"],
            paper_label=row["paper_label"],
            backend=row["backend"],
            window=row["window"],
            repetition=row["repetition"],
            seed=row["seed"],
            vconfig_requested=row["vconfig_requested"],
            vconfig_effective=row["vconfig_effective"],
            feedback_enabled=row["feedback_enabled"],
            build_report_sha256=row["build_report_sha256"],
            manifest_sha256=row["manifest_sha256"],
            backend_sha256=row["backend_sha256"],
            instrumentation_metadata_sha256=row[
                "instrumentation_metadata_sha256"
            ],
            instrumented_cfg_sites=row["instrumented_cfg_sites"],
            memory_metric_version=(
                metric_contract.memory_metric_version if metric_contract else None
            ),
            memory_map_bits=(
                metric_contract.memory_map_bits if metric_contract else None
            ),
            memory_sector_bytes=(
                metric_contract.memory_sector_bytes if metric_contract else None
            ),
            thread_activity_map_bits=(
                metric_contract.thread_activity_map_bits if metric_contract else None
            ),
            memory_hash_contract_version=(
                metric_contract.memory_hash_contract_version
                if metric_contract
                else None
            ),
            raw_telemetry_path=row["raw_telemetry_path"],
            raw_telemetry_sha256=row["raw_telemetry_sha256"],
            include_extended_memory_contract=include_extended_memory_contract,
            cfg_metric_version=(
                cfg_contract.cfg_metric_version if cfg_contract else None
            ),
            include_cfg_metric_contract=include_cfg_metric_contract,
        )
    except KeyError as error:
        raise RuntimeError(f"completed trial missing context field: {error}") from error


def verify_result_directory(output: Path) -> dict[str, Any]:
    output = output.resolve()
    environment = _load_object(output / "environment.json", "campaign environment")
    if environment.get("coverage_recording_mode") != "completion-change-v1":
        raise RuntimeError(
            "environment coverage recording mode must be completion-change-v1"
        )
    snapshot_fields = (
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
    missing_snapshot_fields = [
        field for field in snapshot_fields if field not in environment
    ]
    if missing_snapshot_fields:
        raise RuntimeError(
            f"missing environment snapshot fields: {missing_snapshot_fields!r}"
        )
    for executable_name in ("fuzzer", "fuzzer_async"):
        identity = environment.get(executable_name)
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise RuntimeError(f"environment {executable_name} identity is invalid")
        expected_path = (output / "binaries" / executable_name).resolve()
        if Path(identity["path"]).resolve() != expected_path:
            raise RuntimeError(
                f"environment {executable_name} must use the result snapshot"
            )
        _required_file(
            expected_path,
            identity.get("sha256"),
            executable_name,
        )
    configurations = _load_jsonl(output / "configurations.jsonl")
    samples = _load_jsonl(output / "samples.jsonl")
    trials = _load_jsonl(output / "trials.jsonl")
    failures = _load_jsonl(output / "failures.jsonl")
    if not (output / "logs").is_dir() or not (output / "raw").is_dir():
        raise RuntimeError("result directory must contain logs/ and raw/")

    configuration_ids = environment.get("configurations")
    if not isinstance(configuration_ids, list) or not all(
        isinstance(item, str) for item in configuration_ids
    ):
        raise RuntimeError("environment configurations must be a string array")
    if [row.get("configuration_id") for row in configurations] != configuration_ids:
        raise RuntimeError("configurations.jsonl order differs from environment")
    if len(set(configuration_ids)) != len(configuration_ids):
        raise RuntimeError("configuration IDs must be unique")
    configurations_by_id = {
        row["configuration_id"]: row for row in configurations
    }

    workloads = environment.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise RuntimeError("environment workloads must be a nonempty array")
    if any(not isinstance(row, dict) for row in workloads):
        raise RuntimeError("environment workload rows must be objects")
    workload_ids = [row.get("workload_id") for row in workloads]
    if not all(isinstance(item, str) for item in workload_ids) or len(
        set(workload_ids)
    ) != len(workload_ids):
        raise RuntimeError("environment workload IDs must be unique strings")
    workloads_by_id = {row["workload_id"]: row for row in workloads}
    repetitions = environment.get("repetitions")
    coverage_seconds = environment.get("coverage_seconds")
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
        or not isinstance(coverage_seconds, int)
        or isinstance(coverage_seconds, bool)
        or coverage_seconds <= 0
    ):
        raise RuntimeError("environment campaign durations/repetitions are invalid")

    for workload in workloads:
        _required_file(
            Path(workload["build_report_path"]),
            workload.get("build_report_sha256"),
            "build report",
        )

    terminal_rows = trials + failures
    for ledger_name, rows in (("trials.jsonl", trials), ("failures.jsonl", failures)):
        indices = [row.get("plan_index") for row in rows]
        if indices != sorted(indices):
            raise RuntimeError(f"{ledger_name} plan indices are not append ordered")
    if len(terminal_rows) != repetitions * len(workloads) * len(configurations):
        raise RuntimeError("terminal row count does not cover the complete campaign matrix")
    if any(
        not isinstance(row.get("plan_index"), int)
        or isinstance(row.get("plan_index"), bool)
        for row in terminal_rows
    ):
        raise RuntimeError("terminal rows must contain integer plan_index")
    ordered_terminals = sorted(terminal_rows, key=lambda row: row["plan_index"])
    if [row["plan_index"] for row in ordered_terminals] != list(
        range(len(ordered_terminals))
    ):
        raise RuntimeError("terminal plan indices must be exact and unique")
    expected_keys = [
        (repetition, workload_id, configuration_id)
        for repetition in range(repetitions)
        for workload_id in workload_ids
        for configuration_id in configuration_ids
    ]
    actual_keys = [
        (row.get("repetition"), row.get("workload_id"), row.get("configuration_id"))
        for row in ordered_terminals
    ]
    if actual_keys != expected_keys:
        raise RuntimeError("terminal matrix order differs from campaign contract")

    samples_by_trial: dict[str, list[dict[str, Any]]] = {}
    sample_keys: set[tuple[str, int]] = set()
    for sample in samples:
        trial_id = sample.get("trial_id")
        sequence = sample.get("sequence")
        if not isinstance(trial_id, str) or not isinstance(sequence, int):
            raise RuntimeError("sample has invalid trial_id/sequence")
        key = (trial_id, sequence)
        if key in sample_keys:
            raise RuntimeError(f"duplicate sample key: {key!r}")
        sample_keys.add(key)
        samples_by_trial.setdefault(trial_id, []).append(sample)

    completed = 0
    skipped = 0
    failed = 0
    cfg_metric_versions: set[str] = set()
    seen_trial_ids: set[str] = set()
    for row in ordered_terminals:
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in seen_trial_ids:
            raise RuntimeError("terminal trial IDs must be unique strings")
        seen_trial_ids.add(trial_id)
        configuration = configurations_by_id[row["configuration_id"]]
        workload = workloads_by_id[row["workload_id"]]
        for field in (
            "paper_label",
            "backend",
            "window",
            "feedback_enabled",
        ):
            if row.get(field) != configuration.get(field):
                raise RuntimeError(f"terminal {field} differs from configuration matrix")
        if row.get("vconfig_requested") != configuration.get("vconfig"):
            raise RuntimeError("terminal requested VConfig differs from configuration matrix")
        if row.get("build_report_path") != workload.get("build_report_path") or row.get(
            "build_report_sha256"
        ) != workload.get("build_report_sha256"):
            raise RuntimeError("terminal build report identity differs from environment")
        _artifact_file(row, "build_report_path", "build_report_sha256")
        _artifact_file(row, "manifest_path", "manifest_sha256")
        _artifact_file(row, "backend_library_path", "backend_sha256")
        metric_contract: MemoryMetricContract | None = None
        cfg_contract: CfgMetricContract | None = None
        include_extended_memory_contract = "memory_sector_bytes" in row
        include_cfg_metric_contract = "cfg_metric_version" in row
        if row.get("feedback_enabled"):
            metadata_path = _artifact_file(
                row,
                "instrumentation_metadata_path",
                "instrumentation_metadata_sha256",
            )
            assert metadata_path is not None
            metadata = _load_object(metadata_path, "feedback metadata")
            try:
                metric_contract = parse_memory_metric_contract(
                    metadata,
                    allow_legacy=True,
                )
                cfg_contract = parse_cfg_metric_contract(metadata)
            except ValueError as error:
                raise RuntimeError(
                    f"invalid feedback metric contract: {metadata_path}: {error}"
                ) from error
            if not isinstance(row.get("instrumented_cfg_sites"), int):
                raise RuntimeError("feedback trial lacks instrumented CFG site count")
            if metadata.get("instrumented_cfg_sites") != row["instrumented_cfg_sites"]:
                raise RuntimeError(
                    "feedback trial CFG site count differs from instrumentation metadata"
                )
            assert cfg_contract is not None
            cfg_metric_versions.add(cfg_contract.cfg_metric_version)
            if include_cfg_metric_contract:
                trial_cfg_contract = {
                    "cfg_metric_version": row.get("cfg_metric_version")
                }
                if trial_cfg_contract != asdict(cfg_contract):
                    raise RuntimeError(
                        "feedback trial CFG metric contract differs from metadata"
                    )
            elif cfg_contract.cfg_metric_version != CFG_PRESENCE_METRIC_VERSION:
                raise RuntimeError(
                    "legacy trial without CFG metric version must use presence semantics"
                )
            if "cfg_counter_saturation" in row:
                raise RuntimeError(
                    "feedback trial must not contain cfg_counter_saturation"
                )
            trial_contract = {
                "memory_metric_version": row.get("memory_metric_version"),
                "memory_map_bits": row.get("memory_map_bits"),
                "memory_sector_bytes": row.get("memory_sector_bytes"),
                "thread_activity_map_bits": row.get("thread_activity_map_bits"),
                "memory_hash_contract_version": row.get(
                    "memory_hash_contract_version"
                ),
            }
            expected_contract = asdict(metric_contract)
            if metric_contract.memory_metric_version == SIMT_MEMORY_METRIC_VERSION:
                if trial_contract != expected_contract:
                    raise RuntimeError(
                        "feedback trial memory metric contract differs from metadata"
                    )
            elif any(value is not None for value in trial_contract.values()):
                legacy_expected = {
                    "memory_metric_version": LEGACY_MEMORY_METRIC_VERSION,
                    "memory_map_bits": MEMORY_MAP_BITS,
                    "memory_sector_bytes": None,
                    "thread_activity_map_bits": THREAD_ACTIVITY_MAP_BITS,
                    "memory_hash_contract_version": None,
                }
                if trial_contract != legacy_expected:
                    raise RuntimeError(
                        "legacy trial memory metric contract differs from metadata"
                    )
        elif (
            row.get("instrumentation_metadata_path") is not None
            or row.get("instrumentation_metadata_sha256") is not None
            or row.get("instrumented_cfg_sites") is not None
            or row.get("memory_metric_version") is not None
            or row.get("memory_map_bits") is not None
            or row.get("memory_sector_bytes") is not None
            or row.get("thread_activity_map_bits") is not None
            or row.get("memory_hash_contract_version") is not None
            or row.get("cfg_metric_version") is not None
        ):
            raise RuntimeError("CuFuzz trial must have null instrumentation metadata")

        status = row.get("status")
        skip_required = (
            configuration.get("vconfig") == "on"
            and workload.get("vconfig_effective") is False
        )
        if status == "skipped":
            skipped += 1
            if row not in trials or not skip_required:
                raise RuntimeError("structured skip is only valid for unsupported on rows")
            if (
                row.get("vconfig_effective") != "unsupported"
                or row.get("vconfig_disabled_reason")
                != workload.get("vconfig_disabled_reason")
            ):
                raise RuntimeError("structured skip reason differs from build contract")
            if samples_by_trial.get(trial_id):
                raise RuntimeError("skipped trial must not have samples")
            continue
        if skip_required:
            raise RuntimeError("unsupported on row must be represented as a structured skip")
        if status == "failed":
            failed += 1
            if row not in failures:
                raise RuntimeError("failed terminal row must be in failures.jsonl")
            _result_file(output, row, "stdout_log", "stdout_sha256")
            _result_file(output, row, "stderr_log", "stderr_sha256")
            _result_file(
                output,
                row,
                "raw_telemetry_path",
                "raw_telemetry_sha256",
                required=False,
            )
            if samples_by_trial.get(trial_id):
                raise RuntimeError("failed trial must not have enriched samples")
            continue
        if status != "completed" or row not in trials:
            raise RuntimeError("terminal row has invalid status/ledger placement")
        completed += 1
        _result_file(output, row, "stdout_log", "stdout_sha256")
        _result_file(output, row, "stderr_log", "stderr_sha256")
        raw_path = _result_file(
            output, row, "raw_telemetry_path", "raw_telemetry_sha256"
        )
        assert raw_path is not None
        try:
            expected_samples = validate_and_enrich_samples(
                raw_path,
                _context_from_trial(
                    row,
                    metric_contract,
                    cfg_contract,
                    include_extended_memory_contract=(
                        include_extended_memory_contract
                    ),
                    include_cfg_metric_contract=include_cfg_metric_contract,
                ),
            )
        except ValueError as error:
            raise RuntimeError(f"completed trial has invalid raw telemetry: {error}") from error
        persisted_samples = samples_by_trial.get(trial_id, [])
        if persisted_samples != expected_samples:
            raise RuntimeError("persisted enriched samples differ from raw telemetry")
        final = expected_samples[-1]
        if (
            row.get("sample_count") != len(expected_samples)
            or row.get("final_executions_completed")
            != final["executions_completed"]
            or row.get("final_timestamp_s") != final["timestamp_s"]
        ):
            raise RuntimeError("completed trial summary differs from final sample")
        if final["timestamp_s"] < coverage_seconds:
            raise RuntimeError("completed trial ended before configured coverage duration")

    unknown_sample_trials = set(samples_by_trial) - seen_trial_ids
    if unknown_sample_trials:
        raise RuntimeError(f"samples reference unknown trials: {unknown_sample_trials!r}")
    if len(cfg_metric_versions) > 1:
        raise RuntimeError(
            f"mixed CFG metric versions are not comparable: "
            f"{sorted(cfg_metric_versions)!r}"
        )
    return {
        "schema_version": 1,
        "status": "ok",
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "samples": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_result_directory(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
