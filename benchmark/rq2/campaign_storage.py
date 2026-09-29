"""Persist and resume RQ2 campaign records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.rq2.campaign_model import CampaignInput, Configuration


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _trial_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    workload_id = row.get("workload_id")
    configuration_id = row.get("configuration_id")
    repetition = row.get("repetition")
    if (
        not isinstance(workload_id, str)
        or not isinstance(configuration_id, str)
        or not isinstance(repetition, int)
        or isinstance(repetition, bool)
    ):
        raise RuntimeError("terminal campaign row has an invalid trial key")
    return workload_id, configuration_id, repetition


def load_attempted_trials(output: Path) -> set[tuple[str, str, int]]:
    attempted: set[tuple[str, str, int]] = set()
    for ledger in ("trials.jsonl", "failures.jsonl"):
        for row in _load_jsonl(output / ledger):
            key = _trial_key(row)
            if key in attempted:
                raise RuntimeError(f"duplicate terminal campaign row: {key!r}")
            attempted.add(key)
    return attempted


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_unique_samples(
    path: Path, samples: Sequence[Mapping[str, Any]]
) -> None:
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        trial_id = row.get("trial_id")
        sequence = row.get("sequence")
        if (
            not isinstance(trial_id, str)
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
        ):
            raise RuntimeError("existing sample has an invalid trial_id/sequence")
        key = (trial_id, sequence)
        if key in existing:
            raise RuntimeError(f"duplicate existing sample key: {key!r}")
        existing[key] = row
    for sample in samples:
        row = dict(sample)
        trial_id = row.get("trial_id")
        sequence = row.get("sequence")
        if (
            not isinstance(trial_id, str)
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
        ):
            raise RuntimeError("new sample has an invalid trial_id/sequence")
        key = (trial_id, sequence)
        prior = existing.get(key)
        if prior is not None:
            if prior != row:
                raise RuntimeError(f"sample collision has different content: {key!r}")
            continue
        _append_jsonl(path, row)
        existing[key] = row


def _remove_orphan_samples(output: Path) -> None:
    terminal_trial_ids: set[str] = set()
    for ledger in ("trials.jsonl", "failures.jsonl"):
        for row in _load_jsonl(output / ledger):
            trial_id = row.get("trial_id")
            if not isinstance(trial_id, str) or not trial_id:
                raise RuntimeError("terminal campaign row has an invalid trial_id")
            terminal_trial_ids.add(trial_id)

    samples_path = output / "samples.jsonl"
    samples = _load_jsonl(samples_path)
    retained = []
    seen: set[tuple[str, int]] = set()
    for row in samples:
        trial_id = row.get("trial_id")
        sequence = row.get("sequence")
        if (
            not isinstance(trial_id, str)
            or not trial_id
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
        ):
            raise RuntimeError("existing sample has an invalid trial_id/sequence")
        key = (trial_id, sequence)
        if key in seen:
            raise RuntimeError(f"duplicate existing sample key: {key!r}")
        seen.add(key)
        if trial_id in terminal_trial_ids:
            retained.append(row)

    if len(retained) == len(samples):
        return
    temporary_path = samples_path.with_name(f".{samples_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        for row in retained:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, samples_path)


def _terminal_base(
    campaign_input: CampaignInput,
    configuration: Configuration,
    repetition: int,
    seed: int,
    plan_index: int,
) -> dict[str, Any]:
    artifact = campaign_input.backends[configuration.backend]
    return {
        "schema_version": 1,
        "trial_id": (
            f"r{repetition:03d}-{campaign_input.workload_id}-"
            f"{configuration.configuration_id}"
        ),
        "workload_id": campaign_input.workload_id,
        "configuration_id": configuration.configuration_id,
        "paper_label": configuration.paper_label,
        "backend": configuration.backend,
        "window": configuration.window,
        "repetition": repetition,
        "plan_index": plan_index,
        "seed": seed,
        "vconfig_requested": configuration.vconfig,
        "feedback_enabled": configuration.feedback_enabled,
        "build_report_path": str(campaign_input.build_report_path),
        "build_report_sha256": campaign_input.build_report_sha256,
        "manifest_path": str(campaign_input.manifest_path),
        "manifest_sha256": campaign_input.manifest_sha256,
        "constraints_sha256": campaign_input.constraints_sha256,
        "backend_library_path": str(artifact.library_path),
        "backend_sha256": artifact.library_sha256,
        "instrumentation_metadata_path": (
            str(artifact.instrumentation_metadata_path)
            if artifact.instrumentation_metadata_path is not None
            else None
        ),
        "instrumentation_metadata_sha256": artifact.instrumentation_metadata_sha256,
        "instrumented_cfg_sites": artifact.instrumented_cfg_sites,
        "memory_metric_version": artifact.memory_metric_version,
        "memory_map_bits": artifact.memory_map_bits,
        "memory_sector_bytes": artifact.memory_sector_bytes,
        "thread_activity_map_bits": artifact.thread_activity_map_bits,
        "memory_hash_contract_version": artifact.memory_hash_contract_version,
        "cfg_metric_version": artifact.cfg_metric_version,
        "external_features": None,
    }
