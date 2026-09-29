"""Load and validate RQ2 campaign artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.rq2.schema import load_catalog
from benchmark.rq2.verify_results import (
    CfgMetricContract,
    SIMT_MEMORY_METRIC_VERSION,
    MemoryMetricContract,
    parse_cfg_metric_contract,
    parse_memory_metric_contract,
)
from benchmark.workloads.schema import sha256_file


RQ2_ROOT = Path(__file__).resolve().parent
BACKENDS = ("cufuzz", "origin", "rapid", "rapid2")
VCONFIG_UNSUPPORTED_REASONS = frozenset(
    {"vconfig_barrier_unsupported", "vconfig_inline_asm_unsupported"}
)


@dataclass(frozen=True)
class BackendArtifact:
    library_path: Path
    library_sha256: str
    instrumentation_metadata_path: Path | None
    instrumentation_metadata_sha256: str | None
    instrumented_cfg_sites: int | None
    memory_metric_version: str | None
    memory_map_bits: int | None
    memory_sector_bytes: int | None
    thread_activity_map_bits: int | None
    memory_hash_contract_version: str | None
    cfg_metric_version: str | None = None


@dataclass(frozen=True)
class CampaignInput:
    workload_id: str
    build_report_path: Path
    build_report_sha256: str
    manifest_path: Path
    manifest_sha256: str
    constraints_sha256: str
    vconfig_effective: bool
    vconfig_disabled_reason: str | None
    backends: Mapping[str, BackendArtifact]


@dataclass(frozen=True)
class Configuration:
    configuration_id: str
    paper_label: str
    backend: str
    window: int
    vconfig: str
    feedback_enabled: bool
    asynchronous: bool


CONFIGURATIONS = (
    Configuration("cufuzz-off", "CuFuzz-style", "cufuzz", 1, "off", False, False),
    Configuration("origin-off", "LibAFL+", "origin", 1, "off", True, False),
    Configuration("origin-on", "LibAFL+", "origin", 1, "on", True, False),
    Configuration("rapid-w1-off", "Sys-s (w1)", "rapid", 1, "off", True, False),
    Configuration("rapid-w1-on", "Sys-s (w1)", "rapid", 1, "on", True, False),
    Configuration("rapid-w4-off", "Sys-s (w4)", "rapid", 4, "off", True, False),
    Configuration("rapid-w4-on", "Sys-s (w4)", "rapid", 4, "on", True, False),
    Configuration("rapid2-off", "Sys", "rapid2", 32, "off", True, True),
    Configuration("rapid2-on", "Sys", "rapid2", 32, "on", True, True),
)


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def _artifact_path(backend_root: Path, identity: Mapping[str, Any]) -> Path:
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("build report artifact is missing its path")
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (backend_root / path).resolve()


def _matches_recorded_sha256(actual: str, recorded: object) -> bool:
    if not isinstance(recorded, str):
        return False
    return actual == recorded or actual.removeprefix("sha256:") == recorded


def load_campaign_input(report_path: Path) -> CampaignInput:
    report_path = report_path.resolve()
    report = _read_object(report_path, "RQ2 build report")
    workload_id = report.get("workload_id")
    if not isinstance(workload_id, str) or not workload_id:
        raise RuntimeError(f"build report missing workload_id: {report_path}")
    rq2 = report.get("rq2")
    if not isinstance(rq2, dict) or rq2.get("vconfig_requested") is not True:
        raise RuntimeError(f"build report missing RQ2 VConfig contract: {report_path}")
    vconfig_effective = rq2.get("vconfig_effective")
    if not isinstance(vconfig_effective, bool):
        raise RuntimeError(f"build report has invalid RQ2 VConfig state: {report_path}")
    disabled_reason = rq2.get("vconfig_disabled_reason")
    if vconfig_effective:
        if disabled_reason is not None:
            raise RuntimeError("effective RQ2 VConfig cannot have a disabled reason")
    elif disabled_reason not in VCONFIG_UNSUPPORTED_REASONS:
        raise RuntimeError("ineffective RQ2 VConfig must have a known disabled reason")

    raw_constraints_path = rq2.get("constraints_path")
    if not isinstance(raw_constraints_path, str) or not raw_constraints_path:
        raise RuntimeError(f"build report missing RQ2 constraints path: {report_path}")
    constraints_path = Path(raw_constraints_path)
    if not constraints_path.is_absolute():
        constraints_path = report_path.parent / constraints_path
    constraints_path = constraints_path.resolve()
    constraints_sha256 = f"sha256:{sha256_file(constraints_path)}"
    if not _matches_recorded_sha256(
        constraints_sha256, rq2.get("constraints_sha256")
    ):
        raise RuntimeError(f"constraints hash differs from build report: {constraints_path}")

    shared_phase2 = report.get("shared_phase2")
    if not isinstance(shared_phase2, dict):
        raise RuntimeError(f"build report missing shared_phase2: {report_path}")
    phase2_raw = shared_phase2.get("phase2_dir")
    if not isinstance(phase2_raw, str) or not phase2_raw:
        raise RuntimeError(f"build report missing phase2_dir: {report_path}")
    manifest_path = (Path(phase2_raw).resolve().parent / "manifest.json").resolve()
    manifest_sha256 = f"sha256:{sha256_file(manifest_path)}"
    manifest_identity = shared_phase2.get("artifacts", {}).get("manifest.json", {})
    if not isinstance(manifest_identity, dict) or not _matches_recorded_sha256(
        manifest_sha256, manifest_identity.get("sha256")
    ):
        raise RuntimeError(f"manifest hash differs from build report: {manifest_path}")

    backend_rows = report.get("backends")
    if not isinstance(backend_rows, dict) or set(backend_rows) != set(BACKENDS):
        raise RuntimeError(f"build report must contain exactly {BACKENDS!r}: {report_path}")
    backends: dict[str, BackendArtifact] = {}
    feedback_hashes: set[str] = set()
    feedback_site_counts: set[int] = set()
    feedback_contracts: set[MemoryMetricContract] = set()
    for backend_name in BACKENDS:
        row = backend_rows[backend_name]
        if not isinstance(row, dict) or not isinstance(row.get("artifacts"), dict):
            raise RuntimeError(f"invalid backend row for {backend_name}: {report_path}")
        artifacts = row["artifacts"]
        backend_root = report_path.parent / "backends" / backend_name
        library_identity = artifacts.get("shared_library")
        backend_report_identity = artifacts.get("backend_build.json")
        if not isinstance(library_identity, dict) or not isinstance(
            backend_report_identity, dict
        ):
            raise RuntimeError(f"backend {backend_name} is missing artifact identities")
        library_path = _artifact_path(backend_root, library_identity)
        library_sha256 = f"sha256:{sha256_file(library_path)}"
        if not _matches_recorded_sha256(
            library_sha256, library_identity.get("sha256")
        ):
            raise RuntimeError(f"backend shared-library hash drift: {library_path}")
        backend_report_path = _artifact_path(backend_root, backend_report_identity)
        backend_report_sha256 = f"sha256:{sha256_file(backend_report_path)}"
        if not _matches_recorded_sha256(
            backend_report_sha256, backend_report_identity.get("sha256")
        ):
            raise RuntimeError(f"backend build report hash drift: {backend_report_path}")
        backend_report = _read_object(backend_report_path, "backend build report")

        metadata_path: Path | None = None
        metadata_sha256: str | None = None
        instrumented_cfg_sites: int | None = None
        metric_contract: MemoryMetricContract | None = None
        cfg_contract: CfgMetricContract | None = None
        raw_metadata_path = backend_report.get("feedback_metadata")
        if backend_name == "cufuzz":
            if raw_metadata_path is not None:
                raise RuntimeError("CuFuzz backend must not expose feedback metadata")
        else:
            if not isinstance(raw_metadata_path, str) or not raw_metadata_path:
                raise RuntimeError(
                    f"feedback backend {backend_name} is missing feedback_metadata"
                )
            candidate = Path(raw_metadata_path)
            metadata_path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (backend_report_path.parent / candidate).resolve()
            )
            metadata = _read_object(metadata_path, "feedback metadata")
            try:
                metric_contract = parse_memory_metric_contract(
                    metadata,
                    allow_legacy=False,
                )
                cfg_contract = parse_cfg_metric_contract(metadata)
            except ValueError as error:
                raise RuntimeError(
                    f"invalid feedback metric contract: {metadata_path}: {error}"
                ) from error
            if metric_contract.memory_metric_version != SIMT_MEMORY_METRIC_VERSION:
                raise RuntimeError(
                    f"new RQ2 campaign requires {SIMT_MEMORY_METRIC_VERSION}: "
                    f"{metadata_path}"
                )
            instrumented_cfg_sites = metadata.get("instrumented_cfg_sites")
            if (
                not isinstance(instrumented_cfg_sites, int)
                or isinstance(instrumented_cfg_sites, bool)
                or instrumented_cfg_sites <= 0
            ):
                raise RuntimeError(
                    f"feedback metadata has invalid instrumented_cfg_sites: {metadata_path}"
                )
            metadata_sha256 = f"sha256:{sha256_file(metadata_path)}"
            feedback_hashes.add(metadata_sha256)
            feedback_site_counts.add(instrumented_cfg_sites)
            feedback_contracts.add(metric_contract)
        backends[backend_name] = BackendArtifact(
            library_path=library_path,
            library_sha256=library_sha256,
            instrumentation_metadata_path=metadata_path,
            instrumentation_metadata_sha256=metadata_sha256,
            instrumented_cfg_sites=instrumented_cfg_sites,
            memory_metric_version=(
                metric_contract.memory_metric_version if metric_contract else None
            ),
            memory_map_bits=(metric_contract.memory_map_bits if metric_contract else None),
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
            cfg_metric_version=(
                cfg_contract.cfg_metric_version if cfg_contract else None
            ),
        )
    if (
        len(feedback_hashes) != 1
        or len(feedback_site_counts) != 1
        or len(feedback_contracts) != 1
    ):
        raise RuntimeError(
            f"feedback metadata differ across origin/rapid/rapid2 for {workload_id}"
        )
    return CampaignInput(
        workload_id=workload_id,
        build_report_path=report_path,
        build_report_sha256=f"sha256:{sha256_file(report_path)}",
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        constraints_sha256=constraints_sha256,
        vconfig_effective=vconfig_effective,
        vconfig_disabled_reason=disabled_reason,
        backends=backends,
    )


def load_campaign_inputs(build_root: Path) -> tuple[CampaignInput, ...]:
    summary_path = build_root.resolve() / "build_summary.json"
    summary = _read_object(summary_path, "RQ2 build summary")
    report_paths = summary.get("reports")
    if not isinstance(report_paths, list):
        raise RuntimeError("RQ2 build summary reports must be an array")
    inputs = tuple(
        load_campaign_input(
            Path(raw_path)
            if Path(raw_path).is_absolute()
            else (summary_path.parent / raw_path)
        )
        for raw_path in report_paths
        if isinstance(raw_path, str)
    )
    expected_ids = tuple(item.workload_id for item in load_catalog(RQ2_ROOT).workloads)
    actual_ids = tuple(item.workload_id for item in inputs)
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"RQ2 build reports must follow catalog order: {actual_ids!r} != {expected_ids!r}"
        )
    return inputs


def validate_cfg_presence_contracts(inputs: Sequence[CampaignInput]) -> None:
    found_contract = False
    for campaign_input in inputs:
        for artifact in campaign_input.backends.values():
            if artifact.instrumentation_metadata_path is None:
                continue
            try:
                parse_cfg_metric_contract(
                    {"cfg_metric_version": artifact.cfg_metric_version}
                )
            except ValueError as error:
                raise RuntimeError(
                    f"invalid CFG metric contract for {campaign_input.workload_id}: {error}"
                ) from error
            found_contract = True
    if not found_contract:
        raise RuntimeError("RQ2 campaign has no CFG presence contract")


def select_campaign_inputs(
    inputs: Sequence[CampaignInput],
    requested_workloads: Sequence[str] | None,
) -> tuple[CampaignInput, ...]:
    ordered_inputs = tuple(inputs)
    if requested_workloads is None:
        return ordered_inputs

    requested = tuple(requested_workloads)
    if any(not workload_id for workload_id in requested):
        raise ValueError("RQ2 workload IDs must not be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate RQ2 workload selection")

    available = {item.workload_id for item in ordered_inputs}
    unknown = tuple(workload_id for workload_id in requested if workload_id not in available)
    if unknown:
        raise ValueError(f"unknown RQ2 workload selection: {unknown!r}")

    selected = set(requested)
    return tuple(item for item in ordered_inputs if item.workload_id in selected)
