"""Strict schema and provenance checks for the RQ1 workload catalog."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from benchmark.workloads.schema import sha256_file


EXPECTED_WORKLOAD_IDS = (
    "shoc_reduction",
    "shoc_radix_sort",
    "shoc_scan",
    "cutlass_gemm",
    "flashattention_device1xn",
    "pytorch_batchnorm",
)

FORBIDDEN_SOURCE_TOKENS = (
    "__LIBAFL_EDGE",
    "device_cov_map",
    "rapid_bound_cov_map",
    "rapid_feedback_bind_context",
    "rapid_feedback_prepare_task",
    "rapid_feedback_record",
    "rapid_feedback_merge",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Upstream:
    repository: str
    commit: str
    path: str
    source_file: str
    sha256: str
    license_file: str


@dataclass(frozen=True)
class Extraction:
    kernel_file: str
    kernel_sha256: str
    patch_file: str | None
    patch_sha256: str | None
    constraints_file: str
    constraints_sha256: str


@dataclass(frozen=True)
class ExecutionContract:
    entry: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_bytes: int
    payload_size: int
    vconfig: Mapping[str, Any]


@dataclass(frozen=True)
class Workload:
    workload_id: str
    label: str
    provenance_path: str


@dataclass(frozen=True)
class Catalog:
    schema_version: int
    workloads: tuple[Workload, ...]


@dataclass(frozen=True)
class SyntheticSource:
    description: str
    design: str


@dataclass(frozen=True)
class Provenance:
    schema_version: int
    workload_id: str
    source_kind: str
    upstream: Upstream | None
    synthetic: SyntheticSource | None
    extraction: Extraction
    execution: ExecutionContract


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return result


def _confined_relative(root: Path, relative: str, context: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{context} must be relative and confined below benchmark root")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{context} must be confined below benchmark root")
    return resolved


def _catalog_artifact(root: Path, relative: str, context: str) -> Path:
    """Resolve a suite artifact from the shared benchmark workload corpus.

    Suite catalogs keep lexical paths below their RQ directory for stable
    compatibility, while ``benchmark/<rq>/workloads`` may be a symlink to the
    single canonical ``benchmark/workloads`` corpus.  Reject lexical escapes
    and only permit the resolved shared-corpus destination.
    """

    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{context} must be relative and confined below benchmark root")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    shared_root = (resolved_root.parent / "workloads").resolve()
    if not (
        resolved == resolved_root
        or resolved_root in resolved.parents
        or resolved == shared_root
        or shared_root in resolved.parents
    ):
        raise ValueError(f"{context} must be confined below benchmark root")
    return resolved


def _triple(value: Any, context: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ValueError(f"{context} must be three positive integers")
    return (value[0], value[1], value[2])


def load_catalog(root: Path) -> Catalog:
    root = root.resolve()
    raw = _load_json(root / "catalog.json")
    if raw.get("schema_version") != 1:
        raise ValueError("catalog.schema_version must be 1")
    entries = raw.get("workloads")
    if not isinstance(entries, list):
        raise ValueError("catalog.workloads must be an array")

    workloads: list[Workload] = []
    for index, entry in enumerate(entries):
        context = f"catalog.workloads[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{context} must be an object")
        workload_id = _required_string(entry, "id", context)
        provenance_path = _required_string(entry, "provenance", context)
        _catalog_artifact(root, provenance_path, f"{context}.provenance")
        workloads.append(
            Workload(
                workload_id=workload_id,
                label=_required_string(entry, "label", context),
                provenance_path=provenance_path,
            )
        )
    return Catalog(schema_version=1, workloads=tuple(workloads))


def load_provenance(root: Path, workload: Workload) -> Provenance:
    root = root.resolve()
    path = _catalog_artifact(root, workload.provenance_path, "provenance path")
    raw = _load_json(path)
    try:
        context = str(path.relative_to(root))
    except ValueError:
        context = str(path.relative_to(root.parent))
    if raw.get("schema_version") != 1:
        raise ValueError(f"{context}.schema_version must be 1")
    workload_id = _required_string(raw, "workload_id", context)
    if workload_id != workload.workload_id:
        raise ValueError(
            f"{context}.workload_id is {workload_id!r}, expected {workload.workload_id!r}"
        )

    source_kind = raw.get("source_kind", "upstream")
    if source_kind not in ("upstream", "synthetic"):
        raise ValueError(f"{context}.source_kind must be 'upstream' or 'synthetic'")

    upstream_raw = raw.get("upstream")
    synthetic_raw = raw.get("synthetic")
    extraction_raw = raw.get("extraction")
    execution_raw = raw.get("execution")
    if not isinstance(extraction_raw, dict):
        raise ValueError(f"{context}.extraction must be an object")
    if not isinstance(execution_raw, dict):
        raise ValueError(f"{context}.execution must be an object")

    upstream: Upstream | None = None
    synthetic: SyntheticSource | None = None
    if source_kind == "upstream":
        if not isinstance(upstream_raw, dict):
            raise ValueError(f"{context}.upstream must be an object")
        if synthetic_raw is not None:
            raise ValueError(f"{context}.synthetic is only valid for synthetic sources")
        repository = _required_string(
            upstream_raw, "repository", f"{context}.upstream"
        )
        if not repository.startswith("https://"):
            raise ValueError(f"{context}.upstream.repository must use https://")
        commit = _required_string(upstream_raw, "commit", f"{context}.upstream")
        if not _COMMIT_RE.fullmatch(commit):
            raise ValueError(
                f"{context}.upstream.commit must be 40 lowercase hex characters"
            )
        upstream = Upstream(
            repository=repository,
            commit=commit,
            path=_required_string(upstream_raw, "path", f"{context}.upstream"),
            source_file=_required_string(
                upstream_raw, "source_file", f"{context}.upstream"
            ),
            sha256=_required_string(upstream_raw, "sha256", f"{context}.upstream"),
            license_file=_required_string(
                upstream_raw, "license_file", f"{context}.upstream"
            ),
        )
    else:
        if upstream_raw is not None:
            raise ValueError(f"{context}.upstream is not valid for synthetic sources")
        if not isinstance(synthetic_raw, dict):
            raise ValueError(f"{context}.synthetic must be an object")
        synthetic = SyntheticSource(
            description=_required_string(
                synthetic_raw, "description", f"{context}.synthetic"
            ),
            design=_required_string(synthetic_raw, "design", f"{context}.synthetic"),
        )

    patch_file = extraction_raw.get("patch_file")
    patch_sha256 = extraction_raw.get("patch_sha256")
    if source_kind == "upstream":
        patch_file = _required_string(
            extraction_raw, "patch_file", f"{context}.extraction"
        )
        patch_sha256 = _required_string(
            extraction_raw, "patch_sha256", f"{context}.extraction"
        )
    elif patch_file is not None or patch_sha256 is not None:
        raise ValueError(
            f"{context}.extraction patch fields are not valid for synthetic sources"
        )
    extraction = Extraction(
        kernel_file=_required_string(extraction_raw, "kernel_file", f"{context}.extraction"),
        kernel_sha256=_required_string(extraction_raw, "kernel_sha256", f"{context}.extraction"),
        patch_file=patch_file,
        patch_sha256=patch_sha256,
        constraints_file=_required_string(extraction_raw, "constraints_file", f"{context}.extraction"),
        constraints_sha256=_required_string(extraction_raw, "constraints_sha256", f"{context}.extraction"),
    )
    hashes = [
        ("extraction.kernel_sha256", extraction.kernel_sha256),
        ("extraction.constraints_sha256", extraction.constraints_sha256),
    ]
    if upstream is not None:
        hashes.append(("upstream.sha256", upstream.sha256))
    if extraction.patch_sha256 is not None:
        hashes.append(("extraction.patch_sha256", extraction.patch_sha256))
    for field, value in hashes:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{context}.{field} must be 64 lowercase hex characters")

    dynamic_shared_bytes = execution_raw.get("dynamic_shared_bytes")
    payload_size = execution_raw.get("payload_size")
    vconfig = execution_raw.get("vconfig")
    if not isinstance(dynamic_shared_bytes, int) or dynamic_shared_bytes < 0:
        raise ValueError(f"{context}.execution.dynamic_shared_bytes must be non-negative")
    if not isinstance(payload_size, int) or payload_size <= 0:
        raise ValueError(f"{context}.execution.payload_size must be positive")
    if not isinstance(vconfig, dict):
        raise ValueError(f"{context}.execution.vconfig must be an object")
    execution = ExecutionContract(
        entry=_required_string(execution_raw, "entry", f"{context}.execution"),
        grid=_triple(execution_raw.get("grid"), f"{context}.execution.grid"),
        block=_triple(execution_raw.get("block"), f"{context}.execution.block"),
        dynamic_shared_bytes=dynamic_shared_bytes,
        payload_size=payload_size,
        vconfig=vconfig,
    )
    return Provenance(
        schema_version=1,
        workload_id=workload_id,
        source_kind=source_kind,
        upstream=upstream,
        synthetic=synthetic,
        extraction=extraction,
        execution=execution,
    )


def verify_catalog(root: Path) -> list[str]:
    root = root.resolve()
    try:
        catalog = load_catalog(root)
    except ValueError as error:
        return [str(error)]

    errors: list[str] = []
    actual_ids = tuple(item.workload_id for item in catalog.workloads)
    if actual_ids != EXPECTED_WORKLOAD_IDS:
        errors.append(
            "catalog workload IDs/order differ: "
            f"expected {EXPECTED_WORKLOAD_IDS!r}, got {actual_ids!r}"
        )

    for workload in catalog.workloads:
        try:
            provenance = load_provenance(root, workload)
        except ValueError as error:
            errors.append(str(error))
            continue
        if (
            provenance.source_kind != "upstream"
            or provenance.upstream is None
            or provenance.extraction.patch_file is None
            or provenance.extraction.patch_sha256 is None
        ):
            errors.append(
                f"{workload.workload_id}: RQ1 provenance must be upstream-backed"
            )
            continue
        workload_root = (root / workload.provenance_path).resolve().parent
        paths_and_hashes = [
            (provenance.extraction.kernel_file, provenance.extraction.kernel_sha256, "kernel"),
            (provenance.extraction.constraints_file, provenance.extraction.constraints_sha256, "constraints"),
        ]
        if provenance.upstream is not None:
            paths_and_hashes[:0] = [
                (provenance.upstream.source_file, provenance.upstream.sha256, "upstream source"),
                (provenance.upstream.license_file, None, "upstream license"),
            ]
        if provenance.extraction.patch_file is not None:
            paths_and_hashes.append(
                (
                    provenance.extraction.patch_file,
                    provenance.extraction.patch_sha256,
                    "extraction patch",
                )
            )
        for relative, expected_hash, label in paths_and_hashes:
            try:
                path = _confined_relative(workload_root, relative, f"{workload.workload_id} {label}")
            except ValueError as error:
                errors.append(str(error))
                continue
            if not path.is_file():
                try:
                    display_path = path.relative_to(root)
                except ValueError:
                    display_path = path.relative_to(root.parent)
                errors.append(
                    f"{workload.workload_id}: missing {label}: {display_path}"
                )
                continue
            if expected_hash is not None:
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    errors.append(
                        f"{workload.workload_id}: stale {label} SHA-256: "
                        f"expected {expected_hash}, got {actual_hash}"
                    )

        kernel_path = workload_root / provenance.extraction.kernel_file
        if kernel_path.is_file():
            source = kernel_path.read_text(encoding="utf-8")
            for token in FORBIDDEN_SOURCE_TOKENS:
                if token in source:
                    errors.append(
                        f"{workload.workload_id}: forbidden manual-feedback token {token!r}"
                    )
    return errors
