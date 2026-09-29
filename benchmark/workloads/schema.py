"""Schema and integrity checks for the shared evaluation workload corpus."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_WORKLOAD_IDS = (
    "shoc_reduction",
    "shoc_radix_sort",
    "shoc_scan",
    "cutlass_gemm",
    "flashattention_device1xn",
    "pytorch_batchnorm",
    "apex_maybe_cast",
    "llama_upscale_f32_bilinear",
    "gpurir_generate_rir",
    "apex_index_mul_2d_vgeo",
    "cuda_samples_inverse_cnd",
    "synth_complex",
)
SOURCE_KINDS = frozenset(("standalone", "third_party", "synthetic"))
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Workload:
    workload_id: str
    label: str
    source_kind: str
    root: Path
    provenance: str | None = None
    project: str | None = None
    kernel_id: str | None = None
    support_registry: str | None = None
    constraint_registry: str | None = None
    submodule: str | None = None
    submodule_commit: str | None = None
    upstream_path: str | None = None
    adapter: str | None = None
    adapter_sha256: str | None = None

    def path(self, relative: str) -> Path:
        return _confined_path(self.root, relative, f"{self.workload_id} path")


@dataclass(frozen=True)
class Catalog:
    schema_version: int
    workloads: tuple[Workload, ...]

    def by_id(self) -> dict[str, Workload]:
        return {workload.workload_id: workload for workload in self.workloads}


@dataclass(frozen=True)
class Suite:
    schema_version: int
    workload_ids: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return result


def _confined_path(root: Path, relative: str, context: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{context} must be relative")
    repository_root = root.resolve().parents[1]
    resolved = (root.resolve() / candidate).resolve()
    if resolved != repository_root and repository_root not in resolved.parents:
        raise ValueError(f"{context} must stay inside the repository")
    return resolved


def load_catalog(root: Path) -> Catalog:
    root = root.resolve()
    raw = _load_json(root / "catalog.json")
    if raw.get("schema_version") != 1:
        raise ValueError("workload catalog schema_version must be 1")
    entries = raw.get("workloads")
    if not isinstance(entries, list):
        raise ValueError("workload catalog workloads must be an array")
    workloads: list[Workload] = []
    for index, raw_entry in enumerate(entries):
        context = f"workloads[{index}]"
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{context} must be an object")
        source_kind = _required_string(raw_entry, "source_kind", context)
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"{context}.source_kind is unsupported: {source_kind}")
        common = {
            "workload_id": _required_string(raw_entry, "id", context),
            "label": _required_string(raw_entry, "label", context),
            "source_kind": source_kind,
            "root": root,
        }
        if source_kind in ("standalone", "synthetic"):
            provenance = _required_string(raw_entry, "provenance", context)
            _confined_path(root, provenance, f"{context}.provenance")
            workloads.append(Workload(**common, provenance=provenance))
            continue
        adapter = raw_entry.get("adapter")
        adapter_sha256 = raw_entry.get("adapter_sha256")
        if (adapter is None) != (adapter_sha256 is None):
            raise ValueError(f"{context} adapter and adapter_sha256 must appear together")
        if adapter is not None:
            if not isinstance(adapter, str) or not adapter:
                raise ValueError(f"{context}.adapter must be a non-empty string")
            if not isinstance(adapter_sha256, str) or not _SHA256_RE.fullmatch(adapter_sha256):
                raise ValueError(f"{context}.adapter_sha256 must be lowercase SHA-256")
            _confined_path(root, adapter, f"{context}.adapter")
        workload = Workload(
            **common,
            project=_required_string(raw_entry, "project", context),
            kernel_id=_required_string(raw_entry, "kernel_id", context),
            support_registry=_required_string(raw_entry, "support_registry", context),
            constraint_registry=_required_string(raw_entry, "constraint_registry", context),
            submodule=_required_string(raw_entry, "submodule", context),
            submodule_commit=_required_string(raw_entry, "submodule_commit", context),
            upstream_path=_required_string(raw_entry, "upstream_path", context),
            adapter=adapter,
            adapter_sha256=adapter_sha256,
        )
        if not _COMMIT_RE.fullmatch(workload.submodule_commit or ""):
            raise ValueError(f"{context}.submodule_commit must be lowercase 40-hex")
        for field in (
            "support_registry",
            "constraint_registry",
            "submodule",
            "upstream_path",
        ):
            _confined_path(root, str(getattr(workload, field)), f"{context}.{field}")
        workloads.append(workload)
    return Catalog(schema_version=1, workloads=tuple(workloads))


def load_suite(path: Path, catalog: Catalog) -> Suite:
    raw = _load_json(path)
    if raw.get("schema_version") != 1:
        raise ValueError(f"suite schema_version must be 1: {path}")
    values = raw.get("workloads")
    if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"suite workloads must be non-empty strings: {path}")
    workload_ids = tuple(values)
    if len(workload_ids) != len(set(workload_ids)):
        raise ValueError(f"suite contains duplicate workloads: {path}")
    unknown = sorted(set(workload_ids) - set(catalog.by_id()))
    if unknown:
        raise ValueError(f"suite contains unknown workloads {unknown}: {path}")
    return Suite(schema_version=1, workload_ids=workload_ids)


def _registry_has_kernel(path: Path, kernel_id: str, *, require_run: bool) -> bool:
    raw = _load_json(path)
    entries = raw.get("kernels") if require_run else raw.get("overrides")
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kernel_id") != kernel_id:
            continue
        return not require_run or entry.get("decision") == "run"
    return False


def verify_catalog(root: Path) -> list[str]:
    try:
        catalog = load_catalog(root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [str(error)]
    errors: list[str] = []
    actual_ids = tuple(workload.workload_id for workload in catalog.workloads)
    if actual_ids != EXPECTED_WORKLOAD_IDS:
        errors.append(f"shared workload IDs/order differ: expected {EXPECTED_WORKLOAD_IDS!r}, got {actual_ids!r}")
    for workload in catalog.workloads:
        if workload.source_kind in ("standalone", "synthetic"):
            path = workload.path(workload.provenance or "")
            if not path.is_file():
                errors.append(f"{workload.workload_id}: missing provenance {path}")
                continue
            provenance = _load_json(path)
            if provenance.get("workload_id") != workload.workload_id:
                errors.append(f"{workload.workload_id}: provenance workload ID differs")
            expected_kind = "synthetic" if workload.source_kind == "synthetic" else "upstream"
            if provenance.get("source_kind", "upstream") != expected_kind:
                errors.append(f"{workload.workload_id}: provenance source kind differs")
            continue
        support = workload.path(workload.support_registry or "")
        constraints = workload.path(workload.constraint_registry or "")
        if not support.is_file() or not _registry_has_kernel(support, workload.kernel_id or "", require_run=True):
            errors.append(f"{workload.workload_id}: support registry does not approve kernel")
        if not constraints.is_file() or not _registry_has_kernel(constraints, workload.kernel_id or "", require_run=False):
            errors.append(f"{workload.workload_id}: constraint registry does not select kernel")
        if workload.adapter is not None:
            adapter = workload.path(workload.adapter)
            if not adapter.is_file():
                errors.append(f"{workload.workload_id}: missing adapter {adapter}")
            elif sha256_file(adapter) != workload.adapter_sha256:
                errors.append(f"{workload.workload_id}: stale adapter SHA-256")
        submodule = workload.path(workload.submodule or "")
        if not submodule.is_dir():
            errors.append(f"{workload.workload_id}: missing submodule {submodule}")
            continue
        upstream = submodule / str(workload.upstream_path)
        if any(submodule.iterdir()) and not upstream.is_file():
            errors.append(f"{workload.workload_id}: missing initialized upstream source {upstream}")
    return errors
