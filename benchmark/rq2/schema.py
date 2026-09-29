"""Schema and RQ1-provenance checks for the RQ2 workload catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from benchmark.rq1.schema import (
    EXPECTED_WORKLOAD_IDS as RQ1_EXPECTED_WORKLOAD_IDS,
    Workload as Rq1Workload,
    _load_json,
    _required_string,
    load_catalog as load_rq1_catalog,
    load_provenance,
    sha256_file,
)


SYNTHETIC_WORKLOAD_IDS = ("synth_complex",)
EXPECTED_WORKLOAD_IDS = (*RQ1_EXPECTED_WORKLOAD_IDS, *SYNTHETIC_WORKLOAD_IDS)

ADAPTED_VCONFIG_WORKLOADS: Mapping[str, tuple[tuple[int, int, int], ...]] = {
    "flashattention_device1xn": (
        (32, 1, 1),
        (64, 1, 1),
        (128, 1, 1),
        (256, 1, 1),
    ),
    "shoc_scan": ((32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1)),
    "shoc_reduction": ((32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1)),
}
LOGICAL_BLOCK_CANDIDATES: Mapping[str, tuple[tuple[int, int, int], ...]] = {
    **ADAPTED_VCONFIG_WORKLOADS,
    "shoc_radix_sort": ((64, 1, 1), (96, 1, 1), (128, 1, 1)),
    "synth_complex": ((32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1)),
}


def _arg_value(arg: int) -> dict[str, Any]:
    return {"kind": "arg_value", "arg": arg}


def _product(*factors: Mapping[str, Any]) -> dict[str, Any]:
    expression = dict(factors[0])
    for factor in factors[1:]:
        expression = {
            "kind": "binary",
            "op": "*",
            "lhs": expression,
            "rhs": dict(factor),
        }
    return expression


def _resize_for_float_product(buffer_arg: int, *scalar_args: int) -> dict[str, Any]:
    return {
        "kind": "expression_compare",
        "lhs": _product(
            *(_arg_value(arg) for arg in scalar_args),
            {"kind": "const", "value": 4},
        ),
        "op": "<=",
        "rhs": {"kind": "payload_len", "arg": buffer_arg},
        "repair": {"kind": "resize_payload", "arg": buffer_arg},
    }


RQ2_DOMAIN_OVERRIDES: Mapping[str, Mapping[int, Mapping[str, str]]] = {
    "flashattention_device1xn": {
        0: {"min_len": "4", "max_len": "1024"},
        1: {"min_len": "1024", "max_len": "1024"},
    },
    "shoc_scan": {0: {"min_len": "4"}},
    "shoc_reduction": {
        0: {"min_len": "4"},
        2: {"min": "1"},
    },
    "cutlass_gemm": {
        4: {"min_len": "4"},
        5: {"min": "1"},
        6: {"min_len": "4", "max_len": "16"},
        9: {"min_len": "4", "max_len": "16"},
    },
    "pytorch_batchnorm": {
        0: {"min_len": "4"},
        1: {"min_len": "4"},
        2: {"min_len": "4"},
        3: {"min_len": "4"},
        4: {"min_len": "4"},
        5: {"min_len": "4"},
        6: {"min_len": "4"},
        7: {"min": "1"},
        8: {"min": "1"},
    },
}

RQ2_ADDITIONAL_DOMAINS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "flashattention_device1xn": (
        {
            "arg": 2,
            "name": "n",
            "type": "const unsigned int",
            "domain": {
                "kind": "int_range",
                "min": "1",
                "max": "256",
                "signed": False,
            },
        },
    ),
}

RQ2_ADDITIONAL_CONSTRAINTS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "flashattention_device1xn": (
        {
            "kind": "count_fits_buffer",
            "count_arg": 2,
            "buffer_arg": 0,
            "elem_size_bytes": 4,
        },
        {
            "kind": "scalar_le_logical_block_dim",
            "scalar_arg": 2,
            "dimension": "x",
        },
    ),
    "shoc_scan": (
        {
            "kind": "scalar_le_logical_block_dim",
            "scalar_arg": 1,
            "dimension": "x",
        },
    ),
    "cutlass_gemm": (
        {
            "kind": "scalar_compare_scalar",
            "lhs_arg": 5,
            "op": ">=",
            "rhs_arg": 0,
        },
        _resize_for_float_product(4, 2, 5),
        {
            "kind": "count_fits_buffer",
            "count_arg": 2,
            "buffer_arg": 6,
            "elem_size_bytes": 4,
        },
        {
            "kind": "count_fits_buffer",
            "count_arg": 0,
            "buffer_arg": 9,
            "elem_size_bytes": 4,
        },
    ),
    "pytorch_batchnorm": (
        _resize_for_float_product(0, 7, 8),
        _resize_for_float_product(1, 7, 8),
        _resize_for_float_product(2, 8),
        _resize_for_float_product(3, 8),
        _resize_for_float_product(4, 8),
        _resize_for_float_product(5, 8),
        _resize_for_float_product(6, 7, 8),
    ),
}


@dataclass(frozen=True)
class Workload:
    workload_id: str
    label: str
    rq1_provenance: str | None
    synthetic_provenance: str | None
    kernel: str
    constraints: str
    vconfig_adaptation: str | None
    root: Path

    @property
    def rq1_provenance_path(self) -> Path:
        if self.rq1_provenance is None:
            raise ValueError(f"{self.workload_id}: no RQ1 provenance record")
        return self.root / self.rq1_provenance

    @property
    def synthetic_provenance_path(self) -> Path:
        if self.synthetic_provenance is None:
            raise ValueError(f"{self.workload_id}: no synthetic provenance record")
        return self.root / self.synthetic_provenance

    @property
    def source_provenance_path(self) -> Path:
        if self.synthetic_provenance is not None:
            return self.synthetic_provenance_path
        return self.rq1_provenance_path

    @property
    def kernel_path(self) -> Path:
        return self.root / self.kernel

    @property
    def constraints_path(self) -> Path:
        return self.root / self.constraints

    @property
    def vconfig_adaptation_path(self) -> Path:
        if self.vconfig_adaptation is None:
            raise ValueError(f"{self.workload_id}: no VConfig adaptation record")
        return self.root / self.vconfig_adaptation


@dataclass(frozen=True)
class Catalog:
    schema_version: int
    workloads: tuple[Workload, ...]


def _catalog_relative(relative: str, context: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{context} must be a confined relative path")
    return candidate


def _rq1_root(root: Path) -> Path:
    return root.resolve().parent / "rq1"


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
        kernel = _required_string(entry, "kernel", context)
        constraints = _required_string(entry, "constraints", context)
        _catalog_relative(kernel, f"{context}.kernel")
        _catalog_relative(constraints, f"{context}.constraints")
        rq1_provenance = entry.get("rq1_provenance")
        synthetic_provenance = entry.get("synthetic_provenance")
        if (rq1_provenance is None) == (synthetic_provenance is None):
            raise ValueError(
                f"{context} must define exactly one provenance record"
            )
        if rq1_provenance is not None:
            if not isinstance(rq1_provenance, str) or not rq1_provenance:
                raise ValueError(
                    f"{context}.rq1_provenance must be a non-empty string"
                )
            if Path(rq1_provenance).is_absolute():
                raise ValueError(f"{context}.rq1_provenance must be relative")
        if synthetic_provenance is not None:
            if not isinstance(synthetic_provenance, str) or not synthetic_provenance:
                raise ValueError(
                    f"{context}.synthetic_provenance must be a non-empty string"
                )
            _catalog_relative(
                synthetic_provenance, f"{context}.synthetic_provenance"
            )
        raw_adaptation = entry.get("vconfig_adaptation")
        if raw_adaptation is not None:
            if not isinstance(raw_adaptation, str) or not raw_adaptation:
                raise ValueError(f"{context}.vconfig_adaptation must be a non-empty string")
            _catalog_relative(raw_adaptation, f"{context}.vconfig_adaptation")
        workloads.append(
            Workload(
                workload_id=workload_id,
                label=_required_string(entry, "label", context),
                rq1_provenance=rq1_provenance,
                synthetic_provenance=synthetic_provenance,
                kernel=kernel,
                constraints=constraints,
                vconfig_adaptation=raw_adaptation,
                root=root,
            )
        )
    return Catalog(schema_version=1, workloads=tuple(workloads))


def _expected_constraints(
    rq1_constraints: Mapping[str, Any], workload: Workload, physical_block: tuple[int, int, int]
) -> Mapping[str, Any]:
    expected = json.loads(json.dumps(rq1_constraints))
    expected["project"] = f"rq2_{workload.workload_id}"
    override = expected["overrides"][0]
    domains = {item["arg"]: item["domain"] for item in override["domains"]}
    for arg, domain_override in RQ2_DOMAIN_OVERRIDES.get(workload.workload_id, {}).items():
        domains[arg].update(domain_override)
    override["domains"].extend(
        json.loads(json.dumps(RQ2_ADDITIONAL_DOMAINS.get(workload.workload_id, ())))
    )
    override["constraints"].extend(
        json.loads(json.dumps(RQ2_ADDITIONAL_CONSTRAINTS.get(workload.workload_id, ())))
    )
    launch_policy = override["launch_policy"]
    if workload.workload_id == "flashattention_device1xn":
        launch_policy["block_candidates"] = [256]
        launch_policy["physical_block_max"] = 256
    launch_policy["logical_grid"] = [1, 1, 1]
    launch_policy["logical_block"] = list(physical_block)
    launch_policy["vconfig_reserved"] = True
    launch_policy["vconfig_mutation"] = True
    candidates = LOGICAL_BLOCK_CANDIDATES.get(workload.workload_id)
    if candidates is not None:
        launch_policy["logical_block_candidates"] = [list(item) for item in candidates]
    for evidence in override["evidence"]:
        evidence["file"] = evidence["file"].replace("benchmark/rq1/", "benchmark/rq2/", 1)
    return expected


def _json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def _verify_adaptation(
    *,
    root: Path,
    rq1_root: Path,
    workload: Workload,
    rq1_workload: Any,
    provenance: Any,
) -> list[str]:
    errors: list[str] = []
    if provenance.upstream is None:
        return [f"{workload.workload_id}: adapted source must have upstream provenance"]
    expected_record = f"workloads/{workload.workload_id}/vconfig_adaptation.json"
    if workload.vconfig_adaptation != expected_record:
        return [
            f"{workload.workload_id}: VConfig adaptation must be {expected_record!r}"
        ]
    if workload.kernel_path.is_symlink() or not workload.kernel_path.is_file():
        errors.append(f"{workload.workload_id}: adapted RQ2 kernel must be a regular file")
    try:
        record = _load_json(workload.vconfig_adaptation_path)
    except ValueError as error:
        return errors + [f"{workload.workload_id}: invalid VConfig adaptation: {error}"]

    rq1_kernel = (
        rq1_root / rq1_workload.provenance_path
    ).parent / provenance.extraction.kernel_file
    patch_path = workload.kernel_path.parent / "vconfig_adaptation.patch"
    expected_paths = {
        "base_kernel": str(rq1_kernel.relative_to(root.parent.parent)),
        "adapted_kernel": str(workload.kernel_path.relative_to(root.parent.parent)),
        "adaptation_patch": str(patch_path.relative_to(root.parent.parent)),
    }
    for field, path in expected_paths.items():
        value = record.get(field)
        if not isinstance(value, dict) or value.get("path") != path:
            errors.append(f"{workload.workload_id}: adaptation {field} path mismatch")
            continue
        actual_path = {
            "base_kernel": rq1_kernel,
            "adapted_kernel": workload.kernel_path,
            "adaptation_patch": patch_path,
        }[field]
        if not actual_path.is_file() or value.get("sha256") != sha256_file(actual_path):
            errors.append(f"{workload.workload_id}: adaptation {field} SHA-256 mismatch")

    expected_candidates = [
        list(item) for item in ADAPTED_VCONFIG_WORKLOADS[workload.workload_id]
    ]
    if record.get("logical_block_candidates") != expected_candidates:
        errors.append(f"{workload.workload_id}: adaptation candidates mismatch")
    if record.get("schema_version") != 1 or record.get("workload_id") != workload.workload_id:
        errors.append(f"{workload.workload_id}: adaptation identity mismatch")
    if not isinstance(record.get("reason"), str) or not record["reason"]:
        errors.append(f"{workload.workload_id}: adaptation reason missing")
    expected_upstream = {
        "repository": provenance.upstream.repository,
        "commit": provenance.upstream.commit,
        "path": provenance.upstream.path,
    }
    if record.get("upstream") != expected_upstream:
        errors.append(f"{workload.workload_id}: adaptation upstream provenance mismatch")
    return errors


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

    rq1_root = _rq1_root(root)
    try:
        rq1_catalog = load_rq1_catalog(rq1_root)
    except ValueError as error:
        return errors + [f"cannot load RQ1 catalog: {error}"]
    rq1_by_id = {item.workload_id: item for item in rq1_catalog.workloads}

    for workload in catalog.workloads:
        synthetic = workload.workload_id in SYNTHETIC_WORKLOAD_IDS
        rq1_workload = rq1_by_id.get(workload.workload_id)
        if synthetic:
            expected_provenance = (
                f"workloads/{workload.workload_id}/provenance.json"
            )
            if workload.synthetic_provenance != expected_provenance:
                errors.append(
                    f"{workload.workload_id}: synthetic provenance must be "
                    f"{expected_provenance!r}"
                )
                continue
            if workload.rq1_provenance is not None:
                errors.append(
                    f"{workload.workload_id}: synthetic workload cannot claim RQ1 provenance"
                )
                continue
            source_workload = Rq1Workload(
                workload_id=workload.workload_id,
                label=workload.label,
                provenance_path=expected_provenance,
            )
            source_root = root
            provenance_label = "synthetic"
        else:
            if workload.synthetic_provenance is not None:
                errors.append(
                    f"{workload.workload_id}: RQ1 workload cannot use synthetic provenance"
                )
                continue
            if rq1_workload is None:
                errors.append(f"{workload.workload_id}: missing RQ1 workload")
                continue
            expected_provenance = f"../rq1/{rq1_workload.provenance_path}"
            if workload.rq1_provenance != expected_provenance:
                errors.append(
                    f"{workload.workload_id}: RQ1 provenance must be "
                    f"{expected_provenance!r}"
                )
                continue
            if workload.label != rq1_workload.label:
                errors.append(f"{workload.workload_id}: label differs from RQ1")
            source_workload = rq1_workload
            source_root = rq1_root
            provenance_label = "RQ1"
        try:
            provenance = load_provenance(source_root, source_workload)
        except ValueError as error:
            errors.append(
                f"{workload.workload_id}: invalid {provenance_label} provenance: {error}"
            )
            continue
        if synthetic and provenance.source_kind != "synthetic":
            errors.append(
                f"{workload.workload_id}: provenance source_kind must be 'synthetic'"
            )
            continue
        if not synthetic and provenance.source_kind != "upstream":
            errors.append(
                f"{workload.workload_id}: RQ1 provenance must remain upstream-backed"
            )
            continue

        expected_kernel = f"workloads/{workload.workload_id}/kernel.cu"
        if workload.kernel != expected_kernel:
            errors.append(f"{workload.workload_id}: kernel must be {expected_kernel!r}")
            continue
        kernel_path = root / workload.kernel
        source_dir = (
            source_root / source_workload.provenance_path
        ).resolve().parent
        source_kernel = source_dir / provenance.extraction.kernel_file
        if synthetic:
            if workload.vconfig_adaptation is not None:
                errors.append(f"{workload.workload_id}: unexpected VConfig adaptation")
            if kernel_path.is_symlink() or not kernel_path.is_file():
                errors.append(
                    f"{workload.workload_id}: synthetic kernel must be a regular file"
                )
            elif sha256_file(kernel_path) != provenance.extraction.kernel_sha256:
                errors.append(f"{workload.workload_id}: stale kernel SHA-256")
        elif workload.workload_id in ADAPTED_VCONFIG_WORKLOADS:
            errors.extend(
                _verify_adaptation(
                    root=root,
                    rq1_root=rq1_root,
                    workload=workload,
                    rq1_workload=rq1_workload,
                    provenance=provenance,
                )
            )
        elif workload.vconfig_adaptation is not None:
            errors.append(f"{workload.workload_id}: unexpected VConfig adaptation")
        elif not kernel_path.is_symlink():
            errors.append(f"{workload.workload_id}: RQ2 kernel must be a symlink")
        else:
            link_target = kernel_path.readlink()
            if link_target.is_absolute():
                errors.append(f"{workload.workload_id}: RQ2 kernel symlink must be relative")
            expected_link_target = (
                Path("../../../rq1")
                / Path(source_workload.provenance_path).parent
                / provenance.extraction.kernel_file
            )
            if link_target != expected_link_target:
                errors.append(
                    f"{workload.workload_id}: RQ2 kernel symlink must use the "
                    "canonical relative target"
                )
            if kernel_path.resolve() != source_kernel.resolve():
                errors.append(f"{workload.workload_id}: RQ2 kernel must resolve to the RQ1 kernel")
            elif sha256_file(kernel_path) != sha256_file(source_kernel):
                errors.append(f"{workload.workload_id}: RQ2 kernel differs from the RQ1 kernel")

        expected_constraints = f"workloads/{workload.workload_id}/constraints.json"
        if workload.constraints != expected_constraints:
            errors.append(f"{workload.workload_id}: constraints must be {expected_constraints!r}")
            continue
        source_constraints_path = source_dir / provenance.extraction.constraints_file
        try:
            if synthetic:
                if not source_constraints_path.is_file():
                    raise ValueError(
                        f"missing synthetic base constraints: {source_constraints_path}"
                    )
                if (
                    sha256_file(source_constraints_path)
                    != provenance.extraction.constraints_sha256
                ):
                    raise ValueError("stale synthetic base constraints SHA-256")
            actual_constraints = _load_json(root / workload.constraints)
            expected_constraints_json = _expected_constraints(
                _load_json(source_constraints_path), workload, provenance.execution.block
            )
        except ValueError as error:
            errors.append(f"{workload.workload_id}: cannot validate constraints: {error}")
            continue
        if not _json_equal(actual_constraints, expected_constraints_json):
            errors.append(
                f"{workload.workload_id}: constraints differ from the approved RQ2 derivation"
            )
    return errors
