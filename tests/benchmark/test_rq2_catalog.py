import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.rq1.schema import (
    EXPECTED_WORKLOAD_IDS as RQ1_EXPECTED_WORKLOAD_IDS,
    load_catalog as load_rq1_catalog,
)
from benchmark.rq1.schema import load_provenance as load_rq1_provenance
from benchmark.rq1.schema import sha256_file
from benchmark.rq2.schema import load_catalog, verify_catalog


REPO_ROOT = Path(__file__).resolve().parents[2]
RQ1_ROOT = REPO_ROOT / "benchmark" / "rq1"
RQ2_ROOT = REPO_ROOT / "benchmark" / "rq2"
WORKLOADS_ROOT = REPO_ROOT / "benchmark" / "workloads"
EXPECTED_WORKLOAD_IDS = (*RQ1_EXPECTED_WORKLOAD_IDS, "synth_complex")


@contextlib.contextmanager
def temporary_rq2_root():
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark_root = Path(temp_dir) / "benchmark"
        benchmark_root.mkdir()
        (benchmark_root / "rq1").symlink_to(RQ1_ROOT, target_is_directory=True)
        shutil.copytree(WORKLOADS_ROOT, benchmark_root / "workloads")
        rq2_root = benchmark_root / "rq2"
        shutil.copytree(
            RQ2_ROOT,
            rq2_root,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "build", "results"),
        )
        yield rq2_root


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


OPEN_DOMAIN_BOUNDS = {
    "flashattention_device1xn": {0: (4, 1024), 2: (1, 256)},
    "shoc_scan": {0: (4, 1024)},
    "shoc_reduction": {0: (4, 2048), 2: (1, 512)},
    "cutlass_gemm": {
        4: (4, 64),
        5: (1, 4),
        6: (4, 16),
        9: (4, 16),
    },
    "pytorch_batchnorm": {
        0: (4, 1024),
        1: (4, 1024),
        2: (4, 64),
        3: (4, 64),
        4: (4, 64),
        5: (4, 64),
        6: (4, 1024),
        7: (1, 16),
        8: (1, 16),
    },
    "synth_complex": {0: (128, 4096), 2: (128, 4096)},
}
EXPECTED_REPAIR_KINDS = {
    "flashattention_device1xn": [
        "count_fits_buffer",
        "scalar_le_logical_block_dim",
    ],
    "shoc_scan": ["count_fits_buffer", "scalar_le_logical_block_dim"],
    "shoc_reduction": ["count_fits_buffer"],
    "cutlass_gemm": [
        "scalar_compare_scalar",
        "expression_compare",
        "count_fits_buffer",
        "count_fits_buffer",
    ],
    "pytorch_batchnorm": ["expression_compare"] * 7,
    "synth_complex": ["count_fits_buffer"],
}


def _override(workload_id: str) -> dict[str, object]:
    path = RQ2_ROOT / "workloads" / workload_id / "constraints.json"
    return json.loads(path.read_text(encoding="utf-8"))["overrides"][0]


def _domain_map(override: dict[str, object]) -> dict[int, dict[str, object]]:
    return {item["arg"]: item["domain"] for item in override["domains"]}


def _bound(domain: dict[str, object], name: str) -> int:
    key = name if domain["kind"] == "int_range" else f"{name}_len"
    return int(domain[key])


def _eval_expr(
    expr: dict[str, object], scalars: dict[int, int], buffers: dict[int, int]
) -> int:
    kind = expr["kind"]
    if kind == "arg_value":
        return scalars[expr["arg"]]
    if kind == "payload_len":
        return buffers[expr["arg"]]
    if kind == "const":
        return expr["value"]
    lhs = _eval_expr(expr["lhs"], scalars, buffers)
    rhs = _eval_expr(expr["rhs"], scalars, buffers)
    return {"+": lhs + rhs, "*": lhs * rhs}[expr["op"]]


def _normalize_contract(
    override: dict[str, object],
    scalars: dict[int, int],
    buffers: dict[int, int],
    *,
    logical_block_x: int,
) -> tuple[dict[int, int], dict[int, int]]:
    domains = _domain_map(override)
    scalars = {
        arg: max(_bound(domains[arg], "min"), min(value, _bound(domains[arg], "max")))
        for arg, value in scalars.items()
    }
    buffers = {
        arg: max(_bound(domains[arg], "min"), min(value, _bound(domains[arg], "max")))
        for arg, value in buffers.items()
    }

    constraints = override["constraints"]
    for _ in range(len(constraints) + 1):
        before = (dict(scalars), dict(buffers))
        for constraint in constraints:
            kind = constraint["kind"]
            if kind == "scalar_compare_scalar":
                if constraint["op"] != ">=":
                    raise AssertionError(f"unsupported test repair op: {constraint['op']}")
                lhs = constraint["lhs_arg"]
                rhs = constraint["rhs_arg"]
                if scalars[lhs] < scalars[rhs]:
                    scalars[lhs] = min(scalars[rhs], _bound(domains[lhs], "max"))
                    scalars[rhs] = min(scalars[rhs], scalars[lhs])
            elif kind == "count_fits_buffer":
                count = constraint["count_arg"]
                buffer = constraint["buffer_arg"]
                elem_size = constraint["elem_size_bytes"]
                required = scalars[count] * elem_size
                buffers[buffer] = max(
                    buffers[buffer], min(required, _bound(domains[buffer], "max"))
                )
                scalars[count] = min(scalars[count], buffers[buffer] // elem_size)
            elif kind == "expression_compare":
                if constraint["op"] != "<=":
                    raise AssertionError(f"unsupported test repair op: {constraint['op']}")
                required = _eval_expr(constraint["lhs"], scalars, buffers)
                buffer = constraint["repair"]["arg"]
                buffers[buffer] = max(
                    buffers[buffer], min(required, _bound(domains[buffer], "max"))
                )
        for constraint in constraints:
            if constraint["kind"] == "scalar_le_logical_block_dim":
                scalars[constraint["scalar_arg"]] = min(
                    scalars[constraint["scalar_arg"]], logical_block_x
                )
        if before == (scalars, buffers):
            break
    return scalars, buffers


def _constraint_holds(
    constraint: dict[str, object],
    scalars: dict[int, int],
    buffers: dict[int, int],
    *,
    logical_block_x: int,
) -> bool:
    kind = constraint["kind"]
    if kind == "scalar_compare_scalar":
        return scalars[constraint["lhs_arg"]] >= scalars[constraint["rhs_arg"]]
    if kind == "count_fits_buffer":
        return (
            scalars[constraint["count_arg"]] * constraint["elem_size_bytes"]
            <= buffers[constraint["buffer_arg"]]
        )
    if kind == "expression_compare":
        return _eval_expr(constraint["lhs"], scalars, buffers) <= _eval_expr(
            constraint["rhs"], scalars, buffers
        )
    if kind == "scalar_le_logical_block_dim":
        return scalars[constraint["scalar_arg"]] <= logical_block_x
    raise AssertionError(f"unsupported test constraint: {kind}")


class Rq2CatalogTests(unittest.TestCase):
    def test_only_approved_workloads_use_regular_sources(self) -> None:
        adapted_ids = {
            "flashattention_device1xn",
            "shoc_scan",
            "shoc_reduction",
        }
        regular_source_ids = {*adapted_ids, "synth_complex"}

        for workload in load_catalog(RQ2_ROOT).workloads:
            with self.subTest(workload=workload.workload_id):
                if workload.workload_id in regular_source_ids:
                    self.assertTrue(workload.kernel_path.is_file())
                    self.assertFalse(workload.kernel_path.is_symlink())
                else:
                    self.assertTrue(workload.kernel_path.is_symlink())
                if workload.workload_id in adapted_ids:
                    self.assertIsNotNone(workload.vconfig_adaptation)
                    self.assertTrue(workload.vconfig_adaptation_path.is_file())
                else:
                    self.assertIsNone(workload.vconfig_adaptation)

    def test_catalog_has_six_rq1_workloads_then_the_synthetic_workload(self) -> None:
        catalog = load_catalog(RQ2_ROOT)

        self.assertEqual(
            tuple(workload.workload_id for workload in catalog.workloads),
            EXPECTED_WORKLOAD_IDS,
        )

    def test_catalog_resolves_each_rq1_provenance_artifact(self) -> None:
        catalog = load_catalog(RQ2_ROOT)
        rq1_catalog = load_rq1_catalog(RQ1_ROOT)
        rq1_by_id = {item.workload_id: item for item in rq1_catalog.workloads}

        for workload in catalog.workloads:
            with self.subTest(workload=workload.workload_id):
                if workload.workload_id == "synth_complex":
                    self.assertIsNone(workload.rq1_provenance)
                    self.assertEqual(
                        workload.synthetic_provenance,
                        "workloads/synth_complex/provenance.json",
                    )
                    self.assertTrue(workload.synthetic_provenance_path.is_file())
                    provenance = json.loads(
                        workload.synthetic_provenance_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(provenance["source_kind"], "synthetic")
                    continue
                expected_rq1 = rq1_by_id[workload.workload_id]
                self.assertEqual(workload.label, expected_rq1.label)
                self.assertEqual(
                    workload.rq1_provenance,
                    f"../rq1/{expected_rq1.provenance_path}",
                )
                self.assertTrue(workload.rq1_provenance_path.is_file())

    def test_synth_complex_provenance_uses_canonical_payload_size(self) -> None:
        workload_root = RQ2_ROOT / "workloads" / "synth_complex"
        constraints = json.loads(
            (workload_root / "base_constraints.json").read_text(encoding="utf-8")
        )
        provenance = json.loads(
            (workload_root / "provenance.json").read_text(encoding="utf-8")
        )

        payload_size = 0
        for argument in constraints["overrides"][0]["domains"]:
            domain = argument["domain"]
            if domain["kind"] == "bytes":
                payload_size += 8 + int(domain["min_len"])
            elif domain["kind"] == "int_range":
                self.assertEqual(argument["type"], "const unsigned int")
                payload_size += 4
            else:
                self.fail(f"unexpected synthetic domain kind: {domain['kind']}")

        self.assertEqual(payload_size, 1172)
        self.assertEqual(provenance["execution"]["payload_size"], payload_size)

    def test_canonical_kernel_symlinks_are_byte_identical_to_rq1_sources(self) -> None:
        catalog = load_catalog(RQ2_ROOT)
        rq1_catalog = load_rq1_catalog(RQ1_ROOT)
        rq1_by_id = {item.workload_id: item for item in rq1_catalog.workloads}

        for workload in catalog.workloads:
            if workload.workload_id in {
                "flashattention_device1xn",
                "shoc_scan",
                "shoc_reduction",
                "synth_complex",
            }:
                continue
            with self.subTest(workload=workload.workload_id):
                rq1_provenance = load_rq1_provenance(
                    RQ1_ROOT, rq1_by_id[workload.workload_id]
                )
                rq1_kernel = (
                    RQ1_ROOT
                    / rq1_by_id[workload.workload_id].provenance_path
                ).parent / rq1_provenance.extraction.kernel_file
                self.assertTrue(workload.kernel_path.is_symlink())
                self.assertEqual(
                    sha256_file(workload.kernel_path),
                    sha256_file(rq1_kernel),
                )

    def test_constraints_derive_from_rq1_and_enable_vconfig(self) -> None:
        catalog = load_catalog(RQ2_ROOT)
        rq1_catalog = load_rq1_catalog(RQ1_ROOT)
        rq1_by_id = {item.workload_id: item for item in rq1_catalog.workloads}

        for workload in catalog.workloads:
            with self.subTest(workload=workload.workload_id):
                if workload.workload_id == "synth_complex":
                    source_constraints = (
                        RQ2_ROOT
                        / "workloads"
                        / "synth_complex"
                        / "base_constraints.json"
                    )
                    physical_block = (256, 1, 1)
                else:
                    rq1_workload = rq1_by_id[workload.workload_id]
                    rq1_provenance = load_rq1_provenance(RQ1_ROOT, rq1_workload)
                    source_constraints = (
                        RQ1_ROOT / rq1_workload.provenance_path
                    ).parent / "constraints.json"
                    physical_block = rq1_provenance.execution.block
                rq1_constraints = json.loads(
                    source_constraints.read_text(encoding="utf-8")
                )
                rq2_constraints = json.loads(
                    workload.constraints_path.read_text(encoding="utf-8")
                )
                rq1_override = rq1_constraints["overrides"][0]
                rq2_override = rq2_constraints["overrides"][0]
                rq2_launch = rq2_override["launch_policy"]

                self.assertEqual(rq2_constraints["project"], f"rq2_{workload.workload_id}")
                expected_domains = [
                    (item["arg"], item["name"], item["type"])
                    for item in rq1_override["domains"]
                ]
                if workload.workload_id == "flashattention_device1xn":
                    expected_domains.append((2, "n", "const unsigned int"))
                self.assertEqual(
                    [
                        (item["arg"], item["name"], item["type"])
                        for item in rq2_override["domains"]
                    ],
                    expected_domains,
                )
                self.assertEqual(rq2_launch["logical_grid"], [1, 1, 1])
                self.assertEqual(
                    rq2_launch["logical_block"], list(physical_block)
                )
                self.assertTrue(rq2_launch["vconfig_reserved"])
                self.assertTrue(rq2_launch["vconfig_mutation"])
                expected_candidates = {
                    "shoc_reduction": [[32, 1, 1], [64, 1, 1], [128, 1, 1], [256, 1, 1]],
                    "shoc_radix_sort": [[64, 1, 1], [96, 1, 1], [128, 1, 1]],
                    "shoc_scan": [[32, 1, 1], [64, 1, 1], [128, 1, 1], [256, 1, 1]],
                    "flashattention_device1xn": [
                        [32, 1, 1],
                        [64, 1, 1],
                        [128, 1, 1],
                        [256, 1, 1],
                    ],
                    "synth_complex": [
                        [32, 1, 1],
                        [64, 1, 1],
                        [128, 1, 1],
                        [256, 1, 1],
                    ],
                }.get(workload.workload_id)
                if expected_candidates is None:
                    self.assertNotIn("logical_block_candidates", rq2_launch)
                else:
                    self.assertEqual(
                        rq2_launch["logical_block_candidates"], expected_candidates
                    )
                self.assertTrue(
                    all(
                        item["file"].startswith(
                            f"benchmark/rq2/workloads/{workload.workload_id}/"
                        )
                        for item in rq2_override["evidence"]
                    )
                )

    def test_flashattention_has_exact_domains_repairs_and_launch_policy(self) -> None:
        override = _override("flashattention_device1xn")
        domains = _domain_map(override)

        self.assertEqual(
            domains,
            {
                0: {
                    "kind": "bytes",
                    "min_len": "4",
                    "max_len": "1024",
                    "elem_size_bytes": 4,
                    "nullable": False,
                },
                1: {
                    "kind": "bytes",
                    "min_len": "1024",
                    "max_len": "1024",
                    "elem_size_bytes": 4,
                    "nullable": False,
                },
                2: {
                    "kind": "int_range",
                    "min": "1",
                    "max": "256",
                    "signed": False,
                },
            },
        )
        self.assertEqual(
            override["constraints"],
            [
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
            ],
        )
        launch = override["launch_policy"]
        self.assertEqual(launch["grid"], [1, 1, 1])
        self.assertEqual(launch["block_candidates"], [256])
        self.assertEqual(launch["physical_block_max"], 256)
        self.assertEqual(launch["target_dynamic_shared_bytes"], 0)
        self.assertEqual(launch["logical_grid"], [1, 1, 1])
        self.assertEqual(launch["logical_block"], [128, 1, 1])
        self.assertEqual(
            launch["logical_block_candidates"],
            [[32, 1, 1], [64, 1, 1], [128, 1, 1], [256, 1, 1]],
        )
        self.assertTrue(launch["vconfig_reserved"])
        self.assertTrue(launch["vconfig_mutation"])

    def test_flashattention_repairs_clamp_n_and_are_idempotent(self) -> None:
        override = _override("flashattention_device1xn")
        for logical_block_x in (32, 64, 128, 256):
            with self.subTest(logical_block_x=logical_block_x):
                normalized = _normalize_contract(
                    override,
                    {2: 256},
                    {0: 4, 1: 1024},
                    logical_block_x=logical_block_x,
                )
                self.assertEqual(normalized[0][2], logical_block_x)
                self.assertGreaterEqual(normalized[1][0], 4 * logical_block_x)
                self.assertEqual(normalized[1][1], 1024)
                self.assertEqual(
                    _normalize_contract(
                        override,
                        *normalized,
                        logical_block_x=logical_block_x,
                    ),
                    normalized,
                )

        oversized = _normalize_contract(
            override,
            {2: 5},
            {0: 1024, 1: 1024},
            logical_block_x=32,
        )
        self.assertEqual(oversized, ({2: 5}, {0: 1024, 1: 1024}))

    def test_synth_complex_has_exact_domains_repairs_and_launch_policy(self) -> None:
        override = _override("synth_complex")

        self.assertEqual(
            _domain_map(override),
            {
                0: {
                    "kind": "bytes",
                    "min_len": "128",
                    "max_len": "4096",
                    "elem_size_bytes": 1,
                    "nullable": False,
                },
                1: {
                    "kind": "bytes",
                    "min_len": "1024",
                    "max_len": "1024",
                    "elem_size_bytes": 4,
                    "nullable": False,
                },
                2: {
                    "kind": "int_range",
                    "min": "128",
                    "max": "4096",
                    "signed": False,
                },
            },
        )
        self.assertEqual(
            override["constraints"],
            [
                {
                    "kind": "count_fits_buffer",
                    "count_arg": 2,
                    "buffer_arg": 0,
                    "elem_size_bytes": 1,
                }
            ],
        )
        self.assertEqual(
            override["launch_policy"],
            {
                "grid": [1, 1, 1],
                "block_candidates": [256],
                "physical_block_max": 256,
                "target_dynamic_shared_bytes": 0,
                "coverage_memory": "global",
                "vconfig_reserved": True,
                "logical_grid": [1, 1, 1],
                "logical_block": [256, 1, 1],
                "logical_block_candidates": [
                    [32, 1, 1],
                    [64, 1, 1],
                    [128, 1, 1],
                    [256, 1, 1],
                ],
                "vconfig_mutation": True,
            },
        )

    def test_synth_complex_adversarial_repair_is_idempotent(self) -> None:
        override = _override("synth_complex")

        normalized = _normalize_contract(
            override,
            {2: 4096},
            {0: 129, 1: 1023},
            logical_block_x=32,
        )

        self.assertEqual(normalized, ({2: 4096}, {0: 4096, 1: 1024}))
        self.assertTrue(
            all(
                _constraint_holds(
                    constraint,
                    *normalized,
                    logical_block_x=32,
                )
                for constraint in override["constraints"]
            )
        )
        self.assertEqual(
            _normalize_contract(
                override,
                *normalized,
                logical_block_x=32,
            ),
            normalized,
        )

    def test_synth_complex_has_controlled_select_and_if_forms(self) -> None:
        source = (
            RQ2_ROOT / "workloads" / "synth_complex" / "kernel.cu"
        ).read_text(encoding="utf-8")
        normalized = " ".join(source.split())

        self.assertIn(
            "return condition ? true_value : false_value;",
            normalized,
        )
        self.assertIn("if (condition)", normalized)
        self.assertIn(
            "synth_select_form(paired_condition, left, right)",
            normalized,
        )
        self.assertIn(
            "synth_if_form(paired_condition, left, right, branch_sink + tid)",
            normalized,
        )

    def test_audited_capacity_domains_are_open(self) -> None:
        for workload_id, expected_bounds in OPEN_DOMAIN_BOUNDS.items():
            domains = _domain_map(_override(workload_id))
            for arg, bounds in expected_bounds.items():
                with self.subTest(workload=workload_id, arg=arg):
                    self.assertEqual(
                        (_bound(domains[arg], "min"), _bound(domains[arg], "max")),
                        bounds,
                    )

    def test_open_domain_maxima_fit_all_repair_constraints(self) -> None:
        for workload_id in OPEN_DOMAIN_BOUNDS:
            override = _override(workload_id)
            domains = _domain_map(override)
            scalars = {
                arg: _bound(domain, "max")
                for arg, domain in domains.items()
                if domain["kind"] == "int_range"
            }
            buffers = {
                arg: _bound(domain, "max")
                for arg, domain in domains.items()
                if domain["kind"] == "bytes"
            }
            with self.subTest(workload=workload_id):
                self.assertEqual(
                    [item["kind"] for item in override["constraints"]],
                    EXPECTED_REPAIR_KINDS[workload_id],
                )
                self.assertTrue(
                    all(
                        _constraint_holds(
                            constraint,
                            scalars,
                            buffers,
                            logical_block_x=256,
                        )
                        for constraint in override["constraints"]
                    )
                )
                self.assertLess(sum(buffers.values()) + 4096, 4 * 1024 * 1024)

    def test_extreme_length_mutations_repair_to_an_idempotent_contract(self) -> None:
        for workload_id in OPEN_DOMAIN_BOUNDS:
            override = _override(workload_id)
            domains = _domain_map(override)
            self.assertTrue(
                all(
                    _bound(domains[arg], "min") < _bound(domains[arg], "max")
                    for arg in OPEN_DOMAIN_BOUNDS[workload_id]
                )
            )
            scalars = {
                arg: _bound(domain, "max")
                for arg, domain in domains.items()
                if domain["kind"] == "int_range"
            }
            if workload_id == "cutlass_gemm":
                scalars[5] = _bound(domains[5], "min")
            buffers = {
                arg: _bound(domain, "min") + 1
                for arg, domain in domains.items()
                if domain["kind"] == "bytes"
            }
            if workload_id == "cutlass_gemm":
                self.assertLess(scalars[5], scalars[0])
                for arg in (4, 6, 9):
                    buffers[arg] = _bound(domains[arg], "min")
            logical_block_x = (
                32
                if workload_id in {"flashattention_device1xn", "shoc_scan"}
                else 256
            )

            normalized = _normalize_contract(
                override, scalars, buffers, logical_block_x=logical_block_x
            )
            normalized_twice = _normalize_contract(
                override,
                *normalized,
                logical_block_x=logical_block_x,
            )

            with self.subTest(workload=workload_id):
                if workload_id == "cutlass_gemm":
                    self.assertEqual(normalized[0][5], 4)
                    self.assertEqual(normalized[1][4], 64)
                self.assertTrue(
                    all(
                        _constraint_holds(
                            constraint,
                            *normalized,
                            logical_block_x=logical_block_x,
                        )
                        for constraint in override["constraints"]
                    )
                )
                self.assertEqual(normalized_twice, normalized)

    def test_catalog_verifier_accepts_all_rq2_contracts(self) -> None:
        self.assertEqual(verify_catalog(RQ2_ROOT), [])

    def test_verifier_rejects_altered_rq1_provenance(self) -> None:
        with temporary_rq2_root() as root:
            catalog_path = root / "catalog.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog["workloads"][0]["rq1_provenance"] = (
                "../rq1/workloads/shoc_scan/provenance.json"
            )
            write_json(catalog_path, catalog)

            errors = verify_catalog(root)

        self.assertTrue(
            any(
                error.startswith("shoc_reduction: RQ1 provenance must be")
                for error in errors
            ),
            errors,
        )

    def test_verifier_rejects_relative_symlinks_that_are_not_canonical(self) -> None:
        workload_id = "cutlass_gemm"
        cases = {
            "absolute": (
                RQ1_ROOT / "workloads" / workload_id / "kernel.cu",
                "RQ2 kernel symlink must be relative",
            ),
            "rewritten": (
                Path(
                    f"../../../rq1/workloads/{workload_id}/../{workload_id}/kernel.cu"
                ),
                "RQ2 kernel symlink must use the canonical relative target",
            ),
            "noncanonical": (
                Path("../../../rq1/workloads/shoc_scan/kernel.cu"),
                "RQ2 kernel must resolve to the RQ1 kernel",
            ),
        }
        for name, (target, expected_error) in cases.items():
            with self.subTest(case=name), temporary_rq2_root() as root:
                kernel = root / "workloads" / workload_id / "kernel.cu"
                kernel.unlink()
                kernel.symlink_to(target)

                errors = verify_catalog(root)

                self.assertIn(f"{workload_id}: {expected_error}", errors)

    def test_verifier_rejects_changed_constraint(self) -> None:
        with temporary_rq2_root() as root:
            constraints_path = root / "workloads" / "shoc_scan" / "constraints.json"
            constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
            constraints["overrides"][0]["launch_policy"]["vconfig_mutation"] = False
            write_json(constraints_path, constraints)

            errors = verify_catalog(root)

        self.assertIn(
            "shoc_scan: constraints differ from the approved RQ2 derivation", errors
        )

    def test_verifier_rejects_changed_synthetic_kernel(self) -> None:
        with temporary_rq2_root() as root:
            kernel_path = root / "workloads" / "synth_complex" / "kernel.cu"
            kernel_path.write_text(
                kernel_path.read_text(encoding="utf-8") + "\n// drift\n",
                encoding="utf-8",
            )

            errors = verify_catalog(root)

        self.assertIn("synth_complex: stale kernel SHA-256", errors)

    def test_verifier_rejects_missing_adaptation_record(self) -> None:
        with temporary_rq2_root() as root:
            (root / "workloads" / "shoc_scan" / "vconfig_adaptation.json").unlink()

            errors = verify_catalog(root)

        self.assertTrue(
            any(error.startswith("shoc_scan: invalid VConfig adaptation:") for error in errors),
            errors,
        )

    def test_verifier_rejects_adaptation_hash_mismatches(self) -> None:
        for field in ("base_kernel", "adapted_kernel", "adaptation_patch"):
            with self.subTest(field=field), temporary_rq2_root() as root:
                record_path = (
                    root / "workloads" / "shoc_scan" / "vconfig_adaptation.json"
                )
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record[field]["sha256"] = "0" * 64
                write_json(record_path, record)

                errors = verify_catalog(root)

                self.assertIn(
                    f"shoc_scan: adaptation {field} SHA-256 mismatch", errors
                )

    def test_verifier_rejects_adaptation_contract_mismatches(self) -> None:
        cases = {
            "candidates": (
                "logical_block_candidates",
                [[64, 1, 1], [128, 1, 1], [256, 1, 1]],
                "shoc_scan: adaptation candidates mismatch",
            ),
            "upstream": (
                "upstream",
                {
                    "repository": "https://example.invalid/shoc",
                    "commit": "0" * 40,
                    "path": "scan_kernel.h",
                },
                "shoc_scan: adaptation upstream provenance mismatch",
            ),
        }
        for name, (field, value, expected_error) in cases.items():
            with self.subTest(case=name), temporary_rq2_root() as root:
                record_path = (
                    root / "workloads" / "shoc_scan" / "vconfig_adaptation.json"
                )
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record[field] = value
                write_json(record_path, record)

                errors = verify_catalog(root)

                self.assertIn(expected_error, errors)

    def test_verifier_rejects_unapproved_third_adaptation(self) -> None:
        with temporary_rq2_root() as root:
            catalog_path = root / "catalog.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            radix = next(item for item in catalog["workloads"] if item["id"] == "shoc_radix_sort")
            radix["vconfig_adaptation"] = (
                "workloads/shoc_radix_sort/vconfig_adaptation.json"
            )
            write_json(catalog_path, catalog)

            errors = verify_catalog(root)

        self.assertIn("shoc_radix_sort: unexpected VConfig adaptation", errors)

    def test_verifier_rejects_adapted_kernel_symlink(self) -> None:
        with temporary_rq2_root() as root:
            kernel = root / "workloads" / "shoc_scan" / "kernel.cu"
            kernel.unlink()
            kernel.symlink_to("../../../rq1/workloads/shoc_scan/kernel.cu")

            errors = verify_catalog(root)

        self.assertIn("shoc_scan: adapted RQ2 kernel must be a regular file", errors)

    def test_verifier_rejects_boolean_number_constraint_substitutions(self) -> None:
        with temporary_rq2_root() as root:
            constraints_path = root / "workloads" / "shoc_scan" / "constraints.json"
            constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
            override = constraints["overrides"][0]
            override["domains"][1]["arg"] = True
            override["launch_policy"]["logical_grid"][0] = True
            override["launch_policy"]["vconfig_reserved"] = 1
            write_json(constraints_path, constraints)

            errors = verify_catalog(root)

        self.assertIn(
            "shoc_scan: constraints differ from the approved RQ2 derivation", errors
        )

    def test_verifier_cli_is_runnable_by_path(self) -> None:
        result = subprocess.run(
            [sys.executable, str(RQ2_ROOT / "verify.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
