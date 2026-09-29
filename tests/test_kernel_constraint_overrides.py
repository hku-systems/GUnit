import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.kernel_constraints.overrides import apply_constraint_overrides


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "kernels": [
            {
                "symbol_name": "_Z6kernelPfi",
                "display_name": "kernel",
                "args": [
                    {
                        "index": 0,
                        "name": "buffer",
                        "type": "float *",
                        "kind": "pointer",
                        "pointer_role": "payload_buffer",
                        "pointee_layout": {
                            "index": "buffer.*",
                            "name": "$pointee",
                            "type": "float",
                            "kind": "scalar",
                            "size_bytes": 4,
                            "align_bytes": 4,
                        },
                        "size_bytes": 8,
                        "align_bytes": 8,
                    },
                    {
                        "index": 1,
                        "name": "count",
                        "type": "int",
                        "kind": "scalar",
                        "size_bytes": 4,
                        "align_bytes": 4,
                    },
                ],
                "others": {
                    "source_file": "/repo/third_party/example/kernel.cu",
                    "source_line": 10,
                },
            }
        ],
    }


def _make_phase1_run(root: Path) -> tuple[Path, Path]:
    run_dir = root / "raw"
    kernel_dir = run_dir / "kernels" / "kernel__12345678"
    manifest = _manifest()
    _write_json(
        run_dir / "index.json",
        {
            "schema_version": 1,
            "kernels": [
                {
                    "kernel_id": "kernel__12345678",
                    "symbol_name": "_Z6kernelPfi",
                    "dir": "kernels/kernel__12345678",
                }
            ],
        },
    )
    _write_json(kernel_dir / "manifest.json", manifest)
    _write_json(
        kernel_dir / "metadata.json",
        {
            "schema_version": 1,
            "kernel_id": "kernel__12345678",
            "kernel_symbol": "_Z6kernelPfi",
            "build_status": "built",
            "output_hashes": {"manifest.json": "sha256:stale"},
        },
    )
    (kernel_dir / "kernel.bc").write_bytes(b"bc")
    _write_json(kernel_dir / "phase2" / "metadata.phase2.json", {"phase2_status": "built"})
    _write_json(run_dir / "rewrite_summary.json", {"counts": {"built": 1}})
    return run_dir, kernel_dir


def _add_duplicate_symbol_kernel(run_dir: Path) -> Path:
    index_path = run_dir / "index.json"
    index = json.loads(index_path.read_text())
    kernel_dir = run_dir / "kernels" / "kernel__abcdef12"
    manifest = _manifest()
    manifest["kernels"][0]["others"]["source_line"] = 20
    _write_json(kernel_dir / "manifest.json", manifest)
    _write_json(
        kernel_dir / "metadata.json",
        {
            "schema_version": 1,
            "kernel_id": "kernel__abcdef12",
            "kernel_symbol": "_Z6kernelPfi",
            "build_status": "built",
            "output_hashes": {"manifest.json": "sha256:stale"},
        },
    )
    index["kernels"].append(
        {
            "kernel_id": "kernel__abcdef12",
            "symbol_name": "_Z6kernelPfi",
            "dir": "kernels/kernel__abcdef12",
        }
    )
    _write_json(index_path, index)
    return kernel_dir


def _override_registry(*, expected_type: str = "int") -> dict:
    return {
        "schema_version": 1,
        "project": "example",
        "overrides": [
            {
                "symbol_name": "_Z6kernelPfi",
                "display_name": "kernel",
                "domains": [
                    {
                        "arg": 0,
                        "name": "buffer",
                        "type": "float *",
                        "domain": {
                            "kind": "bytes",
                            "min_len": "4",
                            "max_len": "256",
                            "elem_size_bytes": 4,
                            "nullable": False,
                        },
                    },
                    {
                        "arg": 1,
                        "name": "count",
                        "type": expected_type,
                        "domain": {
                            "kind": "int_range",
                            "min": "1",
                            "max": "64",
                            "signed": True,
                        },
                    },
                ],
                "constraints": [
                    {
                        "kind": "count_fits_buffer",
                        "count_arg": 1,
                        "buffer_arg": 0,
                        "elem_size_bytes": 4,
                    }
                ],
                "evidence": [
                    {
                        "kind": "callsite",
                        "file": "third_party/example/kernel.cu",
                        "line": 42,
                        "note": "launch uses count elements from buffer",
                    }
                ],
            }
        ],
    }


class ConstraintOverrideTest(unittest.TestCase):
    def test_rejects_duplicate_symbols_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry = _override_registry()
            registry["overrides"].append(dict(registry["overrides"][0]))
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)
            out_dir = root / "resolved"

            with self.assertRaisesRegex(ValueError, "duplicate constraint override symbol"):
                apply_constraint_overrides(raw_run, out_dir, registry_path)

            self.assertFalse(out_dir.exists())

    def test_kernel_id_disambiguates_duplicate_phase1_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            _add_duplicate_symbol_kernel(raw_run)
            registry = _override_registry()
            registry["overrides"][0]["kernel_id"] = "kernel__abcdef12"
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)
            out_dir = root / "resolved"

            apply_constraint_overrides(raw_run, out_dir, registry_path)

            first_manifest = json.loads(
                (out_dir / "kernels" / "kernel__12345678" / "manifest.json").read_text()
            )
            second_manifest = json.loads(
                (out_dir / "kernels" / "kernel__abcdef12" / "manifest.json").read_text()
            )
            self.assertNotIn("domain", first_manifest["kernels"][0]["args"][0])
            self.assertEqual(second_manifest["kernels"][0]["args"][0]["domain"]["max_len"], "256")

    def test_ambiguous_duplicate_phase1_symbol_requires_kernel_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            _add_duplicate_symbol_kernel(raw_run)
            registry_path = root / "overrides.json"
            _write_json(registry_path, _override_registry())
            out_dir = root / "resolved"

            with self.assertRaisesRegex(ValueError, "ambiguous duplicate symbol"):
                apply_constraint_overrides(raw_run, out_dir, registry_path)

            self.assertFalse(out_dir.exists())

    def test_rejects_missing_evidence_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry = _override_registry()
            registry["overrides"][0]["evidence"] = []
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)
            out_dir = root / "resolved"

            with self.assertRaisesRegex(ValueError, "must include evidence"):
                apply_constraint_overrides(raw_run, out_dir, registry_path)

            self.assertFalse(out_dir.exists())

    def test_builds_resolved_run_and_updates_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry_path = root / "overrides.json"
            _write_json(registry_path, _override_registry())
            out_dir = root / "resolved"
            raw_manifest_path = raw_run / "kernels" / "kernel__12345678" / "manifest.json"
            raw_hash_before = hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest()

            report = apply_constraint_overrides(raw_run, out_dir, registry_path)
            raw_hash_after = hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest()

            raw_manifest = json.loads(
                (raw_run / "kernels" / "kernel__12345678" / "manifest.json").read_text()
            )
            self.assertEqual(raw_hash_after, raw_hash_before)
            self.assertNotIn("domain", raw_manifest["kernels"][0]["args"][0])
            resolved_kernel_dir = out_dir / "kernels" / "kernel__12345678"
            resolved_manifest_path = resolved_kernel_dir / "manifest.json"
            resolved_manifest = json.loads(resolved_manifest_path.read_text())
            kernel = resolved_manifest["kernels"][0]
            self.assertEqual(kernel["args"][0]["domain"]["max_len"], "256")
            self.assertEqual(kernel["args"][1]["domain"]["min"], "1")
            self.assertEqual(kernel["constraints"][0]["kind"], "count_fits_buffer")
            self.assertFalse((resolved_kernel_dir / "phase2").exists())
            self.assertFalse((out_dir / "rewrite_summary.json").exists())
            self.assertEqual(report["counts"], {"applied": 1, "unmatched": 0})

            metadata = json.loads((resolved_kernel_dir / "metadata.json").read_text())
            expected_hash = "sha256:" + hashlib.sha256(resolved_manifest_path.read_bytes()).hexdigest()
            self.assertEqual(metadata["output_hashes"]["manifest.json"], expected_hash)
            applied = json.loads((resolved_kernel_dir / "constraint_override.applied.json").read_text())
            self.assertEqual(applied["symbol_name"], "_Z6kernelPfi")
            self.assertEqual(applied["evidence"][0]["kind"], "callsite")

    def test_applies_launch_policy_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry = _override_registry()
            registry["overrides"][0]["launch_policy"] = {
                "grid": [1, 1, 1],
                "block_candidates": [128, 64, 32],
                "physical_block_max": 128,
                "logical_grid": [1, 1, 1],
                "logical_block": [8, 8, 1],
                "logical_block_candidates": [[8, 4, 1], [8, 8, 1]],
                "target_dynamic_shared_bytes": 0,
                "coverage_memory": "global",
                "vconfig_reserved": True,
                "vconfig_mutation": False,
            }
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)
            out_dir = root / "resolved"

            apply_constraint_overrides(raw_run, out_dir, registry_path)

            resolved_manifest = json.loads(
                (out_dir / "kernels" / "kernel__12345678" / "manifest.json").read_text()
            )
            self.assertEqual(
                resolved_manifest["kernels"][0]["launch_policy"]["block_candidates"],
                [128, 64, 32],
            )
            self.assertEqual(
                resolved_manifest["kernels"][0]["launch_policy"]["logical_block"],
                [8, 8, 1],
            )
            self.assertEqual(
                resolved_manifest["kernels"][0]["launch_policy"][
                    "logical_block_candidates"
                ],
                [[8, 4, 1], [8, 8, 1]],
            )
            self.assertFalse(
                resolved_manifest["kernels"][0]["launch_policy"]["vconfig_mutation"]
            )

    def test_rejects_unknown_manifest_kernel_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, raw_kernel_dir = _make_phase1_run(root)
            manifest_path = raw_kernel_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["kernels"][0]["unexpected_launch"] = True
            _write_json(manifest_path, manifest)

            registry = _override_registry()
            registry["overrides"][0]["launch_policy"] = {
                "grid": [1, 1, 1],
                "block_candidates": [128, 64, 32],
                "physical_block_max": 128,
                "target_dynamic_shared_bytes": 512,
                "coverage_memory": "global",
                "vconfig_reserved": True,
            }
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)
            out_dir = root / "resolved"

            with self.assertRaisesRegex(ValueError, "unexpected_launch"):
                apply_constraint_overrides(raw_run, out_dir, registry_path)

            self.assertFalse(out_dir.exists())

    def test_rejects_unknown_registry_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry = _override_registry()
            registry["overrides"][0]["unexpected_launch"] = True
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)
            out_dir = root / "resolved"

            with self.assertRaisesRegex(ValueError, "unexpected_launch"):
                apply_constraint_overrides(raw_run, out_dir, registry_path)

            self.assertFalse(out_dir.exists())

    def test_rejects_unknown_launch_policy_registry_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry = _override_registry()
            registry["overrides"][0]["launch_policy"] = {
                "grid": [1, 1, 1],
                "block_candidates": [128, 64, 32],
                "physical_block_max": 128,
                "target_dynamic_shared_bytes": 512,
                "coverage_memory": "global",
                "vconfig_reserved": True,
                "unknown": True,
            }
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)
            out_dir = root / "resolved"

            with self.assertRaisesRegex(ValueError, "launch_policy.unknown"):
                apply_constraint_overrides(raw_run, out_dir, registry_path)

            self.assertFalse(out_dir.exists())

    def test_rejects_argument_signature_drift_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry_path = root / "overrides.json"
            _write_json(registry_path, _override_registry(expected_type="size_t"))
            out_dir = root / "resolved"

            with self.assertRaisesRegex(ValueError, "argument signature mismatch"):
                apply_constraint_overrides(raw_run, out_dir, registry_path)

            self.assertFalse(out_dir.exists())

    def test_applies_domain_to_nested_layout_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, raw_kernel_dir = _make_phase1_run(root)
            manifest = json.loads((raw_kernel_dir / "manifest.json").read_text())
            manifest["kernels"][0]["args"][1] = {
                "index": 1,
                "name": "config",
                "type": "Config",
                "kind": "opaque_val",
                "size_bytes": 4,
                "align_bytes": 4,
                "type_layout": {
                    "layout_status": "complete",
                    "fields": [
                        {
                            "index": "config.count",
                            "name": "count",
                            "type": "int",
                            "kind": "scalar",
                            "size_bytes": 4,
                            "align_bytes": 4,
                        }
                    ],
                },
            }
            _write_json(raw_kernel_dir / "manifest.json", manifest)
            registry = _override_registry()
            registry["overrides"][0]["domains"] = [
                {
                    "arg": 1,
                    "name": "config",
                    "type": "Config",
                    "path": ["count"],
                    "domain": {
                        "kind": "int_range",
                        "min": "1",
                        "max": "64",
                        "signed": True,
                    },
                }
            ]
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)

            apply_constraint_overrides(raw_run, root / "resolved", registry_path)

            resolved = json.loads(
                (
                    root
                    / "resolved"
                    / "kernels"
                    / "kernel__12345678"
                    / "manifest.json"
                ).read_text()
            )
            field = resolved["kernels"][0]["args"][1]["type_layout"]["fields"][0]
            self.assertEqual(field["domain"]["min"], "1")
            self.assertNotIn("domain", resolved["kernels"][0]["args"][1])

    def test_applies_layout_alignment_patch_to_manifest_node(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry = _override_registry()
            registry["overrides"][0]["layouts"] = [
                {
                    "arg": 0,
                    "name": "buffer",
                    "type": "float *",
                    "align_bytes": 16,
                }
            ]
            registry_path = root / "overrides.json"
            _write_json(registry_path, registry)

            apply_constraint_overrides(raw_run, root / "resolved", registry_path)

            resolved = json.loads(
                (
                    root
                    / "resolved"
                    / "kernels"
                    / "kernel__12345678"
                    / "manifest.json"
                ).read_text()
            )
            self.assertEqual(resolved["kernels"][0]["args"][0]["align_bytes"], 16)
            self.assertEqual(resolved["kernels"][0]["args"][1]["align_bytes"], 4)

    def test_cli_writes_resolved_run_and_reports_applied_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_run, _ = _make_phase1_run(root)
            registry_path = root / "overrides.json"
            _write_json(registry_path, _override_registry())
            out_dir = root / "resolved"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.kernel_constraints.cli",
                    "--run-dir",
                    str(raw_run),
                    "--out-dir",
                    str(out_dir),
                    "--overrides",
                    str(registry_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("applied=1", result.stdout)
            self.assertTrue((out_dir / "constraint_overrides.report.json").is_file())

    def test_real_phantom_registry_uses_precise_product_constraints(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/phantom_fhe.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }

        for display_name, n_min in (
            ("bconv_mult_unroll2_kernel", "2"),
            ("bconv_mult_unroll4_kernel", "4"),
        ):
            with self.subTest(display_name=display_name):
                override = overrides[display_name]
                domains = {domain["name"]: domain["domain"] for domain in override["domains"]}
                self.assertEqual(domains["n"]["min"], n_min)
                self.assertIn(
                    _expr_le_payload_product_constraint(buffer_arg=0, lhs_args=(5, 6), elem_size=8),
                    override["constraints"],
                )
                self.assertIn(
                    _expr_le_payload_product_constraint(buffer_arg=1, lhs_args=(5, 6), elem_size=8),
                    override["constraints"],
                )
                for buffer_arg in (2, 3):
                    self.assertIn(
                        {
                            "kind": "scalar_le_buffer_len",
                            "scalar_arg": 5,
                            "buffer_arg": buffer_arg,
                            "unit": "elements",
                        },
                        override["constraints"],
                    )

    def test_real_phantom_registry_constrains_moddown_and_decompose_bounds(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/phantom_fhe.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }

        moddown = overrides["moddown_bconv_single_p_kernel"]
        moddown_domains = {domain["name"]: domain["domain"] for domain in moddown["domains"]}
        self.assertEqual(moddown_domains["n"]["min"], "128")
        self.assertEqual(moddown_domains["n"]["max"], "128")
        self.assertEqual(moddown_domains["size_QlP"]["min"], "2")
        self.assertIn(
            {
                "kind": "expression_compare",
                "lhs": {
                    "kind": "binary",
                    "op": "*",
                    "lhs": {
                        "kind": "binary",
                        "op": "*",
                        "lhs": {"kind": "arg_value", "arg": 2},
                        "rhs": {
                            "kind": "binary",
                            "op": "-",
                            "lhs": {"kind": "arg_value", "arg": 4},
                            "rhs": {"kind": "const", "value": 1},
                        },
                    },
                    "rhs": {"kind": "const", "value": 8},
                },
                "op": "<=",
                "rhs": {"kind": "payload_len", "arg": 0},
                "repair": {"kind": "resize_payload", "arg": 0},
            },
            moddown["constraints"],
        )

        for display_name in (
            "decompose_array_uint64_kernel",
            "decompose_array_uint128_kernel",
        ):
            with self.subTest(display_name=display_name):
                override = overrides[display_name]
                domains = {domain["name"]: domain["domain"] for domain in override["domains"]}
                self.assertEqual(domains["poly_degree"]["min"], "128")
                self.assertGreaterEqual(int(domains["dst"]["min_len"]), 1024)
                self.assertEqual(domains["modulus"]["min_len"], "24")
                self.assertEqual(domains["modulus"]["max_len"], "24")
                self.assertTrue(
                    any(
                        evidence["kind"] == "call_site"
                        and evidence["file"] == "third_party/phantom-fhe/src/rns_base.cu"
                        and evidence["line"] == 157
                        for evidence in override["evidence"]
                    )
                )

        self.assertTrue(
            any(
                evidence["kind"] == "call_site"
                and evidence["file"] == "third_party/phantom-fhe/src/rns_bconv.cu"
                and evidence["line"] == 796
                for evidence in moddown["evidence"]
            )
        )

    def test_real_phantom_registry_aligns_vectorized_uint64_payloads(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/phantom_fhe.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }

        for display_name, pointer_names in (
            ("bconv_mult_unroll2_kernel", ("dst", "src")),
            ("bconv_mult_unroll4_kernel", ("dst", "src")),
            ("bconv_matmul_unroll2_kernel", ("dst", "xi_qiHatInv_mod_qi")),
            ("bconv_matmul_unroll4_kernel", ("dst", "xi_qiHatInv_mod_qi")),
            ("bconv_matmul_padded_unroll2_kernel", ("dst", "xi_qiHatInv_mod_qi")),
            ("bconv_matmul_padded_unroll4_kernel", ("dst", "xi_qiHatInv_mod_qi")),
            ("base_convert_matmul_hps_unroll2_kernel", ("dst", "xi_qiHatInv_mod_qi")),
            ("base_convert_matmul_hps_unroll4_kernel", ("dst", "xi_qiHatInv_mod_qi")),
            (
                "phantom::bconv_fuse_sub_mul_unroll2_kernel",
                ("dst", "xi_qiHatInv_mod_qi", "input_base_Bsk"),
            ),
        ):
            with self.subTest(display_name=display_name):
                layouts = {
                    layout["name"]: layout
                    for layout in overrides[display_name].get("layouts", [])
                }
                for pointer_name in pointer_names:
                    self.assertEqual(layouts[pointer_name]["align_bytes"], 16)

    def test_real_phantom_registry_fixes_unroll_width_domains(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/phantom_fhe.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }

        for display_name, n_value in (
            ("bconv_matmul_unroll2_kernel", "2"),
            ("bconv_matmul_unroll4_kernel", "4"),
            ("bconv_matmul_padded_unroll2_kernel", "2"),
            ("bconv_matmul_padded_unroll4_kernel", "4"),
            ("base_convert_matmul_hps_unroll2_kernel", "2"),
            ("base_convert_matmul_hps_unroll4_kernel", "4"),
            ("phantom::bconv_fuse_sub_mul_unroll2_kernel", "2"),
        ):
            with self.subTest(display_name=display_name):
                domains = {
                    domain["name"]: domain["domain"]
                    for domain in overrides[display_name]["domains"]
                }
                self.assertEqual(domains["n"]["min"], n_value)
                self.assertEqual(domains["n"]["max"], n_value)

    def test_real_phantom_registry_sets_extern_shared_launch_bytes(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/phantom_fhe.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }

        for display_name in (
            "bconv_matmul_unroll2_kernel",
            "bconv_matmul_unroll4_kernel",
            "bconv_matmul_padded_unroll2_kernel",
            "bconv_matmul_padded_unroll4_kernel",
            "base_convert_matmul_hps_unroll2_kernel",
            "base_convert_matmul_hps_unroll4_kernel",
            "phantom::bconv_fuse_sub_mul_unroll2_kernel",
        ):
            with self.subTest(display_name=display_name):
                launch = overrides[display_name]["launch_policy"]
                self.assertEqual(launch["coverage_memory"], "global")
                self.assertGreaterEqual(launch["target_dynamic_shared_bytes"], 128)
                self.assertLessEqual(max(launch["block_candidates"]), launch["physical_block_max"])

        for display_name in (
            "inplace_special_ffft_base_kernel",
            "inplace_special_ifft_base_kernel",
        ):
            with self.subTest(display_name=display_name):
                launch = overrides[display_name]["launch_policy"]
                self.assertEqual(launch["target_dynamic_shared_bytes"], 1024)
                self.assertEqual(launch["block_candidates"], [32, 16])

        launch = overrides["inplace_fnwt_radix2_opt"]["launch_policy"]
        self.assertEqual(launch["target_dynamic_shared_bytes"], 512)
        self.assertEqual(launch["block_candidates"], [32])
        self.assertEqual(launch["physical_block_max"], 32)
        self.assertEqual(launch["logical_block"], [32, 1, 1])
        self.assertEqual(launch["logical_block_candidates"], [[32, 1, 1]])
        self.assertTrue(launch["vconfig_reserved"])

        for display_name in ("inplace_fnwt_radix2", "inplace_inwt_radix2"):
            with self.subTest(display_name=display_name):
                launch = overrides[display_name]["launch_policy"]
                self.assertEqual(launch["grid"], [1, 1, 1])
                self.assertEqual(launch["block_candidates"], [2, 1])
                self.assertEqual(launch["physical_block_max"], 2)
                self.assertEqual(launch["target_dynamic_shared_bytes"], 32)
                self.assertFalse(launch["vconfig_reserved"])

    def test_real_phantom_registry_constrains_tensor_prod_mxn_products(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/phantom_fhe.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }
        constraints = overrides["tensor_prod_mxn_rns_poly"]["constraints"]

        for buffer_arg, size_arg in ((0, 1), (2, 3), (5, 6)):
            with self.subTest(buffer_arg=buffer_arg):
                self.assertIn(
                    _expr_le_payload_product_constraint(
                        buffer_arg=buffer_arg,
                        lhs_args=(7, 8, size_arg),
                        elem_size=8,
                    ),
                    constraints,
                )

        domains = {domain["name"]: domain["domain"] for domain in overrides["tensor_prod_mxn_rns_poly"]["domains"]}
        self.assertEqual(domains["op1_size"]["min"], "2")
        self.assertEqual(domains["op1_size"]["max"], "2")
        self.assertEqual(domains["op2_size"]["min"], "3")
        self.assertEqual(domains["op2_size"]["max"], "3")
        self.assertEqual(domains["res_size"]["min"], "4")
        self.assertEqual(domains["res_size"]["max"], "4")

    def test_real_phantom_registry_uses_native_small_radix2_shape(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/phantom_fhe.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }

        for display_name, coeff_name, start_name, n_name in (
            ("inplace_fnwt_radix2", "coeff_mod_size", "start_mod_idx", "n"),
            ("inplace_inwt_radix2", "coeff_mod_size", "start_mod_idx", "n"),
        ):
            with self.subTest(display_name=display_name):
                domains = {
                    domain["name"]: domain["domain"]
                    for domain in overrides[display_name]["domains"]
                }
                self.assertEqual(domains[coeff_name]["min"], "1")
                self.assertEqual(domains[coeff_name]["max"], "1")
                self.assertEqual(domains[start_name]["min"], "0")
                self.assertEqual(domains[start_name]["max"], "0")
                self.assertEqual(domains[n_name]["min"], "4")
                self.assertEqual(domains[n_name]["max"], "4")

    def test_real_gpurir_registry_runs_reducible_extern_shared_kernel(self) -> None:
        registry = json.loads(Path("third_party/fuzz/kernel_constraints/gpurir.json").read_text())
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }
        override = overrides["reduceRIR_kernel"]
        domains = {domain["name"]: domain["domain"] for domain in override["domains"]}

        self.assertEqual(domains["M"]["min"], "1")
        self.assertEqual(domains["T"]["min"], "1")
        self.assertEqual(domains["N"]["min"], "128")
        self.assertEqual(override["launch_policy"]["target_dynamic_shared_bytes"], 512)
        self.assertEqual(override["launch_policy"]["block_candidates"], [128, 64, 32])
        self.assertIn(
            _expr_le_payload_product_constraint(buffer_arg=0, lhs_args=(2, 3, 4), elem_size=4),
            override["constraints"],
        )
        self.assertIn(
            _expr_le_payload_product_constraint(buffer_arg=1, lhs_args=(2, 3, 5), elem_size=4),
            override["constraints"],
        )

    def test_real_gpurir_registry_models_vconfig_multidimensional_kernels(self) -> None:
        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/gpurir.json").read_text()
        )
        overrides = {
            item["display_name"]: item
            for item in registry["overrides"]
            if item.get("display_name")
        }

        expected_launches = {
            "calcAmpTau_kernel": ([4, 4, 4], [[2, 4, 4], [4, 4, 4]]),
            "diffRev_kernel": ([16, 4, 2], [[8, 4, 1], [8, 4, 2], [16, 4, 2]]),
            "envPred_kernel": ([4, 4, 1], [[2, 2, 1], [4, 2, 1], [4, 4, 1]]),
            "generateRIR_kernel": ([32, 4, 1], [[8, 4, 1], [16, 4, 1], [32, 4, 1]]),
            "h2RIR_to_floatRIR_kernel": (
                [128, 1, 1],
                [[32, 1, 1], [64, 1, 1], [128, 1, 1]],
            ),
        }
        for display_name, (logical_block, candidates) in expected_launches.items():
            with self.subTest(display_name=display_name):
                launch = overrides[display_name]["launch_policy"]
                self.assertEqual(launch["grid"], [1, 1, 1])
                self.assertEqual(launch["logical_grid"], [1, 1, 1])
                self.assertEqual(launch["logical_block"], logical_block)
                self.assertEqual(launch["logical_block_candidates"], candidates)
                self.assertTrue(launch["vconfig_mutation"])

        diff = overrides["diffRev_kernel"]
        for scalar_arg, dimension in ((7, "x"), (4, "y"), (5, "z")):
            self.assertIn(
                {
                    "kind": "scalar_le_logical_block_dim",
                    "scalar_arg": scalar_arg,
                    "dimension": dimension,
                },
                diff["constraints"],
            )
        self.assertIn(
            _expr_le_payload_product_constraint(
                buffer_arg=0, lhs_args=(4, 5, 7), elem_size=4
            ),
            diff["constraints"],
        )
        for buffer_arg in (1, 2, 3):
            self.assertIn(
                _expr_le_payload_product_constraint(
                    buffer_arg=buffer_arg, lhs_args=(4, 5), elem_size=4
                ),
                diff["constraints"],
            )

        env = overrides["envPred_kernel"]
        env_domains = {
            domain["name"]: domain["domain"] for domain in env["domains"]
        }
        self.assertEqual(env_domains["tau_dp"]["pattern_hex"], "00000000")
        self.assertEqual(env_domains["Fs"]["min"], 100.0)
        self.assertEqual(env_domains["Fs"]["max"], 100.0)
        for buffer_arg in (0, 1, 3):
            self.assertIn(
                _expr_le_payload_product_constraint(
                    buffer_arg=buffer_arg, lhs_args=(4, 5), elem_size=4
                ),
                env["constraints"],
            )
        self.assertIn(
            _expr_le_payload_product_constraint(
                buffer_arg=2, lhs_args=(4, 5, 6), elem_size=4
            ),
            env["constraints"],
        )

        generate = overrides["generateRIR_kernel"]
        self.assertIn(
            {
                "kind": "scalar_compare_scalar",
                "lhs_arg": 5,
                "op": "<=",
                "rhs_arg": 7,
            },
            generate["constraints"],
        )
        self.assertIn(
            _expr_le_payload_product_constraint(
                buffer_arg=0, lhs_args=(4, 3, 6), elem_size=4
            ),
            generate["constraints"],
        )
        for buffer_arg in (1, 2):
            self.assertIn(
                _expr_le_payload_product_constraint(
                    buffer_arg=buffer_arg, lhs_args=(4, 5), elem_size=4
                ),
                generate["constraints"],
            )

        half = overrides["h2RIR_to_floatRIR_kernel"]
        half_domains = {
            domain["name"]: domain["domain"] for domain in half["domains"]
        }
        self.assertEqual(half_domains["M"]["min"], "1")
        self.assertEqual(half_domains["M"]["max"], "1")
        self.assertIn(
            _expr_le_payload_product_constraint(
                buffer_arg=0, lhs_args=(2, 3), elem_size=4
            ),
            half["constraints"],
        )
        self.assertIn(
            _expr_le_payload_product_constraint(
                buffer_arg=1, lhs_args=(2, 3), elem_size=8
            ),
            half["constraints"],
        )

        calc = overrides["calcAmpTau_kernel"]
        calc_domains = {
            domain["name"]: domain["domain"] for domain in calc["domains"]
        }
        for pattern in ("spkr_pattern", "mic_pattern"):
            self.assertEqual(calc_domains[pattern]["min"], "0")
            self.assertEqual(calc_domains[pattern]["max"], "0")
        self.assertEqual(calc_domains["g_pos_src"]["pattern_hex"], "0000803f")
        self.assertEqual(calc_domains["g_pos_rcv"]["pattern_hex"], "00000000")
        self.assertGreater(calc_domains["c"]["min"], 0)
        self.assertGreater(calc_domains["Fs"]["min"], 0)
        for scalar_arg, dimension in ((18, "x"), (19, "y"), (20, "z")):
            self.assertIn(
                {
                    "kind": "scalar_le_logical_block_dim",
                    "scalar_arg": scalar_arg,
                    "dimension": dimension,
                },
                calc["constraints"],
            )
        for buffer_arg in (0, 1):
            self.assertIn(
                _expr_le_payload_product_constraint(
                    buffer_arg=buffer_arg,
                    lhs_args=(21, 22, 18, 19, 20),
                    elem_size=4,
                ),
                calc["constraints"],
            )
        for buffer_arg, lhs_args, elem_size in (
            (2, (21, 22), 4),
            (3, (21,), 12),
            (4, (22,), 12),
        ):
            self.assertIn(
                _expr_le_payload_product_constraint(
                    buffer_arg=buffer_arg,
                    lhs_args=lhs_args,
                    elem_size=elem_size,
                ),
                calc["constraints"],
            )
        self.assertEqual(calc["launch_policy"]["target_dynamic_shared_bytes"], 96)


def _expr_le_payload_product_constraint(
    *,
    buffer_arg: int,
    lhs_args: tuple[int, ...],
    elem_size: int,
) -> dict:
    assert lhs_args
    lhs: dict = {"kind": "arg_value", "arg": lhs_args[0]}
    for arg in lhs_args[1:]:
        lhs = {
            "kind": "binary",
            "op": "*",
            "lhs": lhs,
            "rhs": {"kind": "arg_value", "arg": arg},
        }
    return {
        "kind": "expression_compare",
        "lhs": {
            "kind": "binary",
            "op": "*",
            "lhs": lhs,
            "rhs": {"kind": "const", "value": elem_size},
        },
        "op": "<=",
        "rhs": {"kind": "payload_len", "arg": buffer_arg},
        "repair": {"kind": "resize_payload", "arg": buffer_arg},
    }


if __name__ == "__main__":
    unittest.main()
