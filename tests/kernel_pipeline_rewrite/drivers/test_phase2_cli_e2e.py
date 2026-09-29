import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_DIR = Path(__file__).resolve().parent
SMOKE_DRIVERS_DIR = REPO_ROOT / "tests" / "kernel_pipeline_smoke" / "drivers"
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
if str(SMOKE_DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(SMOKE_DRIVERS_DIR))

from shared_e2e import cuda_device_skip_reason, get_capture_e2e_context  # noqa: E402
from shared_phase2 import collect_phase2_kernel_results, find_phase2_kernel_results_by_display_name, run_phase2_for_run_dir  # noqa: E402


def _cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-A function-casts-as-integer -A unstable-name-collisions")
    return env


def _manifest_path_for_record(record: dict) -> Path:
    return Path(record["kernel_dir"]) / "manifest.json"


def _phase2_dir_for_record(record: dict) -> Path:
    return Path(record["phase2_dir"])


def _manifest_path_by_display_name(run_dir: Path, display_name: str) -> Path:
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    for entry in index.get("kernels", []):
        kernel_dir = Path(entry["dir"])
        if not kernel_dir.is_absolute():
            kernel_dir = run_dir / kernel_dir
        manifest_path = kernel_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kernels = manifest.get("kernels") or []
        if kernels and kernels[0].get("display_name") == display_name:
            return manifest_path
    raise RuntimeError(f"kernel display_name not found in phase1 outputs: {display_name}")


def _inject_phase3_nested_array_constraints(run_dir: Path) -> None:
    manifest_path = _manifest_path_by_display_name(run_dir, "layout_struct_array_nested_ptr_kernel")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kernels"][0]["constraints"] = [
        {
            "kind": "count_fits_buffer",
            "count_arg": 1,
            "count_path": ["items", "1", "inner", "count"],
            "buffer_arg": 1,
            "buffer_path": ["items", "1", "inner", "data"],
            "elem_size_bytes": 4,
        },
        {
            "kind": "scalar_compare_const",
            "scalar_arg": 1,
            "scalar_path": ["items", "1", "inner", "count"],
            "op": ">=",
            "value": 1,
        },
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _phase3_dump_seed(manifest_path: Path) -> str:
    fuzzer_bin = REPO_ROOT / "cuda-fuzzer" / "target" / "debug" / "fuzzer"
    subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(REPO_ROOT / "cuda-fuzzer" / "Cargo.toml"),
            "--bin",
            "fuzzer",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=_cargo_env(),
    )

    result = subprocess.run(
        [
            str(fuzzer_bin),
            "/tmp/rapid-phase3-dump-seed-does-not-load.so",
            "--manifest",
            str(manifest_path),
            "--dump-seed",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def _build_origin_backend(phase2_dir: Path) -> Path:
    out_dir = phase2_dir / "backends" / "origin-phase3-e2e"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cuda-kernel" / "origin" / "build.py"),
            "--phase2-dir",
            str(phase2_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "origin backend build failed:\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    build = json.loads((out_dir / "backend_build.json").read_text(encoding="utf-8"))
    return Path(build["shared_lib"])


def _build_rapid_backend(phase2_dir: Path) -> Path:
    out_dir = phase2_dir / "backends" / "rapid-phase3-e2e"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cuda-kernel" / "rapid" / "build.py"),
            "--phase2-dir",
            str(phase2_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "rapid backend build failed:\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    build = json.loads((out_dir / "backend_build.json").read_text(encoding="utf-8"))
    return Path(build["shared_lib"])


def _build_rapid2_backend(phase2_dir: Path) -> Path:
    out_dir = phase2_dir / "backends" / "rapid2-phase3-e2e"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "cuda-kernel" / "rapid2" / "build.py"),
            "--phase2-dir",
            str(phase2_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "rapid2 backend build failed:\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    build = json.loads((out_dir / "backend_build.json").read_text(encoding="utf-8"))
    return Path(build["shared_lib"])


def _phase3_fixed_smoke(shared_lib: Path, manifest_path: Path) -> str:
    _phase3_dump_seed(manifest_path)
    result = subprocess.run(
        [
            str(REPO_ROOT / "cuda-fuzzer" / "target" / "debug" / "fuzzer"),
            str(shared_lib),
            "--manifest",
            str(manifest_path),
            "--no-mutate",
            "--runs",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    return result.stdout + result.stderr


def _phase3_no_mutate_runs_smoke(shared_lib: Path, manifest_path: Path) -> str:
    _phase3_dump_seed(manifest_path)
    result = subprocess.run(
        [
            str(REPO_ROOT / "cuda-fuzzer" / "target" / "debug" / "fuzzer"),
            str(shared_lib),
            "--manifest",
            str(manifest_path),
            "--no-mutate",
            "--runs",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    return result.stdout + result.stderr


def _phase3_async_fixed_smoke(shared_lib: Path, manifest_path: Path) -> str:
    _phase3_dump_seed(manifest_path)
    fuzzer_async_bin = REPO_ROOT / "cuda-fuzzer" / "target" / "debug" / "fuzzer_async"
    subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(REPO_ROOT / "cuda-fuzzer" / "Cargo.toml"),
            "--bin",
            "fuzzer_async",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=_cargo_env(),
    )
    result = subprocess.run(
        [
            str(fuzzer_async_bin),
            str(shared_lib),
            "--manifest",
            str(manifest_path),
            "--no-mutate",
            "--runs",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "async fixed Phase3 smoke failed:\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return result.stdout + result.stderr


def _phase3_async_no_mutate_runs_smoke(shared_lib: Path, manifest_path: Path) -> str:
    _phase3_dump_seed(manifest_path)
    subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(REPO_ROOT / "cuda-fuzzer" / "Cargo.toml"),
            "--bin",
            "fuzzer_async",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=_cargo_env(),
    )
    result = subprocess.run(
        [
            str(REPO_ROOT / "cuda-fuzzer" / "target" / "debug" / "fuzzer_async"),
            str(shared_lib),
            "--manifest",
            str(manifest_path),
            "--no-mutate",
            "--runs",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    return result.stdout + result.stderr


def _assert_hex_seed(testcase: unittest.TestCase, seed_hex: str) -> None:
    testcase.assertNotEqual(seed_hex, "")
    testcase.assertEqual(len(seed_hex) % 2, 0)
    testcase.assertTrue(all(ch in "0123456789abcdefABCDEF" for ch in seed_hex), seed_hex)


class Phase2CliEndToEndTest(unittest.TestCase):
    def test_phase2_cli_consumes_phase1_e2e_outputs(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        _inject_phase3_nested_array_constraints(ctx.run_dir)
        summary = run_phase2_for_run_dir(ctx.run_dir)
        summary_path = ctx.run_dir / "rewrite_summary.json"
        self.assertTrue(summary_path.is_file(), "missing rewrite_summary.json")
        written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(written_summary, summary)
        self.assertGreater(summary["counts"]["total"], 0)
        self.assertGreater(summary["counts"]["built"], 0)
        self.assertEqual(summary["counts"]["skipped"], 0)

        status_counts = Counter(result["status"] for result in summary["results"])
        self.assertEqual(summary["counts"]["built"], status_counts["built"])
        self.assertEqual(summary["counts"]["failed"], status_counts["failed"])
        self.assertEqual(summary["counts"]["skipped"], status_counts["skipped"])
        self.assertTrue(
            all("failure_detail" in result for result in summary["results"]),
            "rewrite_summary results must preserve per-kernel failure_detail",
        )

        add_kernel_records = find_phase2_kernel_results_by_display_name(ctx.run_dir, "add_kernel")
        self.assertEqual(len(add_kernel_records), 1)
        add_kernel = add_kernel_records[0]
        self.assertEqual(add_kernel["build_status"], "built")
        self.assertEqual(add_kernel["phase2_status"], "built")
        self.assertIsNone(add_kernel["failure_reason"])
        self.assertIsNone(add_kernel["failure_detail"])

        uint3_records = find_phase2_kernel_results_by_display_name(ctx.run_dir, "uint3_write_sum")
        self.assertEqual(len(uint3_records), 1)
        uint3_kernel = uint3_records[0]
        self.assertEqual(uint3_kernel["build_status"], "built")
        self.assertEqual(uint3_kernel["phase2_status"], "built")
        self.assertIsNone(uint3_kernel["failure_reason"])
        self.assertIsNone(uint3_kernel["failure_detail"])

        record_ptr_records = find_phase2_kernel_results_by_display_name(ctx.run_dir, "record_ptr_copy")
        self.assertEqual(len(record_ptr_records), 1)
        record_ptr_kernel = record_ptr_records[0]
        self.assertEqual(record_ptr_kernel["build_status"], "built")
        self.assertEqual(record_ptr_kernel["phase2_status"], "built")
        self.assertIsNone(record_ptr_kernel["failure_reason"])
        self.assertIsNone(record_ptr_kernel["failure_detail"])

        for display_name in (
            "layout_plain_val_kernel",
            "layout_outer_struct_kernel",
            "layout_struct_array_val_kernel",
            "layout_struct_array_ptr_kernel",
            "layout_struct_array_nested_kernel",
            "layout_struct_array_nested_ptr_kernel",
            "layout_anonymous_ptr_kernel",
            "layout_derived_params_kernel",
        ):
            with self.subTest(display_name=display_name):
                records = find_phase2_kernel_results_by_display_name(ctx.run_dir, display_name)
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(record["build_status"], "built")
                self.assertEqual(record["phase2_status"], "built")
                self.assertIsNone(record["failure_reason"])
                self.assertIsNone(record["failure_detail"])

        for display_name in (
            "layout_struct_array_ptr_kernel",
            "layout_struct_array_nested_ptr_kernel",
            "layout_anonymous_ptr_kernel",
            "layout_derived_params_kernel",
        ):
            with self.subTest(phase3_seed_display_name=display_name):
                record = find_phase2_kernel_results_by_display_name(ctx.run_dir, display_name)[0]
                seed_hex = _phase3_dump_seed(_manifest_path_for_record(record))
                _assert_hex_seed(self, seed_hex)

        nested_ptr_record = find_phase2_kernel_results_by_display_name(
            ctx.run_dir, "layout_struct_array_nested_ptr_kernel"
        )[0]
        nested_ptr_decode = (
            _phase2_dir_for_record(nested_ptr_record) / "gen" / "fuzzer_decode.v1.cuh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "RAPID_DECODE_ASSERT((decoded.cfg.items[1].inner.count * 4) <= cfg_items_1_inner_data_len_u64);",
            nested_ptr_decode,
        )
        self.assertIn(
            "RAPID_DECODE_ASSERT(decoded.cfg.items[1].inner.count >= 1);",
            nested_ptr_decode,
        )
        if cuda_device_skip_reason() is None:
            nested_ptr_lib = _build_origin_backend(_phase2_dir_for_record(nested_ptr_record))
            nested_ptr_smoke = _phase3_fixed_smoke(
                nested_ptr_lib, _manifest_path_for_record(nested_ptr_record)
            )
            self.assertIn("Loaded manifest:", nested_ptr_smoke)
            self.assertIn("Running bounded fixed-seed LibAFL loop: runs=1", nested_ptr_smoke)
            self.assertIn("Arg-pack stats:", nested_ptr_smoke)

            nested_ptr_no_mutate_smoke = _phase3_no_mutate_runs_smoke(
                nested_ptr_lib, _manifest_path_for_record(nested_ptr_record)
            )
            self.assertIn("Loaded manifest:", nested_ptr_no_mutate_smoke)
            self.assertIn("Running bounded fixed-seed LibAFL loop: runs=1", nested_ptr_no_mutate_smoke)
            self.assertIn("Arg-pack stats:", nested_ptr_no_mutate_smoke)

            nested_ptr_rapid_lib = _build_rapid_backend(_phase2_dir_for_record(nested_ptr_record))
            nested_ptr_rapid_smoke = _phase3_fixed_smoke(
                nested_ptr_rapid_lib, _manifest_path_for_record(nested_ptr_record)
            )
            self.assertIn("Loaded manifest:", nested_ptr_rapid_smoke)
            self.assertIn("Running bounded fixed-seed LibAFL loop: runs=1", nested_ptr_rapid_smoke)
            self.assertIn("Arg-pack stats:", nested_ptr_rapid_smoke)

            nested_ptr_rapid2_lib = _build_rapid2_backend(_phase2_dir_for_record(nested_ptr_record))
            nested_ptr_async_smoke = _phase3_async_fixed_smoke(
                nested_ptr_rapid2_lib, _manifest_path_for_record(nested_ptr_record)
            )
            self.assertIn("Loaded manifest:", nested_ptr_async_smoke)
            self.assertIn("Running bounded async fixed-seed LibAFL loop: runs=1", nested_ptr_async_smoke)
            self.assertIn("Final statistics:", nested_ptr_async_smoke)
            self.assertIn("Arg-pack stats:", nested_ptr_async_smoke)

            nested_ptr_async_no_mutate_smoke = _phase3_async_no_mutate_runs_smoke(
                nested_ptr_rapid2_lib, _manifest_path_for_record(nested_ptr_record)
            )
            self.assertIn("Loaded manifest:", nested_ptr_async_no_mutate_smoke)
            self.assertIn(
                "Running bounded async fixed-seed LibAFL loop: runs=1",
                nested_ptr_async_no_mutate_smoke,
            )
            self.assertIn("Final statistics:", nested_ptr_async_no_mutate_smoke)
            self.assertRegex(
                nested_ptr_async_no_mutate_smoke,
                r"Total executions: [1-9][0-9]*",
            )
            self.assertIn("Arg-pack stats:", nested_ptr_async_no_mutate_smoke)

        complex_records = find_phase2_kernel_results_by_display_name(ctx.run_dir, "complex_args_kernel")
        self.assertEqual(len(complex_records), 1)
        complex_kernel = complex_records[0]
        self.assertEqual(complex_kernel["build_status"], "built")
        self.assertEqual(complex_kernel["phase2_status"], "failed")
        self.assertEqual(complex_kernel["failure_reason"], "phase2_input_invalid")
        self.assertEqual(complex_kernel["failure_detail"], "params.nested.choice:union_not_supported")

        namespace_records = [
            record
            for record in collect_phase2_kernel_results(ctx.run_dir)
            if str(record.get("display_name", "")).endswith("ns_record_ptr_copy")
        ]
        self.assertEqual(len(namespace_records), 1)
        namespace_kernel = namespace_records[0]
        self.assertEqual(namespace_kernel["build_status"], "built")
        self.assertEqual(namespace_kernel["phase2_status"], "built")
        self.assertIsNone(namespace_kernel["failure_reason"])
        self.assertIsNone(namespace_kernel["failure_detail"])

        header_scoped_records = [
            record
            for record in collect_phase2_kernel_results(ctx.run_dir)
            if str(record.get("display_name", "")).endswith("header_nested_params_kernel")
        ]
        self.assertEqual(len(header_scoped_records), 1)
        header_scoped_kernel = header_scoped_records[0]
        self.assertEqual(header_scoped_kernel["build_status"], "built")
        self.assertEqual(header_scoped_kernel["phase2_status"], "built")
        self.assertIsNone(header_scoped_kernel["failure_reason"])
        self.assertIsNone(header_scoped_kernel["failure_detail"])

        scoped_records = [
            record
            for record in collect_phase2_kernel_results(ctx.run_dir)
            if str(record.get("display_name", "")).endswith("class_nested_record_ptr_copy")
        ]
        self.assertEqual(len(scoped_records), 1)
        scoped_kernel = scoped_records[0]
        self.assertEqual(scoped_kernel["build_status"], "built")
        self.assertEqual(scoped_kernel["phase2_status"], "built")
        self.assertIsNone(scoped_kernel["failure_reason"])
        self.assertIsNone(scoped_kernel["failure_detail"])

        material_failure_expectations = {
            "material_const_field_kernel": "const_assignment_blocker",
            "material_private_field_kernel": "private_data_field",
            "material_reference_field_kernel": "reference_field",
            "material_virtual_method_kernel": "non_trivially_copyable",
        }
        for display_name, reason_code in material_failure_expectations.items():
            with self.subTest(display_name=display_name):
                records = find_phase2_kernel_results_by_display_name(ctx.run_dir, display_name)
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(record["build_status"], "built")
                self.assertEqual(record["phase2_status"], "failed")
                self.assertEqual(record["failure_reason"], "phase2_input_invalid")
                self.assertTrue(str(record["failure_detail"]).endswith(f":{reason_code}"))

        public_methods_records = find_phase2_kernel_results_by_display_name(
            ctx.run_dir, "material_public_methods_kernel"
        )
        self.assertEqual(len(public_methods_records), 1)
        public_methods_kernel = public_methods_records[0]
        self.assertEqual(public_methods_kernel["phase2_status"], "built")
        self.assertIsNone(public_methods_kernel["failure_reason"])
        self.assertIsNone(public_methods_kernel["failure_detail"])

if __name__ == "__main__":
    unittest.main()
