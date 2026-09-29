import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "kernel_pipeline_smoke" / "fixtures"
DRIVERS_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = REPO_ROOT / "scripts" / "kernel-smoke"
if str(DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVERS_DIR))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import cli as kernel_smoke_cli  # type: ignore[import-not-found]  # noqa: E402
from shared_e2e import get_capture_e2e_context  # noqa: E402


def _collect_manifest_symbols(run_dir: Path) -> set[str]:
    symbols: set[str] = set()
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    for entry in index.get("kernels", []):
        kernel_dir = Path(entry["dir"])
        if not kernel_dir.is_absolute():
            kernel_dir = run_dir / kernel_dir
        manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        for kernel in manifest.get("kernels", []):
            name = kernel.get("symbol_name")
            if isinstance(name, str) and name:
                symbols.add(name)
    return symbols


def _collect_manifest_display_names(run_dir: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    for entry in index.get("kernels", []):
        kernel_dir = Path(entry["dir"])
        if not kernel_dir.is_absolute():
            kernel_dir = run_dir / kernel_dir
        manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        for kernel in manifest.get("kernels", []):
            symbol = kernel.get("symbol_name")
            display_name = kernel.get("display_name")
            if isinstance(symbol, str) and symbol and isinstance(display_name, str):
                names[symbol] = display_name
    return names


def _collect_manifest_kernels(run_dir: Path) -> list[dict]:
    kernels: list[dict] = []
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    for entry in index.get("kernels", []):
        kernel_dir = Path(entry["dir"])
        if not kernel_dir.is_absolute():
            kernel_dir = run_dir / kernel_dir
        manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernels.extend(manifest.get("kernels", []))
    return kernels


class EmitAndResumeTest(unittest.TestCase):
    def test_capture_mode_e2e_summary_and_symbols(self) -> None:
        """Main path: one shared capture E2E run, then assert outputs."""
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        self.assertTrue(ctx.commands_jsonl.is_file(), "capture commands.jsonl missing")

        discover = json.loads((ctx.run_dir / "discover.json").read_text(encoding="utf-8"))
        provenance = discover.get("provenance", {})
        diagnostics = discover.get("diagnostics", {})
        self.assertEqual(provenance.get("mode"), "artifact")
        self.assertEqual(provenance.get("jobs"), 1)
        # Grouped subsections must exist.
        self.assertIn("discovery", provenance)
        self.assertIn("artifact_diagnostics", provenance)
        self.assertIn("variant_failures_by_variant", diagnostics)
        self.assertIn("ptx_ast_mismatch_by_variant", diagnostics)

        for path in (
            "variant_progress.json",
            "discover.json",
            "summary.json",
        ):
            self.assertTrue((ctx.run_dir / path).is_file(), f"missing {path}")

        # Removed files must not be present.
        for stale_file in ("checkpoint.json", "stage_results.json"):
            self.assertFalse(
                (ctx.run_dir / stale_file).is_file(),
                f"{stale_file} should not exist in simplified flow",
            )

        summary = json.loads((ctx.run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary.get("capture_dir"), str(ctx.capture_dir))
        self.assertGreater(summary["counts"]["discovered"], 0)
        self.assertGreater(summary["counts"]["succeeded"], 0)

        symbols = _collect_manifest_symbols(ctx.run_dir)
        self.assertTrue(any("direct_kernel" in sym for sym in symbols))
        self.assertTrue(any("lib_only_axpy" in sym for sym in symbols))
        self.assertTrue(any("lib_only_clamp_u32" in sym for sym in symbols))

        display_names = _collect_manifest_display_names(ctx.run_dir)
        self.assertEqual(
            display_names.get("_ZN3ns13ns26ns_addEPiii"),
            "ns1::ns2::ns_add",
        )
        self.assertEqual(
            display_names.get("_ZN3ns13ns28ns_scaleEPfff"),
            "ns1::ns2::ns_scale",
        )

        kernels = discover.get("kernels", [])
        self.assertGreater(len(kernels), 0)
        self.assertTrue(all(k.get("selection_reason") == "replayable_cuda_compile" for k in kernels))

    def test_capture_mode_resume_keeps_progress(self) -> None:
        """Resume run preserves summary counts and reconciliation.

        Verifies:
        - A no-op --resume produces the same summary counts as the original.
        - Removed files (checkpoint.json, stage_results.json) are absent.
        - emit_results/manifest_results survive the round-trip via discover.json.
        """
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        # Read pre-resume summary for comparison.
        summary_before = json.loads(
            (ctx.run_dir / "summary.json").read_text(encoding="utf-8"),
        )
        counts_before = summary_before["counts"]
        reconciliation_before = summary_before.get("reconciliation")

        cli = REPO_ROOT / "scripts" / "kernel-smoke" / "cli.py"
        cmd = [
            sys.executable,
            str(cli),
            "run",
            "--capture-dir",
            str(ctx.capture_dir),
            "--out-root",
            str(ctx.out_root),
            "--run-id",
            ctx.run_id,
            "--target-lib",
            "fixtures_all_e2e",
            "--mode",
            "artifact",
            "--jobs",
            "1",
            "--resume",
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

        summary = json.loads((ctx.run_dir / "summary.json").read_text(encoding="utf-8"))
        variant_progress = json.loads((ctx.run_dir / "variant_progress.json").read_text(encoding="utf-8"))
        completed = variant_progress.get("completed_variants", [])
        order = variant_progress.get("variant_order", [])

        self.assertGreater(summary["counts"]["discovered"], 0)
        self.assertGreater(len(completed), 0)
        self.assertLessEqual(len(completed), len(order))

        # Summary counts must be identical after a no-op resume.
        self.assertEqual(
            summary["counts"]["discovered"],
            counts_before["discovered"],
            "discovered count changed after resume",
        )
        self.assertEqual(
            summary["counts"]["succeeded"],
            counts_before["succeeded"],
            "succeeded count changed after resume",
        )
        self.assertEqual(
            summary["counts"]["failed"],
            counts_before["failed"],
            "failed count changed after resume",
        )
        self.assertEqual(
            summary["counts"]["skipped"],
            counts_before["skipped"],
            "skipped count changed after resume",
        )

        self.assertEqual(
            summary.get("reconciliation"),
            reconciliation_before,
            "reconciliation changed after resume",
        )

        # Removed files must not exist.
        for stale_file in ("checkpoint.json", "stage_results.json"):
            self.assertFalse(
                (ctx.run_dir / stale_file).is_file(),
                f"{stale_file} should not exist in simplified flow",
            )

    def test_template_instantiations_keep_distinct_args(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        kernels = _collect_manifest_kernels(ctx.run_dir)
        templated = [
            kernel for kernel in kernels
            if kernel.get("display_name") == "templated_entry_kernel"
        ]

        self.assertGreaterEqual(len(templated), 2)
        arg_types = {
            kernel.get("args", [{}])[0].get("type")
            for kernel in templated
            if kernel.get("args")
        }
        self.assertIn("const template_box<int>", arg_types)
        self.assertIn("const template_box<float>", arg_types)


class DiagnosticPersistenceTest(unittest.TestCase):
    def test_parse_run_args_rejects_semantic_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ksmoke-mode-") as td:
            args = type(
                "Args",
                (),
                {
                    "capture_dir": td,
                    "out_root": "out",
                    "target_lib": "target",
                    "run_id": "run",
                    "resume": False,
                    "mode": "semantic",
                    "jobs": 1,
                    "profile": False,
                },
            )()

            result = kernel_smoke_cli._parse_run_args(args)

        self.assertIsInstance(result, str)
        self.assertIn("only artifact mode is supported", result)

    def test_variant_failure_persists_reason_detail_and_selection(self) -> None:
        ds = kernel_smoke_cli._DiscoverState()
        result_obj = {
            "variant_id": "deadbeefcafe",
            "source_file": "/tmp/kernel.cu",
            "discovered_kernels": [],
            "preprocess": [{"variant_id": "deadbeefcafe", "reason": "preprocess_failed", "detail": "bad macro"}],
            "ast_log": [{"variant_id": "deadbeefcafe", "reason": "ast_failed", "detail": "bad ast"}],
            "replay_log": [{"variant_id": "deadbeefcafe", "reason": "toolchain_mismatch", "detail": "unsupported compiler: gcc"}],
            "emit_results": [],
            "manifest_results": [],
            "ast_kernels": 0,
            "fallback_kernels": 0,
            "ast_failures": 1,
            "ir_symbol_count": 0,
            "ptx_entry_count": 0,
            "ptx_missing_or_empty": True,
            "ptx_diag_reason": "toolchain_mismatch",
            "ptx_diag_detail": "unsupported compiler: gcc",
            "ptx_diag_phase": "device_bc",
            "ptx_diag_command": [],
            "ptx_ast_mismatch": None,
            "selection_reason": "replayable_cuda_compile",
            "compiler": "/usr/bin/gcc",
            "record_id": 7,
        }

        kernel_smoke_cli._accumulate_variant_result(ds, result_obj)

        with tempfile.TemporaryDirectory(prefix="ksmoke-diag-") as td:
            run_dir = Path(td)
            kernel_smoke_cli._persist_discover_progress(
                run_dir,
                ds,
                variant_order=["deadbeefcafe"],
                mode="artifact",
                jobs=1,
            )
            discover = json.loads((run_dir / "discover.json").read_text(encoding="utf-8"))

        diag = discover["diagnostics"]["variant_failures_by_variant"]["deadbeefcafe"]
        self.assertEqual(diag["phase"], "device_bc")
        self.assertEqual(diag["reason"], "toolchain_mismatch")
        self.assertEqual(diag["detail"], "unsupported compiler: gcc")
        self.assertEqual(diag["compiler"], "/usr/bin/gcc")
        self.assertEqual(diag["selection_reason"], "replayable_cuda_compile")

if __name__ == "__main__":
    unittest.main()
