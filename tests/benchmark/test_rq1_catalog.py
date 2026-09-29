import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.rq1.schema import (
    EXPECTED_WORKLOAD_IDS,
    FORBIDDEN_SOURCE_TOKENS,
    load_catalog,
    verify_catalog,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RQ1_ROOT = REPO_ROOT / "benchmark" / "rq1"
WORKLOADS_ROOT = REPO_ROOT / "benchmark" / "workloads"


def copy_catalog_fixture(temp_dir: str) -> tuple[Path, Path]:
    benchmark_root = Path(temp_dir) / "benchmark"
    root = benchmark_root / "rq1"
    root.mkdir(parents=True)
    shutil.copy2(RQ1_ROOT / "catalog.json", root / "catalog.json")
    (root / "workloads").symlink_to("../workloads")
    shutil.copytree(WORKLOADS_ROOT, benchmark_root / "workloads")
    return root, benchmark_root


class Rq1CatalogTests(unittest.TestCase):
    def test_catalog_has_exactly_six_approved_workloads(self) -> None:
        catalog = load_catalog(RQ1_ROOT)

        self.assertEqual(
            tuple(item.workload_id for item in catalog.workloads),
            EXPECTED_WORKLOAD_IDS,
        )

    def test_catalog_accepts_complete_current_corpus(self) -> None:
        errors = verify_catalog(RQ1_ROOT)

        self.assertEqual(errors, [])

    def test_catalog_reports_missing_kernel_through_workloads_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, benchmark_root = copy_catalog_fixture(temp_dir)
            workload_id = EXPECTED_WORKLOAD_IDS[0]
            (benchmark_root / "workloads" / workload_id / "kernel.cu").unlink()

            errors = verify_catalog(root)

        self.assertIn(
            f"{workload_id}: missing kernel: workloads/{workload_id}/kernel.cu",
            errors,
        )

    def test_catalog_rejects_synthetic_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, _ = copy_catalog_fixture(temp_dir)
            workload_id = EXPECTED_WORKLOAD_IDS[0]
            provenance_path = root / "workloads" / workload_id / "provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["source_kind"] = "synthetic"
            provenance["synthetic"] = {
                "description": "invalid replacement for upstream provenance",
                "design": "RQ1 verifier regression",
            }
            provenance.pop("upstream")
            provenance["extraction"].pop("patch_file")
            provenance["extraction"].pop("patch_sha256")
            provenance_path.write_text(
                json.dumps(provenance, indent=2) + "\n",
                encoding="utf-8",
            )

            self.assertIn(
                f"{workload_id}: RQ1 provenance must be upstream-backed",
                verify_catalog(root),
            )

    def test_workload_sources_have_no_manual_feedback(self) -> None:
        for path in (RQ1_ROOT / "workloads").glob("*/kernel.cu"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertFalse(
                    any(token in source for token in FORBIDDEN_SOURCE_TOKENS),
                    path,
                )

    def test_catalog_rejects_paths_outside_benchmark_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workloads": [
                            {
                                "id": workload_id,
                                "provenance": (
                                    "../outside.json"
                                    if index == 0
                                    else f"workloads/{workload_id}/provenance.json"
                                ),
                            }
                            for index, workload_id in enumerate(
                                EXPECTED_WORKLOAD_IDS
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "confined"):
                load_catalog(root)

    def test_verifier_cli_is_runnable_by_path(self) -> None:
        result = subprocess.run(
            [sys.executable, str(RQ1_ROOT / "verify.py")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
