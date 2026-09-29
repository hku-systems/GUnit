import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIZER_PATH = REPO_ROOT / "scripts" / "categorize_phase1_results.py"


def load_categorizer():
    spec = importlib.util.spec_from_file_location("categorize_phase1_results", CATEGORIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Phase1ResultCategorizerTest(unittest.TestCase):
    def test_writes_supported_unsupported_error_outputs(self):
        categorizer = load_categorizer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results_path = root / "e2e_results.json"
            out_dir = root / "out"
            results_path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "candidate": str(REPO_ROOT / "third_party" / "ok.cu"),
                                "stages": {
                                    "phase1": {
                                        "ok": True,
                                        "run_dir": str(root / "phase1" / "ok"),
                                        "kernel_count": 2,
                                    }
                                },
                            },
                            {
                                "candidate": str(REPO_ROOT / "third_party" / "unsupported.cu"),
                                "stages": {
                                    "phase1": {
                                        "ok": False,
                                        "failure_reason": "union_not_supported",
                                        "status": "failed",
                                    }
                                },
                            },
                            {
                                "candidate": str(REPO_ROOT / "third_party" / "error.cu"),
                                "stages": {
                                    "phase1": {
                                        "ok": False,
                                        "failure_reason": "command_failed",
                                        "status": "failed",
                                    }
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = categorizer.categorize_results(results_path, out_dir)

            self.assertEqual((out_dir / "supported_kernels.txt").read_text().splitlines(), ["third_party/ok.cu"])
            self.assertIn("third_party/unsupported.cu\tunion_not_supported", (out_dir / "unsupported_kernels.txt").read_text())
            self.assertIn("third_party/error.cu\tcommand_failed", (out_dir / "error_kernels.txt").read_text())
            self.assertEqual(report["stats"]["supported"], 1)
            self.assertEqual(report["stats"]["unsupported"], 1)
            self.assertEqual(report["stats"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
