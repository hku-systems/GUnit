import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = REPO_ROOT / "scripts" / "analyze_e2e_results.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_e2e_results", ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E2EResultAnalyzerTest(unittest.TestCase):
    def test_aggregates_failures_and_writes_summary(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "fixture" / "e2e_results.json"
            second = root / "third_party" / "e2e_results.json"
            out = root / "SUMMARY.md"
            first.parent.mkdir()
            second.parent.mkdir()

            first.write_text(
                json.dumps(
                    {
                        "candidate_count": 2,
                        "fixed_seed": 42,
                        "fuzzer_mode": "fixed",
                        "results": [
                            {
                                "candidate": str(REPO_ROOT / "tests" / "ok.cu"),
                                "stages": {"phase1": {"ok": True}, "phase2": {"ok": True}},
                                "kernels": [
                                    {
                                        "display_name": "ok_kernel",
                                        "kernel_id": "ok__1",
                                        "phase2_status": "built",
                                        "stages": {
                                            "build_rapid2": {"ok": True, "status": "ok"},
                                            "fuzzer": {"ok": True, "status": "ok"},
                                        },
                                    }
                                ],
                            },
                            {
                                "candidate": str(REPO_ROOT / "tests" / "phase1.cu"),
                                "stages": {
                                    "phase1": {
                                        "ok": False,
                                        "status": "failed",
                                        "failure_reason": "union_not_supported",
                                    }
                                },
                                "kernels": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "candidate_count": 2,
                        "fixed_seed": 42,
                        "fuzzer_mode": "fixed",
                        "results": [
                            {
                                "candidate": str(REPO_ROOT / "third_party" / "mixed.cu"),
                                "stages": {
                                    "phase1": {"ok": True},
                                    "phase2": {
                                        "ok": True,
                                        "records": [
                                            {
                                                "kernel_id": "phase2_failed__1",
                                                "display_name": "phase2_failed",
                                                "failure_reason": "phase2_input_invalid",
                                                "failure_context": {
                                                    "reason_code": "protected_data_field",
                                                    "support_path": "scoreShift.__x",
                                                },
                                                "failure_detail": "scoreShift.__x:protected_data_field",
                                            }
                                        ],
                                    },
                                },
                                "kernels": [
                                    {
                                        "display_name": "phase2_failed",
                                        "kernel_id": "phase2_failed__1",
                                        "phase2_status": "failed",
                                        "stages": {},
                                    },
                                    {
                                        "display_name": "build_failed",
                                        "kernel_id": "build_failed__1",
                                        "phase2_status": "built",
                                        "stages": {
                                            "build_rapid2": {
                                                "ok": False,
                                                "status": "failed",
                                                "stderr": "ld: cannot find -lcuda",
                                            }
                                        },
                                    },
                                    {
                                        "display_name": "fuzzer_failed",
                                        "kernel_id": "fuzzer_failed__1",
                                        "phase2_status": "built",
                                        "stages": {
                                            "build_rapid2": {"ok": True, "status": "ok"},
                                            "fuzzer": {
                                                "ok": False,
                                                "status": "failed",
                                                "stderr": "cudaGetDeviceCount failed: no CUDA-capable device is detected",
                                            },
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = analyzer.analyze_results([first, second], out)
            summary = out.read_text(encoding="utf-8")

            self.assertEqual(report["overall"]["candidates_total"], 4)
            self.assertEqual(report["overall"]["kernels_total"], 4)
            self.assertEqual(report["overall"]["kernels_success"], 1)
            self.assertEqual(report["overall"]["kernels_failed"], 3)
            self.assertEqual(report["phase1_failures"]["union_not_supported"], 1)
            self.assertEqual(report["rewrite_failures"]["protected_data_field"], 1)
            self.assertEqual(report["build_failures"]["link_error"], 1)
            self.assertEqual(report["fuzzer_failures"]["no_gpu"], 1)
            self.assertIn("# E2E Results Summary", summary)
            self.assertIn("| Total kernels | 4 |", summary)
            self.assertIn("## Rewrite Failures", summary)
            self.assertIn("| protected_data_field | 1 |", summary)
            self.assertIn("- `tests/ok.cu` :: `ok_kernel` (`ok__1`)", summary)
            self.assertIn("## Recommendations", summary)

if __name__ == "__main__":
    unittest.main()
