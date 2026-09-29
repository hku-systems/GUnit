import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_DIR = Path(__file__).resolve().parent
SMOKE_DRIVERS_DIR = REPO_ROOT / "tests" / "kernel_pipeline_smoke" / "drivers"
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
if str(SMOKE_DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(SMOKE_DRIVERS_DIR))

from shared_e2e import cuda_device_skip_reason, get_capture_e2e_context  # noqa: E402
from shared_phase2 import run_phase2_for_run_dir  # noqa: E402
from shared_runtime import RuntimeSmokeRunner, discover_runtime_cases  # noqa: E402


class Phase2RuntimeEndToEndTest(unittest.TestCase):
    def test_runtime_cases_execute_against_phase2_outputs(self) -> None:
        if reason := cuda_device_skip_reason():
            self.skipTest(reason)
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        summary = run_phase2_for_run_dir(ctx.run_dir)
        self.assertGreater(summary["counts"]["built"], 0)

        runner = RuntimeSmokeRunner()
        for case in discover_runtime_cases():
            with self.subTest(case=case["case_name"]):
                result = runner.run(run_dir=ctx.run_dir, case_name=case["case_name"])
                runtime_json = Path(result["runtime_dir"]) / "runtime_smoke.json"
                self.assertTrue(runtime_json.is_file(), "missing runtime_smoke.json")
                result_json = json.loads(runtime_json.read_text(encoding="utf-8"))
                self.assertEqual(result.get("returncode"), case["expected_exit_code"])
                self.assertIn(case["expected_stdout"], result.get("stdout", ""))
                self.assertEqual(result_json.get("returncode"), case["expected_exit_code"])
                self.assertEqual(result_json.get("expected_exit_code"), case["expected_exit_code"])


if __name__ == "__main__":
    unittest.main()
