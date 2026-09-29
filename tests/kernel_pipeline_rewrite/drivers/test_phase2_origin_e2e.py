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
from shared_origin import run_origin_case  # noqa: E402
from shared_phase2 import run_phase2_for_run_dir  # noqa: E402


class Phase2OriginEndToEndTest(unittest.TestCase):
    def test_origin_shared_library_executes_generated_kernel(self) -> None:
        if reason := cuda_device_skip_reason():
            self.skipTest(reason)
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        summary = run_phase2_for_run_dir(ctx.run_dir)
        self.assertGreater(summary["counts"]["built"], 0)

        result = run_origin_case(run_dir=ctx.run_dir, case_name="add_kernel")
        self.assertEqual(result["output"], result["expected"])
        self.assertEqual(result["output_size"], result["expected_size"])


if __name__ == "__main__":
    unittest.main()
