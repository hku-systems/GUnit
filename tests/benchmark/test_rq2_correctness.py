import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.rq2.correctness import build_and_run_correctness


REPO_ROOT = Path(__file__).resolve().parents[2]


class Rq2CorrectnessTests(unittest.TestCase):
    def test_builds_and_runs_all_four_drivers_on_requested_gpu(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="ok\n", stderr="")
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "benchmark.rq2.correctness.subprocess.run", return_value=completed
        ) as run:
            build_and_run_correctness(
                repo_root=REPO_ROOT,
                cuda_path="/cuda",
                gpu_device=2,
                build_dir=Path(temp_dir),
            )

        calls = run.call_args_list
        self.assertEqual(len(calls), 8)
        compile_calls = calls[0], calls[2], calls[4], calls[6]
        run_calls = calls[1], calls[3], calls[5], calls[7]
        workload_ids = (
            "shoc_scan",
            "shoc_reduction",
            "flashattention_device1xn",
            "synth_complex",
        )
        for workload_id, call in zip(workload_ids, compile_calls):
            command = call.args[0]
            self.assertEqual(command[0], "/cuda/bin/nvcc")
            self.assertIn("-std=c++17", command)
            self.assertIn("-O3", command)
            self.assertIn("-DNDEBUG", command)
            self.assertIn(
                str(REPO_ROOT / "benchmark" / "rq2" / "correctness" / f"{workload_id}_driver.cu"),
                command,
            )
            self.assertIn(
                str(REPO_ROOT / "benchmark" / "rq2" / "workloads" / workload_id / "kernel.cu"),
                command,
            )
            self.assertEqual(command[-2], "-o")
            self.assertTrue(command[-1].endswith(f"/{workload_id}_correctness"))
        for workload_id, call in zip(workload_ids, run_calls):
            self.assertTrue(call.args[0][0].endswith(f"/{workload_id}_correctness"))
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(call.kwargs["capture_output"], True)
            self.assertEqual(call.kwargs["text"], True)

    def test_driver_failure_is_reported_with_output(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 7, stdout="scan out", stderr="scan err"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "benchmark.rq2.correctness.subprocess.run", side_effect=results
        ):
            with self.assertRaisesRegex(RuntimeError, "scan out.*scan err"):
                build_and_run_correctness(
                    repo_root=REPO_ROOT,
                    cuda_path="/cuda",
                    gpu_device=0,
                    build_dir=Path(temp_dir),
                )


if __name__ == "__main__":
    unittest.main()
