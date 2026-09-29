import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAPID_DIR = REPO_ROOT / "cuda-kernel" / "rapid"


class RapidTaskLocalFeedbackTest(unittest.TestCase):
    def test_worker_copies_feedback_into_the_current_task_before_publication(self) -> None:
        source = (RAPID_DIR / "harness.cpp").read_text(encoding="utf-8")
        worker = source[
            source.index("static void gpu_producer_worker()") : source.index(
                "static void ensure_initialized()"
            )
        ]

        self.assertIn("cuMemcpyDtoHAsync(task->coverage.data()", worker)
        self.assertIn("cuMemcpyDtoHAsync(task->simt_memcov.data()", worker)
        self.assertIn("cuMemcpyDtoHAsync(task->output.get()", worker)
        self.assertIn("publish_completion(std::move(task), status)", worker)
        self.assertNotIn("libafl_cov_map", worker)
        self.assertNotIn("libafl_simt_memcov_bits", worker)

    def test_terminal_completion_is_counted_at_publication_not_submission(self) -> None:
        source = (RAPID_DIR / "harness.cpp").read_text(encoding="utf-8")
        publish_start = source.index("inline bool publish_completion")
        publish = source[
            publish_start : source.index(
                "inline void print_rapid_feedback_stats", publish_start
            )
        ]
        submit = source[
            source.index("libafl_submit_with_id") : source.index(
                "libafl_poll_results"
            )
        ]

        self.assertIn("completed_inputs.fetch_add", publish)
        self.assertIn("pending_inputs.fetch_sub", publish)
        self.assertNotIn("completed_inputs.fetch_add", submit)
        self.assertIn("pending_inputs.fetch_add", submit)

    def test_rapid_has_no_cumulative_feedback_or_legacy_sync_abi(self) -> None:
        source = (RAPID_DIR / "harness.cpp").read_text(encoding="utf-8")

        self.assertNotIn("feedback_accumulator", source)
        self.assertNotIn("RapidFeedbackAccumulator", source)
        self.assertNotIn("libafl_target(", source)
        self.assertNotIn("libafl_get_last_run_status", source)
        self.assertIn("libafl_submit_with_id", source)
        self.assertIn("libafl_poll_results", source)
        self.assertIn("libafl_release_tasks", source)


if __name__ == "__main__":
    unittest.main()
