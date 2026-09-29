import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.rq1 import runtime_verify
from benchmark.rq1.runtime_verify import validate_runtime_report


class Rq1RuntimeReportTests(unittest.TestCase):
    def test_rapid_no_feedback_uses_the_ordered_runtime_worker(self) -> None:
        with (
            patch.object(
                runtime_verify,
                "_rapid_ordered_worker",
                return_value={"worker": "ordered"},
            ) as ordered,
            patch.object(
                runtime_verify,
                "_sync_worker",
                return_value={"worker": "sync"},
            ) as sync,
        ):
            result = runtime_verify._worker(
                "rapid-no-feedback", Path("target.so"), b"seed", 10, 2
            )

        self.assertEqual(result, {"worker": "ordered"})
        ordered.assert_called_once_with(Path("target.so"), b"seed", 10, 2)
        sync.assert_not_called()

    def test_runtime_report_requires_completed_runs_feedback_and_allocations(self) -> None:
        runs = 10
        shared = {
            "runs": runs,
            "submitted": runs,
            "completed": runs,
            "failed": 0,
            "timeouts": 0,
            "invalid_inputs": 0,
            "seed_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "payload_size": 64,
            "vconfig": {"fixed": True},
        }
        report = {
            "runs": runs,
            "backends": {
                "cufuzz": {
                    **shared,
                    "coverage_nonzero_bytes": 0,
                    "simt_memcov_nonzero_bits": 0,
                    "allocation_stats_delta": {
                        "runs_started": runs,
                        "allocation_attempts": 2 * runs,
                        "allocations_succeeded": 2 * runs,
                        "frees": 2 * runs,
                        "runs_completed": runs,
                    },
                },
                "origin-no-feedback": {
                    **shared,
                    "coverage_nonzero_bytes": 0,
                    "simt_memcov_nonzero_bits": 0,
                },
                "origin": {
                    **shared,
                    "coverage_nonzero_bytes": 3,
                    "simt_memcov_nonzero_bits": 7,
                },
                "rapid-no-feedback": {
                    **shared,
                    "coverage_nonzero_bytes": 0,
                    "simt_memcov_nonzero_bits": 0,
                    "window_size": 2,
                    "pending": 0,
                    "completed_queue": 0,
                    "outstanding": 0,
                    "retired_task_ids": list(range(1, runs + 1)),
                },
                "rapid": {
                    **shared,
                    "coverage_nonzero_bytes": 3,
                    "simt_memcov_nonzero_bits": 7,
                    "window_size": 2,
                    "pending": 0,
                    "completed_queue": 0,
                    "outstanding": 0,
                    "retired_task_ids": list(range(1, runs + 1)),
                },
                "rapid2-no-feedback": {
                    **shared,
                    "coverage_nonzero_bytes": 0,
                    "simt_memcov_nonzero_bits": 0,
                },
                "rapid2": {
                    **shared,
                    "coverage_nonzero_bytes": 3,
                    "simt_memcov_nonzero_bits": 7,
                },
            },
        }

        validate_runtime_report(report)

        report["backends"]["rapid"]["outstanding"] = 1
        with self.assertRaisesRegex(RuntimeError, "rapid.*outstanding"):
            validate_runtime_report(report)
        report["backends"]["rapid"]["outstanding"] = 0
        report["backends"]["rapid"]["retired_task_ids"] = [2, 1] + list(
            range(3, runs + 1)
        )
        with self.assertRaisesRegex(RuntimeError, "rapid.*FIFO"):
            validate_runtime_report(report)

    def test_runtime_report_rejects_no_feedback_backend_with_coverage(self) -> None:
        report = {
            "runs": 1,
            "backends": {
                name: {
                    "runs": 1,
                    "submitted": 1,
                    "completed": 1,
                    "failed": 0,
                    "timeouts": 0,
                    "invalid_inputs": 0,
                    "seed_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                    "payload_size": 1,
                    "vconfig": {},
                    "coverage_nonzero_bytes": 1 if name == "rapid-no-feedback" else 0,
                    "simt_memcov_nonzero_bits": 0,
                    **(
                        {
                            "allocation_stats_delta": {
                                "runs_started": 1,
                                "allocation_attempts": 2,
                                "allocations_succeeded": 2,
                                "frees": 2,
                                "runs_completed": 1,
                            }
                        }
                        if name == "cufuzz"
                        else {}
                    ),
                }
                for name in (
                    "cufuzz",
                    "origin-no-feedback",
                    "origin",
                    "rapid-no-feedback",
                    "rapid",
                    "rapid2-no-feedback",
                    "rapid2",
                )
            },
        }
        report["backends"]["origin"]["coverage_nonzero_bytes"] = 1
        report["backends"]["rapid"]["coverage_nonzero_bytes"] = 1
        report["backends"]["rapid2"]["coverage_nonzero_bytes"] = 1

        with self.assertRaisesRegex(RuntimeError, "rapid-no-feedback.*zero feedback"):
            validate_runtime_report(report)

    def test_runtime_report_rejects_cufuzz_missing_free(self) -> None:
        report = {
            "runs": 2,
            "backends": {
                name: {
                    "runs": 2,
                    "submitted": 2,
                    "completed": 2,
                    "failed": 0,
                    "timeouts": 0,
                    "invalid_inputs": 0,
                    "seed_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                    "payload_size": 1,
                    "vconfig": {},
                    "coverage_nonzero_bytes": 0 if "feedback" in name or name == "cufuzz" else 1,
                    "simt_memcov_nonzero_bits": 0,
                    **(
                        {
                            "allocation_stats_delta": {
                                "runs_started": 2,
                                "allocation_attempts": 4,
                                "allocations_succeeded": 4,
                                "frees": 3,
                                "runs_completed": 2,
                            }
                        }
                        if name == "cufuzz"
                        else {}
                    ),
                }
                for name in (
                    "cufuzz",
                    "origin-no-feedback",
                    "origin",
                    "rapid-no-feedback",
                    "rapid",
                    "rapid2-no-feedback",
                    "rapid2",
                )
            },
        }
        for name in ("origin", "rapid", "rapid2"):
            report["backends"][name]["coverage_nonzero_bytes"] = 1

        with self.assertRaisesRegex(RuntimeError, "allocation lifecycle"):
            validate_runtime_report(report)


if __name__ == "__main__":
    unittest.main()
