import unittest
from dataclasses import asdict
from pathlib import Path

from .cufuzz_runtime_probe import AllocationStats, validate_allocation_delta
from .run_campaign import (
    CampaignResult,
    _parse_rapid_counters,
    backend_environment,
    build_fuzzer_command,
    throughput_execution_count,
)
from .run_rapid2_benchmark import (
    benchmark_case_from_campaign,
    build_benchmark_report,
    validate_build_identity,
)


class CampaignRunnerTest(unittest.TestCase):
    def test_fuzzer_command_is_mutating_with_run_cap(self) -> None:
        command_args = {
            "backend": "rapid2",
            "fuzzer_root": Path("/tmp/fuzzers"),
            "library": Path("/tmp/librapid2_target.so"),
            "manifest": Path("/tmp/manifest.json"),
            "runs": 100_000,
        }
        command = build_fuzzer_command(**command_args)

        self.assertIn("--runs", command)
        self.assertNotIn("--no-mutate", command)

    def test_cufuzz_allocation_delta_accepts_two_allocations_per_run(self) -> None:
        before = AllocationStats(
            runs_started=3,
            allocation_attempts=6,
            allocations_succeeded=6,
            frees=6,
            runs_completed=3,
        )
        after = AllocationStats(
            runs_started=8,
            allocation_attempts=16,
            allocations_succeeded=16,
            frees=16,
            runs_completed=8,
        )

        self.assertEqual(
            validate_allocation_delta(before, after, expected_runs=5),
            AllocationStats(
                runs_started=5,
                allocation_attempts=10,
                allocations_succeeded=10,
                frees=10,
                runs_completed=5,
            ),
        )

    def test_cufuzz_allocation_delta_rejects_missing_free(self) -> None:
        before = AllocationStats(0, 0, 0, 0, 0)
        after = AllocationStats(2, 4, 4, 3, 2)

        with self.assertRaisesRegex(RuntimeError, "allocation lifecycle mismatch"):
            validate_allocation_delta(before, after, expected_runs=2)

    def test_backend_environment_assigns_distinct_gpus(self) -> None:
        envs = {
            backend: backend_environment(backend, ["0", "1", "2"])
            for backend in ("origin", "rapid", "rapid2")
        }

        self.assertNotEqual(
            envs["origin"]["CUDA_VISIBLE_DEVICES"],
            envs["rapid"]["CUDA_VISIBLE_DEVICES"],
        )
        self.assertNotEqual(
            envs["rapid"]["CUDA_VISIBLE_DEVICES"],
            envs["rapid2"]["CUDA_VISIBLE_DEVICES"],
        )

    def test_rapid_counters_are_parsed_and_completed_count_drives_throughput(self) -> None:
        counters = _parse_rapid_counters(
            (
                "rapid_submitted=12000 rapid_completed=11000 "
                "rapid_pending=1000 rapid_failed=7\n",
                "rapid_submitted=11900 rapid_completed=10900 rapid_pending=1000\n",
            )
        )

        self.assertEqual(
            counters,
            {
                "rapid_submitted": 12000,
                "rapid_completed": 11000,
                "rapid_pending": 1000,
                "rapid_failed": 7,
            },
        )
        rapid = self._campaign_result(
            backend="rapid",
            executions=12000,
            client_joined=True,
            rapid_submitted=counters["rapid_submitted"],
            rapid_completed=counters["rapid_completed"],
            rapid_pending=counters["rapid_pending"],
            rapid_failed=counters["rapid_failed"],
        )
        origin = self._campaign_result(
            backend="origin", executions=8000, client_joined=True
        )

        self.assertEqual(throughput_execution_count(rapid), 11000)
        self.assertEqual(throughput_execution_count(origin), 8000)
        self.assertEqual(asdict(rapid)["rapid_completed"], 11000)

    def test_rapid_counters_preserve_one_coherent_latest_snapshot(self) -> None:
        counters = _parse_rapid_counters(
            (
                "rapid_submitted=12000 rapid_completed=11000 "
                "rapid_pending=1000 rapid_failed=3\n"
                "rapid_submitted=12500 rapid_completed=12400 "
                "rapid_pending=100 rapid_failed=4\n"
                "rapid_submitted=0 rapid_completed=0 "
                "rapid_pending=0 rapid_failed=0\n",
            )
        )

        self.assertEqual(
            counters,
            {
                "rapid_submitted": 12500,
                "rapid_completed": 12400,
                "rapid_pending": 100,
                "rapid_failed": 4,
            },
        )

    def test_rapid2_benchmark_calculates_rates_and_slowdown(self) -> None:
        instrumented = self._campaign_result(executions=80, client_joined=True)
        uninstrumented = self._campaign_result(executions=100, client_joined=True)

        report = build_benchmark_report(
            instrumented_build={
                "kernel_id": "k",
                "display_name": "kernel",
                "phase2_dir": "/tmp/k/phase2",
                "feedback_instrumentation": "enabled",
                "shared_lib": "/tmp/enabled/librapid2_target.so",
            },
            uninstrumented_build={
                "kernel_id": "k",
                "display_name": "kernel",
                "phase2_dir": "/tmp/k/phase2",
                "feedback_instrumentation": "disabled",
                "shared_lib": "/tmp/disabled/librapid2_target.so",
            },
            instrumented=instrumented,
            uninstrumented=uninstrumented,
            observed_seconds=10.0,
        )

        self.assertEqual(report["kernel_id"], "k")
        self.assertEqual(report["instrumented"]["executions_per_second"], 8.0)
        self.assertEqual(report["uninstrumented"]["executions_per_second"], 10.0)
        self.assertAlmostEqual(report["slowdown"], 0.2)
        self.assertAlmostEqual(report["slowdown_percent"], 20.0)

    def test_rapid2_benchmark_rejects_missing_client_or_zero_executions(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "client did not join"):
            benchmark_case_from_campaign(
                "instrumented",
                self._campaign_result(executions=1, client_joined=False),
                observed_seconds=1.0,
            )

        with self.assertRaisesRegex(RuntimeError, "positive executions"):
            benchmark_case_from_campaign(
                "uninstrumented",
                self._campaign_result(executions=0, client_joined=True),
                observed_seconds=1.0,
            )

    def test_rapid2_benchmark_rejects_mismatched_build_identity(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "same kernel_id"):
            validate_build_identity(
                {
                    "kernel_id": "a",
                    "display_name": "kernel",
                    "phase2_dir": "/tmp/a/phase2",
                    "feedback_instrumentation": "enabled",
                },
                {
                    "kernel_id": "b",
                    "display_name": "kernel",
                    "phase2_dir": "/tmp/a/phase2",
                    "feedback_instrumentation": "disabled",
                },
            )

    def _campaign_result(
        self,
        *,
        executions: int,
        client_joined: bool,
        backend: str = "rapid2",
        rapid_submitted: int = 0,
        rapid_completed: int = 0,
        rapid_pending: int = 0,
        rapid_failed: int = 0,
    ) -> CampaignResult:
        return CampaignResult(
            backend=backend,
            command=["/tmp/fuzzer_async"],
            cuda_visible_devices="0",
            broker_returncode=0,
            client_returncode=0,
            client_joined=client_joined,
            executions=executions,
            cfg_sites=1,
            simt_memcov_bits=1,
            logical_thread_bits=1,
            broker_log="/tmp/broker.log",
            client_log="/tmp/client.log",
            timed_out=False,
            rapid_submitted=rapid_submitted,
            rapid_completed=rapid_completed,
            rapid_pending=rapid_pending,
            rapid_failed=rapid_failed,
        )


if __name__ == "__main__":
    unittest.main()
