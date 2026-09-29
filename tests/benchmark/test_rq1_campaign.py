from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.rq1 import campaign

from benchmark.rq1.campaign import (
    BACKENDS,
    CONFIGURATIONS,
    BenchmarkRecord,
    aggregate_records,
    artifact_backend,
    benchmark_command,
    parse_benchmark_result,
    trial_order,
)


class Rq1CampaignTests(unittest.TestCase):
    def test_optional_environment_tool_absence_is_recorded(self) -> None:
        with patch.object(
            campaign.subprocess,
            "run",
            side_effect=FileNotFoundError("missing-tool"),
        ):
            captured = campaign._capture(["missing-tool", "--version"])

        self.assertIn("unavailable", captured)
        self.assertIn("missing-tool", captured)

    def test_build_summary_accepts_a_selected_workload_subset(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            report_paths = []
            for workload_id in ("first", "second"):
                report_path = root / workload_id / "build_report.json"
                report_path.parent.mkdir()
                report_path.write_text(
                    json.dumps(
                        {
                            "workload_id": workload_id,
                            "backends": {backend: {} for backend in BACKENDS},
                        }
                    ),
                    encoding="utf-8",
                )
                report_paths.append(str(report_path))
            (root / "build_summary.json").write_text(
                json.dumps({"reports": report_paths}), encoding="utf-8"
            )
            (root / "runtime_verification_summary.json").write_text(
                json.dumps({"reports": []}), encoding="utf-8"
            )

            reports = campaign._load_build_reports(root)

        self.assertEqual(
            [report[1]["workload_id"] for report in reports],
            ["first", "second"],
        )

    def test_requested_window_size_is_reflected_in_configuration_labels(self) -> None:
        configurations = campaign.configurations(4)

        self.assertEqual(
            configurations,
            (
                "cufuzz",
                "origin-no-feedback",
                "origin",
                "rapid-w1",
                "rapid-w4",
                "rapid-no-feedback-w1",
                "rapid-no-feedback-w4",
                "rapid2",
                "rapid2-no-feedback",
            ),
        )
        self.assertEqual(artifact_backend("rapid-w4"), "rapid")
        self.assertEqual(
            artifact_backend("rapid-no-feedback-w4"), "rapid-no-feedback"
        )
        self.assertEqual(len(campaign.configurations(1)), 7)

    def test_window_size_accepts_32_and_rejects_33(self) -> None:
        common_args = [
            "campaign.py",
            "--build-root",
            "build",
            "--output",
            "output",
            "--repetitions",
            "1",
            "--warmup-runs",
            "1",
            "--benchmark-seconds",
            "1",
            "--window-size",
        ]
        with patch.object(campaign, "run_campaign") as run_campaign:
            with patch("sys.argv", common_args + ["32"]):
                campaign.main()
            self.assertEqual(run_campaign.call_args.args[0].window_size, 32)

            run_campaign.reset_mock()
            with patch("sys.argv", common_args + ["33"]):
                with self.assertRaisesRegex(SystemExit, r"1\.\.=32"):
                    campaign.main()
            run_campaign.assert_not_called()

    def test_resume_treats_failed_trials_as_attempted_without_aggregating_them(self) -> None:
        if not hasattr(campaign, "load_attempted_trials"):
            self.fail("campaign resume does not load failed trial keys")

        with tempfile.TemporaryDirectory() as raw_tmp:
            output = Path(raw_tmp)
            (output / "trials.jsonl").write_text(
                json.dumps(
                    {
                        "workload_id": "scan",
                        "backend": "rapid-w1",
                        "repetition": 0,
                        "executions_per_second": 12.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output / "failures.jsonl").write_text(
                json.dumps(
                    {
                        "workload_id": "scan",
                        "backend": "rapid-w2",
                        "repetition": 0,
                        "failure": "timeout",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            attempted, records = campaign.load_attempted_trials(output)

        self.assertEqual(
            attempted,
            {("scan", "rapid-w1", 0), ("scan", "rapid-w2", 0)},
        )
        self.assertEqual(
            records,
            [BenchmarkRecord("scan", "rapid-w1", 0, 12.5)],
        )

    def test_nonzero_exit_is_recorded_and_campaign_continues(self) -> None:
        benchmark = {
            "requested_min_duration_ns": 1,
            "measured_iterations": 1,
            "completed": 4,
            "elapsed_ns": 1,
            "corpus_size": 1,
            "solutions": 0,
            "pending": 0,
            "mutation_enabled": False,
        }
        failed = SimpleNamespace(returncode=17, stdout="failed\n", stderr="error\n")
        succeeded = SimpleNamespace(
            returncode=0,
            stdout="RAPID_BENCHMARK_RESULT " + json.dumps(benchmark) + "\n",
            stderr="",
        )
        report = {
            "workload_id": "scan",
            "shared_phase2": {
                "artifacts": {"manifest.json": {"sha256": "manifest-sha"}}
            },
            "backends": {
                backend: {
                    "artifacts": {
                        "shared_library": {"sha256": f"{backend}-sha"}
                    }
                }
                for backend in ("origin", "rapid2")
            },
            "source_provenance": {},
        }

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            args = SimpleNamespace(
                build_root=root / "build",
                output=root / "output",
                fuzzer=root / "fuzzer",
                fuzzer_async=root / "fuzzer_async",
                gpu_device="0",
                window_size=2,
                repetitions=1,
                warmup_runs=1,
                benchmark_seconds=1,
                timeout=10,
            )
            with (
                patch.object(campaign, "_load_build_reports", return_value=[(root, report)]),
                patch.object(campaign, "_environment", return_value={}),
                patch.object(campaign, "_manifest", return_value=root / "manifest.json"),
                patch.object(campaign, "_backend_library", return_value=root / "target.so"),
                patch.object(campaign, "_seed_sha256", return_value="input-sha"),
                patch.object(campaign, "configurations", return_value=("origin", "rapid2")),
                patch.object(campaign.subprocess, "run", side_effect=(failed, succeeded)) as run,
                self.assertLogs("benchmark.rq1.campaign", level="WARNING") as logs,
            ):
                campaign.run_campaign(args)

            failures = [
                json.loads(line)
                for line in (args.output / "failures.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            trials = [
                json.loads(line)
                for line in (args.output / "trials.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(run.call_count, 2)
        self.assertEqual(failures[0]["failure"], "nonzero_exit")
        self.assertEqual(failures[0]["exit_code"], 17)
        self.assertEqual(trials[0]["backend"], "rapid2")
        self.assertTrue(any("scan origin" in message for message in logs.output))

    def test_benchmark_command_adds_only_the_configured_window(self) -> None:
        common = {
            "executable": "fuzzer",
            "library": "target.so",
            "manifest": "manifest.json",
            "warmup_runs": 100,
            "benchmark_seconds": 30,
        }

        rapid_w1 = benchmark_command(window_size=1, **common)
        rapid_w2 = benchmark_command(window_size=2, **common)
        origin = benchmark_command(window_size=None, **common)

        self.assertEqual(rapid_w1[-2:], ["--window-size", "1"])
        self.assertEqual(rapid_w2[-2:], ["--window-size", "2"])
        self.assertEqual(artifact_backend("rapid-w1"), "rapid")
        self.assertEqual(artifact_backend("rapid-w2"), "rapid")
        self.assertEqual(
            artifact_backend("rapid-no-feedback-w1"), "rapid-no-feedback"
        )
        self.assertEqual(
            artifact_backend("rapid-no-feedback-w2"), "rapid-no-feedback"
        )
        self.assertNotIn("--window-size", origin)
        for command in (rapid_w1, rapid_w2, origin):
            self.assertNotIn("--retire-worker", command)
            self.assertNotIn("--supply-threads", command)

    def test_parse_benchmark_result_requires_one_valid_marker(self) -> None:
        payload = {
            "requested_min_duration_ns": 30_000_000_000,
            "measured_iterations": 4200,
            "completed": 71,
            "elapsed_ns": 30_002_000_000,
            "corpus_size": 2,
            "solutions": 0,
            "pending": 0,
            "mutation_enabled": False,
        }
        output = "setup\nRAPID_BENCHMARK_RESULT " + json.dumps(payload) + "\ncleanup\n"

        self.assertEqual(parse_benchmark_result(output), payload)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            parse_benchmark_result("no result")
        with self.assertRaisesRegex(RuntimeError, "completed"):
            parse_benchmark_result(
                "RAPID_BENCHMARK_RESULT "
                + json.dumps({**payload, "completed": 0})
            )
        with self.assertRaisesRegex(RuntimeError, "minimum duration"):
            parse_benchmark_result(
                "RAPID_BENCHMARK_RESULT "
                + json.dumps({**payload, "elapsed_ns": 29_999_999_999})
            )

    def test_trial_order_rotates_all_nine_configurations_once(self) -> None:
        self.assertEqual(
            BACKENDS,
            (
                "cufuzz",
                "origin-no-feedback",
                "origin",
                "rapid-no-feedback",
                "rapid",
                "rapid2-no-feedback",
                "rapid2",
            ),
        )
        self.assertEqual(
            CONFIGURATIONS,
            (
                "cufuzz",
                "origin-no-feedback",
                "origin",
                "rapid-w1",
                "rapid-w2",
                "rapid-no-feedback-w1",
                "rapid-no-feedback-w2",
                "rapid2",
                "rapid2-no-feedback",
            ),
        )
        self.assertEqual(trial_order(0), CONFIGURATIONS)
        self.assertEqual(
            trial_order(1), CONFIGURATIONS[1:] + CONFIGURATIONS[:1]
        )
        self.assertEqual(set(trial_order(7)), set(CONFIGURATIONS))

    def test_aggregate_records_reports_median_and_quartiles(self) -> None:
        records = [
            BenchmarkRecord("w", "origin", repetition, 10.0 + repetition)
            for repetition in range(5)
        ]

        rows = aggregate_records(records)

        self.assertEqual(
            rows,
            [
                {
                    "workload_id": "w",
                    "backend": "origin",
                    "repetitions": 5,
                    "median_exec_per_sec": 12.0,
                    "q1_exec_per_sec": 11.0,
                    "q3_exec_per_sec": 13.0,
                }
            ],
        )

    def test_aggregate_single_smoke_repetition_uses_the_observation_as_quartiles(self) -> None:
        rows = aggregate_records([BenchmarkRecord("w", "rapid2", 0, 42.0)])

        self.assertEqual(rows[0]["median_exec_per_sec"], 42.0)
        self.assertEqual(rows[0]["q1_exec_per_sec"], 42.0)
        self.assertEqual(rows[0]["q3_exec_per_sec"], 42.0)

    def test_parse_benchmark_result_rejects_pending_work(self) -> None:
        payload = {
            "requested_min_duration_ns": 1,
            "measured_iterations": 1,
            "completed": 4,
            "elapsed_ns": 1,
            "corpus_size": 1,
            "solutions": 0,
            "pending": 1,
            "mutation_enabled": False,
        }
        with self.assertRaisesRegex(RuntimeError, "pending"):
            parse_benchmark_result(
                "RAPID_BENCHMARK_RESULT " + json.dumps(payload)
            )

    def test_parse_benchmark_result_requires_mutation_to_be_disabled(self) -> None:
        payload = {
            "requested_min_duration_ns": 1,
            "measured_iterations": 1,
            "completed": 4,
            "elapsed_ns": 1,
            "corpus_size": 1,
            "solutions": 0,
            "pending": 0,
        }
        with self.assertRaisesRegex(RuntimeError, "mutation_enabled=false"):
            parse_benchmark_result(
                "RAPID_BENCHMARK_RESULT " + json.dumps(payload)
            )
        with self.assertRaisesRegex(RuntimeError, "mutation_enabled=false"):
            parse_benchmark_result(
                "RAPID_BENCHMARK_RESULT "
                + json.dumps({**payload, "mutation_enabled": True})
            )


if __name__ == "__main__":
    unittest.main()
