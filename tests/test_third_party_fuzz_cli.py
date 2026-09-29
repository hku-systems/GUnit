from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.third_party_fuzz.cli as cli


class ThirdPartyFuzzCliTest(unittest.TestCase):
    def test_project_filter_limits_collect_to_selected_fresh_runs(self) -> None:
        available = {"gpurir": object(), "cudasift": object()}
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(
                cli, "project_runs_for_campaign", return_value=available
            ),
            mock.patch.object(cli, "collect_kernel_records", return_value=[]) as collect,
            mock.patch.object(cli, "save_kernel_results"),
            mock.patch.object(cli, "_print_json"),
        ):
            result = cli.main(
                [
                    "--campaign-dir",
                    td,
                    "--project",
                    "cudasift",
                    "collect",
                ]
            )

        self.assertEqual(result, 0)
        collect.assert_called_once_with(runs={"cudasift": available["cudasift"]})

    def test_backend_matrix_dispatches_only_selected_backends(self) -> None:
        records = [
            {
                "support_decision": "run",
                "backend_results": {
                    "origin": {"status": "passed"},
                    "rapid2": {"status": "passed"},
                },
            }
        ]
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(cli, "project_runs_for_campaign", return_value={}),
            mock.patch.object(
                cli, "run_backend_matrix_stage", return_value=records
            ) as run_matrix,
            mock.patch.object(cli, "_print_json"),
        ):
            result = cli.main(
                [
                    "--campaign-dir",
                    td,
                    "backend-matrix",
                    "--backend",
                    "origin",
                    "--backend",
                    "rapid2",
                    "--cuda-arch",
                    "sm_90",
                    "--timeout-seconds",
                    "123",
                ]
            )

        self.assertEqual(result, 0)
        run_matrix.assert_called_once_with(
            campaign_dir=Path(td),
            runs={},
            backends=("origin", "rapid2"),
            kernel_selectors=None,
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_90",
            timeout_seconds=123,
        )

    def test_coverage_matrix_dispatches_gpu_seed_and_applicability_cells(
        self,
    ) -> None:
        records = [
            {
                "kernel_id": "enabled__11111111",
                "display_name": "enabled",
                "support_decision": "run",
                "coverage_results": {
                    "rapid2": {
                        "off": {"status": "passed"},
                        "on": {
                            "status": "skipped",
                            "skip_kind": "applicability",
                        },
                    }
                },
            },
            {
                "kernel_id": "deferred__22222222",
                "display_name": "deferred",
                "support_decision": "run",
            },
        ]
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(cli, "project_runs_for_campaign", return_value={}),
            mock.patch.object(
                cli, "run_coverage_matrix_stage", return_value=records
            ) as run_matrix,
            mock.patch.object(cli, "_print_json"),
        ):
            result = cli.main(
                [
                    "--campaign-dir",
                    td,
                    "--kernel",
                    "enabled",
                    "coverage-matrix",
                    "--backend",
                    "rapid2",
                    "--vconfig",
                    "off",
                    "--vconfig",
                    "on",
                    "--coverage-seconds",
                    "5",
                    "--timeout-seconds",
                    "40",
                    "--seed",
                    "17",
                    "--gpu-device",
                    "3",
                    "--fuzzer",
                    "/tmp/fuzzer",
                    "--fuzzer-async",
                    "/tmp/fuzzer_async",
                ]
            )

        self.assertEqual(result, 0)
        run_matrix.assert_called_once_with(
            campaign_dir=Path(td),
            runs={},
            backends=("rapid2",),
            kernel_selectors=("enabled",),
            vconfigs=("off", "on"),
            coverage_seconds=5,
            timeout_seconds=40,
            seed=17,
            fuzzer=Path("/tmp/fuzzer"),
            fuzzer_async=Path("/tmp/fuzzer_async"),
            gpu_device="3",
        )


if __name__ == "__main__":
    unittest.main()
