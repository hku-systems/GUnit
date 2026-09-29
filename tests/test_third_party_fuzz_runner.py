import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import scripts.third_party_fuzz.runner as runner
from scripts.third_party_fuzz.runner import (
    BACKEND_SPECS,
    RUNS,
    _timeout_output_text,
    backend_stage_if_fresh,
    backend_dir_for,
    backend_library_for,
    build_backend_command,
    build_coverage_command,
    build_fuzzer_command,
    classify_fuzz_failure,
    collect_kernel_records,
    enrich_runtime_classifications,
    evaluate_fixed_fuzz_output,
    evaluate_mutation_fuzz_output,
    ensure_fuzzer_async,
    evaluate_coverage_log,
    run_build_stage,
    iter_phase2_built,
    mutation_stage_passes,
    parse_fuzzer_stats,
    project_runs_for_campaign,
    run_backend_build,
    run_backend_matrix_stage,
    verify_backend_exports,
)


class ThirdPartyFuzzRunnerTest(unittest.TestCase):
    def test_select_kernel_records_accepts_ids_or_display_names(self) -> None:
        records = [
            {"kernel_id": "first__11111111", "display_name": "first"},
            {"kernel_id": "second__22222222", "display_name": "shared"},
            {"kernel_id": "third__33333333", "display_name": "shared"},
        ]

        selected = runner.select_kernel_records(
            records, ("first__11111111", "shared")
        )

        self.assertEqual(
            [record["kernel_id"] for record in selected],
            ["first__11111111", "second__22222222", "third__33333333"],
        )
        with self.assertRaisesRegex(ValueError, "unknown kernel selector"):
            runner.select_kernel_records(records, ("missing",))

    def test_backend_specs_cover_all_runtime_abis(self) -> None:
        self.assertEqual(set(BACKEND_SPECS), {"origin", "rapid", "rapid2"})
        self.assertEqual(BACKEND_SPECS["origin"].library_name, "libphase2_origin_target.so")
        self.assertEqual(BACKEND_SPECS["rapid"].library_name, "libphase2_rapid_target.so")
        self.assertEqual(BACKEND_SPECS["rapid2"].library_name, "librapid2_target.so")
        self.assertEqual(BACKEND_SPECS["origin"].fuzzer_binary, "fuzzer")
        self.assertEqual(BACKEND_SPECS["rapid"].fuzzer_binary, "fuzzer")
        self.assertEqual(BACKEND_SPECS["rapid2"].fuzzer_binary, "fuzzer_async")
        self.assertIn("libafl_target", BACKEND_SPECS["origin"].required_exports)
        self.assertIn(
            "libafl_get_ordered_queue_counts",
            BACKEND_SPECS["rapid"].required_exports,
        )
        self.assertIn(
            "libafl_wait_for_completion",
            BACKEND_SPECS["rapid2"].required_exports,
        )

    def test_backend_commands_and_artifacts_are_backend_specific(self) -> None:
        record = {"filesystem_project": "demo", "kernel_id": "kernel__12345678"}
        campaign = Path("/tmp/campaign")
        for backend_name, library_name in (
            ("origin", "libphase2_origin_target.so"),
            ("rapid", "libphase2_rapid_target.so"),
            ("rapid2", "librapid2_target.so"),
        ):
            out_dir = backend_dir_for(campaign, record, backend=backend_name)
            command = build_backend_command(
                backend=backend_name,
                phase2_dir=Path("/tmp/phase2"),
                out_dir=out_dir,
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
            )

            self.assertIn(f"cuda-kernel/{backend_name}/build.py", command[1])
            self.assertEqual(
                out_dir,
                campaign / "demo/kernels/kernel__12345678/backends" / backend_name,
            )
            self.assertEqual(
                backend_library_for(campaign, record, backend=backend_name),
                out_dir / library_name,
            )

    def test_coverage_commands_match_each_backend_frontend(self) -> None:
        common = {
            "manifest": Path("/tmp/manifest.json"),
            "coverage_seconds": 60,
            "coverage_log": Path("/tmp/coverage.jsonl"),
            "vconfig": "on",
        }
        origin = build_coverage_command(
            backend="origin",
            fuzzer=Path("/tmp/fuzzer"),
            library=Path("/tmp/libphase2_origin_target.so"),
            window_size=None,
            **common,
        )
        rapid = build_coverage_command(
            backend="rapid",
            fuzzer=Path("/tmp/fuzzer"),
            library=Path("/tmp/libphase2_rapid_target.so"),
            window_size=4,
            **common,
        )
        rapid2 = build_coverage_command(
            backend="rapid2",
            fuzzer=Path("/tmp/fuzzer_async"),
            library=Path("/tmp/librapid2_target.so"),
            window_size=32,
            **common,
        )

        for command in (origin, rapid, rapid2):
            self.assertIn("--coverage-seconds", command)
            self.assertIn("--coverage-log", command)
            self.assertIn("--vconfig", command)
            self.assertEqual(command[command.index("--vconfig") + 1], "on")
        self.assertNotIn("--window-size", origin)
        self.assertEqual(rapid[-2:], ["--window-size", "4"])
        self.assertEqual(rapid2[-2:], ["--window-size", "32"])

    def test_coverage_command_rejects_invalid_backend_configuration(self) -> None:
        common = {
            "backend": "rapid",
            "fuzzer": Path("/tmp/fuzzer"),
            "library": Path("/tmp/libphase2_rapid_target.so"),
            "manifest": Path("/tmp/manifest.json"),
            "coverage_seconds": 60,
            "coverage_log": Path("/tmp/coverage.jsonl"),
            "vconfig": "on",
        }
        with self.assertRaisesRegex(ValueError, "window_size"):
            build_coverage_command(window_size=None, **common)
        with self.assertRaisesRegex(ValueError, "vconfig"):
            build_coverage_command(window_size=4, **{**common, "vconfig": "maybe"})

    def test_coverage_log_requires_completion_driven_terminal_sample(self) -> None:
        baseline = {
            "schema_version": 1,
            "sequence": 0,
            "timestamp_s": 0.0,
            "final_sample": False,
            "executions_submitted": 0,
            "executions_completed": 0,
            "cfg_sites": 0,
            "memory_features": 0,
            "thread_activity_features": 0,
            "feedback_features_total": 0,
            "cfg_sites_delta": 0,
            "memory_features_delta": 0,
            "thread_activity_features_delta": 0,
            "feedback_features_total_delta": 0,
        }
        final = {
            **baseline,
            "sequence": 1,
            "timestamp_s": 1.0,
            "final_sample": True,
            "executions_submitted": 10,
            "executions_completed": 10,
            "cfg_sites": 4,
            "memory_features": 7,
            "thread_activity_features": 2,
            "feedback_features_total": 13,
            "cfg_sites_delta": 4,
            "memory_features_delta": 7,
            "thread_activity_features_delta": 2,
            "feedback_features_total_delta": 13,
        }
        with tempfile.TemporaryDirectory() as td:
            coverage_log = Path(td) / "coverage.jsonl"
            coverage_log.write_text(
                "\n".join(json.dumps(row) for row in (baseline, final)) + "\n",
                encoding="utf-8",
            )
            summary = evaluate_coverage_log(coverage_log)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["executions_completed"], 10)
            self.assertEqual(summary["cfg_sites"], 4)
            self.assertEqual(summary["memory_features"], 7)

            coverage_log.write_text(json.dumps(baseline) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final sample"):
                evaluate_coverage_log(coverage_log)

            zero_execution_final = {**baseline, "sequence": 1, "final_sample": True}
            coverage_log.write_text(
                "\n".join(
                    json.dumps(row) for row in (baseline, zero_execution_final)
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "zero executions"):
                evaluate_coverage_log(coverage_log)

    def test_coverage_cell_skips_vconfig_on_when_phase2_disabled_it(self) -> None:
        record = {
            "filesystem_project": "phantom-fhe",
            "kernel_id": "ntt__12345678",
            "manifest": "/tmp/manifest.json",
            "vconfig_enabled": False,
            "vconfig_disabled_reason": "vconfig_barrier_unsupported",
            "backend_results": {
                "origin": {"status": "passed"},
            },
        }
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner.subprocess, "run") as run,
        ):
            result = runner.run_coverage_cell(
                record,
                backend="origin",
                vconfig="on",
                campaign_dir=Path(td),
                fuzzer=Path("/tmp/fuzzer"),
                fuzzer_async=Path("/tmp/fuzzer_async"),
                coverage_seconds=10,
                timeout_seconds=30,
                window_size=None,
                seed=7,
            )

        self.assertEqual(result["status"], "skipped")
        self.assertTrue(result["passed"])
        self.assertEqual(result["skip_kind"], "applicability")
        self.assertEqual(
            result["failure_reason"], "vconfig_barrier_unsupported"
        )
        run.assert_not_called()

    def test_coverage_cell_runs_with_fixed_seed_and_validates_terminal_log(
        self,
    ) -> None:
        record = {
            "filesystem_project": "demo",
            "kernel_id": "kernel__12345678",
            "manifest": "/tmp/manifest.json",
            "vconfig_enabled": True,
            "vconfig_disabled_reason": None,
            "backend_results": {
                "rapid2": {"status": "passed"},
            },
        }
        summary = {
            "status": "passed",
            "samples": 3,
            "executions_submitted": 20,
            "executions_completed": 20,
            "cfg_sites": 9,
            "memory_features": 11,
            "thread_activity_features": 4,
            "feedback_features_total": 24,
            "final_timestamp_s": 2.0,
        }
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(
                runner.subprocess,
                "run",
                return_value=mock.Mock(
                    returncode=0, stdout="coverage complete\n", stderr=""
                ),
            ) as run,
            mock.patch.object(
                runner, "evaluate_coverage_log", return_value=summary
            ) as evaluate,
        ):
            campaign = Path(td)
            result = runner.run_coverage_cell(
                record,
                backend="rapid2",
                vconfig="off",
                campaign_dir=campaign,
                fuzzer=Path("/tmp/fuzzer"),
                fuzzer_async=Path("/tmp/fuzzer_async"),
                coverage_seconds=2,
                timeout_seconds=30,
                window_size=32,
                seed=19,
                gpu_device="3",
            )

        workdir = (
            campaign
            / "demo/kernels/kernel__12345678/coverage/rapid2/off"
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["cfg_sites"], 9)
        self.assertEqual(result["seed"], 19)
        self.assertEqual(result["coverage_log"], str(workdir / "coverage.jsonl"))
        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]).name, "fuzzer_async")
        self.assertEqual(command[command.index("--vconfig") + 1], "off")
        self.assertEqual(
            run.call_args.kwargs["env"]["RAPID_FIXED_SEED"], "19"
        )
        self.assertEqual(
            run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3"
        )
        self.assertEqual(run.call_args.kwargs["cwd"], workdir)
        evaluate.assert_called_once_with(workdir / "coverage.jsonl")

    def test_parse_fuzzer_stats_requires_final_statistics_and_arg_pack_stats(self) -> None:
        stats = parse_fuzzer_stats(
            "\n".join(
                [
                    "Final statistics:",
                    "  Corpus size: 3",
                    "  Crashes found: 0",
                    "  Total executions: 100",
                    "Arg-pack stats: normalize_calls=101, normalize_repack_count=100, invalid_repair_count=0, mutation_calls=0, seed_generation_count=1, payload_clamp_count=4",
                ]
            )
        )

        self.assertEqual(stats["total_executions"], 100)
        self.assertEqual(stats["mutation_calls"], 0)
        self.assertEqual(stats["payload_clamp_count"], 4)

        with self.assertRaisesRegex(ValueError, "Final statistics"):
            parse_fuzzer_stats("Arg-pack stats: normalize_calls=1")

    def test_build_fuzzer_command_separates_fixed_and_mutation_modes(self) -> None:
        fixed = build_fuzzer_command(
            fuzzer=Path("/repo/cuda-fuzzer/target/release/fuzzer_async"),
            backend=Path("/tmp/librapid2_target.so"),
            manifest=Path("/tmp/manifest.json"),
            mode="fixed",
            runs=100,
        )
        mutation = build_fuzzer_command(
            fuzzer=Path("/repo/cuda-fuzzer/target/release/fuzzer_async"),
            backend=Path("/tmp/librapid2_target.so"),
            manifest=Path("/tmp/manifest.json"),
            mode="mutation",
            runs=1000,
        )

        self.assertIn("--no-mutate", fixed)
        self.assertEqual(fixed[-2:], ["--runs", "100"])
        self.assertNotIn("--no-mutate", mutation)
        self.assertEqual(mutation[-2:], ["--runs", "1000"])

    def test_build_fuzzer_command_absolutizes_paths_for_isolated_workdirs(self) -> None:
        command = build_fuzzer_command(
            fuzzer=Path("cuda-fuzzer/target/release/fuzzer_async"),
            backend=Path("build/backend/librapid2_target.so"),
            manifest=Path("build/kernel/manifest.json"),
            mode="fixed",
            runs=100,
        )

        self.assertTrue(Path(command[0]).is_absolute())
        self.assertTrue(Path(command[1]).is_absolute())
        self.assertTrue(Path(command[3]).is_absolute())

    def test_iter_phase2_built_uses_authoritative_kernel_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_json(
                run_dir / "index.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "kernel_id": "kernel_a__11111111",
                            "symbol_name": "_Z1av",
                            "dir": "kernels/kernel_a__11111111",
                        }
                    ],
                },
            )
            _write_json(
                run_dir / "rewrite_summary.json",
                {
                    "results": [
                        {
                            "kernel_id": "kernel_a__11111111",
                            "status": "built",
                            "phase2_dir": str(run_dir / "kernels/kernel_a__11111111/phase2"),
                        }
                    ]
                },
            )

            built = list(iter_phase2_built(run_dir))

            self.assertEqual(len(built), 1)
            self.assertEqual(built[0]["kernel_id"], "kernel_a__11111111")
            self.assertEqual(built[0]["symbol_name"], "_Z1av")

    def test_iter_phase2_built_rejects_rewrite_summary_without_results_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_json(
                run_dir / "index.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "kernel_id": "kernel_a__11111111",
                            "symbol_name": "_Z1av",
                            "dir": "kernels/kernel_a__11111111",
                        }
                    ],
                },
            )
            _write_json(
                run_dir / "rewrite_summary.json",
                {
                    "kernels": [
                        {
                            "kernel_id": "kernel_a__11111111",
                            "phase2_status": "built",
                        }
                    ]
                },
            )

            with self.assertRaisesRegex(ValueError, "rewrite summary results must be a list"):
                list(iter_phase2_built(run_dir))

    def test_backend_command_uses_kernel_id_backend_dir(self) -> None:
        command = build_backend_command(
            phase2_dir=Path("/tmp/run/kernels/kernel_a__11111111/phase2"),
            out_dir=Path("/tmp/campaign/gpurir/kernels/kernel_a__11111111/backend"),
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
        )

        self.assertIn("cuda-kernel/rapid2/build.py", command[1])
        self.assertIn("--phase2-dir", command)
        self.assertIn("/tmp/run/kernels/kernel_a__11111111/phase2", command)
        self.assertIn("--out-dir", command)
        self.assertIn("/tmp/campaign/gpurir/kernels/kernel_a__11111111/backend", command)

    def test_backend_export_gate_rejects_loader_symbol_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shared_lib = Path(td) / "librapid2_target.so"
            shared_lib.write_text("", encoding="utf-8")
            nm_output = "\n".join(
                [
                    "T libafl_submit_with_id",
                    "T libafl_poll_results",
                    "T libafl_get_queue_counts",
                    "T libafl_set_target_timeout_ms",
                    "T libafl_wait_for_completion",
                    "T libafl_wait",
                    "T libafl_stop",
                    "B libafl_cov_map",
                ]
            )

            with mock.patch.object(
                runner.subprocess,
                "run",
                return_value=mock.Mock(returncode=0, stdout=nm_output),
            ):
                ok, missing, _ = verify_backend_exports(shared_lib)

        self.assertFalse(ok)
        self.assertEqual(missing, ["libafl_release_tasks"])

    def test_backend_export_gate_uses_selected_loader_contract(self) -> None:
        exports = {
            "origin": (
                "libafl_cov_map",
                "libafl_simt_memcov_bits",
                "libafl_target",
                "libafl_get_last_run_status",
            ),
            "rapid": BACKEND_SPECS["rapid"].required_exports,
            "rapid2": BACKEND_SPECS["rapid2"].required_exports,
        }
        with tempfile.TemporaryDirectory() as td:
            shared_lib = Path(td) / "libtarget.so"
            shared_lib.write_text("", encoding="utf-8")
            for backend, symbols in exports.items():
                nm_output = "\n".join(f"T {symbol}" for symbol in symbols)
                with (
                    self.subTest(backend=backend),
                    mock.patch.object(
                        runner.subprocess,
                        "run",
                        return_value=mock.Mock(returncode=0, stdout=nm_output),
                    ),
                ):
                    ok, missing, _ = verify_backend_exports(
                        shared_lib, backend=backend
                    )
                    self.assertTrue(ok)
                    self.assertEqual(missing, [])

    def test_backend_export_gate_reports_selected_contract_when_nm_fails(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner.subprocess, "run", side_effect=OSError("nm")),
        ):
            shared_lib = Path(td) / "libtarget.so"
            shared_lib.write_text("", encoding="utf-8")
            ok, missing, detail = verify_backend_exports(
                shared_lib, backend="origin"
            )

        self.assertFalse(ok)
        self.assertEqual(missing, list(BACKEND_SPECS["origin"].required_exports))
        self.assertIn("nm failed", detail)

    def test_run_backend_build_uses_selected_backend_layout(self) -> None:
        record = {
            "filesystem_project": "demo",
            "kernel_id": "kernel__12345678",
            "phase2_status": "built",
            "phase2_dir": "/tmp/phase2",
        }
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(
                runner.subprocess,
                "run",
                return_value=mock.Mock(returncode=0, stdout="built"),
            ),
            mock.patch.object(
                runner,
                "verify_backend_exports",
                return_value=(True, [], "exports"),
            ) as verify,
            mock.patch.object(
                runner, "_record_backend_freshness", return_value=True
            ) as freshness,
        ):
            campaign = Path(td)
            result = run_backend_build(
                record,
                backend="origin",
                campaign_dir=campaign,
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
                timeout_seconds=10,
            )

        expected_dir = campaign / "demo/kernels/kernel__12345678/backends/origin"
        self.assertEqual(result["status"], "passed")
        self.assertEqual(
            result["shared_lib"],
            str(expected_dir / "libphase2_origin_target.so"),
        )
        self.assertIn("cuda-kernel/origin/build.py", result["command"][1])
        verify.assert_called_once_with(
            expected_dir / "libphase2_origin_target.so", backend="origin"
        )
        freshness.assert_called_once_with(
            record,
            backend="origin",
            out_dir=expected_dir,
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
        )

    def test_backend_freshness_inputs_are_backend_specific(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            common = root / "cuda-kernel/backend_build.py"
            origin = root / "cuda-kernel/origin/harness.cpp"
            rapid = root / "cuda-kernel/rapid/harness.cpp"
            rapid2 = root / "cuda-kernel/rapid2/harness.cpp"
            for path in (common, origin, rapid, rapid2):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("// source\n", encoding="utf-8")

            with mock.patch.object(runner, "REPO_ROOT", root):
                origin_inputs = runner._backend_source_inputs(backend="origin")
                rapid_inputs = runner._backend_source_inputs(backend="rapid")
                rapid2_inputs = runner._backend_source_inputs(backend="rapid2")

        self.assertIn(origin, origin_inputs)
        self.assertNotIn(rapid, origin_inputs)
        self.assertNotIn(rapid2, origin_inputs)
        self.assertIn(rapid, rapid_inputs)
        self.assertNotIn(origin, rapid_inputs)
        self.assertIn(rapid2, rapid2_inputs)
        self.assertNotIn(origin, rapid2_inputs)

    def test_backend_matrix_builds_each_backend_without_fanning_out_structural_skips(
        self,
    ) -> None:
        records = [
            {
                "filesystem_project": "demo",
                "kernel_id": "run__11111111",
                "phase2_status": "built",
                "phase2_dir": "/tmp/phase2",
                "support_decision": "run",
            },
            {
                "filesystem_project": "demo",
                "kernel_id": "skip__22222222",
                "phase2_status": "built",
                "support_decision": "skip",
                "skip_reason": "pointer_to_pointer_not_supported",
            },
        ]

        def built(_record: dict, *, backend: str, **_kwargs: object) -> dict:
            return {"status": "passed", "passed": True, "backend": backend}

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "collect_kernel_records", return_value=records),
            mock.patch.object(runner, "backend_stage_if_fresh", return_value=None),
            mock.patch.object(runner, "run_backend_build", side_effect=built) as build,
            mock.patch.object(runner, "save_kernel_results"),
        ):
            result = run_backend_matrix_stage(
                campaign_dir=Path(td),
                backends=("origin", "rapid2"),
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
                timeout_seconds=10,
            )

        self.assertEqual(
            set(result[0]["backend_results"]), {"origin", "rapid2"}
        )
        self.assertEqual(result[0]["backend_results"]["origin"]["status"], "passed")
        self.assertEqual(result[1]["support_decision"], "skip")
        self.assertNotIn("backend_results", result[1])
        self.assertEqual(build.call_count, 2)

    def test_backend_matrix_limits_work_to_selected_kernel(self) -> None:
        records = [
            {
                "filesystem_project": "demo",
                "kernel_id": "keep__11111111",
                "display_name": "keep",
                "phase2_status": "built",
                "support_decision": "run",
            },
            {
                "filesystem_project": "demo",
                "kernel_id": "defer__22222222",
                "display_name": "defer",
                "phase2_status": "built",
                "support_decision": "run",
            },
        ]
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "collect_kernel_records", return_value=records),
            mock.patch.object(runner, "backend_stage_if_fresh", return_value=None),
            mock.patch.object(
                runner,
                "run_backend_build",
                return_value={"status": "passed", "passed": True},
            ) as build,
            mock.patch.object(runner, "save_kernel_results"),
        ):
            result = run_backend_matrix_stage(
                campaign_dir=Path(td),
                backends=("origin",),
                kernel_selectors=("keep",),
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
                timeout_seconds=10,
            )

        self.assertEqual(build.call_count, 1)
        self.assertIn("backend_results", result[0])
        self.assertNotIn("backend_results", result[1])

    def test_coverage_matrix_records_all_cells_with_the_same_seed(self) -> None:
        records = [
            {
                "filesystem_project": "demo",
                "kernel_id": "run__11111111",
                "phase2_status": "built",
                "support_decision": "run",
                "vconfig_enabled": True,
                "backend_results": {
                    name: {"status": "passed"}
                    for name in ("origin", "rapid", "rapid2")
                },
            },
            {
                "filesystem_project": "demo",
                "kernel_id": "skip__22222222",
                "phase2_status": "built",
                "support_decision": "skip",
                "skip_reason": "pointer_to_pointer_not_supported",
            },
            {
                "filesystem_project": "demo",
                "kernel_id": "defer__33333333",
                "display_name": "defer",
                "phase2_status": "built",
                "support_decision": "run",
                "vconfig_enabled": True,
                "backend_results": {
                    name: {"status": "passed"}
                    for name in ("origin", "rapid", "rapid2")
                },
            },
        ]

        def covered(
            _record: dict, *, backend: str, vconfig: str, seed: int, **_kwargs: object
        ) -> dict:
            return {
                "status": "passed",
                "passed": True,
                "backend": backend,
                "vconfig": vconfig,
                "seed": seed,
            }

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "load_kernel_results", return_value=records),
            mock.patch.object(
                runner, "run_coverage_cell", side_effect=covered
            ) as run_cell,
            mock.patch.object(runner, "save_kernel_results"),
        ):
            result = runner.run_coverage_matrix_stage(
                campaign_dir=Path(td),
                runs={},
                backends=("origin", "rapid", "rapid2"),
                kernel_selectors=("run__11111111",),
                vconfigs=("off", "on"),
                coverage_seconds=10,
                timeout_seconds=30,
                seed=41,
                fuzzer=Path("/tmp/fuzzer"),
                fuzzer_async=Path("/tmp/fuzzer_async"),
            )

        self.assertEqual(run_cell.call_count, 6)
        self.assertEqual(
            set(result[0]["coverage_results"]),
            {"origin", "rapid", "rapid2"},
        )
        self.assertEqual(
            set(result[0]["coverage_results"]["rapid2"]), {"off", "on"}
        )
        self.assertEqual(
            {
                call.kwargs["seed"]
                for call in run_cell.call_args_list
            },
            {41},
        )
        self.assertEqual(
            {
                (call.kwargs["backend"], call.kwargs["window_size"])
                for call in run_cell.call_args_list
            },
            {("origin", None), ("rapid", 1), ("rapid2", 32)},
        )
        self.assertNotIn("coverage_results", result[1])
        self.assertNotIn("coverage_results", result[2])

    def test_ensure_fuzzer_async_rebuilds_when_sources_are_newer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "cuda-fuzzer/src/bin/fuzzer_async.rs"
            target = root / "cuda-fuzzer/target/release/fuzzer_async"
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            (root / "cuda-fuzzer/Cargo.toml").write_text("[package]\nname='demo'\n", encoding="utf-8")
            source.write_text("fn main() {}\n", encoding="utf-8")
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            os.utime(root / "cuda-fuzzer/Cargo.toml", (100.0, 100.0))
            os.utime(target, (100.0, 100.0))
            os.utime(source, (200.0, 200.0))

            with (
                mock.patch.object(runner, "REPO_ROOT", root),
                mock.patch.object(runner.subprocess, "run") as cargo,
            ):
                self.assertEqual(ensure_fuzzer_async(release=True), target)

            cargo.assert_called_once_with(
                [
                    "cargo",
                    "build",
                    "--manifest-path",
                    "cuda-fuzzer/Cargo.toml",
                    "--release",
                ],
                cwd=root,
                env=mock.ANY,
                check=True,
            )

    def test_ensure_fuzzer_async_reuses_fresh_binary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "cuda-fuzzer/src/bin/fuzzer_async.rs"
            target = root / "cuda-fuzzer/target/release/fuzzer_async"
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            (root / "cuda-fuzzer/Cargo.toml").write_text("[package]\nname='demo'\n", encoding="utf-8")
            source.write_text("fn main() {}\n", encoding="utf-8")
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            os.utime(root / "cuda-fuzzer/Cargo.toml", (100.0, 100.0))
            os.utime(source, (100.0, 100.0))
            os.utime(target, (200.0, 200.0))

            with (
                mock.patch.object(runner, "REPO_ROOT", root),
                mock.patch.object(runner.subprocess, "run") as cargo,
            ):
                self.assertEqual(ensure_fuzzer_async(release=True), target)

            cargo.assert_not_called()

    def test_build_stage_skips_non_runnable_kernels_and_saves_incrementally(self) -> None:
        records = [
            {
                "filesystem_project": "demo",
                "kernel_id": "skip_kernel__11111111",
                "phase2_status": "built",
                "support_decision": "skip",
                "skip_reason": "vconfig_required",
                "backend": {"status": "not_attempted"},
            },
            {
                "filesystem_project": "demo",
                "kernel_id": "run_kernel__22222222",
                "phase2_status": "built",
                "support_decision": "run",
                "phase2_dir": "/tmp/run/kernels/run_kernel__22222222/phase2",
                "backend": {"status": "not_attempted"},
            },
        ]

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "collect_kernel_records", return_value=records),
            mock.patch.object(runner, "backend_stage_if_fresh", return_value=None),
            mock.patch.object(
                runner,
                "run_backend_build",
                return_value={"status": "passed", "passed": True},
            ) as build,
            mock.patch.object(runner, "save_kernel_results") as save,
        ):
            result = run_build_stage(
                campaign_dir=Path(td),
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
                timeout_seconds=900,
            )

        self.assertEqual(result[0]["backend"]["status"], "skipped")
        self.assertEqual(result[0]["backend"]["failure_reason"], "vconfig_required")
        self.assertEqual(result[1]["backend"]["status"], "passed")
        build.assert_called_once_with(
            records[1],
            campaign_dir=Path(td),
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
            timeout_seconds=900,
        )
        self.assertEqual(save.call_count, 2)

    def test_build_stage_preclassifies_skips_before_incremental_save(self) -> None:
        records = [
            {
                "filesystem_project": "demo",
                "kernel_id": "run_kernel__11111111",
                "phase2_status": "built",
                "support_decision": "run",
                "phase2_dir": "/tmp/run/kernels/run_kernel__11111111/phase2",
                "backend": {"status": "not_attempted"},
            },
            {
                "filesystem_project": "demo",
                "kernel_id": "skip_kernel__22222222",
                "phase2_status": "built",
                "support_decision": "skip",
                "skip_reason": "vconfig_required",
                "backend": {"status": "not_attempted"},
            },
        ]
        snapshots: list[list[dict]] = []

        def capture_save(_campaign_dir: Path, saved_records: list[dict]) -> None:
            snapshots.append(json.loads(json.dumps(saved_records)))

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(runner, "collect_kernel_records", return_value=records),
            mock.patch.object(runner, "backend_stage_if_fresh", return_value=None),
            mock.patch.object(
                runner,
                "run_backend_build",
                return_value={"status": "passed", "passed": True},
            ),
            mock.patch.object(runner, "save_kernel_results", side_effect=capture_save),
        ):
            run_build_stage(
                campaign_dir=Path(td),
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
                timeout_seconds=900,
            )

        self.assertGreaterEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0][1]["backend"]["status"], "skipped")
        self.assertEqual(snapshots[0][1]["backend"]["failure_reason"], "vconfig_required")

    def test_backend_freshness_rejects_backend_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            campaign = root / "campaign"
            run_dir = root / "run"
            kernel_dir = run_dir / "kernels/run_kernel__11111111"
            phase2_dir = kernel_dir / "phase2"
            out_dir = campaign / "demo/kernels/run_kernel__11111111/backend"
            phase2_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)

            rewrite_summary = run_dir / "rewrite_summary.json"
            manifest = kernel_dir / "manifest.json"
            build_spec = phase2_dir / "build_spec.json"
            device_bc = phase2_dir / "kernel.device.bc"
            invoke_header = phase2_dir / "gen/fuzzer_invoke.v1.cuh"
            target_layout_header = phase2_dir / "gen/rapid_target_layout.v1.h"
            runtime_source = root / "cuda-kernel/rapid2/multi_pipeline_manager.cuh"
            vconfig_source = root / "tools/rapid-vconfig-instrument/rapid_vconfig_instrument.cpp"
            vconfig_build = root / "tools/rapid-vconfig-instrument/build.py"
            vconfig_pass_source = root / "tools/rapid-vconfig-instrument/virtual_dim_pass.cpp"
            vconfig_pass_header = root / "tools/rapid-vconfig-instrument/virtual_dim_pass.h"
            vconfig_plugin_source = root / "tools/rapid-vconfig-instrument/virtual_dim_plugin.cpp"
            virtual_dim_source = root / "rapid-llvm/llvm/lib/Transforms/Scalar/VirtualDim.cpp"
            virtual_dim_plugin = root / "rapid-llvm/llvm/lib/Transforms/Scalar/VirtualDimPlugin.cpp"
            virtual_dim_header = root / "rapid-llvm/llvm/include/llvm/Transforms/Scalar/VirtualDim.h"
            for path, content in (
                (rewrite_summary, "{}\n"),
                (
                    build_spec,
                    '{"vconfig_enabled": true, "vconfig_warp_aligned": true}\n',
                ),
                (device_bc, "bitcode"),
                (invoke_header, "// invoke\n"),
                (target_layout_header, "// layout\n"),
                (runtime_source, "// runtime v1\n"),
                (root / "cuda-kernel/backend_build.py", "# builder\n"),
                (root / "cuda-kernel/launch_config.py", "# launch\n"),
                (root / "scripts/kernel-rewrite/common.py", "# common\n"),
                (
                    root / "scripts/kernel-rewrite/contracts/kernel.py",
                    "# contract\n",
                ),
                (root / "cuda-kernel/utils/status/run_status.h", "// status\n"),
                (root / "tools/rapid-feedback-instrument/build.py", "# instrumenter\n"),
                (vconfig_source, "// vconfig tool v1\n"),
                (vconfig_build, "# vconfig builder v1\n"),
                (vconfig_pass_source, "// vconfig pass v1\n"),
                (vconfig_pass_header, "// vconfig pass header v1\n"),
                (vconfig_plugin_source, "// vconfig plugin v1\n"),
                (virtual_dim_source, "// rapid-llvm virtual dim v1\n"),
                (virtual_dim_plugin, "// rapid-llvm virtual dim plugin v1\n"),
                (virtual_dim_header, "// rapid-llvm virtual dim header v1\n"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            _write_json(
                manifest,
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "symbol_name": "_Z3runv",
                            "display_name": "run_kernel",
                            "launch_policy": {
                                "grid": [1, 1, 1],
                                "block_candidates": [128, 64, 32],
                                "physical_block_max": 128,
                                "target_dynamic_shared_bytes": 512,
                                "coverage_memory": "global",
                                "vconfig_reserved": True,
                            },
                        }
                    ],
                },
            )
            record = {
                "filesystem_project": "demo",
                "kernel_id": "run_kernel__11111111",
                "phase2_dir": str(phase2_dir),
                "manifest": str(manifest),
            }
            build_record = {
                "schema_version": 1,
                "phase2_dir": str(phase2_dir),
                "device_bc": str(device_bc),
                "invoke_header": str(invoke_header),
                "target_layout_header": str(target_layout_header),
                "type_shim_include_dirs": [],
                "launch_config": {
                    "grid": [1, 1, 1],
                    "block_candidates": [128, 64, 32],
                    "physical_block_max": 128,
                    "logical_grid": [1, 1, 1],
                    "logical_block": [1, 1, 1],
                    "has_logical_vconfig_bounds": False,
                    "target_dynamic_shared_bytes": 512,
                    "coverage_memory": "global",
                    "vconfig_reserved": True,
                    "vconfig_enabled": True,
                    "vconfig_warp_aligned": True,
                    "vconfig_mutation": True,
                },
            }
            (out_dir / "librapid2_target.so").write_text("", encoding="utf-8")

            with (
                mock.patch.object(runner, "REPO_ROOT", root),
                mock.patch.object(runner, "_backend_tool_identities", return_value={}),
            ):
                _write_json(out_dir / "backend_build.json", build_record)
                self.assertTrue(
                    runner._record_backend_freshness(
                        record,
                        out_dir=out_dir,
                        cuda_path="/usr/local/cuda",
                        cuda_arch="sm_86",
                    )
                )
                build_record = json.loads(
                    (out_dir / "backend_build.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    build_record["campaign_freshness"]["schema_version"], 1
                )
                self.assertEqual(
                    build_record["campaign_freshness"]["backend"], "rapid2"
                )
                expected_freshness = runner._backend_freshness_record(
                    record,
                    build_record,
                    cuda_path="/usr/local/cuda",
                    cuda_arch="sm_86",
                )
                self.assertEqual(
                    build_record["campaign_freshness"]["sha256"],
                    expected_freshness["sha256"],
                )
                with mock.patch.object(
                    runner,
                    "verify_backend_exports",
                    return_value=(True, [], ""),
                ):
                    self.assertIsNotNone(
                        backend_stage_if_fresh(
                            record,
                            campaign_dir=campaign,
                            cuda_path="/usr/local/cuda",
                            cuda_arch="sm_86",
                        )
                    )
                    for changed_path, changed_content in (
                        (runtime_source, "// runtime v2\n"),
                        (vconfig_source, "// vconfig tool v2\n"),
                        (vconfig_pass_source, "// vconfig pass v2\n"),
                        (device_bc, "bitcode v2"),
                    ):
                        original_content = changed_path.read_text(encoding="utf-8")
                        changed_path.write_text(changed_content, encoding="utf-8")
                        with self.subTest(changed_path=changed_path):
                            self.assertIsNone(
                                backend_stage_if_fresh(
                                    record,
                                    campaign_dir=campaign,
                                    cuda_path="/usr/local/cuda",
                                    cuda_arch="sm_86",
                                )
                            )
                        changed_path.write_text(original_content, encoding="utf-8")
                        self.assertIsNotNone(
                            backend_stage_if_fresh(
                                record,
                                campaign_dir=campaign,
                                cuda_path="/usr/local/cuda",
                                cuda_arch="sm_86",
                            )
                        )

                    original_content = virtual_dim_source.read_text(encoding="utf-8")
                    virtual_dim_source.write_text(
                        "// rapid-llvm virtual dim unrelated v2\n",
                        encoding="utf-8",
                    )
                    self.assertIsNotNone(
                        backend_stage_if_fresh(
                            record,
                            campaign_dir=campaign,
                            cuda_path="/usr/local/cuda",
                            cuda_arch="sm_86",
                        )
                    )
                    virtual_dim_source.write_text(original_content, encoding="utf-8")

    def test_project_profiles_use_expected_resolved_runs(self) -> None:
        self.assertEqual(
            RUNS["gpurir"].resolved_run,
            Path("build/e2e/third-party-fuzz/20260719-final/gpurir/phase1/resolved/gpurir"),
        )
        self.assertEqual(RUNS["phantom_fhe"].filesystem_project, "phantom-fhe")
        self.assertEqual(RUNS["tensorrt_clip"].filesystem_project, "tensorrt_clip")
        self.assertEqual(RUNS["cutlass_basic"].support_registry, Path("third_party/fuzz/support/cutlass_basic.json"))

    def test_project_profiles_can_be_rebased_to_clean_campaign_dir(self) -> None:
        campaign = Path("/tmp/rapid-thirdparty-clean")

        runs = project_runs_for_campaign(campaign)

        self.assertEqual(
            runs["gpurir"].resolved_run,
            campaign / "gpurir/phase1/resolved/gpurir",
        )
        self.assertEqual(
            runs["phantom_fhe"].resolved_run,
            campaign / "phantom-fhe/phase1/resolved/phantom_fhe",
        )
        self.assertEqual(
            runs["cudasift"].resolved_run,
            campaign / "cudasift/phase1/resolved/cudasift",
        )
        self.assertEqual(
            runs["lietorch"].resolved_run,
            campaign / "lietorch/phase1/resolved/lietorch",
        )
        self.assertEqual(
            runs["tensorrt_clip"].resolved_run,
            campaign / "tensorrt_clip/phase1/resolved/existing_capture",
        )
        self.assertEqual(
            runs["cutlass_basic"].resolved_run,
            campaign / "cutlass_basic/phase1/resolved/cutlass_basic",
        )
        self.assertEqual(runs["phantom_fhe"].support_registry, RUNS["phantom_fhe"].support_registry)

    def test_collect_kernel_records_preserves_phase2_failures_and_support_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "final/demo/phase1/resolved/demo"
            _write_json(
                run_dir / "index.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "kernel_id": "run_kernel__11111111",
                            "symbol_name": "_Z3runv",
                            "dir": "kernels/run_kernel__11111111",
                        },
                        {
                            "kernel_id": "skip_kernel__22222222",
                            "symbol_name": "_Z4skipv",
                            "display_name": "skip_kernel",
                            "dir": "kernels/skip_kernel__22222222",
                        },
                        {
                            "kernel_id": "failed_kernel__33333333",
                            "symbol_name": "_Z6failedv",
                            "display_name": "failed_kernel",
                            "dir": "kernels/failed_kernel__33333333",
                        },
                    ],
                },
            )
            _write_json(
                run_dir / "kernels/run_kernel__11111111/manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "symbol_name": "_Z3runv",
                            "display_name": "run_kernel",
                        }
                    ],
                },
            )
            _write_json(
                run_dir / "rewrite_summary.json",
                {
                    "results": [
                        {
                            "kernel_id": "run_kernel__11111111",
                            "status": "built",
                            "phase2_dir": str(
                                run_dir / "kernels/run_kernel__11111111/phase2"
                            ),
                        },
                        {
                            "kernel_id": "skip_kernel__22222222",
                            "status": "built",
                            "phase2_dir": str(
                                run_dir / "kernels/skip_kernel__22222222/phase2"
                            ),
                        },
                        {
                            "kernel_id": "failed_kernel__33333333",
                            "status": "failed",
                            "failure_reason": "protected_data_field",
                            "failure_detail": "field data_ is protected",
                        },
                    ]
                },
            )
            _write_json(
                run_dir / "kernels/run_kernel__11111111/phase2/build_spec.json",
                {
                    "vconfig_requested": True,
                    "vconfig_enabled": False,
                    "vconfig_warp_aligned": True,
                    "vconfig_disabled_reason": "vconfig_barrier_unsupported",
                },
            )
            _write_json(
                run_dir / "kernels/skip_kernel__22222222/phase2/build_spec.json",
                {
                    "vconfig_requested": True,
                    "vconfig_enabled": True,
                    "vconfig_warp_aligned": True,
                    "vconfig_disabled_reason": None,
                },
            )
            support_path = root / "support/demo.json"
            _write_json(
                support_path,
                {
                    "schema_version": 1,
                    "project": "demo",
                    "kernels": [
                        {
                            "kernel_id": "run_kernel__11111111",
                            "symbol_name": "_Z3runv",
                            "decision": "run",
                            "evidence": [_evidence()],
                        },
                        {
                            "kernel_id": "skip_kernel__22222222",
                            "symbol_name": "_Z4skipv",
                            "decision": "skip",
                            "reason_code": "vconfig_required",
                            "detail": "uses threadIdx.y",
                            "evidence": [_evidence()],
                        },
                    ],
                },
            )

            records = collect_kernel_records(
                runs={
                    "demo": RUNS["gpurir"].__class__(
                        project_id="demo",
                        filesystem_project="demo",
                        resolved_run=run_dir,
                        support_registry=support_path,
                    )
                }
            )

            by_id = {record["kernel_id"]: record for record in records}
            self.assertEqual(by_id["run_kernel__11111111"]["display_name"], "run_kernel")
            self.assertEqual(by_id["run_kernel__11111111"]["support_decision"], "run")
            self.assertTrue(by_id["run_kernel__11111111"]["vconfig_requested"])
            self.assertFalse(by_id["run_kernel__11111111"]["vconfig_enabled"])
            self.assertEqual(
                by_id["run_kernel__11111111"]["vconfig_disabled_reason"],
                "vconfig_barrier_unsupported",
            )
            self.assertEqual(by_id["skip_kernel__22222222"]["skip_reason"], "vconfig_required")
            self.assertEqual(by_id["failed_kernel__33333333"]["phase2_status"], "failed")
            self.assertEqual(by_id["failed_kernel__33333333"]["skip_reason"], "protected_data_field")

    def test_collect_kernel_records_rejects_failed_rewrite_record_without_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "final/demo/phase1/resolved/demo"
            _write_json(
                run_dir / "index.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "kernel_id": "failed_kernel__33333333",
                            "symbol_name": "_Z6failedv",
                            "display_name": "failed_kernel",
                            "dir": "kernels/failed_kernel__33333333",
                        },
                    ],
                },
            )
            _write_json(
                run_dir / "rewrite_summary.json",
                {
                    "results": [
                        {
                            "kernel_id": "failed_kernel__33333333",
                            "status": "failed",
                        },
                    ]
                },
            )
            support_path = root / "support/demo.json"
            _write_json(
                support_path,
                {
                    "schema_version": 1,
                    "project": "demo",
                    "kernels": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "rewrite failed entry missing failure_reason"):
                collect_kernel_records(
                    runs={
                        "demo": RUNS["gpurir"].__class__(
                            project_id="demo",
                            filesystem_project="demo",
                            resolved_run=run_dir,
                            support_registry=support_path,
                        )
                    }
                )

    def test_collect_kernel_records_rejects_stale_constraint_registry_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "final/demo/phase1/resolved/demo"
            registry_path = root / "kernel_constraints/demo.json"
            _write_json(registry_path, {"schema_version": 1, "project": "demo", "overrides": []})
            _write_json(
                run_dir / "constraint_overrides.report.json",
                {
                    "schema_version": 1,
                    "project": "demo",
                    "registry": str(registry_path),
                    "registry_sha256": "sha256:stale",
                    "counts": {"applied": 0, "unmatched": 0},
                    "applied": [],
                },
            )

            with self.assertRaisesRegex(ValueError, "constraint overrides.*stale"):
                collect_kernel_records(
                    runs={
                        "demo": RUNS["gpurir"].__class__(
                            project_id="demo",
                            filesystem_project="demo",
                            resolved_run=run_dir,
                            support_registry=root / "support/demo.json",
                            constraint_registry=registry_path,
                        )
                    }
                )

    def test_evaluate_fuzzer_output_enforces_fixed_and_mutation_gates(self) -> None:
        fixed = evaluate_fixed_fuzz_output(
            "\n".join(
                [
                    "Final statistics:",
                    "  Corpus size: 1",
                    "  Crashes found: 0",
                    "  Total executions: 100",
                    "Arg-pack stats: normalize_calls=101, normalize_repack_count=100, invalid_repair_count=0, mutation_calls=0, seed_generation_count=1, payload_clamp_count=0",
                ]
            ),
            expected_runs=100,
        )
        self.assertTrue(fixed["passed"])

        bad_fixed = evaluate_fixed_fuzz_output(
            "\n".join(
                [
                    "Final statistics:",
                    "  Corpus size: 1",
                    "  Crashes found: 0",
                    "  Total executions: 99",
                    "Arg-pack stats: normalize_calls=99, mutation_calls=0, seed_generation_count=1",
                ]
            ),
            expected_runs=100,
        )
        self.assertFalse(bad_fixed["passed"])
        self.assertIn("executions", bad_fixed["failure_reason"])

        mutation = evaluate_mutation_fuzz_output(
            "\n".join(
                [
                    "Final statistics:",
                    "  Corpus size: 3",
                    "  Crashes found: 0",
                    "  Total executions: 77",
                    "Arg-pack stats: normalize_calls=78, mutation_calls=12, seed_generation_count=1",
                ]
            )
        )
        self.assertTrue(mutation["passed"])

        no_mutation = evaluate_mutation_fuzz_output(
            "\n".join(
                [
                    "Final statistics:",
                    "  Corpus size: 3",
                    "  Crashes found: 0",
                    "  Total executions: 77",
                    "Arg-pack stats: normalize_calls=78, mutation_calls=0, seed_generation_count=1",
                ]
            )
        )
        self.assertFalse(no_mutation["passed"])
        self.assertIn("mutation", no_mutation["failure_reason"])

    def test_mutation_stage_timeout_is_observation_bound_not_failure(self) -> None:
        self.assertTrue(
            mutation_stage_passes(
                {"passed": True, "failure_reason": None},
                client_joined=True,
                timed_out=True,
                client_returncode=-2,
            )["passed"]
        )
        self.assertFalse(
            mutation_stage_passes(
                {"passed": False, "failure_reason": "mutation calls did not increase"},
                client_joined=True,
                timed_out=True,
                client_returncode=-2,
            )["passed"]
        )

    def test_mutation_non_timeout_client_failure_is_not_masked_by_progress(self) -> None:
        decision = mutation_stage_passes(
            {"passed": True, "failure_reason": None},
            client_joined=True,
            timed_out=False,
            client_returncode=-6,
        )

        self.assertFalse(decision["passed"])
        self.assertIn("client exited with -6", decision["failure_reason"])

    def test_mutation_timeout_without_final_stats_uses_progress_evidence(self) -> None:
        output = "\n".join(
            [
                "[Client Heartbeat #1]  (GLOBAL) run time: 1s, clients: 1, corpus: 0, objectives: 0, executions: 0",
                "[Testcase    #1]  (GLOBAL) run time: 1s, clients: 1, corpus: 2, objectives: 0, executions: 1, submitted: 10, evaluated: 1",
                "[UserStats   #1]  (GLOBAL) run time: 1s, clients: 1, corpus: 2, objectives: 0, executions: 4, submitted: 11, evaluated: 4",
            ]
        )

        evaluation = evaluate_mutation_fuzz_output(output)
        decision = mutation_stage_passes(
            evaluation,
            client_joined=True,
            timed_out=True,
            client_returncode=-2,
        )

        self.assertTrue(decision["passed"])
        self.assertEqual(evaluation["stats"]["observed_executions"], 4)
        self.assertEqual(evaluation["stats"]["observed_evaluated"], 4)

    def test_mutation_timeout_with_cuda_error_remains_failure(self) -> None:
        evaluation = evaluate_mutation_fuzz_output(
            "\n".join(
                [
                    "[UserStats   #1] executions: 4, submitted: 11, evaluated: 4",
                    "CUDA Error:",
                    "    Error code: 700",
                    "    Error text: an illegal memory access was encountered",
                ]
            )
        )

        decision = mutation_stage_passes(
            evaluation,
            client_joined=True,
            timed_out=True,
            client_returncode=-6,
        )

        self.assertFalse(decision["passed"])
        self.assertIn("CUDA", decision["failure_reason"])

    def test_timeout_output_text_decodes_bytes_and_strings(self) -> None:
        self.assertEqual(_timeout_output_text(b"out", "err"), "outerr")
        self.assertEqual(_timeout_output_text(None, b"err"), "err")

    def test_classify_fuzz_failure_uses_log_evidence(self) -> None:
        self.assertEqual(
            classify_fuzz_failure(
                "missing Final statistics block; fuzzer exited with -6",
                "cuLaunchKernel(rapid2_persistent_kernel) failed: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            )["kind"],
            "launch_out_of_resources",
        )
        self.assertEqual(
            classify_fuzz_failure(
                "fuzzer exited with -6",
                "Error code: 700\nError text: an illegal memory access was encountered",
            )["kind"],
            "cuda_illegal_memory_access",
        )
        self.assertEqual(
            classify_fuzz_failure(
                "fixed fuzz timed out after 45s",
                "CUDA Error:\n    Error code: 716\n    Error text: misaligned address",
            )["kind"],
            "cuda_misaligned_address",
        )
        self.assertEqual(
            classify_fuzz_failure("fixed fuzz timed out after 30s", "")["kind"],
            "fixed_timeout",
        )
        self.assertEqual(
            classify_fuzz_failure(
                "no executions completed",
                "[RAPID2] cudaGetDeviceCount failed: no CUDA-capable device is detected",
            )["kind"],
            "cuda_device_unavailable",
        )

    def test_enrich_runtime_classifications_marks_fixed_failures_runtime_skips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "fixed.log"
            log.write_text("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES\n", encoding="utf-8")
            records = [
                {
                    "support_decision": "run",
                    "fixed": {
                        "status": "failed",
                        "failure_reason": "missing Final statistics block; fuzzer exited with -6",
                        "log": str(log),
                    },
                    "mutation": {"status": "not_attempted"},
                },
                {
                    "support_decision": "run",
                    "fixed": {"status": "passed"},
                    "mutation": {"status": "not_attempted"},
                },
            ]

            enrich_runtime_classifications(records)

            self.assertEqual(records[0]["runtime_skip_reason"], "launch_out_of_resources")
            self.assertEqual(records[0]["mutation"]["status"], "skipped")
            self.assertEqual(records[1]["runtime_decision"], "run")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _evidence() -> dict:
    return {
        "kind": "kernel_body",
        "file": "third_party/demo.cu",
        "line": 10,
        "note": "test evidence",
    }


if __name__ == "__main__":
    unittest.main()
