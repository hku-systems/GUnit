import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def mutating_result_line(**overrides: int) -> str:
    result = {
        "requested_seconds": 30,
        "executions": 1,
        "corpus_size": 2,
        "solutions": 0,
        "mutation_calls": 1,
        "coverage_nonzero_bytes": 3,
        "simt_memcov_nonzero_bits": 7,
        "pending": 0,
        "completed": 0,
        "outstanding": 0,
        "in_flight": 0,
        "queued_submissions": 0,
    }
    result.update(overrides)
    return "RAPID_MUTATING_RESULT " + json.dumps(result) + "\n"


class RQ4ProfileTests(unittest.TestCase):
    def test_profile_facade_exports_internal_module_contracts(self) -> None:
        from benchmark.rq4 import profile, profile_capture, profile_model

        self.assertIs(profile.profile_identity, profile_model.profile_identity)
        self.assertIs(
            profile.load_profiling_records,
            profile_capture.load_profiling_records,
        )

    def test_unified_profile_schema_feeds_rq4_and_phase_views(self) -> None:
        from benchmark.rq4.profile import load_profiling_records, profiling_summary
        from benchmark.rq4.profile_diagnostic import diagnostic_segments

        records = [
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "main_loop",
                "segment": segment,
                "unit": "ns",
                "count": 1,
                "total": total,
                "max": total,
            }
            for segment, total in (
                ("submit", 10),
                ("poll", 20),
                ("coverage", 30),
                ("evaluate", 40),
                ("release", 50),
                ("scheduler_stage", 60),
                ("other_idle", 70),
            )
        ]
        records.extend(
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "cpu_feedback",
                "segment": segment,
                "unit": "ns",
                "count": count,
                "total": total,
                "max": total,
            }
            for segment, count, total in (
                ("predicate", 3, 11),
                ("metadata", 2, 13),
            )
        )
        records.extend(
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "device_kernel",
                "segment": segment,
                "unit": "cycles",
                "count": 4,
                "total": total,
                "max": None,
            }
            for segment, total in (
                ("idle", 1),
                ("feedback_init", 2),
                ("input_decode", 3),
                ("feedback_prepare", 4),
                ("target_execution", 5),
                ("feedback_merge", 6),
                ("signal", 7),
                ("bookkeeping", 8),
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profiling.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            loaded = load_profiling_records(path)

        summary = profiling_summary(loaded)
        phases = {row["segment"]: row for row in diagnostic_segments(loaded)}
        self.assertEqual(
            summary["cpu_feedback_timing_ns"],
            {"calls": 3, "predicate_ns": 11, "metadata_ns": 13, "total_ns": 24},
        )
        self.assertEqual(summary["device_timing_cycles"]["iterations"], 4)
        self.assertEqual(summary["device_timing_cycles"]["target_execution"], 5)
        self.assertEqual(summary["wall_time_ns"], 304)
        self.assertEqual(phases["evaluate"]["total_ns"], 64)
        self.assertEqual(sum(row["total_ns"] for row in phases.values()), 304)

        invalid = [dict(records[0], schema_version=1)]
        with self.assertRaisesRegex(RuntimeError, "schema_version"):
            profiling_summary(invalid)

    def test_timeout_kills_nsys_process_group_and_preserves_logs(self) -> None:
        from benchmark.rq4 import profile

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child_pid_path = root / "child.pid"
            nsys_stub = root / "nsys"
            nsys_stub.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)'])\n"
                "with open(os.environ['RQ4_TEST_CHILD_PID'], 'w') as stream:\n"
                "    stream.write(str(child.pid))\n"
                "print('partial stdout', flush=True)\n"
                "print('partial stderr', file=sys.stderr, flush=True)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            nsys_stub.chmod(0o755)
            args = SimpleNamespace(
                output=root / "output",
                fuzzer=root / "fuzzer",
                fuzzer_async=root / "fuzzer_async",
                timing_build_root=None,
                workload=None,
                build_root=[root / "build"],
                gpu_device="0",
                window_size=2,
                repetitions=1,
                warmup_runs=1,
                profile_seconds=1,
                mutate=False,
                mutate_seconds=30,
                nsys=nsys_stub,
                timeout=0.2,
                keep_sqlite=False,
            )
            config = SimpleNamespace(
                name="rapid2",
                artifact_backend="rapid2",
                async_frontend=True,
                window_size=2,
            )
            child_pid = None
            try:
                with (
                    patch.object(
                        profile,
                        "_load_reports",
                        return_value=[(root, {"workload_id": "workload"})],
                    ),
                    patch.object(
                        profile.rq1,
                        "_manifest",
                        return_value=root / "manifest.json",
                    ),
                    patch.object(
                        profile.rq1,
                        "_backend_library",
                        return_value=root / "target.so",
                    ),
                    patch.object(
                        profile,
                        "configurations",
                        return_value=(config,),
                    ),
                    patch.object(
                        profile, "benchmark_command", return_value=["fuzzer"]
                    ),
                    patch.dict(
                        os.environ, {"RQ4_TEST_CHILD_PID": str(child_pid_path)}
                    ),
                ):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        profile.run_profiles(args)

                child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline and self._process_is_running(child_pid):
                    time.sleep(0.01)
                self.assertFalse(self._process_is_running(child_pid))

                stem = args.output / "raw/workload-rapid2-r1"
                self.assertEqual(
                    stem.with_suffix(".stdout.log").read_text(encoding="utf-8"),
                    "partial stdout\n",
                )
                self.assertEqual(
                    stem.with_suffix(".stderr.log").read_text(encoding="utf-8"),
                    "partial stderr\n",
                )
            finally:
                if child_pid is None and child_pid_path.exists():
                    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    @staticmethod
    def _process_is_running(pid: int) -> bool:
        try:
            state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
        except FileNotFoundError:
            return False
        return state != "Z"

    def test_rq4_runner_is_syntax_valid_and_parameterizes_build_roots(self) -> None:
        root = Path(__file__).resolve().parents[2]
        profiling = root / "benchmark/rq4/run_profile.sh"
        completed = subprocess.run(
            ["bash", "-n", str(profiling)], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        profile_source = profiling.read_text(encoding="utf-8")
        self.assertIn("-m benchmark.rq4.profile", profile_source)
        self.assertIn("--features profiling", profile_source)
        self.assertIn("rq4_overhead_breakdown.csv", profile_source)
        self.assertIn("rq4_phase_diagnostic.csv", profile_source)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            python_stub = temp / "python"
            python_stub.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$*\" == *\"-m unittest\"* ]]; then exit 0; fi\n"
                "printf 'seed=%s\\n' \"${RAPID_FIXED_SEED:-unset}\"\n"
                "printf '%s\\n' \"$*\"\n"
                "if [[ \"$*\" == *\"-m benchmark.rq4.build\"* ]]; then exit 0; fi\n"
                "exit 42\n",
                encoding="utf-8",
            )
            python_stub.chmod(0o755)
            cargo_stub = temp / "cargo"
            cargo_stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            cargo_stub.chmod(0o755)
            env = os.environ.copy()
            env.update(RQ4_PYTHON=str(python_stub), RQ4_CARGO=str(cargo_stub))
            env.pop("RQ4_BUILD_ROOTS", None)

            missing = subprocess.run(
                ["bash", str(profiling)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("RQ4_BUILD_ROOTS must list", missing.stderr)

            env["RQ4_BUILD_ROOTS"] = "build/standalone build/imported"
            configured = subprocess.run(
                ["bash", str(profiling)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(configured.returncode, 42)
            revision = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            timing_build = f"benchmark/rq4/build/timing-w2-feedback-v2-{revision}"
            pilot = (
                "benchmark/rq4/results/"
                f"diagnostic-profile-pilot-apex-mutating-30s-seed1-r1-w2-feedback-v3-{revision}"
            )
            self.assertIn(
                "-m benchmark.rq4.build --build-root build/standalone "
                f"--build-root build/imported --output {timing_build}",
                configured.stdout,
            )
            self.assertIn(
                "-m benchmark.rq4.profile --build-root build/standalone "
                f"--build-root build/imported --timing-build-root {timing_build}",
                configured.stdout,
            )
            self.assertIn(
                f"--workload apex_maybe_cast --output {pilot} "
                "--mutate --mutate-seconds 30",
                configured.stdout,
            )
            self.assertIn("seed=1", configured.stdout)

            env["RQ4_MODE"] = "fixed"
            fixed = subprocess.run(
                ["bash", str(profiling)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(fixed.returncode, 42)
            self.assertIn(
                "--workload apex_maybe_cast --output "
                "benchmark/rq4/results/"
                f"diagnostic-profile-pilot-apex-5s-r1-w2-feedback-v2-{revision} "
                "--profile-seconds 5",
                fixed.stdout,
            )
            self.assertNotIn("--mutate", fixed.stdout)

    def test_nsys_command_uses_low_overhead_cuda_trace(self) -> None:
        from benchmark.rq4.profile import nsys_profile_command

        command = nsys_profile_command(
            nsys=Path("nsys"),
            output=Path("profile/kernel"),
            application=["fuzzer", "target.so"],
        )
        self.assertEqual(command[:2], ["nsys", "profile"])
        self.assertIn("--trace=cuda,nvtx", command)
        self.assertIn("--sample=none", command)
        self.assertIn("--cpuctxsw=none", command)
        self.assertIn("--capture-range=cudaProfilerApi", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertEqual(command[-2:], ["fuzzer", "target.so"])

    def test_mutating_nsys_command_finalizes_without_killing_client(self) -> None:
        from benchmark.rq4.profile import nsys_profile_command

        command = nsys_profile_command(
            nsys=Path("nsys"),
            output=Path("profile/kernel"),
            application=["fuzzer", "target.so"],
            finalize_on_stop=True,
        )

        self.assertIn("--capture-range-end=stop-shutdown", command)
        self.assertIn("--kill=none", command)
        self.assertNotIn("--trace-fork-before-exec=true", command)
        self.assertNotIn("--capture-range-end=stop", command)

    def test_nsys_export_preserves_interval_trace_as_sqlite(self) -> None:
        from benchmark.rq4.profile import nsys_export_command

        command = nsys_export_command(
            nsys=Path("nsys"),
            report=Path("raw/kernel-r1.nsys-rep"),
            sqlite_path=Path("raw/kernel-r1.sqlite"),
        )

        self.assertEqual(
            command,
            [
                "nsys",
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                "--output",
                "raw/kernel-r1.sqlite",
                "raw/kernel-r1.nsys-rep",
            ],
        )

    def test_profile_resume_identity_includes_repetition(self) -> None:
        from benchmark.rq4.profile import profile_identity, profile_stem

        first = profile_identity("apex_maybe_cast", "rapid2", 1)
        second = profile_identity("apex_maybe_cast", "rapid2", 2)

        self.assertEqual(first, ("apex_maybe_cast", "rapid2", 1))
        self.assertNotEqual(first, second)
        self.assertEqual(
            profile_stem(Path("raw"), *second),
            Path("raw/apex_maybe_cast-rapid2-r2"),
        )

    def test_profile_environment_enables_range_capture_only_in_child(self) -> None:
        from benchmark.rq4.profile import profile_environment

        parent = {"PATH": "/usr/bin"}
        child = profile_environment(
            parent, gpu_device="3", profiling_output=Path("raw/profile.jsonl")
        )

        self.assertEqual(parent, {"PATH": "/usr/bin"})
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(child["RAPID_PROFILE"], "1")
        self.assertEqual(child["RAPID_PROFILE_OUTPUT"], "raw/profile.jsonl")

    def test_profile_cli_defaults_to_one_repetition(self) -> None:
        from benchmark.rq4.profile import _parser

        args = _parser().parse_args(
            ["--build-root", "build", "--output", "results"]
        )

        self.assertEqual(args.repetitions, 1)
        self.assertFalse(args.keep_sqlite)
        self.assertFalse(args.mutate)
        self.assertEqual(args.mutate_seconds, 30)

    def test_mutating_command_uses_a_timed_campaign_budget(self) -> None:
        from benchmark.rq4.profile import mutating_command

        command = mutating_command(
            executable=Path("fuzzer_async"),
            library=Path("librapid2.so"),
            manifest=Path("manifest.json"),
            mutate_seconds=30,
            window_size=2,
        )

        self.assertEqual(
            command,
            [
                "fuzzer_async",
                "librapid2.so",
                "--manifest",
                "manifest.json",
                "--mutate-seconds",
                "30",
                "--window-size",
                "2",
            ],
        )

    def test_mutating_result_uses_time_budget_and_positive_progress(self) -> None:
        from benchmark.rq4.profile import parse_mutating_result

        output = mutating_result_line()

        result = parse_mutating_result(output, expected_seconds=30)

        self.assertEqual(
            result,
            {
                "requested_seconds": 30,
                "executions": 1,
                "corpus_size": 2,
                "solutions": 0,
                "mutation_calls": 1,
                "coverage_nonzero_bytes": 3,
                "simt_memcov_nonzero_bits": 7,
                "pending": 0,
                "completed": 0,
                "outstanding": 0,
                "in_flight": 0,
                "queued_submissions": 0,
            },
        )

    def test_mutating_result_allows_static_corpus_without_feedback(self) -> None:
        from benchmark.rq4.profile import parse_mutating_result

        output = mutating_result_line(
            executions=129072,
            corpus_size=1,
            mutation_calls=121689,
            coverage_nonzero_bytes=0,
            simt_memcov_nonzero_bits=0,
        )

        result = parse_mutating_result(
            output, expected_seconds=30, require_feedback_activity=False
        )

        self.assertEqual(result["corpus_size"], 1)

    def test_mutating_result_allows_static_corpus_with_feedback_activity(self) -> None:
        from benchmark.rq4.profile import parse_mutating_result

        output = mutating_result_line(
            executions=129072,
            corpus_size=1,
            mutation_calls=121689,
            simt_memcov_nonzero_bits=0,
        )

        with self.assertLogs("benchmark.rq4.profile", level="WARNING"):
            result = parse_mutating_result(
                output, expected_seconds=30, require_feedback_activity=True
            )

        self.assertEqual(result["corpus_size"], 1)

    def test_mutating_result_rejects_dead_feedback_channel(self) -> None:
        from benchmark.rq4.profile import parse_mutating_result

        output = mutating_result_line(
            executions=129072,
            corpus_size=1,
            mutation_calls=121689,
            coverage_nonzero_bytes=0,
            simt_memcov_nonzero_bits=0,
        )

        with self.assertRaisesRegex(RuntimeError, "feedback produced no coverage"):
            parse_mutating_result(
                output, expected_seconds=30, require_feedback_activity=True
            )

    def test_mutating_profile_uses_port_readiness_and_releases_between_cells(
        self,
    ) -> None:
        from benchmark.rq4.profile import _run_mutating_profile

        broker = [
            sys.executable,
            "-c",
            (
                "import signal, socket, sys, time; "
                "listener = socket.socket(); "
                "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                "listener.bind(('127.0.0.1', 1337)); "
                "listener.listen(); "
                "print('monitor flood', flush=True); "
                "print('broker error', file=sys.stderr, flush=True); "
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
                "time.sleep(60)"
            ),
        ]
        client = [sys.executable, "-c", "print('client ran')"]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for cell in range(2):
                completed, _pgid = _run_mutating_profile(
                    client,
                    broker_command=broker,
                    cwd=root,
                    env=os.environ.copy(),
                    broker_log=root / f"broker-{cell}.log",
                    timeout=1,
                    budget_seconds=1,
                )
                self.assertEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, "client ran\n")
                self.assertEqual(
                    (root / f"broker-{cell}.log").read_text(encoding="utf-8"),
                    "broker error\n",
                )

    def test_mutating_profile_rejects_a_stale_broker_port(self) -> None:
        from benchmark.rq4.profile import _run_mutating_profile

        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 1337))
        listener.listen()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                with self.assertRaisesRegex(
                    RuntimeError, "port 1337 is already in use.*stale broker"
                ):
                    _run_mutating_profile(
                        [sys.executable, "-c", "raise SystemExit(99)"],
                        broker_command=[
                            sys.executable,
                            "-c",
                            "import time; time.sleep(60)",
                        ],
                        cwd=root,
                        env=os.environ.copy(),
                        broker_log=root / "broker.log",
                        timeout=0.2,
                        budget_seconds=0.2,
                    )
        finally:
            listener.close()

    def test_mutating_port_probe_ignores_closed_connection_time_wait(self) -> None:
        from benchmark.rq4.profile import _mutating_broker_port_is_available

        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 1337))
        listener.listen()
        client = socket.create_connection(("127.0.0.1", 1337))
        accepted, _ = listener.accept()
        accepted.close()
        client.close()
        listener.close()

        self.assertTrue(_mutating_broker_port_is_available())

    def test_resume_loader_keeps_repetitions_independent(self) -> None:
        from benchmark.rq4.profile import load_completed_profiles

        with tempfile.TemporaryDirectory() as temp_dir:
            rows = Path(temp_dir) / "profiles.jsonl"
            rows.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "mode": "fixed",
                            "workload_id": "apex_maybe_cast",
                            "configuration": "rapid2",
                            "repetition": repetition,
                            "window_size": 2,
                        }
                    )
                    for repetition in (1, 2)
                )
                + "\n",
                encoding="utf-8",
            )

            completed = load_completed_profiles(rows, mode="fixed")

        self.assertEqual(
            completed,
            {
                ("apex_maybe_cast", "rapid2", 1),
                ("apex_maybe_cast", "rapid2", 2),
            },
        )

    def test_mutating_resume_rejects_incomplete_result_rows(self) -> None:
        from benchmark.rq4.profile import load_completed_profiles

        stale_result = {
            "requested_seconds": 30,
            "executions": 195325,
            "corpus_size": 333,
            "solutions": 0,
            "mutation_calls": 183330,
            "pending": 0,
            "completed": 0,
            "outstanding": 0,
            "in_flight": 0,
            "queued_submissions": 0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = Path(temp_dir) / "profiles.jsonl"
            rows.write_text(
                json.dumps(
                    {
                        "mode": "mutating",
                        "workload_id": "apex_maybe_cast",
                        "configuration": "libafl-plus",
                        "repetition": 1,
                        "mutating": stale_result,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "incomplete mutating profile.*apex_maybe_cast libafl-plus.*coverage_nonzero_bytes",
            ):
                load_completed_profiles(rows, mode="mutating")

    def test_cuda_api_names_are_grouped_into_paper_categories(self) -> None:
        from benchmark.rq4.trace_extract import cuda_api_category

        expected = {
            "cudaMalloc": "allocation",
            "cudaFree": "free",
            "cudaMemcpyAsync": "memcpy",
            "cuMemcpyDtoH_v2": "memcpy",
            "cudaLaunchKernel": "launch",
            "cuLaunchKernel": "launch",
            "cudaDeviceSynchronize": "synchronization",
            "cuStreamSynchronize": "synchronization",
            "cudaGetDevice": "other",
        }
        for name, category in expected.items():
            with self.subTest(name=name):
                self.assertEqual(cuda_api_category(name), category)

    def test_compact_evidence_produces_audit_totals_without_nsys_stats(self) -> None:
        from benchmark.rq4.profile import summarize_trace_evidence

        evidence = SimpleNamespace(
            cuda_api_time_ns={
                "RAPID2-Coll-0": {"memcpy": 80, "other": 20},
                "RAPID2-Disp-0": {"memcpy": 120, "launch": 10},
            },
            cuda_api_calls={
                "RAPID2-Coll-0": {"memcpy": 2, "other": 1},
                "RAPID2-Disp-0": {"memcpy": 3, "launch": 1},
            },
            kernels=(SimpleNamespace(duration_ns=300),),
            gpu_memcpy_time_ns=150,
        )

        summary = summarize_trace_evidence(evidence)

        self.assertEqual(summary["cuda_api"]["memcpy"], {"time_ns": 200, "calls": 5})
        self.assertEqual(summary["cuda_api"]["launch"], {"time_ns": 10, "calls": 1})
        self.assertEqual(summary["gpu_kernel_time_ns"], 300)
        self.assertEqual(summary["gpu_mem_time_ns"], 150)

    def test_mutating_result_crash_cell_passes_threshold(self) -> None:
        from benchmark.rq4.profile import parse_mutating_result

        output = mutating_result_line(
            executions=11,
            corpus_size=1,
            solutions=5,
            mutation_calls=0,
            coverage_nonzero_bytes=0,
            simt_memcov_nonzero_bits=0,
        )

        result = parse_mutating_result(
            output, expected_seconds=30, require_feedback_activity=False
        )

        self.assertEqual(result["solutions"], 5)
        self.assertEqual(result["mutation_calls"], 0)

    def test_mutating_result_no_crash_no_mutation_fails(self) -> None:
        from benchmark.rq4.profile import parse_mutating_result

        output = mutating_result_line(
            executions=11,
            corpus_size=1,
            mutation_calls=0,
            coverage_nonzero_bytes=0,
            simt_memcov_nonzero_bits=0,
        )

        with self.assertRaisesRegex(RuntimeError, "no mutations"):
            parse_mutating_result(
                output, expected_seconds=30, require_feedback_activity=False
            )

    def test_crash_loop_terminates_process_group_at_cell_timeout(self) -> None:
        """A crash-loop cell is killed when the full cell timeout expires."""
        from benchmark.rq4.profile import CrashLoopTimeout, _run_mutating_profile

        respawner_script = (
            "import os, signal, subprocess, sys, time\n"
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
            "while True:\n"
            "    child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(0.01); raise SystemExit(1)'])\n"
            "    child.wait()\n"
        )
        client = [sys.executable, "-c", respawner_script]
        broker = [
            sys.executable,
            "-c",
            (
                "import signal, socket, sys, time; "
                "listener = socket.socket(); "
                "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                "listener.bind(('127.0.0.1', 1337)); "
                "listener.listen(); "
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
                "time.sleep(60)"
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            t0 = time.monotonic()
            with self.assertRaises(CrashLoopTimeout) as ctx:
                _run_mutating_profile(
                    client,
                    broker_command=broker,
                    cwd=root,
                    env=os.environ.copy(),
                    broker_log=root / "broker.log",
                    timeout=1.5,
                    budget_seconds=0.5,
                )
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 5.0)
            self.assertGreaterEqual(ctx.exception.elapsed, 1.5)

    def test_mutating_profile_allows_finalize_within_cell_timeout(self) -> None:
        from benchmark.rq4.profile import _run_mutating_profile

        broker = [
            sys.executable,
            "-c",
            (
                "import signal, socket, sys, time; "
                "listener = socket.socket(); "
                "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                "listener.bind(('127.0.0.1', 1337)); "
                "listener.listen(); "
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
                "time.sleep(60)"
            ),
        ]
        client = [
            sys.executable,
            "-c",
            "import time; time.sleep(1.2); print('finalized')",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            completed, _pgid = _run_mutating_profile(
                client,
                broker_command=broker,
                cwd=root,
                env=os.environ.copy(),
                broker_log=root / "broker.log",
                timeout=2.5,
                budget_seconds=0.5,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "finalized\n")

    def test_incomplete_crash_loop_row_preserves_partial_artifacts(self) -> None:
        from benchmark.rq4.profile import _capture_incomplete_row

        config = SimpleNamespace(
            name="libafl-plus",
            label="LibAFL+",
            artifact_backend="origin",
            window_size=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            stem = Path(temp_dir) / "shoc_radix_sort-libafl-plus-r1"
            profiling_path = stem.with_suffix(".profiling.jsonl")
            profiling_path.write_text("", encoding="utf-8")

            row = _capture_incomplete_row(
                workload_id="shoc_radix_sort",
                config=config,
                repetition=1,
                stem=stem,
                command=["nsys", "profile", "fuzzer_async"],
                library=Path("/lib/origin.so"),
                manifest=Path("/manifest.json"),
                mutate_seconds=30,
                application=["fuzzer_async", "origin.so"],
                failure={"crash_loop": True, "crash_loop_elapsed": 62.3},
                nsys=Path("nsys"),
                keep_sqlite=False,
                mode="mutating",
                profile_seconds=10,
            )

            self.assertTrue(row["capture_incomplete"])
            self.assertTrue(row["crash_loop"])
            self.assertAlmostEqual(row["crash_loop_elapsed"], 62.3, places=1)
            self.assertIsNone(row["mutating"])
            self.assertNotIn("solutions", row)
            self.assertEqual(row["workload_id"], "shoc_radix_sort")
            self.assertEqual(row["configuration"], "libafl-plus")
            json.dumps(row)

    def test_cell_preflight_rejects_port_holding_residual(self) -> None:
        from benchmark.rq4.profile import CellResidualError, _verify_cell_boundary

        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 1337))
        listener.listen()
        try:
            with self.assertRaises(CellResidualError) as ctx:
                _verify_cell_boundary(gpu_device="0")
            self.assertTrue(ctx.exception.port_busy)
        finally:
            listener.close()

    def test_cell_boundary_degrades_persistent_gpu_lock_to_warning(self) -> None:
        import fcntl
        from unittest.mock import patch

        from benchmark.rq4.profile import (
            _gpu_client_lock_path,
            _verify_cell_boundary,
        )

        path = _gpu_client_lock_path("0")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with patch(
                "benchmark.rq4.profile.LOCK_HOLDER_TOTAL_TIMEOUT", 0.05
            ), self.assertLogs("benchmark.rq4.profile", level="WARNING") as logs:
                result = _verify_cell_boundary(gpu_device="0")
            self.assertEqual(result, "lock_residual")
            self.assertIn("lock_residual", "\n".join(logs.output))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_cell_cleanup_handles_port_holding_child_in_process_group(self) -> None:
        from benchmark.rq4.profile import (
            _find_cell_residual_pids,
            _mutating_broker_port_is_available,
            _teardown_cell,
        )

        script = (
            "import socket, time; "
            "s = socket.socket(); "
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
            "s.bind(('127.0.0.1', 1337)); "
            "s.listen(); "
            "time.sleep(60)"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", script],
            start_new_session=True,
        )
        pgid = holder.pid
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if not _mutating_broker_port_is_available():
                    break
                time.sleep(0.02)
            self.assertFalse(_mutating_broker_port_is_available())
            _teardown_cell(pgid, gpu_device="0")
            self.assertTrue(_mutating_broker_port_is_available())
            self.assertFalse(self._process_is_running(pgid))
            self.assertEqual(_find_cell_residual_pids(pgid), [])
        finally:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            holder.wait()

    def test_cell_cleanup_kills_grandchild_inheriting_lock(self) -> None:
        """Parent acquires lock, forks grandchild that inherits fd, parent exits.
        Cleanup must find the grandchild via /proc/*/fd scan and kill it."""
        from benchmark.rq4.profile import (
            _find_lock_fd_holders,
            _gpu_client_lock_is_available,
            _gpu_client_lock_path,
            _teardown_cell,
        )

        lock_path = _gpu_client_lock_path("0")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        script = (
            "import fcntl, os, sys, time\n"
            f"fd = os.open({str(lock_path)!r}, os.O_CREAT | os.O_RDWR)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "child_pid = os.fork()\n"
            "if child_pid == 0:\n"
            "    time.sleep(120)\n"
            "    os._exit(0)\n"
            "sys.stdout.write(f'{child_pid}\\n')\n"
            "sys.stdout.flush()\n"
            "os._exit(0)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            text=True,
        )
        grandchild_pid = int(parent.stdout.readline().strip())
        parent.stdout.close()
        parent.wait()

        try:
            self.assertFalse(_gpu_client_lock_is_available("0"))
            holders = _find_lock_fd_holders("0")
            self.assertIn(grandchild_pid, holders)

            dummy = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            dummy_pgid = dummy.pid
            try:
                _teardown_cell(dummy_pgid, gpu_device="0")
                self.assertTrue(_gpu_client_lock_is_available("0"))
            finally:
                try:
                    os.killpg(dummy_pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                dummy.wait()
        finally:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                os.waitpid(grandchild_pid, 0)
            except ChildProcessError:
                pass

    def test_cell_teardown_kills_process_tree_of_lock_holder(self) -> None:
        from benchmark.rq4.profile import (
            _gpu_client_lock_is_available,
            _gpu_client_lock_path,
            _teardown_cell,
        )

        lock_path = _gpu_client_lock_path("0")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        script = (
            "import fcntl, os, sys, time\n"
            f"fd = os.open({str(lock_path)!r}, os.O_CREAT | os.O_RDWR)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            "    os.close(fd)\n"
            "    grandchild = os.fork()\n"
            "    if grandchild == 0:\n"
            "        time.sleep(120)\n"
            "        os._exit(0)\n"
            "    sys.stdout.write(f'{grandchild}\\n')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(120)\n"
            "    os._exit(0)\n"
            "sys.stdout.write(f'{child}\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(120)\n"
        )
        holder = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            text=True,
        )
        descendant_pids = [
            int(holder.stdout.readline().strip()),
            int(holder.stdout.readline().strip()),
        ]
        holder.stdout.close()
        try:
            self.assertFalse(_gpu_client_lock_is_available("0"))
            dummy = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            dummy_pgid = dummy.pid
            try:
                _teardown_cell(dummy_pgid, gpu_device="0")
                self.assertTrue(_gpu_client_lock_is_available("0"))
                for pid in descendant_pids:
                    self.assertFalse(self._process_is_running(pid))
            finally:
                try:
                    os.killpg(dummy_pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                dummy.wait()
        finally:
            for pid in (holder.pid, *descendant_pids):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            holder.wait()

    def test_mutating_cell_timeout_formula_boundaries(self) -> None:
        from benchmark.rq4.profile import mutating_cell_timeout

        for seconds, base_timeout, expected in (
            (30, 180, 180.0),
            (40, 60, 180.0),
            (60, 180, 240.0),
        ):
            with self.subTest(seconds=seconds, base_timeout=base_timeout):
                self.assertEqual(
                    mutating_cell_timeout(seconds, base_timeout=base_timeout),
                    expected,
                )

    def test_timeout_cell_produces_capture_incomplete_row(self) -> None:
        """A mutating cell that times out writes a capture_incomplete row
        and does not abort the entire run."""
        from benchmark.rq4 import profile

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = SimpleNamespace(
                output=root / "output",
                fuzzer=root / "fuzzer",
                fuzzer_async=root / "fuzzer_async",
                timing_build_root=None,
                workload=None,
                build_root=[root / "build"],
                gpu_device="0",
                window_size=2,
                repetitions=1,
                warmup_runs=1,
                profile_seconds=1,
                mutate=True,
                mutate_seconds=30,
                nsys=root / "nsys_stub",
                timeout=180,
                keep_sqlite=False,
            )
            config = SimpleNamespace(
                name="gunit",
                label="GUnit",
                artifact_backend="rapid2",
                async_frontend=True,
                window_size=2,
                feedback_enabled=True,
            )

            nsys_stub = root / "nsys_stub"
            nsys_stub.write_text(
                "#!/usr/bin/env python3\n"
                "import time, sys\n"
                "print('partial stdout', flush=True)\n"
                "print('partial stderr', file=sys.stderr, flush=True)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            nsys_stub.chmod(0o755)

            broker_script = root / "broker.py"
            broker_script.write_text(
                "import signal, socket, sys, time\n"
                "listener = socket.socket()\n"
                "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                "listener.bind(('127.0.0.1', 1337))\n"
                "listener.listen()\n"
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )

            with (
                patch.object(
                    profile,
                    "_load_reports",
                    return_value=[(root, {"workload_id": "apex_index"})],
                ),
                patch.object(
                    profile.rq1,
                    "_manifest",
                    return_value=root / "manifest.json",
                ),
                patch.object(
                    profile.rq1,
                    "_backend_library",
                    return_value=root / "target.so",
                ),
                patch.object(
                    profile,
                    "configurations",
                    return_value=(config,),
                ),
                patch.object(
                    profile,
                    "mutating_command",
                    return_value=[
                        sys.executable, "-c",
                        "import signal, socket, sys, time; "
                        "listener = socket.socket(); "
                        "listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                        "listener.bind(('127.0.0.1', 1337)); "
                        "listener.listen(); "
                        "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
                        "time.sleep(60)",
                    ],
                ),
                patch.object(
                    profile,
                    "mutating_cell_timeout",
                    return_value=2.0,
                ),
            ):
                profile.run_profiles(args)

            rows_path = args.output / "profiles.jsonl"
            self.assertTrue(rows_path.exists())
            rows = [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(row["capture_incomplete"])
            self.assertTrue(row["timeout_expired"])
            self.assertEqual(row["workload_id"], "apex_index")
            self.assertEqual(row["configuration"], "gunit")
            self.assertIsNone(row["mutating"])

    def test_export_failure_marks_cell_incomplete_and_continues(self) -> None:
        from benchmark.rq4 import profile

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = SimpleNamespace(
                output=root / "output",
                fuzzer=root / "fuzzer",
                fuzzer_async=root / "fuzzer_async",
                timing_build_root=None,
                workload=None,
                build_root=[root / "build"],
                gpu_device="0",
                window_size=2,
                repetitions=1,
                warmup_runs=1,
                profile_seconds=1,
                mutate=True,
                mutate_seconds=30,
                nsys=root / "nsys_stub",
                timeout=180,
                keep_sqlite=False,
            )
            config = SimpleNamespace(
                name="libafl-plus",
                label="LibAFL+",
                artifact_backend="origin",
                async_frontend=False,
                window_size=None,
                feedback_enabled=True,
            )
            args.nsys.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "counter = pathlib.Path(__file__).with_suffix('.count')\n"
                "attempt = int(counter.read_text()) if counter.exists() else 0\n"
                "counter.write_text(str(attempt + 1))\n"
                "if attempt == 0:\n"
                "    raise SystemExit(7)\n"
                "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "output.touch()\n",
                encoding="utf-8",
            )
            args.nsys.chmod(0o755)

            mutating_output = mutating_result_line(
                executions=11,
                corpus_size=1,
                coverage_nonzero_bytes=1,
                simt_memcov_nonzero_bits=0,
            )

            def run_cell(
                command: list[str], **_kwargs: object
            ) -> tuple[subprocess.CompletedProcess[str], None]:
                stem = Path(command[command.index("--output") + 1])
                stem.with_suffix(".nsys-rep").touch()
                return subprocess.CompletedProcess(command, 0, mutating_output, ""), None

            evidence = SimpleNamespace(
                cuda_api_time_ns={},
                cuda_api_calls={},
                kernels=(),
                gpu_memcpy_time_ns=0,
            )
            profiling = {"device_timing_cycles": {"target_execution": 1}}
            reports = [
                (root, {"workload_id": "shoc_radix_sort"}),
                (root, {"workload_id": "apex_index"}),
            ]

            with (
                patch.object(profile, "_load_reports", return_value=reports),
                patch.object(
                    profile.rq1, "_manifest", return_value=root / "manifest.json"
                ),
                patch.object(
                    profile.rq1,
                    "_backend_library",
                    return_value=root / "target.so",
                ),
                patch.object(profile, "configurations", return_value=(config,)),
                patch.object(profile, "_verify_cell_boundary"),
                patch.object(profile, "_run_mutating_profile", side_effect=run_cell),
                patch.object(profile, "_teardown_cell"),
                patch.object(profile, "write_compact_evidence", return_value=evidence),
                patch.object(profile, "load_profiling_records", return_value=[]),
                patch.object(profile, "profiling_summary", return_value=profiling),
            ):
                profile.run_profiles(args)

            rows = [
                json.loads(line)
                for line in (args.output / "profiles.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["workload_id"], "shoc_radix_sort")
            self.assertTrue(rows[0]["capture_incomplete"])
            self.assertEqual(rows[0]["postprocess_stage"], "nsys_export")
            self.assertEqual(rows[1]["workload_id"], "apex_index")
            self.assertFalse(rows[1].get("capture_incomplete", False))

if __name__ == "__main__":
    unittest.main()
