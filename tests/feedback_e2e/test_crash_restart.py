from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from .feedback_runtime import FeedbackArtifact, discover_feedback_artifact
from .process_utils import (
    _cargo_env,
    _has_crash_artifact,
    _read_log,
    _tcp_port_is_listening,
    _terminate_process_group,
    _wait_for,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_TESTS_ENABLED = os.environ.get("RAPID_RUN_GPU_TESTS") == "1"
OOB_FIXTURE = REPO_ROOT / "tests/feedback_e2e/fixtures/oob_crash_kernel.cu"


@dataclass(frozen=True)
class CrashRestartScenario:
    backend: str
    fuzzer_build_args: tuple[str, ...]
    fuzzer_binary: Path
    pipeline_run_id: str
    pipeline_extra_args: tuple[str, ...]
    library_attribute: str
    broker_uses_tcp: bool
    restart_markers: tuple[str, ...]
    forbidden_marker: str


SCENARIOS = (
    CrashRestartScenario(
        backend="origin",
        fuzzer_build_args=("--bin", "fuzzer"),
        fuzzer_binary=REPO_ROOT / "cuda-fuzzer/target/debug/fuzzer",
        pipeline_run_id="origin-oob-crash-restart",
        pipeline_extra_args=(),
        library_attribute="origin_library",
        broker_uses_tcp=False,
        restart_markers=("ORIGIN NON-RECOVERABLE STATUS",),
        forbidden_marker="cuMemcpyHtoD(envelope) failed",
    ),
    CrashRestartScenario(
        backend="rapid2",
        fuzzer_build_args=("--bin", "fuzzer_async", "--release"),
        fuzzer_binary=REPO_ROOT / "cuda-fuzzer/target/release/fuzzer_async",
        pipeline_run_id="oob-crash-restart",
        pipeline_extra_args=("--build-profile", "release"),
        library_attribute="rapid2_library",
        broker_uses_tcp=True,
        restart_markers=(
            "RAPID2 NON-RECOVERABLE COMPLETION",
            "RAPID2 BACKEND WATCHDOG TIMEOUT",
        ),
        forbidden_marker="Storing state in crashed fuzzer instance did not work",
    ),
)


def _restart_marker_count(log_text: str, scenario: CrashRestartScenario) -> int:
    return sum(log_text.count(marker) for marker in scenario.restart_markers)


@unittest.skipUnless(GPU_TESTS_ENABLED, "set RAPID_RUN_GPU_TESTS=1 to run CUDA e2e")
class CrashRestartE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_root = tempfile.TemporaryDirectory(
            prefix="rapid-oob-crash-restart-"
        )
        root = Path(cls._temporary_root.name)
        cls.artifacts: dict[str, FeedbackArtifact] = {}

        for scenario in SCENARIOS:
            subprocess.run(
                [
                    "cargo",
                    "build",
                    "--manifest-path",
                    str(REPO_ROOT / "cuda-fuzzer/Cargo.toml"),
                    *scenario.fuzzer_build_args,
                ],
                check=True,
                cwd=REPO_ROOT,
                env=_cargo_env(),
            )
            out_root = root / f"pipeline-{scenario.backend}"
            subprocess.run(
                [
                    str(REPO_ROOT / ".venv/bin/python"),
                    str(REPO_ROOT / "cuda-kernel/builtin_phase_pipeline.py"),
                    "--out-root",
                    str(out_root),
                    "--run-id",
                    scenario.pipeline_run_id,
                    "--source",
                    str(OOB_FIXTURE),
                    "--cuda-arch",
                    os.environ.get("KSMOKE_CUDA_ARCH", "sm_86"),
                    "--cuda-path",
                    os.environ.get("CUDA_PATH", "/usr/local/cuda"),
                    *scenario.pipeline_extra_args,
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            cls.artifacts[scenario.backend] = discover_feedback_artifact(
                out_root / "out" / scenario.pipeline_run_id
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_root.cleanup()

    def test_backends_restart_with_a_fresh_context_after_cuda_crash(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(backend=scenario.backend):
                self._run_crash_restart(scenario)

    def _run_crash_restart(self, scenario: CrashRestartScenario) -> None:
        artifact = self.artifacts[scenario.backend]
        library = getattr(artifact, scenario.library_attribute)
        command = [
            str(scenario.fuzzer_binary),
            str(library),
            "--manifest",
            str(artifact.manifest),
        ]
        env = _cargo_env()
        env.setdefault("RUST_LOG", "info")
        env.setdefault("RAPID_FIXED_SEED", "1")

        work_dir = (
            Path(self._temporary_root.name) / f"broker-client-{scenario.backend}"
        )
        work_dir.mkdir(parents=True)
        broker_log = work_dir / "broker.log"
        client_log = work_dir / "client.log"
        with broker_log.open("wb") as broker_output, client_log.open(
            "wb"
        ) as client_output:
            broker = subprocess.Popen(
                command,
                cwd=work_dir,
                env=env,
                stdout=broker_output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            client: subprocess.Popen[bytes] | None = None
            try:
                broker_ready = _wait_for(
                    lambda: self._broker_is_ready(
                        scenario,
                        broker,
                        broker_log,
                    ),
                    timeout=15,
                )
                self.assertTrue(
                    broker_ready,
                    f"{scenario.backend} broker did not become ready:\n"
                    f"{_read_log(broker_log)}",
                )
                self.assertIsNone(
                    broker.poll(),
                    f"{scenario.backend} broker exited before client start:\n"
                    f"{_read_log(broker_log)}",
                )

                client = subprocess.Popen(
                    command,
                    cwd=work_dir,
                    env=env,
                    stdout=client_output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                observed = _wait_for(
                    lambda: (
                        _restart_marker_count(_read_log(client_log), scenario) >= 2
                        and _read_log(client_log).count(
                            "RAPID FUZZER CLIENT LOADED CUDA BACKEND"
                        )
                        >= 2
                        and _has_crash_artifact(work_dir)
                    )
                    or client.poll() is not None,
                    timeout=35,
                    interval=0.2,
                )
                log_text = _read_log(client_log)
                self.assertTrue(
                    observed,
                    f"{scenario.backend} client did not persist and restart after "
                    f"crash:\n{log_text}",
                )
                self.assertGreaterEqual(
                    _restart_marker_count(log_text, scenario),
                    2,
                    log_text,
                )
                self.assertGreaterEqual(
                    log_text.count("RAPID FUZZER CLIENT LOADED CUDA BACKEND"),
                    2,
                    log_text,
                )
                self.assertTrue(_has_crash_artifact(work_dir), log_text)
                self.assertNotIn(scenario.forbidden_marker, log_text)
                self.assertIsNone(
                    client.poll(),
                    f"{scenario.backend} restarter exited instead of supervising a "
                    f"new child:\n{log_text}",
                )
            finally:
                if client is not None:
                    _terminate_process_group(client)
                _terminate_process_group(broker)

    @staticmethod
    def _broker_is_ready(
        scenario: CrashRestartScenario,
        broker: subprocess.Popen[bytes],
        broker_log: Path,
    ) -> bool:
        if broker.poll() is not None:
            return True
        if scenario.broker_uses_tcp:
            return _tcp_port_is_listening(1337)
        return "Doing broker things" in _read_log(broker_log)


if __name__ == "__main__":
    unittest.main()
