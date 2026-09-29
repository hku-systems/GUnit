import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from .feedback_runtime import discover_feedback_artifact, generate_default_seed
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
TIMEOUT_FIXTURE = REPO_ROOT / "tests/feedback_e2e/fixtures/timeout_kernel.cu"

LIBAFL_RUN_STATUS_TIMEOUT = 2
LIBAFL_BACKEND_STAGE_EXECUTE = 4


def _probe_timeout(library: Path, seed: bytes, timeout_ms: int) -> dict[str, object]:
    worker = textwrap.dedent(
        """
        import ctypes
        import json
        import os
        import sys
        import time

        from tests.feedback_e2e.feedback_runtime import (
            EDGES_MAP_SIZE,
            SIMT_MEMCOV_STORAGE_SIZE,
            _TaskResult,
        )

        target = ctypes.CDLL(sys.argv[1])
        timeout_ms = int(sys.argv[3])

        set_timeout = target.libafl_set_target_timeout_ms
        set_timeout.argtypes = [ctypes.c_uint64]
        set_timeout.restype = None
        set_timeout(timeout_ms)

        submit = target.libafl_submit_with_id
        submit.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        submit.restype = ctypes.c_uint64
        poll = target.libafl_poll_results
        poll.argtypes = [ctypes.POINTER(_TaskResult), ctypes.c_size_t]
        poll.restype = ctypes.c_size_t
        wait_for_completion = getattr(target, "libafl_wait_for_completion", None)
        if wait_for_completion is not None:
            wait_for_completion.argtypes = []
            wait_for_completion.restype = None
        release = target.libafl_release_tasks
        release.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        release.restype = ctypes.c_size_t

        seed = bytes.fromhex(sys.argv[2])
        seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)
        task_id = submit(seed_buffer, len(seed))
        if task_id == 0:
            raise RuntimeError("CUDA backend rejected timeout probe submission")

        result_buffer = (_TaskResult * 1)()
        if wait_for_completion is not None:
            wait_for_completion()
            if poll(result_buffer, 1) != 1:
                raise RuntimeError("completion wait returned before timeout result was pollable")
        else:
            deadline = time.monotonic() + 5.0
            result_count = poll(result_buffer, 1)
            while result_count != 1 and time.monotonic() < deadline:
                time.sleep(0.001)
                result_count = poll(result_buffer, 1)
            if result_count != 1:
                raise TimeoutError("CUDA backend did not publish a timeout completion")
        result = result_buffer[0]
        task_ids = (ctypes.c_uint64 * 1)(result.task_id)
        release(task_ids, 1)
        print(
            json.dumps(
                {
                    "task_id": result.task_id,
                    "status": {
                        "code": result.status.code,
                        "stage": result.status.stage,
                        "detail": result.status.detail,
                    },
                    "edge_size": result.edge_size,
                    "simt_memcov_size": result.simt_memcov_size,
                    "edge_ptr_is_null": not bool(result.edge_ptr),
                    "simt_memcov_ptr_is_null": not bool(result.simt_memcov_ptr),
                    "exec_time_ns": result.exec_time_ns,
                }
            ),
            flush=True,
        )
        os._exit(0)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", worker, str(library), seed.hex(), str(timeout_ms)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=REPO_ROOT,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def _probe_timeout_configuration_gate(library: Path, seed: bytes) -> dict[str, int]:
    worker = textwrap.dedent(
        """
        import ctypes
        import json
        import os
        import sys

        target = ctypes.CDLL(sys.argv[1])
        seed = bytes.fromhex(sys.argv[2])
        seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)

        submit = target.libafl_submit_with_id
        submit.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        submit.restype = ctypes.c_uint64
        set_timeout = target.libafl_set_target_timeout_ms
        set_timeout.argtypes = [ctypes.c_uint64]
        set_timeout.restype = None

        unconfigured_task_id = submit(seed_buffer, len(seed))
        set_timeout(100)
        configured_task_id = submit(seed_buffer, len(seed))
        print(
            json.dumps(
                {
                    "unconfigured_task_id": unconfigured_task_id,
                    "configured_task_id": configured_task_id,
                }
            ),
            flush=True,
        )
        os._exit(0)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", worker, str(library), seed.hex()],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=REPO_ROOT,
    )
    return json.loads(completed.stdout.splitlines()[-1])


@unittest.skipUnless(GPU_TESTS_ENABLED, "set RAPID_RUN_GPU_TESTS=1 to run CUDA e2e")
class Rapid2TimeoutFeedbackE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(REPO_ROOT / "cuda-fuzzer/Cargo.toml"),
                "--bin",
                "fuzzer",
            ],
            check=True,
            cwd=REPO_ROOT,
            env=_cargo_env(),
        )

        cls._temporary_root = tempfile.TemporaryDirectory(prefix="rapid-timeout-feedback-")
        out_root = Path(cls._temporary_root.name) / "pipeline"
        subprocess.run(
            [
                str(REPO_ROOT / ".venv/bin/python"),
                str(REPO_ROOT / "cuda-kernel/builtin_phase_pipeline.py"),
                "--out-root",
                str(out_root),
                "--run-id",
                "timeout-feedback",
                "--source",
                str(TIMEOUT_FIXTURE),
                "--cuda-arch",
                os.environ.get("KSMOKE_CUDA_ARCH", "sm_86"),
                "--cuda-path",
                os.environ.get("CUDA_PATH", "/usr/local/cuda"),
                "--build-profile",
                "release",
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        cls.artifact = discover_feedback_artifact(out_root / "out/timeout-feedback")
        cls.seed = generate_default_seed(
            REPO_ROOT / "cuda-fuzzer/target/release/fuzzer",
            cls.artifact.manifest,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_root.cleanup()

    def test_watchdog_publishes_timeout_completion_for_feedback(self) -> None:
        timeout_ms = 100

        result = _probe_timeout(
            self.artifact.rapid2_library,
            self.seed,
            timeout_ms,
        )

        self.assertEqual(result["status"]["code"], LIBAFL_RUN_STATUS_TIMEOUT)
        self.assertEqual(result["status"]["stage"], LIBAFL_BACKEND_STAGE_EXECUTE)
        self.assertGreaterEqual(result["status"]["detail"], timeout_ms)
        self.assertEqual(result["edge_size"], 65_536)
        self.assertEqual(result["simt_memcov_size"], 8_192)
        self.assertFalse(result["edge_ptr_is_null"])
        self.assertFalse(result["simt_memcov_ptr_is_null"])
        self.assertGreaterEqual(result["exec_time_ns"], timeout_ms * 1_000_000)

    def test_rapid_ordered_publishes_timeout_completion_for_feedback(self) -> None:
        timeout_ms = 100

        result = _probe_timeout(
            self.artifact.rapid_library,
            self.seed,
            timeout_ms,
        )

        self.assertEqual(result["status"]["code"], LIBAFL_RUN_STATUS_TIMEOUT)
        self.assertEqual(result["status"]["stage"], LIBAFL_BACKEND_STAGE_EXECUTE)
        self.assertGreaterEqual(result["status"]["detail"], timeout_ms)
        self.assertEqual(result["edge_size"], 65_536)
        self.assertEqual(result["simt_memcov_size"], 8_192)
        self.assertFalse(result["edge_ptr_is_null"])
        self.assertFalse(result["simt_memcov_ptr_is_null"])
        self.assertGreaterEqual(result["exec_time_ns"], timeout_ms * 1_000_000)

    def test_rapid_timeout_feedback_restarts_with_a_fresh_backend(self) -> None:
        work_dir = Path(self._temporary_root.name) / "rapid-timeout-restart"
        work_dir.mkdir()
        fuzzer = REPO_ROOT / "cuda-fuzzer/target/release/fuzzer"
        command = [
            str(fuzzer),
            str(self.artifact.rapid_library),
            "--manifest",
            str(self.artifact.manifest),
            "--window-size",
            "4",
            "--no-mutate",
        ]
        env = _cargo_env()
        env.setdefault("RUST_LOG", "info")
        env.setdefault("RAPID_FIXED_SEED", "1")
        broker_log = work_dir / "broker.log"
        client_log = work_dir / "client.log"

        with broker_log.open("wb") as broker_output, client_log.open("wb") as client_output:
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
                self.assertTrue(
                    _wait_for(
                        lambda: _tcp_port_is_listening(1337)
                        or broker.poll() is not None,
                        timeout=15,
                    ),
                    _read_log(broker_log),
                )
                self.assertIsNone(broker.poll(), _read_log(broker_log))
                client = subprocess.Popen(
                    command,
                    cwd=work_dir,
                    env=env,
                    stdout=client_output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                restarted = _wait_for(
                    lambda: (
                        "RAPID BACKEND WATCHDOG TIMEOUT" in _read_log(client_log)
                        and "RAPID NON-RECOVERABLE COMPLETION" in _read_log(client_log)
                        and _read_log(client_log).count(
                            "RAPID FUZZER CLIENT LOADED ORDERED CUDA BACKEND"
                        )
                        >= 2
                        and _has_crash_artifact(work_dir)
                    )
                    or client.poll() is not None,
                    timeout=25,
                    interval=0.2,
                )
                log_text = _read_log(client_log)
                self.assertTrue(restarted, log_text)
                self.assertGreaterEqual(
                    log_text.count("RAPID FUZZER CLIENT LOADED ORDERED CUDA BACKEND"),
                    2,
                    log_text,
                )
                self.assertTrue(_has_crash_artifact(work_dir), log_text)
                self.assertIsNone(client.poll(), log_text)
            finally:
                if client is not None:
                    _terminate_process_group(client)
                _terminate_process_group(broker)

    def test_submit_requires_explicit_positive_timeout_configuration(self) -> None:
        result = _probe_timeout_configuration_gate(
            self.artifact.rapid2_library,
            self.seed,
        )

        self.assertEqual(result["unconfigured_task_id"], 0)
        self.assertGreater(result["configured_task_id"], 0)


if __name__ == "__main__":
    unittest.main()
