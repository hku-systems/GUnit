import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from .process_utils import (
    _cargo_env,
    _has_crash_artifact,
    _read_log,
    _terminate_process_group,
    _wait_for,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class AsyncSubmitFailureE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(REPO_ROOT / "cuda-fuzzer/Cargo.toml"),
                "--bin",
                "fuzzer_async",
            ],
            check=True,
            cwd=REPO_ROOT,
            env=_cargo_env(),
        )

        cls._temporary_root = tempfile.TemporaryDirectory(
            prefix="rapid-async-submit-failure-"
        )
        root = Path(cls._temporary_root.name)
        source = root / "reject_submit.c"
        cls.library = root / "libreject_submit.so"
        cls.manifest = root / "manifest.json"
        cls.work_dir = root / "broker-client"
        cls.work_dir.mkdir()

        source.write_text(
            """
#include <stddef.h>
#include <stdint.h>

uint8_t libafl_cov_map[65536];

uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {
  (void)input;
  (void)size;
  return 0;
}

size_t libafl_poll_results(void *results, size_t max_count) {
  (void)results;
  (void)max_count;
  return 0;
}

size_t libafl_release_tasks(const uint64_t *task_ids, size_t count) {
  (void)task_ids;
  return count;
}

void libafl_get_queue_counts(void *counts) {
  size_t *values = (size_t *)counts;
  values[0] = 0;
  values[1] = 0;
}

void libafl_set_target_timeout_ms(uint64_t timeout_ms) { (void)timeout_ms; }
void libafl_wait(void) {}
void libafl_wait_for_completion(void) {}
void libafl_stop(void) {}
""",
            encoding="utf-8",
        )
        subprocess.run(
            ["cc", "-shared", "-fPIC", str(source), "-o", str(cls.library)],
            check=True,
            cwd=REPO_ROOT,
        )

        cls.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "symbol_name": "reject_submit_kernel",
                            "display_name": "reject_submit_kernel",
                            "args": [
                                {
                                    "index": 0,
                                    "name": "input",
                                    "type": "uint8_t *",
                                    "kind": "pointer",
                                    "pointer_role": "payload_buffer",
                                    "pointee_layout": {
                                        "index": "input.*",
                                        "name": "$pointee",
                                        "type": "uint8_t",
                                        "kind": "scalar",
                                        "size_bytes": 1,
                                        "align_bytes": 1,
                                    },
                                    "size_bytes": 8,
                                    "align_bytes": 8,
                                }
                            ],
                            "constraints": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_root.cleanup()

    def test_rejected_submit_is_persisted_and_restarts_client(self) -> None:
        command = [
            str(REPO_ROOT / "cuda-fuzzer/target/debug/fuzzer_async"),
            str(self.library),
            "--manifest",
            str(self.manifest),
        ]
        env = _cargo_env()
        env["RUST_LOG"] = "info,cuda_fuzzer::gpu_executor=off"
        env.setdefault("RAPID_FIXED_SEED", "1")

        broker_log = self.work_dir / "broker.log"
        client_log = self.work_dir / "client.log"
        with broker_log.open("wb") as broker_output, client_log.open("wb") as client_output:
            broker = subprocess.Popen(
                command,
                cwd=self.work_dir,
                env=env,
                stdout=broker_output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            client: subprocess.Popen[bytes] | None = None
            try:
                broker_ready = _wait_for(
                    lambda: "Doing broker things" in _read_log(broker_log)
                    or broker.poll() is not None,
                    timeout=15,
                )
                self.assertTrue(
                    broker_ready,
                    f"broker did not become ready:\n{_read_log(broker_log)}",
                )
                self.assertIsNone(
                    broker.poll(),
                    f"broker exited before client start:\n{_read_log(broker_log)}",
                )

                client = subprocess.Popen(
                    command,
                    cwd=self.work_dir,
                    env=env,
                    stdout=client_output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                observed = _wait_for(
                    lambda: (
                        _read_log(client_log).count("RAPID2 SUBMIT FAILURE") >= 2
                        and _read_log(client_log).count(
                            "RAPID FUZZER CLIENT LOADED CUDA BACKEND"
                        )
                        >= 2
                        and _has_crash_artifact(self.work_dir)
                    )
                    or client.poll() is not None,
                    timeout=10,
                    interval=0.1,
                )
                client_text = _read_log(client_log)
                combined_log = _read_log(broker_log) + client_text
                self.assertTrue(
                    observed,
                    f"submit rejection was not persisted and restarted:\n{combined_log}",
                )
                self.assertGreaterEqual(client_text.count("RAPID2 SUBMIT FAILURE"), 2)
                self.assertGreaterEqual(
                    client_text.count("RAPID FUZZER CLIENT LOADED CUDA BACKEND"), 2
                )
                self.assertTrue(_has_crash_artifact(self.work_dir), combined_log)
                self.assertRegex(combined_log, r"objectives:\s*[1-9]")
                self.assertIsNone(
                    client.poll(),
                    f"client restarter exited instead of supervising:\n{client_text}",
                )
            finally:
                if client is not None:
                    _terminate_process_group(client)
                _terminate_process_group(broker)


if __name__ == "__main__":
    unittest.main()
