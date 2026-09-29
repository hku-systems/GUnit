from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .feedback_runtime import discover_feedback_artifact, generate_default_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_TESTS_ENABLED = os.environ.get("RAPID_RUN_GPU_TESTS") == "1"
CANARY_FIXTURE = REPO_ROOT / "tests/feedback_e2e/fixtures/canary_corrupt_kernel.cu"

RAPID_TASK_ENVELOPE_HEADER_BYTES = 32
POINTER_LENGTH_OFFSET = 16
POINTER_DATA_OFFSET = 24
LIBAFL_RUN_STATUS_OK = 0
LIBAFL_RUN_STATUS_CUDA_ERROR = 1
LIBAFL_BACKEND_STAGE_COLLECT = 5
CANARY_STATUS_DETAIL = 0x43414E59


def _patch_canary_args(seed: bytes, *, nbytes: int) -> bytes:
    payload = bytearray(seed[RAPID_TASK_ENVELOPE_HEADER_BYTES:])
    pointer_size = int.from_bytes(
        payload[POINTER_LENGTH_OFFSET:POINTER_DATA_OFFSET], "little"
    )
    if POINTER_DATA_OFFSET + pointer_size != len(payload):
        raise ValueError("canary fixture pointer payload is not the final arg-pack slot")

    payload[0:8] = pointer_size.to_bytes(8, "little")
    payload[8:12] = nbytes.to_bytes(4, "little", signed=True)
    return seed[:RAPID_TASK_ENVELOPE_HEADER_BYTES] + payload


def _run_backend(backend: str, library: Path, seed: bytes) -> tuple[dict, str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.feedback_e2e.feedback_runtime",
            "--worker",
            "--backend",
            backend,
            "--library",
            str(library),
            "--seed-hex",
            seed.hex(),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} canary worker failed: "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line), completed.stderr
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        f"{backend} canary worker produced no JSON: stdout={completed.stdout!r}"
    )


def _status_for(backend: str, result: dict) -> dict:
    if backend == "rapid2":
        return result["results"][0]["status"]
    return result["status"]


@unittest.skipUnless(GPU_TESTS_ENABLED, "set RAPID_RUN_GPU_TESTS=1 to run CUDA e2e")
class CanaryE2ETest(unittest.TestCase):
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
        )

        cls._temporary_root = tempfile.TemporaryDirectory(prefix="rapid-canary-e2e-")
        out_root = Path(cls._temporary_root.name)
        subprocess.run(
            [
                str(REPO_ROOT / ".venv/bin/python"),
                str(REPO_ROOT / "cuda-kernel/builtin_phase_pipeline.py"),
                "--out-root",
                str(out_root),
                "--run-id",
                "canary-e2e",
                "--source",
                str(CANARY_FIXTURE),
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
        cls.artifact = discover_feedback_artifact(out_root / "out/canary-e2e")
        default_seed = generate_default_seed(
            REPO_ROOT / "cuda-fuzzer/target/release/fuzzer",
            cls.artifact.manifest,
        )
        cls.clean_seed = _patch_canary_args(default_seed, nbytes=0)
        cls.corrupt_seed = _patch_canary_args(default_seed, nbytes=1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_root.cleanup()

    def test_origin_and_rapid2_detect_corrupt_canary(self) -> None:
        for backend, library in (
            ("origin", self.artifact.origin_library),
            ("rapid2", self.artifact.rapid2_library),
        ):
            with self.subTest(backend=backend):
                clean, clean_stderr = _run_backend(
                    backend, library, self.clean_seed
                )
                corrupt, corrupt_stderr = _run_backend(
                    backend, library, self.corrupt_seed
                )

                self.assertEqual(
                    _status_for(backend, clean)["code"], LIBAFL_RUN_STATUS_OK
                )
                self.assertNotIn("canary corrupted", clean_stderr)
                self.assertEqual(
                    _status_for(backend, corrupt),
                    {
                        "code": LIBAFL_RUN_STATUS_CUDA_ERROR,
                        "stage": LIBAFL_BACKEND_STAGE_COLLECT,
                        "detail": CANARY_STATUS_DETAIL,
                    },
                )
                self.assertRegex(
                    corrupt_stderr,
                    r"canary corrupted: task_id=[1-9][0-9]* offset=0 "
                    r"expected=0x52 actual=0xff",
                )


if __name__ == "__main__":
    unittest.main()
