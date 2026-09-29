import importlib.util
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from .feedback_runtime import (
    discover_feedback_artifact,
    generate_default_seed,
    run_feedback_worker,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_TESTS_ENABLED = os.environ.get("RAPID_RUN_GPU_TESTS") == "1"
FIXTURE_DIR = REPO_ROOT / "tests/feedback_e2e/fixtures"
BLOCKDIM_FIXTURE = FIXTURE_DIR / "vconfig_blockdim_kernel.cu"
SYNC_LOOP_FIXTURE = FIXTURE_DIR / "vconfig_sync_loop_kernel.cu"
WARP_FIXTURE = FIXTURE_DIR / "vconfig_warp_kernel.cu"


def _load_builtin_pipeline_module():
    path = REPO_ROOT / "cuda-kernel/builtin_phase_pipeline.py"
    spec = importlib.util.spec_from_file_location("rapid_builtin_phase_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_block_x(seed: bytes, block_x: int) -> bytes:
    return struct.pack("<6I", 1, 1, 1, block_x, 1, 1) + seed[24:]


def _inject_launch_policy(
    run_dir: Path,
    *,
    block_candidates: list[int],
    physical_block_max: int,
    logical_block: list[int] | None = None,
    vconfig_reserved: bool = True,
) -> None:
    manifest_path = next((run_dir / "kernels").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernel = manifest["kernels"][0]
    kernel["launch_policy"] = {
        "grid": [1, 1, 1],
        "block_candidates": block_candidates,
        "physical_block_max": physical_block_max,
        "coverage_memory": "global",
        "vconfig_reserved": vconfig_reserved,
    }
    if logical_block is not None:
        kernel["launch_policy"]["logical_block"] = logical_block
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _single_kernel_dir(run_dir: Path) -> Path:
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    kernels = index.get("kernels", [])
    if len(kernels) != 1:
        raise AssertionError(f"expected one kernel artifact, got {len(kernels)}")
    kernel_dir = Path(kernels[0]["dir"])
    if not kernel_dir.is_absolute():
        kernel_dir = run_dir / kernel_dir
    return kernel_dir


def _build_vconfig_run(
    out_root: Path,
    *,
    fixture: Path,
    run_id: str,
    block_candidates: list[int] | None = None,
    physical_block_max: int = 64,
    logical_block: list[int] | None = None,
    vconfig_reserved: bool = True,
) -> Path:
    pipeline = _load_builtin_pipeline_module()
    capture_dir = out_root / f"capture-{run_id}"
    run_dir = out_root / "out" / run_id
    object_path = out_root / "build" / f"{fixture.stem}.o"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    if block_candidates is None:
        block_candidates = [64, 32]

    env = os.environ.copy()
    env["RAPID_CAPTURE_DIR"] = str(capture_dir)
    cuda_arch = os.environ.get("KSMOKE_CUDA_ARCH", "sm_86")
    cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    python = pipeline._pick_python()

    pipeline._run(
        pipeline._capture_compile_command(
            wrapped_compiler=pipeline._pick_wrapped_cuda_compiler(),
            source_path=fixture.resolve(),
            object_path=object_path,
            cuda_path=cuda_path,
            cuda_arch=cuda_arch,
        ),
        env=env,
    )
    pipeline._run(
        [
            python,
            str(pipeline.KERNEL_SMOKE_CLI),
            "run",
            "--capture-dir",
            str(capture_dir),
            "--out-root",
            str(out_root / "out"),
            "--run-id",
            run_id,
            "--target-lib",
            run_id.replace("-", "_"),
            "--mode",
            "artifact",
            "--jobs",
            "1",
        ],
        env=env,
    )
    _inject_launch_policy(
        run_dir,
        block_candidates=block_candidates,
        physical_block_max=physical_block_max,
        logical_block=logical_block,
        vconfig_reserved=vconfig_reserved,
    )
    pipeline._run([python, str(pipeline.KERNEL_REWRITE_CLI), "--run-dir", str(run_dir)], env=env)

    kernel_dir = _single_kernel_dir(run_dir)
    backends_dir = kernel_dir / "phase2/backends"
    backends_dir.mkdir(parents=True, exist_ok=True)
    pipeline._build_backend(
        python=python,
        cli=pipeline.ORIGIN_BUILD_CLI,
        phase2_dir=kernel_dir / "phase2",
        out_dir=backends_dir / "origin",
        cuda_arch=cuda_arch,
        cuda_path=cuda_path,
        env=env,
        build_profile="release",
    )
    pipeline._build_backend(
        python=python,
        cli=pipeline.RAPID_BUILD_CLI,
        phase2_dir=kernel_dir / "phase2",
        out_dir=backends_dir / "rapid",
        cuda_arch=cuda_arch,
        cuda_path=cuda_path,
        env=env,
        build_profile="release",
    )
    pipeline._build_backend(
        python=python,
        cli=pipeline.RAPID2_BUILD_CLI,
        phase2_dir=kernel_dir / "phase2",
        out_dir=backends_dir / "rapid2",
        cuda_arch=cuda_arch,
        cuda_path=cuda_path,
        env=env,
        build_profile="release",
    )
    return run_dir


def _read_build_spec(artifact) -> dict:
    return json.loads(
        (artifact.kernel_dir / "phase2/build_spec.json").read_text(encoding="utf-8")
    )


def _edge_hash(result: dict) -> str:
    return result["result"]["edge_sha256"]


def _assert_success(testcase: unittest.TestCase, result: dict) -> None:
    testcase.assertEqual(result["status"]["code"], 0, result["status"])


def _run_single_feedback(backend: str, library: Path, seed: bytes) -> dict:
    return run_feedback_worker(backend, library, [seed], timeout=45.0)


@unittest.skipUnless(GPU_TESTS_ENABLED, "set RAPID_RUN_GPU_TESTS=1 to run CUDA e2e")
class VConfigFeedbackE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(REPO_ROOT / "cuda-fuzzer/Cargo.toml"),
                "--bin",
                "fuzzer",
            ],
            check=True,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "RUSTFLAGS": "-A function-casts-as-integer -A unstable-name-collisions",
            },
        )
        cls._temporary_root = tempfile.TemporaryDirectory(prefix="rapid-vconfig-e2e-")
        cls.run_dir = _build_vconfig_run(
            Path(cls._temporary_root.name),
            fixture=BLOCKDIM_FIXTURE,
            run_id="vconfig-blockdim-e2e",
        )
        cls.artifact = discover_feedback_artifact(cls.run_dir)
        default_seed = generate_default_seed(
            REPO_ROOT / "cuda-fuzzer/target/debug/fuzzer",
            cls.artifact.manifest,
        )
        cls.seed32 = _patch_block_x(default_seed, 32)
        cls.seed64 = _patch_block_x(default_seed, 64)

        cls.origin_disabled_run_dir = _build_vconfig_run(
            Path(cls._temporary_root.name),
            fixture=BLOCKDIM_FIXTURE,
            run_id="vconfig-origin-disabled-e2e",
            vconfig_reserved=False,
        )
        cls.origin_disabled_artifact = discover_feedback_artifact(
            cls.origin_disabled_run_dir
        )
        origin_disabled_seed = generate_default_seed(
            REPO_ROOT / "cuda-fuzzer/target/debug/fuzzer",
            cls.origin_disabled_artifact.manifest,
        )
        cls.origin_disabled_seed32 = _patch_block_x(origin_disabled_seed, 32)
        cls.origin_disabled_seed64 = _patch_block_x(origin_disabled_seed, 64)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_root.cleanup()

    def _compare_backend(self, backend: str, library: Path) -> None:
        result32 = _run_single_feedback(backend, library, self.seed32)
        result64 = _run_single_feedback(backend, library, self.seed64)

        _assert_success(self, result32)
        _assert_success(self, result64)
        self.assertNotEqual(
            _edge_hash(result32),
            _edge_hash(result64),
        )
        self.assertGreaterEqual(result32["result"]["cfg_sites"], 2)
        self.assertGreaterEqual(result64["result"]["cfg_sites"], 2)

    def test_origin_uses_patched_vconfig_blockdim(self) -> None:
        self._compare_backend("origin", self.artifact.origin_library)

    def test_origin_disabled_vconfig_uses_physical_relaunch(self) -> None:
        build_spec = _read_build_spec(self.origin_disabled_artifact)
        self.assertIs(build_spec.get("vconfig_requested"), False)
        self.assertIs(build_spec.get("vconfig_enabled"), False)
        self.assertNotIn("vconfig_disabled_reason", build_spec)

        result32 = _run_single_feedback(
            "origin",
            self.origin_disabled_artifact.origin_library,
            self.origin_disabled_seed32,
        )
        result64 = _run_single_feedback(
            "origin",
            self.origin_disabled_artifact.origin_library,
            self.origin_disabled_seed64,
        )
        _assert_success(self, result32)
        _assert_success(self, result64)
        self.assertNotEqual(_edge_hash(result32), _edge_hash(result64))

    def test_rapid_uses_patched_vconfig_blockdim(self) -> None:
        self._compare_backend("rapid", self.artifact.rapid_library)

    def test_rapid2_uses_patched_vconfig_blockdim(self) -> None:
        result = run_feedback_worker(
            "rapid2",
            self.artifact.rapid2_library,
            [self.seed32, self.seed64],
            timeout=45.0,
        )

        first, second = result["results"]
        _assert_success(self, first)
        _assert_success(self, second)
        self.assertNotEqual(
            _edge_hash(first),
            _edge_hash(second),
        )
        self.assertGreaterEqual(first["result"]["cfg_sites"], 2)
        self.assertGreaterEqual(second["result"]["cfg_sites"], 2)


@unittest.skipUnless(GPU_TESTS_ENABLED, "set RAPID_RUN_GPU_TESTS=1 to run CUDA e2e")
class VConfigSyncWarpE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [
                "cargo",
                "build",
                "--manifest-path",
                str(REPO_ROOT / "cuda-fuzzer/Cargo.toml"),
                "--bin",
                "fuzzer",
            ],
            check=True,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "RUSTFLAGS": "-A function-casts-as-integer -A unstable-name-collisions",
            },
        )
        cls._temporary_root = tempfile.TemporaryDirectory(prefix="rapid-vconfig-sync-warp-e2e-")
        out_root = Path(cls._temporary_root.name)
        cls.sync_run_dir = _build_vconfig_run(
            out_root,
            fixture=SYNC_LOOP_FIXTURE,
            run_id="vconfig-sync-loop-e2e",
        )
        cls.sync_artifact = discover_feedback_artifact(cls.sync_run_dir)
        sync_default_seed = generate_default_seed(
            REPO_ROOT / "cuda-fuzzer/target/debug/fuzzer",
            cls.sync_artifact.manifest,
        )
        cls.sync_seed32 = _patch_block_x(sync_default_seed, 32)
        cls.sync_seed64 = _patch_block_x(sync_default_seed, 64)

        cls.warp_run_dir = _build_vconfig_run(
            out_root,
            fixture=WARP_FIXTURE,
            run_id="vconfig-warp-e2e",
            logical_block=[16, 4, 1],
        )
        cls.warp_artifact = discover_feedback_artifact(cls.warp_run_dir)
        warp_default_seed = generate_default_seed(
            REPO_ROOT / "cuda-fuzzer/target/debug/fuzzer",
            cls.warp_artifact.manifest,
        )
        cls.warp_seed = warp_default_seed

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_root.cleanup()

    def test_sync_loop_vconfig_stays_enabled(self) -> None:
        build_spec = _read_build_spec(self.sync_artifact)

        self.assertIs(build_spec.get("vconfig_enabled"), True)
        self.assertIs(build_spec.get("vconfig_warp_aligned"), False)
        self.assertIsNone(build_spec.get("vconfig_disabled_reason"))

    def _compare_sync_loop_backend(self, backend: str, library: Path) -> None:
        result32 = _run_single_feedback(backend, library, self.sync_seed32)
        result64 = _run_single_feedback(backend, library, self.sync_seed64)

        _assert_success(self, result32)
        _assert_success(self, result64)
        self.assertNotEqual(_edge_hash(result32), _edge_hash(result64))
        self.assertGreaterEqual(result32["result"]["cfg_sites"], 2)
        self.assertGreaterEqual(result64["result"]["cfg_sites"], 2)

    def test_origin_replays_sync_loop_without_inactive_deadlock(self) -> None:
        self._compare_sync_loop_backend("origin", self.sync_artifact.origin_library)

    def test_rapid_replays_sync_loop_without_inactive_deadlock(self) -> None:
        self._compare_sync_loop_backend("rapid", self.sync_artifact.rapid_library)

    def test_rapid2_replays_sync_loop_without_inactive_deadlock(self) -> None:
        result = run_feedback_worker(
            "rapid2",
            self.sync_artifact.rapid2_library,
            [self.sync_seed32, self.sync_seed64],
            timeout=45.0,
        )

        first, second = result["results"]
        _assert_success(self, first)
        _assert_success(self, second)
        self.assertNotEqual(_edge_hash(first), _edge_hash(second))
        self.assertGreaterEqual(first["result"]["cfg_sites"], 2)
        self.assertGreaterEqual(second["result"]["cfg_sites"], 2)

    def test_warp_collective_vconfig_stays_enabled_and_marked_aligned(self) -> None:
        build_spec = _read_build_spec(self.warp_artifact)

        self.assertIs(build_spec.get("vconfig_enabled"), True)
        self.assertIs(build_spec.get("vconfig_warp_aligned"), True)
        self.assertIsNone(build_spec.get("vconfig_disabled_reason"))

    def _compare_warp_backend(self, backend: str, library: Path) -> None:
        result = _run_single_feedback(backend, library, self.warp_seed)

        _assert_success(self, result)
        self.assertGreaterEqual(result["result"]["cfg_sites"], 2)

    def test_origin_preserves_multidimensional_warp_layout(self) -> None:
        self._compare_warp_backend("origin", self.warp_artifact.origin_library)

    def test_rapid_preserves_multidimensional_warp_layout(self) -> None:
        self._compare_warp_backend("rapid", self.warp_artifact.rapid_library)

    def test_rapid2_preserves_multidimensional_warp_layout(self) -> None:
        result = run_feedback_worker(
            "rapid2",
            self.warp_artifact.rapid2_library,
            [self.warp_seed],
            timeout=45.0,
        )

        (only,) = result["results"]
        _assert_success(self, only)
        self.assertGreaterEqual(only["result"]["cfg_sites"], 2)


if __name__ == "__main__":
    unittest.main()
