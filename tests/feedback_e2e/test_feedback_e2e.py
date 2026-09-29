import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from .cufuzz_runtime_probe import probe_cufuzz_library
from .feedback_runtime import (
    EDGES_MAP_SIZE,
    SIMT_MEMCOV_STORAGE_SIZE,
    THREAD_ACTIVITY_BYTE_BASE,
    discover_feedback_artifact,
    generate_default_seed,
    patch_trailing_u64,
    run_feedback_worker,
    summarize_maps,
    _worker_main,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_TESTS_ENABLED = os.environ.get("RAPID_RUN_GPU_TESTS") == "1"


class FeedbackRuntimeUnitTest(unittest.TestCase):
    def _compile_ordered_feedback_library(self, directory: Path) -> Path:
        compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
        if compiler is None:
            self.skipTest("C compiler unavailable")
        source = directory / "ordered_feedback.c"
        library = directory / "libordered_feedback.so"
        source.write_text(
            r"""
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint16_t code;
  uint16_t stage;
  uint32_t detail;
} RunStatus;

typedef struct {
  uint64_t task_id;
  size_t input_ptr;
  uint8_t *edge_ptr;
  uint8_t *simt_memcov_ptr;
  uint32_t edge_size;
  uint32_t simt_memcov_size;
  RunStatus status;
  uint64_t exec_time_ns;
} TaskResult;

typedef struct {
  size_t pending;
  size_t completed;
  size_t outstanding;
} OrderedQueueCounts;

uint8_t libafl_cov_map[65536];
uint8_t libafl_simt_memcov_bits[8192];

static uint8_t edge_storage[65536];
static uint8_t mem_storage[8192];
static uint64_t active_task_id = 0;
static int acquired = 0;
static int released = 0;

void libafl_set_target_timeout_ms(uint64_t timeout_ms) {
  (void)timeout_ms;
}

uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {
  (void)size;
  edge_storage[7] = input[0];
  mem_storage[0] = 0x3;
  active_task_id = 9;
  acquired = 0;
  released = 0;
  return active_task_id;
}

size_t libafl_poll_results(TaskResult *results, size_t max_count) {
  if (max_count == 0 || active_task_id == 0 || acquired || released) {
    return 0;
  }
  results[0].task_id = active_task_id;
  results[0].input_ptr = 0;
  results[0].edge_ptr = edge_storage;
  results[0].simt_memcov_ptr = mem_storage;
  results[0].edge_size = 65536;
  results[0].simt_memcov_size = 8192;
  results[0].status.code = 0;
  results[0].status.stage = 4;
  results[0].status.detail = 123;
  results[0].exec_time_ns = 456;
  acquired = 1;
  return 1;
}

size_t libafl_release_tasks(const uint64_t *task_ids, size_t count) {
  if (count == 1 && task_ids[0] == active_task_id && acquired && !released) {
    released = 1;
    return 1;
  }
  return 0;
}

void libafl_get_ordered_queue_counts(OrderedQueueCounts *counts) {
  counts->pending = 0;
  counts->completed = acquired && !released ? 1 : 0;
  counts->outstanding = 0;
}

void libafl_wait(void) {}
void libafl_stop(void) {}
""",
            encoding="utf-8",
        )
        result = subprocess.run(
            [compiler, "-shared", "-fPIC", str(source), "-o", str(library)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return library

    def test_discover_artifact_selects_requested_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            kernel_dir = run_dir / "kernels/chosen"
            (kernel_dir / "phase2/backends/origin").mkdir(parents=True)
            (kernel_dir / "phase2/backends/rapid").mkdir(parents=True)
            (kernel_dir / "phase2/backends/rapid2").mkdir(parents=True)
            (kernel_dir / "manifest.json").touch()
            (kernel_dir / "phase2/backends/origin/libphase2_origin_target.so").touch()
            (kernel_dir / "phase2/backends/rapid/libphase2_rapid_target.so").touch()
            (kernel_dir / "phase2/backends/rapid2/librapid2_target.so").touch()
            (run_dir / "index.json").write_text(
                json.dumps(
                    {
                        "kernels": [
                            {"kernel_id": "other", "dir": "kernels/other"},
                            {"kernel_id": "chosen", "dir": "kernels/chosen"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            artifact = discover_feedback_artifact(run_dir, kernel_id="chosen")

        self.assertEqual(artifact.kernel_dir, kernel_dir)

    def test_summarize_maps_counts_feedback_partitions(self) -> None:
        edge_map = bytearray(EDGES_MAP_SIZE)
        bitset = bytearray(SIMT_MEMCOV_STORAGE_SIZE)
        edge_map[7] = 1
        bitset[0] = 0b00000011
        bitset[THREAD_ACTIVITY_BYTE_BASE] = 0b00000100

        result = summarize_maps("origin", bytes(edge_map), bytes(bitset))

        self.assertEqual(result.cfg_sites, 1)
        self.assertEqual(result.simt_memcov_bits, 2)
        self.assertEqual(result.logical_thread_bits, 1)
        self.assertEqual(result.simt_memcov_set_bits, (0, 1))
        self.assertEqual(result.logical_thread_set_bits, (2,))

    def test_summarize_maps_rejects_wrong_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "edge map size"):
            summarize_maps("origin", b"", bytes(SIMT_MEMCOV_STORAGE_SIZE))
        with self.assertRaisesRegex(ValueError, "memory/index bitset size"):
            summarize_maps("origin", bytes(EDGES_MAP_SIZE), b"")

    def test_patch_trailing_u64_preserves_arg_pack_payload(self) -> None:
        seed = bytes(range(24))

        patched = patch_trailing_u64(seed, 8)

        self.assertEqual(patched[:-8], seed[:-8])
        self.assertEqual(int.from_bytes(patched[-8:], "little"), 8)

    def test_rapid_worker_uses_ordered_abi_without_sync_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rapid-feedback-runtime-") as temp_dir:
            library = self._compile_ordered_feedback_library(Path(temp_dir))

            result = _worker_main("rapid", library, ["05"])

        self.assertEqual(result["backend"], "rapid")
        self.assertEqual(result["result"]["backend"], "rapid")
        self.assertEqual(result["result"]["cfg_sites"], 1)
        self.assertEqual(result["result"]["simt_memcov_bits"], 2)
        self.assertEqual(
            result["status"],
            {
                "code": 0,
                "stage": 4,
                "detail": 123,
            },
        )
        self.assertEqual(
            result["queue_counts"],
            {"pending": 0, "completed": 0, "outstanding": 0},
        )


@unittest.skipUnless(GPU_TESTS_ENABLED, "set RAPID_RUN_GPU_TESTS=1 to run CUDA e2e")
class FeedbackBackendE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured_run_dir = os.environ.get("RAPID_FEEDBACK_E2E_RUN_DIR")
        cls._temporary_root: tempfile.TemporaryDirectory[str] | None = None
        if configured_run_dir:
            cls.run_dir = Path(configured_run_dir)
        else:
            cls._temporary_root = tempfile.TemporaryDirectory(
                prefix="rapid-feedback-e2e-"
            )
            out_root = Path(cls._temporary_root.name)
            subprocess.run(
                [
                    str(REPO_ROOT / ".venv/bin/python"),
                    str(REPO_ROOT / "cuda-kernel/builtin_phase_pipeline.py"),
                    "--out-root",
                    str(out_root),
                    "--run-id",
                    "feedback-e2e",
                    "--source",
                    str(REPO_ROOT / "tests/feedback_e2e/fixtures/feedback_kernels.cu"),
                    "--cuda-arch",
                    os.environ.get("KSMOKE_CUDA_ARCH", "sm_86"),
                    "--cuda-path",
                    os.environ.get("CUDA_PATH", "/usr/local/cuda"),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            cls.run_dir = out_root / "out/feedback-e2e"

        cls.artifact = discover_feedback_artifact(cls.run_dir)
        cls.cufuzz_library = (
            cls.artifact.kernel_dir
            / "phase2/backends/cufuzz/libphase2_cufuzz_target.so"
        )
        cls.origin_no_feedback_library = (
            cls.artifact.kernel_dir
            / "phase2/backends/origin-no-feedback/libphase2_origin_target.so"
        )
        fuzzer = REPO_ROOT / "cuda-fuzzer/target/release/fuzzer"
        default_seed = generate_default_seed(fuzzer, cls.artifact.manifest)
        cls.active_seed = patch_trailing_u64(default_seed, 8)
        cls.idle_seed = patch_trailing_u64(default_seed, 0)
        cls.origin = run_feedback_worker(
            "origin", cls.artifact.origin_library, [cls.active_seed]
        )
        cls.origin_no_feedback = run_feedback_worker(
            "origin", cls.origin_no_feedback_library, [cls.active_seed]
        )
        cls.cufuzz = probe_cufuzz_library(
            cls.cufuzz_library,
            cls.active_seed,
            runs=10,
        )
        cls.rapid = run_feedback_worker(
            "rapid", cls.artifact.rapid_library, [cls.active_seed]
        )
        cls.rapid2 = run_feedback_worker(
            "rapid2",
            cls.artifact.rapid2_library,
            [cls.active_seed, cls.idle_seed, cls.idle_seed],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._temporary_root is not None:
            cls._temporary_root.cleanup()

    def test_fixture_has_real_payload_memory_probes(self) -> None:
        for backend in ("origin", "rapid", "rapid2"):
            metadata_path = (
                self.artifact.kernel_dir
                / f"phase2/backends/{backend}/intermediates/feedback_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertGreater(metadata["instrumented_cfg_sites"], 0)
            self.assertGreater(metadata["instrumented_memory_sites"], 0)
            self.assertEqual(
                metadata["memory_metric_version"],
                "rapid-simt-memcov-v1",
            )

    def test_clean_llvm_baselines_have_no_device_feedback_baggage(self) -> None:
        for backend in ("cufuzz", "origin-no-feedback"):
            backend_dir = self.artifact.kernel_dir / f"phase2/backends/{backend}"
            build = json.loads(
                (backend_dir / "backend_build.json").read_text(encoding="utf-8")
            )
            ptx = (backend_dir / "intermediates/module.ptx").read_text(
                encoding="utf-8"
            )
            with self.subTest(backend=backend):
                self.assertEqual(build["llvm_feedback_instrumentation"], "disabled")
                self.assertEqual(build["inner_cuda_feedback"], "disabled")
                self.assertNotIn("__rapid_feedback_", ptx)
                self.assertNotIn("device_cov_map", ptx)
                self.assertNotIn("rapid_bound_cov_map", ptx)

    def test_cufuzz_allocates_and_frees_twice_per_target_call(self) -> None:
        self.assertEqual(
            self.cufuzz["allocation_stats_delta"],
            {
                "runs_started": 10,
                "allocation_attempts": 20,
                "allocations_succeeded": 20,
                "frees": 20,
                "runs_completed": 10,
            },
        )
        self.assertEqual(self.cufuzz["coverage_nonzero_bytes"], 0)
        self.assertEqual(self.cufuzz["simt_memcov_nonzero_bits"], 0)

    def test_origin_feedback_disabled_and_enabled_are_observably_distinct(self) -> None:
        disabled = self.origin_no_feedback["result"]
        enabled = self.origin["result"]

        self.assertEqual(disabled["cfg_sites"], 0)
        self.assertEqual(disabled["simt_memcov_bits"], 0)
        self.assertEqual(disabled["logical_thread_bits"], 0)
        self.assertGreater(enabled["cfg_sites"], 0)
        self.assertGreater(enabled["logical_thread_bits"], 0)

    def test_backends_report_all_feedback_channels_and_equivalent_maps(self) -> None:
        origin = self.origin["result"]
        rapid = self.rapid["result"]
        rapid2 = self.rapid2["results"][0]["result"]

        for result in (origin, rapid, rapid2):
            with self.subTest(backend=result["backend"]):
                self.assertGreater(result["cfg_sites"], 0)
                self.assertGreater(result["simt_memcov_bits"], 0)
                self.assertGreater(result["logical_thread_bits"], 0)
        self.assertEqual(origin["edge_sha256"], rapid["edge_sha256"])
        self.assertEqual(origin["edge_sha256"], rapid2["edge_sha256"])
        self.assertEqual(origin["simt_memcov_sha256"], rapid["simt_memcov_sha256"])
        self.assertEqual(origin["simt_memcov_sha256"], rapid2["simt_memcov_sha256"])

    def test_rapid2_reuses_slots_without_mutating_acquired_results(self) -> None:
        task_a, task_b, task_c = self.rapid2["results"]

        self.assertNotEqual(
            task_a["result"]["edge_sha256"], task_b["result"]["edge_sha256"]
        )
        self.assertNotEqual(
            task_a["result"]["simt_memcov_sha256"], task_b["result"]["simt_memcov_sha256"]
        )
        self.assertEqual(task_b["result"], task_c["result"])
        self.assertTrue(self.rapid2["first_result_lifetime_preserved"])

    def test_rapid2_completion_order_matches_submission_across_slot_reuse(self) -> None:
        self.assertEqual(self.rapid2["submitted_task_ids"], [1, 2, 3])
        self.assertEqual(
            [result["task_id"] for result in self.rapid2["results"]],
            [1, 2, 3],
        )

    def test_rapid2_stream_completion_round_trip_with_four_slots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rapid2-stream-four-slot-") as td:
            out_dir = Path(td) / "rapid2"
            subprocess.run(
                [
                    str(REPO_ROOT / ".venv/bin/python"),
                    str(REPO_ROOT / "cuda-kernel/rapid2/build.py"),
                    "--phase2-dir",
                    str(self.artifact.kernel_dir / "phase2"),
                    "--out-dir",
                    str(out_dir),
                    "--cuda-arch",
                    os.environ.get("KSMOKE_CUDA_ARCH", "sm_86"),
                    "--cuda-path",
                    os.environ.get("CUDA_PATH", "/usr/local/cuda"),
                    "--slot-count",
                    "4",
                    "--completion-mode",
                    "stream",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            rapid2 = run_feedback_worker(
                "rapid2",
                out_dir / "librapid2_target.so",
                [
                    self.active_seed,
                    self.idle_seed,
                    self.idle_seed,
                    self.idle_seed,
                    self.idle_seed,
                ],
            )

        self.assertEqual(rapid2["submitted_task_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(
            [result["task_id"] for result in rapid2["results"]],
            [1, 2, 3, 4, 5],
        )
        for result in rapid2["results"]:
            self.assertEqual(result["status"]["code"], 0)


if __name__ == "__main__":
    unittest.main()
