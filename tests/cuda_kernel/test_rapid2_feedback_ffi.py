import shutil
import subprocess
import tempfile
import unittest
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAPID2_DIR = REPO_ROOT / "cuda-kernel" / "rapid2"


def _load_rapid2_build_module():
    spec = importlib.util.spec_from_file_location("rapid2_build_for_test", RAPID2_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Rapid2FeedbackFfiTest(unittest.TestCase):
    def test_fourth_slot_generation_unblocks_stream_wait(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        cuda_root = Path("/usr/local/cuda/targets/x86_64-linux")
        cuda_include = cuda_root / "include"
        cuda_library = cuda_root / "lib"
        if (
            compiler is None
            or not cuda_include.is_dir()
            or not (cuda_library / "libcudart.so").is_file()
        ):
            self.skipTest("C++ compiler, CUDA headers, or libcudart is unavailable")

        source = r"""
#include "double_buffer_sync.cuh"

#include <cassert>
#include <cstdint>

int main() {
  DoubleBufferSync sync;
  cudaStream_t dispatch_stream = nullptr;
  cudaStream_t completion_stream = nullptr;
  cudaStream_t signal_stream = nullptr;
  cudaEvent_t completion_event = nullptr;
  assert(cudaStreamCreate(&dispatch_stream) == cudaSuccess);
  assert(cudaStreamCreate(&completion_stream) == cudaSuccess);
  assert(cudaStreamCreate(&signal_stream) == cudaSuccess);
  assert(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming) ==
         cudaSuccess);
  assert(sync.enqueueBufferReady(3, 7, dispatch_stream) == CUDA_SUCCESS);
  assert(cudaStreamSynchronize(dispatch_stream) == cudaSuccess);

  const DeviceSyncPtrs ptrs = sync.getDevicePtrs();
  uint64_t ready = 0;
  assert(cudaMemcpy(&ready, (const void *)ptrs.buffer_ready[3], sizeof(ready),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
  assert(ready == 7);

  assert(sync.enqueueCompletionWait(3, 7, completion_stream) == CUDA_SUCCESS);
  assert(cudaEventRecord(completion_event, completion_stream) == cudaSuccess);
  assert(cuStreamWriteValue64(reinterpret_cast<CUstream>(signal_stream),
                              reinterpret_cast<CUdeviceptr>(
                                  ptrs.buffer_done[3]),
                              7, CU_STREAM_WRITE_VALUE_DEFAULT) ==
         CUDA_SUCCESS);
  assert(cudaEventSynchronize(completion_event) == cudaSuccess);

  assert(cudaEventDestroy(completion_event) == cudaSuccess);
  assert(cudaStreamDestroy(signal_stream) == cudaSuccess);
  assert(cudaStreamDestroy(completion_stream) == cudaSuccess);
  assert(cudaStreamDestroy(dispatch_stream) == cudaSuccess);
  return 0;
}
"""

        with tempfile.TemporaryDirectory(prefix="rapid2-sequence-sync-") as td:
            temp_dir = Path(td)
            source_path = temp_dir / "sequence_sync_test.cpp"
            binary_path = temp_dir / "sequence_sync_test"
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-DRAPID2_SLOT_COUNT=4",
                    str(source_path),
                    "-I",
                    str(RAPID2_DIR),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel"),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils"),
                    "-I",
                    str(cuda_include),
                    "-L",
                    str(cuda_library),
                    f"-Wl,-rpath,{cuda_library}",
                    "-lcuda",
                    "-lcudart",
                    "-o",
                    str(binary_path),
                ],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            completed = subprocess.run(
                [str(binary_path)],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_builder_rejects_unsupported_slot_count(self) -> None:
        rapid2_build = _load_rapid2_build_module()
        with self.assertRaisesRegex(RuntimeError, "slot count must be 2 or 4"):
            rapid2_build.build_rapid2_target(
                phase2_dir=Path("/does/not/exist"),
                out_dir=None,
                cuda_arch="sm_86",
                cuda_path="/usr/local/cuda",
                slot_count=3,
            )

    def test_builder_rejects_unsupported_completion_mode(self) -> None:
        rapid2_build = _load_rapid2_build_module()
        with self.assertRaisesRegex(
            RuntimeError, "completion mode must be polling or stream"
        ):
            rapid2_build.build_rapid2_target(
                phase2_dir=Path("/does/not/exist"),
                out_dir=None,
                cuda_arch="sm_86",
                cuda_path="/usr/local/cuda",
                completion_mode="callback",
            )

    def test_rapid2_feedback_uses_one_contiguous_pinned_slab(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        cuda_root = Path("/usr/local/cuda/targets/x86_64-linux")
        cuda_include = cuda_root / "include"
        cuda_library = cuda_root / "lib"
        if (
            compiler is None
            or not cuda_include.is_dir()
            or not (cuda_library / "libcudart.so").is_file()
        ):
            self.skipTest("C++ compiler, CUDA headers, or libcudart is unavailable")

        source = r"""
#include "rapid_target_layout.v1.h"
#include "combined_input.cuh"
#include "coverage/coverage_constants.h"
#include "feedback/feedback_constants.h"

#include <cassert>
#include <cstdint>

int main() {
  auto combined = HostCombinedInput::allocate(1036);
  auto feedback = HostFeedbackBuffer::allocate();
  unsigned flags = 0;
  assert(cudaHostGetFlags(&flags, combined.get()) == cudaSuccess);
  assert(cudaHostGetFlags(&flags, feedback.data()) == cudaSuccess);
  assert(feedback.edgeData() == feedback.data());
  assert(feedback.memIndexData() == feedback.data() + MAP_SIZE);
  assert(feedback.size() == MAP_SIZE + SIMT_MEMCOV_STORAGE_SIZE);
  return 0;
}
"""

        with tempfile.TemporaryDirectory(prefix="rapid2-pinned-host-") as td:
            temp_dir = Path(td)
            source_path = temp_dir / "pinned_host_test.cpp"
            binary_path = temp_dir / "pinned_host_test"
            (temp_dir / "rapid_target_layout.v1.h").write_text(
                "#ifndef RAPID_TARGET_LAYOUT_V1_H\n"
                "#define RAPID_TARGET_LAYOUT_V1_H\n"
                "#define RAPID_PAYLOAD_SLOT_COUNT 1u\n"
                "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT 1u\n"
                "#endif\n",
                encoding="utf-8",
            )
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    str(source_path),
                    "-I",
                    str(RAPID2_DIR),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel"),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils"),
                    "-I",
                    str(cuda_include),
                    "-L",
                    str(cuda_library),
                    f"-Wl,-rpath,{cuda_library}",
                    "-lcudart",
                    "-o",
                    str(binary_path),
                ],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            subprocess.run(
                [str(binary_path)],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

    def test_completed_queue_applies_lossless_backpressure(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        source = r"""
#include "config.h"
#include "task_queue.cuh"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <thread>

struct Item {
  explicit Item(int value) : id(value) {}
  int id;
};

int main() {
  static_assert(!Rapid2CompletedQueue<Item>::kDropOldestOnFull);

  Rapid2CompletedQueue<Item> queue(2);
  assert(queue.push(std::make_unique<Item>(1)));
  assert(queue.push(std::make_unique<Item>(2)));

  std::atomic<bool> pushed_third{false};
  std::thread producer([&] {
    assert(queue.push(std::make_unique<Item>(3)));
    pushed_third.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  assert(!pushed_third.load(std::memory_order_acquire));

  auto first = queue.try_pop();
  assert(first && first->id == 1);
  producer.join();
  assert(pushed_third.load(std::memory_order_acquire));

  auto second = queue.try_pop();
  auto third = queue.try_pop();
  assert(second && second->id == 2);
  assert(third && third->id == 3);
  assert(queue.try_pop() == nullptr);
}
"""

        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            source_path = temp_dir / "completed_queue_test.cpp"
            binary_path = temp_dir / "completed_queue_test"
            source_path.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    str(source_path),
                    "-I",
                    str(RAPID2_DIR),
                    "-o",
                    str(binary_path),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            subprocess.run(
                [str(binary_path)],
                check=True,
                cwd=REPO_ROOT,
            )

    def test_completed_queue_wait_only_observes_published_slots(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        source = r"""
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#define private public
#include "config.h"
#include "task_queue.cuh"
#undef private

struct Item {
  explicit Item(uint64_t value) : id(value) {}
  uint64_t id;
};

int main() {
  // Reproduce the reservation/publication window deterministically. The old
  // size()-based empty() treated enqueue_pos_ reservation as a visible item.
  Rapid2CompletedQueue<Item> reserved_queue(8);
  reserved_queue.enqueue_pos_.store(1, std::memory_order_relaxed);

  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  std::atomic<bool> waiter_entered{false};
  std::atomic<bool> waiter_finished{false};
  std::atomic<bool> woke_without_item{false};
  std::thread waiter([&] {
    std::unique_lock<std::mutex> lock(completion_mutex);
    waiter_entered.store(true, std::memory_order_release);
    completion_cv.wait(lock, [&] { return !reserved_queue.empty(); });
    auto item = reserved_queue.try_pop();
    woke_without_item.store(!item, std::memory_order_release);
    waiter_finished.store(true, std::memory_order_release);
  });

  while (!waiter_entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  assert(!waiter_finished.load(std::memory_order_acquire));

  reserved_queue.slots_[0].ptr = new Item(1);
  reserved_queue.slots_[0].seq.store(1, std::memory_order_release);
  completion_cv.notify_one();
  waiter.join();
  assert(!woke_without_item.load(std::memory_order_acquire));

  // Exercise the real push path under bounded MPMC contention. This mirrors
  // MultiPipelineManager::waitForCompletion(): every successful predicate
  // wake must be followed by a successful non-blocking poll.
  constexpr uint64_t kProducerCount = 4;
  constexpr uint64_t kItemsPerProducer = 10000;
  constexpr uint64_t kTotalItems = kProducerCount * kItemsPerProducer;
  Rapid2CompletedQueue<Item> queue(1024);
  std::atomic<uint64_t> next_id{1};
  std::atomic<uint64_t> producers_done{0};
  std::thread producers[kProducerCount];
  for (uint64_t producer = 0; producer < kProducerCount; ++producer) {
    producers[producer] = std::thread([&] {
      for (uint64_t item = 0; item < kItemsPerProducer; ++item) {
        const uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        assert(queue.push(std::make_unique<Item>(id)));
        completion_cv.notify_one();
      }
      producers_done.fetch_add(1, std::memory_order_release);
      completion_cv.notify_one();
    });
  }

  uint64_t consumed = 0;
  while (consumed < kTotalItems) {
    std::unique_lock<std::mutex> lock(completion_mutex);
    const bool woke = completion_cv.wait_for(
        lock, std::chrono::seconds(2), [&] {
          return !queue.empty() ||
                 producers_done.load(std::memory_order_acquire) ==
                     kProducerCount;
        });
    assert(woke);
    lock.unlock();

    auto item = queue.try_pop();
    assert(item);
    ++consumed;
  }

  for (auto &producer : producers) {
    producer.join();
  }
  assert(producers_done.load(std::memory_order_acquire) == kProducerCount);
  assert(queue.empty());
  return 0;
}
"""

        with tempfile.TemporaryDirectory(prefix="rapid2-queue-visibility-") as td:
            temp_dir = Path(td)
            source_path = temp_dir / "queue_visibility_test.cpp"
            binary_path = temp_dir / "queue_visibility_test"
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    str(source_path),
                    "-I",
                    str(RAPID2_DIR),
                    "-o",
                    str(binary_path),
                ],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            completed = subprocess.run(
                [str(binary_path)],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_rapid2_ffi_contains_exceptions_and_rejects_multiple_pipelines(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        cuda_root = Path("/usr/local/cuda/targets/x86_64-linux")
        cuda_include = cuda_root / "include"
        cuda_library = cuda_root / "lib"
        if (
            compiler is None
            or not cuda_include.is_dir()
            or not (cuda_library / "libcudart.so").is_file()
        ):
            self.skipTest("C++ compiler, CUDA headers, or libcudart is unavailable")

        source = r"""
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define private public
#include "harness.cpp"
#undef private

namespace rapid2_embedded_ptx {
const unsigned char kEmbeddedPtx[] = {0};
const size_t kEmbeddedPtxSize = 0;
} // namespace rapid2_embedded_ptx

static std::atomic<bool> fail_host_alloc{false};
static std::atomic<size_t> free_host_calls{0};

extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void **ptr, size_t size,
                                                unsigned int flags) {
  (void)flags;
  if (fail_host_alloc.load(std::memory_order_acquire)) {
    return cudaErrorMemoryAllocation;
  }
  *ptr = std::malloc(size);
  return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  free_host_calls.fetch_add(1, std::memory_order_relaxed);
  std::free(ptr);
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void **ptr, size_t size) {
  *ptr = std::malloc(size);
  return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t CUDARTAPI cudaFree(void *ptr) {
  std::free(ptr);
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemset(void *ptr, int value,
                                              size_t size) {
  std::memset(ptr, value, size);
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI
cudaHostGetDevicePointer(void **device_ptr, void *host_ptr,
                         unsigned int flags) {
  (void)flags;
  *device_ptr = host_ptr;
  return cudaSuccess;
}

int main() {
  assert(!rapid2_ensure_initialized(2, MAX_INPUT_SIZE));
  rapid2_initialize(2);
  assert(!g_initialized.load(std::memory_order_acquire));
  assert(!g_pipeline_manager);

  auto manager = std::make_unique<MultiPipelineManager>(1, 8);
  manager->running_.store(true, std::memory_order_release);
  manager->setTargetTimeoutMs(1000);
  g_pipeline_manager = std::move(manager);
  g_initialized.store(true, std::memory_order_release);
  g_target_timeout_ms.store(1000, std::memory_order_release);

  RapidTaskEnvelopeHeader header{};
  header.payload_size = 1;
  std::array<uint8_t, sizeof(header) + 1> input{};
  std::memcpy(input.data(), &header, sizeof(header));
  input.back() = 0x5a;

  const size_t frees_before_blocked_submit =
      free_host_calls.load(std::memory_order_relaxed);
  {
    SingleKernelPipeline pipeline;
    pipeline.initialized_ = true;
    pipeline.kernel_launched_.store(true, std::memory_order_release);
    pipeline.accepting_tasks_ = false;
    auto blocked_task =
        std::make_unique<TaskData>(99, input.data(), input.size());
    assert(!pipeline.waitAndSubmitTask(std::move(blocked_task)));
    assert(free_host_calls.load(std::memory_order_relaxed) ==
           frees_before_blocked_submit);
    pipeline.kernel_launched_.store(false, std::memory_order_release);
    pipeline.initialized_ = false;
  }

  fail_host_alloc.store(true, std::memory_order_release);
  assert(libafl_submit_with_id(input.data(), input.size()) == 0);
  assert(g_pipeline_manager->next_task_id_.load(std::memory_order_acquire) == 1);

  fail_host_alloc.store(false, std::memory_order_release);
  assert(libafl_submit_with_id(input.data(), input.size()) == 1);
  assert(g_pipeline_manager->next_task_id_.load(std::memory_order_acquire) == 2);

  g_pipeline_manager->running_.store(false, std::memory_order_release);
  g_pipeline_manager.reset();
  g_initialized.store(false, std::memory_order_release);
  g_target_timeout_ms.store(0, std::memory_order_release);
  return 0;
}
"""

        with tempfile.TemporaryDirectory(prefix="rapid2-ffi-guard-") as td:
            temp_dir = Path(td)
            source_path = temp_dir / "ffi_guard_test.cpp"
            binary_path = temp_dir / "ffi_guard_test"
            (temp_dir / "rapid_target_layout.v1.h").write_text(
                "#define RAPID_PAYLOAD_SLOT_COUNT 1u\n"
                "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT 1u\n",
                encoding="utf-8",
            )
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    str(source_path),
                    "-I",
                    str(temp_dir),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel"),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils"),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils" / "coverage"),
                    "-I",
                    str(RAPID2_DIR),
                    "-I",
                    str(cuda_include),
                    "-L",
                    str(cuda_library),
                    f"-Wl,-rpath,{cuda_library}",
                    "-L/usr/lib/x86_64-linux-gnu",
                    "-lcuda",
                    "-lcudart",
                    "-o",
                    str(binary_path),
                ],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            completed = subprocess.run(
                [str(binary_path)],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "[RAPID2] Initialization failed: exactly one pipeline is required (got 2)",
                completed.stderr,
            )
            self.assertIn(
                "[RAPID2] input submission failed: out of memory",
                completed.stderr,
            )

    def test_builder_rejects_multi_block_physical_grid_until_runtime_support_exists(self) -> None:
        build = _load_rapid2_build_module()

        with self.assertRaisesRegex(RuntimeError, "physical grid"):
            build._launch_policy_config(
                {
                    "launch_policy": {
                        "grid": [2, 1, 1],
                        "block_candidates": [128, 64, 32],
                        "coverage_memory": "global",
                    }
                }
            )

    def test_builder_rejects_unknown_manifest_kernel_field(self) -> None:
        build = _load_rapid2_build_module()

        with self.assertRaisesRegex(RuntimeError, "unexpected_launch"):
            build._launch_policy_config(
                {
                    "unexpected_launch": True,
                }
            )

    def test_builder_rejects_unknown_launch_policy_field(self) -> None:
        build = _load_rapid2_build_module()

        with self.assertRaisesRegex(RuntimeError, "launch_policy.unknown"):
            build._launch_policy_config(
                {
                    "launch_policy": {
                        "grid": [1, 1, 1],
                        "block_candidates": [128, 64, 32],
                        "physical_block_max": 128,
                        "target_dynamic_shared_bytes": 512,
                        "coverage_memory": "global",
                        "unknown": True,
                    }
                }
            )

    def test_rapid2_harness_compiles_with_layout_assertions(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        cuda_include = Path("/usr/local/cuda/targets/x86_64-linux/include")
        if compiler is None or not cuda_include.is_dir():
            self.skipTest("C++ compiler or CUDA headers are unavailable")

        with tempfile.TemporaryDirectory() as td:
            layout_dir = Path(td)
            (layout_dir / "rapid_target_layout.v1.h").write_text(
                "#define RAPID_PAYLOAD_SLOT_COUNT 1u\n"
                "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT 1u\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-fsyntax-only",
                    str(RAPID2_DIR / "harness.cpp"),
                    "-I",
                    str(layout_dir),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel"),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils"),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils" / "coverage"),
                    "-I",
                    str(RAPID2_DIR),
                    "-I",
                    str(cuda_include),
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
