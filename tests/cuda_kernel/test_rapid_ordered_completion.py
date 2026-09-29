import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAPID_DIR = REPO_ROOT / "cuda-kernel" / "rapid"


class RapidOrderedCompletionQueueTest(unittest.TestCase):
    def test_idle_spin_observes_new_work_before_parking(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        source = textwrap.dedent(
            r"""
            #include "worker_idle_wait.h"

            #include <atomic>
            #include <cassert>
            #include <chrono>
            #include <cstdint>
            #include <thread>

            int main() {
              using namespace std::chrono_literals;

              std::atomic<uint64_t> pending_inputs{0};
              std::atomic<bool> stop_requested{false};

              assert(rapid::spin_until_work(
                         pending_inputs, stop_requested, 0ns) ==
                     rapid::IdleSpinResult::kTimedOut);

              pending_inputs.store(1, std::memory_order_release);
              assert(rapid::spin_until_work(
                         pending_inputs, stop_requested, 10ms) ==
                     rapid::IdleSpinResult::kWorkAvailable);

              pending_inputs.store(0, std::memory_order_release);
              std::thread producer([&] {
                std::this_thread::sleep_for(1ms);
                pending_inputs.store(1, std::memory_order_release);
              });
              assert(rapid::spin_until_work(
                         pending_inputs, stop_requested, 100ms) ==
                     rapid::IdleSpinResult::kWorkAvailable);
              producer.join();

              pending_inputs.store(0, std::memory_order_release);
              stop_requested.store(true, std::memory_order_release);
              assert(rapid::spin_until_work(
                         pending_inputs, stop_requested, 10ms) ==
                     rapid::IdleSpinResult::kStopRequested);
              return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="rapid-worker-idle-wait-") as raw_tmp:
            tmp = Path(raw_tmp)
            source_path = tmp / "test.cpp"
            binary_path = tmp / "test"
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    str(source_path),
                    "-I",
                    str(RAPID_DIR),
                    "-o",
                    str(binary_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            subprocess.run([str(binary_path)], check=True, cwd=REPO_ROOT)

    def test_rapid_tasks_own_distinct_feedback_and_output_buffers(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        source = textwrap.dedent(
            r"""
            #include "rapid_task_data.h"

            #include <cassert>
            #include <cstdint>

            int main() {
              const uint8_t first_envelope[] = {1, 2, 3, 4};
              const uint8_t second_envelope[] = {5, 6, 7, 8};
              RapidTaskData first(1, 0x1111, first_envelope,
                                  sizeof(first_envelope), 2);
              RapidTaskData second(2, 0x2222, second_envelope,
                                   sizeof(second_envelope), 2);

              assert(first.task_id == 1);
              assert(second.task_id == 2);
              assert(first.input_ptr == 0x1111);
              assert(second.input_ptr == 0x2222);
              assert(first.envelope.get() != second.envelope.get());
              assert(first.output.get() != second.output.get());
              assert(first.coverage.data() != second.coverage.data());
              assert(first.simt_memcov.data() != second.simt_memcov.data());

              first.output[0] = 0x31;
              first.coverage[7] = 0x41;
              first.simt_memcov[9] = 0x51;
              assert(second.output[0] == 0);
              assert(second.coverage[7] == 0);
              assert(second.simt_memcov[9] == 0);
              assert(first.status.code == LIBAFL_RUN_STATUS_OK);
              assert(second.status.code == LIBAFL_RUN_STATUS_OK);
              return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="rapid-task-data-") as raw_tmp:
            tmp = Path(raw_tmp)
            source_path = tmp / "test.cpp"
            binary_path = tmp / "test"
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    str(source_path),
                    "-I",
                    str(RAPID_DIR),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils"),
                    "-o",
                    str(binary_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            subprocess.run([str(binary_path)], check=True, cwd=REPO_ROOT)

    def test_completed_tasks_block_without_dropping_and_acquire_fifo(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        source = textwrap.dedent(
            r"""
            #include "ordered_completion_queue.h"

            #include <atomic>
            #include <cassert>
            #include <chrono>
            #include <cstdint>
            #include <memory>
            #include <thread>

            struct Item {
              Item(uint64_t task_id, uint8_t marker)
                  : task_id(task_id), marker(marker) {}
              uint64_t task_id;
              uint8_t marker;
            };

            int main() {
              OrderedCompletionQueue<Item> queue(2);
              assert(queue.push(std::make_unique<Item>(1, 0x11)));
              assert(queue.push(std::make_unique<Item>(2, 0x22)));

              std::atomic<bool> third_published{false};
              std::thread producer([&] {
                assert(queue.push(std::make_unique<Item>(3, 0x33)));
                third_published.store(true, std::memory_order_release);
              });

              std::this_thread::sleep_for(std::chrono::milliseconds(20));
              assert(!third_published.load(std::memory_order_acquire));
              assert(queue.completed_size() == 2);
              assert(queue.outstanding_size() == 0);

              Item *first = queue.try_acquire();
              assert(first != nullptr);
              assert(first->task_id == 1);
              assert(first->marker == 0x11);
              assert(queue.outstanding_size() == 1);

              producer.join();
              assert(third_published.load(std::memory_order_acquire));
              assert(first->marker == 0x11);
              assert(!queue.release(99));
              assert(queue.release(1));

              Item *second = queue.try_acquire();
              Item *third = queue.try_acquire();
              assert(second != nullptr && second->task_id == 2);
              assert(third != nullptr && third->task_id == 3);
              assert(queue.try_acquire() == nullptr);
              assert(queue.release(2));
              assert(queue.release(3));
              assert(queue.completed_size() == 0);
              assert(queue.outstanding_size() == 0);

              queue.shutdown();
              assert(!queue.push(std::make_unique<Item>(4, 0x44)));
              return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory(prefix="rapid-ordered-completion-") as raw_tmp:
            tmp = Path(raw_tmp)
            source_path = tmp / "test.cpp"
            binary_path = tmp / "test"
            source_path.write_text(source, encoding="utf-8")
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    str(source_path),
                    "-I",
                    str(RAPID_DIR),
                    "-o",
                    str(binary_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            subprocess.run([str(binary_path)], check=True, cwd=REPO_ROOT)

    def test_rapid_harness_defines_ordered_task_ffi(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        nm = shutil.which("nm")
        cuda_include = Path("/usr/local/cuda/targets/x86_64-linux/include")
        if compiler is None or nm is None or not cuda_include.is_dir():
            self.skipTest("C++ compiler, nm, or CUDA headers are unavailable")

        with tempfile.TemporaryDirectory(prefix="rapid-ordered-ffi-") as raw_tmp:
            tmp = Path(raw_tmp)
            (tmp / "rapid_target_layout.v1.h").write_text(
                "#define RAPID_PAYLOAD_SLOT_COUNT 1u\n"
                "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT 1u\n",
                encoding="utf-8",
            )
            object_path = tmp / "harness.o"
            build = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    "-fPIC",
                    "-c",
                    str(RAPID_DIR / "harness.cpp"),
                    "-I",
                    str(tmp),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel"),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel" / "utils"),
                    "-I",
                    str(RAPID_DIR),
                    "-I",
                    str(cuda_include),
                    "-o",
                    str(object_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            symbols = subprocess.run(
                [nm, "-g", "--defined-only", str(object_path)],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            ).stdout

        for symbol in (
            "libafl_submit_with_id",
            "libafl_poll_results",
            "libafl_release_tasks",
            "libafl_get_ordered_queue_counts",
            "libafl_set_target_timeout_ms",
            "libafl_wait",
            "libafl_stop",
        ):
            self.assertIn(symbol, symbols)

    def test_completion_flag_is_published_after_system_fence(self) -> None:
        signal = (RAPID_DIR / "signal_helper.cuh").read_text(encoding="utf-8")
        body = signal[signal.index("__device__ void gpu_signal") : signal.index(
            "// Host-side semaphore wait operation"
        )]
        self.assertLess(body.index("__threadfence_system()"), body.index("atomicExch(data_processed, 1)"))

        wrapper = (RAPID_DIR / "wrapper.cu").read_text(encoding="utf-8")
        wake = wrapper.index("gpu_wait(semaphore)")
        invoke = wrapper.index("fuzzer_invoke_v1", wake)
        self.assertIn("if (!*run)", wrapper[wake:invoke])


if __name__ == "__main__":
    unittest.main()
