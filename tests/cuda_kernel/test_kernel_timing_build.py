import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_KERNEL = REPO_ROOT / "cuda-kernel"
if str(CUDA_KERNEL) not in sys.path:
    sys.path.insert(0, str(CUDA_KERNEL))

from backend_build import kernel_timing_defines  # noqa: E402


class KernelTimingBuildTests(unittest.TestCase):
    def test_timing_define_is_opt_in(self) -> None:
        self.assertEqual(kernel_timing_defines(False), ())
        self.assertEqual(kernel_timing_defines(True), ("-DENABLE_KERNEL_TIMING=1",))

    def test_all_profiled_builders_expose_timing_switch(self) -> None:
        origin_source = (CUDA_KERNEL / "origin/build.py").read_text(encoding="utf-8")
        rapid_source = (CUDA_KERNEL / "rapid/build.py").read_text(encoding="utf-8")
        rapid2_source = (CUDA_KERNEL / "rapid2/build.py").read_text(encoding="utf-8")
        for source in (origin_source, rapid_source, rapid2_source):
            self.assertIn("enable_kernel_timing: bool = False", source)
            self.assertIn('parser.add_argument("--enable-kernel-timing"', source)
            self.assertIn('"kernel_timing_enabled": enable_kernel_timing', source)

    def test_device_timing_contract_separates_feedback_from_target_execution(self) -> None:
        source = (CUDA_KERNEL / "utils/timing_stats.cuh").read_text(encoding="utf-8")
        self.assertIn("decode_cycles", source)
        self.assertIn("feedback_prepare_cycles", source)

    def test_rapid_backend_dumps_device_timing_before_module_unload(self) -> None:
        source = (CUDA_KERNEL / "rapid/harness.cpp").read_text(encoding="utf-8")
        self.assertIn("dump_kernel_timing_stats", source)
        self.assertLess(
            source.index("dump_kernel_timing_stats();"),
            source.index("cuModuleUnload(gpu_resources.module)"),
        )

    def test_rapid_timing_worker_has_a_stable_profile_lane_name(self) -> None:
        source = (CUDA_KERNEL / "rapid/harness.cpp").read_text(encoding="utf-8")
        worker = source[source.index("static void gpu_producer_worker()") :]
        timing_guard = worker.index("#if ENABLE_KERNEL_TIMING")
        thread_name = worker.index('pthread_setname_np(pthread_self(), "RAPID-Sync")')
        timing_end = worker.index("#endif", timing_guard)
        self.assertLess(timing_guard, thread_name)
        self.assertLess(thread_name, timing_end)

    def test_persistent_backends_expose_a_profile_timing_reset_hook(self) -> None:
        rapid = (CUDA_KERNEL / "rapid/harness.cpp").read_text(encoding="utf-8")
        rapid2 = (
            CUDA_KERNEL / "rapid2/libafl_interface_shared.cuh"
        ).read_text(encoding="utf-8")
        for source in (rapid, rapid2):
            self.assertIn("libafl_profile_timing_start", source)
            self.assertIn("reset", source[source.index("libafl_profile_timing_start") :])

    def test_persistent_timing_snapshots_use_stream_scoped_async_copies(self) -> None:
        rapid = (CUDA_KERNEL / "rapid/harness.cpp").read_text(encoding="utf-8")
        rapid_snapshot = rapid[
            rapid.index("inline bool copy_kernel_timing_stats") :
            rapid.index("inline bool reset_kernel_timing_stats")
        ]
        rapid2 = (CUDA_KERNEL / "rapid2/module_runtime.cuh").read_text(
            encoding="utf-8"
        )
        rapid2_snapshot = rapid2[
            rapid2.index("inline bool copy_timing_stats_from_device") :
            rapid2.index("inline bool reset_timing_stats_on_device")
        ]
        for snapshot in (rapid_snapshot, rapid2_snapshot):
            self.assertIn("cuMemcpyDtoHAsync", snapshot)
            self.assertIn("cuStreamSynchronize", snapshot)

    def test_rapid_timing_reset_publish_and_snapshot_are_ordered(self) -> None:
        wrapper = (CUDA_KERNEL / "rapid/wrapper.cu").read_text(encoding="utf-8")
        loop = wrapper[wrapper.index("while (*run)") :]
        reset = loop.index("KERNEL_TIMING_RESET_REQUEST")
        self.assertLess(loop.index("gpu_wait(semaphore)"), reset)
        self.assertLess(
            loop.index("publish_rapid_timing_stats(stats)"),
            loop.index("atomicExch(data_processed, 1)"),
        )

        harness = (CUDA_KERNEL / "rapid/harness.cpp").read_text(encoding="utf-8")
        snapshot = harness[harness.index("libafl_profile_timing_snapshot") :]
        self.assertLess(
            snapshot.index("shutdown_and_join_worker"),
            snapshot.index("final_timing_stats"),
        )
        self.assertNotIn("attempt < 1000", snapshot)

    def test_rapid2_timing_snapshot_is_published_on_request(self) -> None:
        kernel = (CUDA_KERNEL / "rapid2/pipelined_kernel_impl.cuh").read_text(
            encoding="utf-8"
        )
        accounting = kernel[kernel.index("stats.iterations++;") :]
        accounting = accounting[: accounting.index("#endif")]
        self.assertNotIn("publish_rapid2_timing_stats", accounting)
        self.assertIn("KERNEL_TIMING_SNAPSHOT_REQUEST", kernel)

        runtime = (CUDA_KERNEL / "rapid2/module_runtime.cuh").read_text(
            encoding="utf-8"
        )
        snapshot = runtime[
            runtime.index("inline bool snapshot_timing_stats_from_device") :
            runtime.index("inline bool reset_timing_stats_on_device")
        ]
        self.assertIn("KERNEL_TIMING_SNAPSHOT_REQUEST", snapshot)
        self.assertIn("request_timing_stats_on_device", snapshot)
        self.assertIn("copy_timing_stats_from_device", snapshot)


if __name__ == "__main__":
    unittest.main()
