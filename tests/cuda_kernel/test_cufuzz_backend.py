import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_KERNEL_DIR = REPO_ROOT / "cuda-kernel"
CUFUZZ_DIR = CUDA_KERNEL_DIR / "cufuzz"


class CuFuzzBackendContractTest(unittest.TestCase):
    def test_builtin_kernel_uses_llvm_feedback_only(self) -> None:
        source = (CUDA_KERNEL_DIR / "kernel.cu").read_text(encoding="utf-8")

        self.assertNotIn("__LIBAFL_EDGE", source)
        self.assertNotIn('coverage/coverage.cuh', source)

    def test_cufuzz_wrapper_is_pure_decode_and_invoke(self) -> None:
        source = (CUFUZZ_DIR / "wrapper.cu").read_text(encoding="utf-8")

        self.assertIn('extern "C" __global__ void cufuzz_wrapper', source)
        self.assertIn("fuzzer_decode_v1", source)
        self.assertIn("fuzzer_invoke_v1", source)
        for forbidden in (
            "rapid_bind_coverage_map",
            "init_shared_coverage",
            "merge_shared_coverage",
            "fuzzer_feedback_prepare_v1",
            "rapid_feedback_record_thread_activity",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_cufuzz_harness_exports_sync_abi_and_allocation_stats(self) -> None:
        source = (CUFUZZ_DIR / "harness.cpp").read_text(encoding="utf-8")

        for symbol in (
            "libafl_target",
            "libafl_get_last_run_status",
            "libafl_get_last_output_size",
            "libafl_copy_last_output",
            "libafl_get_cufuzz_allocation_stats",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, source)
        self.assertIn("cuMemAlloc", source)
        self.assertIn("cuMemFree", source)
        self.assertIn("runs_started", source)
        self.assertIn("allocation_attempts", source)
        self.assertIn("allocations_succeeded", source)
        self.assertIn("runs_completed", source)

    def test_cufuzz_builder_records_honest_provenance(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CUFUZZ_DIR / "build.py"), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--phase2-dir", result.stdout)
        source = (CUFUZZ_DIR / "build.py").read_text(encoding="utf-8")
        self.assertIn('"backend": "cufuzz"', source)
        self.assertIn('"runner": "libafl"', source)
        self.assertIn('"execution_model": "per_input_allocate_launch_free"', source)
        self.assertIn('"llvm_feedback_instrumentation": "disabled"', source)
        self.assertIn('"inner_cuda_feedback": "disabled"', source)
        self.assertIn('"backend_added_inner_feedback": "disabled"', source)
        self.assertIn('"device_feedback_transport": "disabled"', source)
        self.assertNotIn('"device_feedback_observed"', source)
        self.assertNotIn('"source_embedded_feedback"', source)
        self.assertIn("validate_phase2_context_abi", source)
        self.assertIn("libphase2_cufuzz_target.so", source)
        self.assertNotIn("select_feedback_device_bitcode(", source)

    def test_builtin_pipeline_builds_cufuzz_and_origin_no_feedback(self) -> None:
        source = (CUDA_KERNEL_DIR / "builtin_phase_pipeline.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("CUFUZZ_BUILD_CLI", source)
        self.assertIn('backends_dir / "cufuzz"', source)
        self.assertIn('backends_dir / "origin-no-feedback"', source)
        self.assertIn('"--feedback-instrumentation", "disabled"', source)

    def test_origin_feedback_mode_guards_wrapper_and_host_transport(self) -> None:
        wrapper = (CUDA_KERNEL_DIR / "origin" / "wrapper.cu").read_text(
            encoding="utf-8"
        )
        harness = (CUDA_KERNEL_DIR / "origin" / "harness.cpp").read_text(
            encoding="utf-8"
        )
        builder = (CUDA_KERNEL_DIR / "origin" / "build.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("RAPID_ENABLE_INNER_FEEDBACK", wrapper)
        self.assertIn("RAPID_ENABLE_INNER_FEEDBACK", harness)
        self.assertIn(
            'f"-DRAPID_ENABLE_INNER_FEEDBACK={inner_feedback_enabled}"',
            builder,
        )
        self.assertIn('result["inner_cuda_feedback"] = feedback.mode', builder)
        self.assertIn(
            'result["backend_added_inner_feedback"] = feedback.mode', builder
        )
        self.assertIn(
            'result["device_feedback_transport"] = feedback.mode', builder
        )
        self.assertIn(
            'result["llvm_feedback_instrumentation"] = feedback.mode', builder
        )
        self.assertNotIn('result["device_feedback_observed"]', builder)
        self.assertNotIn('result["source_embedded_feedback"]', builder)

    def test_persistent_feedback_mode_guards_device_and_host_transport(self) -> None:
        runtime_sources = {
            "rapid": (
                CUDA_KERNEL_DIR / "rapid" / "wrapper.cu",
                CUDA_KERNEL_DIR / "rapid" / "harness.cpp",
            ),
            "rapid2": (
                CUDA_KERNEL_DIR / "rapid2" / "pipelined_kernel_impl.cuh",
                CUDA_KERNEL_DIR / "rapid2" / "single_kernel_pipeline.cuh",
            ),
        }

        for backend, sources in runtime_sources.items():
            with self.subTest(backend=backend):
                builder = (CUDA_KERNEL_DIR / backend / "build.py").read_text(
                    encoding="utf-8"
                )
                wrapper = (CUDA_KERNEL_DIR / backend / "wrapper.cu").read_text(
                    encoding="utf-8"
                )
                runtime = "\n".join(
                    source.read_text(encoding="utf-8") for source in sources
                )

                self.assertIn("RAPID_ENABLE_INNER_FEEDBACK", wrapper)
                self.assertIn("RAPID_ENABLE_INNER_FEEDBACK", runtime)
                self.assertIn(
                    'f"-DRAPID_ENABLE_INNER_FEEDBACK={inner_feedback_enabled}"',
                    builder,
                )
                self.assertIn(
                    'result["device_feedback_transport"] = feedback.mode', builder
                )


if __name__ == "__main__":
    unittest.main()
