import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_UTILS_DIR = REPO_ROOT / "cuda-kernel" / "utils"


class FeedbackHeadersTest(unittest.TestCase):
    def compile_cpp(self, source: str, *, payload_slot_count: int = 1) -> None:
        compiler = shutil.which("clang++-22") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("clang++ is unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "feedback_layout.cpp"
            (tmp_path / "rapid_target_layout.v1.h").write_text(
                "#define RAPID_PAYLOAD_SLOT_COUNT "
                f"{payload_slot_count}u\n"
                "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT "
                "(RAPID_PAYLOAD_SLOT_COUNT == 0u ? 1u : "
                "RAPID_PAYLOAD_SLOT_COUNT)\n",
                encoding="utf-8",
            )
            source_path.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-fsyntax-only",
                    "-I",
                    str(tmp_path),
                    "-I",
                    str(CUDA_UTILS_DIR),
                    str(source_path),
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

    def compile_cuda(
        self,
        source: str,
        *,
        payload_slot_count: int = 1,
        cuda_arch: str = "sm_86",
    ) -> None:
        compiler = shutil.which("clang++-22") or shutil.which("clang++")
        cuda_path = Path("/usr/local/cuda")
        if compiler is None or not (cuda_path / "include").is_dir():
            self.skipTest("clang++ or CUDA headers are unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "feedback_device.cu"
            output_path = tmp_path / "feedback_device.bc"
            (tmp_path / "rapid_target_layout.v1.h").write_text(
                "#define RAPID_PAYLOAD_SLOT_COUNT "
                f"{payload_slot_count}u\n"
                "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT "
                "(RAPID_PAYLOAD_SLOT_COUNT == 0u ? 1u : "
                "RAPID_PAYLOAD_SLOT_COUNT)\n",
                encoding="utf-8",
            )
            source_path.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    compiler,
                    "-x",
                    "cuda",
                    "--cuda-device-only",
                    f"--cuda-path={cuda_path}",
                    f"--cuda-gpu-arch={cuda_arch}",
                    "-emit-llvm",
                    "-c",
                    "-I",
                    str(tmp_path),
                    "-I",
                    str(REPO_ROOT / "cuda-kernel"),
                    "-I",
                    str(CUDA_UTILS_DIR),
                    str(source_path),
                    "-o",
                    str(output_path),
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

    def test_feedback_constants_and_context_layout_compile(self) -> None:
        source = r'''
#include "rapid_target_layout.v1.h"
#include "feedback/feedback_context.cuh"

int main() { return 0; }
'''
        for payload_slot_count in (0, 1, 3):
            with self.subTest(payload_slot_count=payload_slot_count):
                self.compile_cpp(
                    source,
                    payload_slot_count=payload_slot_count,
                )

        header = (CUDA_UTILS_DIR / "feedback" / "feedback_context.cuh").read_text(
            encoding="utf-8"
        )
        for declaration in (
            "uint32_t elem_size;",
            "uint16_t arg_slot;",
            "uint16_t flags;",
            "uint32_t reserved;",
        ):
            self.assertNotIn(declaration, header)
        for field in (
            "uint32_t grid_x;",
            "uint32_t grid_y;",
            "uint32_t grid_z;",
            "uint32_t block_x;",
            "uint32_t block_y;",
            "uint32_t block_z;",
        ):
            self.assertIn(field, header)

    def test_edge_helper_uses_set_like_atomic_or(self) -> None:
        coverage_path = CUDA_UTILS_DIR / "coverage" / "coverage.cuh"
        coverage_text = coverage_path.read_text(encoding="utf-8")
        helper_path = CUDA_UTILS_DIR / "coverage" / "coverage_helper.cuh"
        helper_text = helper_path.read_text(encoding="utf-8")

        self.assertIn("atomicOr", helper_text)
        self.assertNotIn("atomicAdd", helper_text)
        self.assertNotIn("_index < MAP_SIZE", helper_text)
        self.assertNotIn("extern __shared__ uint8_t shared_cov_map", coverage_text)
        self.assertIn("rapid_bind_coverage_map", helper_text)
        self.assertIn("rapid_current_coverage_map", helper_text)
        self.assertIn("__libafl_edge(_index, rapid_current_coverage_map())", coverage_text)

    def test_feedback_globals_exports_aligned_bitset(self) -> None:
        globals_path = CUDA_UTILS_DIR / "feedback" / "feedback_globals.cpp"
        globals_text = globals_path.read_text(encoding="utf-8")

        self.assertIn('extern "C"', globals_text)
        self.assertIn("visibility(\"default\")", globals_text)
        self.assertIn("aligned(4096)", globals_text)
        self.assertIn("libafl_simt_memcov_bits[SIMT_MEMCOV_STORAGE_SIZE]", globals_text)

    def test_feedback_device_helpers_compile(self) -> None:
        source = r'''
#define RAPID_DEFINE_DEVICE_COVERAGE_MAP 1
#include "rapid_target_layout.v1.h"
#include "feedback/feedback.cuh"

extern "C" __global__ void feedback_compile_smoke(
    RapidKernelContext *context, const uint8_t *ptr) {
  rapid_bind_coverage_map(device_cov_map);
  rapid_feedback_clear_task_maps(context);
  __rapid_feedback_bb(7u);
  __rapid_feedback_mem(context, 11u, 0u, RAPID_FEEDBACK_READ, 4u, ptr);
  rapid_feedback_record_thread_activity(context);
  merge_shared_coverage();
}
'''
        for cuda_arch in ("sm_75", "sm_86"):
            with self.subTest(cuda_arch=cuda_arch):
                self.compile_cuda(source, cuda_arch=cuda_arch)

    def test_task_envelope_layout_compiles_as_shared_pod(self) -> None:
        source = r'''
#include "rapid_target_layout.v1.h"
#include "input/task_envelope.cuh"

#include <cstddef>
#include <type_traits>

static_assert(sizeof(RapidTaskEnvelopeHeader) == 32u);
static_assert(offsetof(RapidTaskEnvelopeHeader, payload_size) == 24u);
static_assert(std::is_standard_layout_v<RapidTaskEnvelopeHeader>);
static_assert(std::is_trivially_copyable_v<RapidTaskEnvelopeHeader>);

int main() {
  RapidTaskEnvelopeHeader envelope{};
  return rapid_task_payload(&envelope) ==
                 reinterpret_cast<uint8_t *>(&envelope + 1)
             ? 0
             : 1;
}
'''
        self.compile_cpp(source)


if __name__ == "__main__":
    unittest.main()
