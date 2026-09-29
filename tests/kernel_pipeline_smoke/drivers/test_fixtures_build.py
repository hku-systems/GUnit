import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "kernel_pipeline_smoke" / "fixtures"
DRIVERS_DIR = Path(__file__).resolve().parent
if str(DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVERS_DIR))

from shared_e2e import cuda_device_skip_reason, get_capture_e2e_context  # noqa: E402


def _has_cuda_compiler() -> bool:
    return shutil.which("clang++") is not None or shutil.which("nvcc") is not None


def _has_nm() -> bool:
    return shutil.which("llvm-nm") is not None or shutil.which("nm") is not None


def _nm_cmd() -> str:
    return shutil.which("llvm-nm") or shutil.which("nm") or "nm"


class FixturesMakefileTest(unittest.TestCase):
    """Validate the fixtures-level Makefile and its build targets."""

    def test_makefile_exists(self) -> None:
        self.assertTrue(
            (FIXTURES_DIR / "Makefile").is_file(),
            "fixtures/Makefile missing",
        )

    def test_makefile_declares_expected_targets(self) -> None:
        """Makefile must declare all, compile_fixtures, run, clean targets."""
        text = (FIXTURES_DIR / "Makefile").read_text(encoding="utf-8")
        for target in ("all:", "compile_fixtures:", "run:", "clean:"):
            with self.subTest(target=target):
                self.assertIn(target, text, f"target '{target}' not found")

    def test_no_rdc_flags(self) -> None:
        """Makefile must NOT enable relocatable device code."""
        text = (FIXTURES_DIR / "Makefile").read_text(encoding="utf-8")
        self.assertNotIn("-rdc=true", text)
        self.assertNotIn(" -dc ", text)
        self.assertNotIn("--relocatable-device-code", text)

    def test_fixture_api_header_exists(self) -> None:
        """fixture_api.cuh must exist and declare wrapper functions."""
        hdr = FIXTURES_DIR / "fixture_api.cuh"
        self.assertTrue(hdr.is_file(), "fixture_api.cuh missing")
        text = hdr.read_text(encoding="utf-8")
        for fn in (
            "test_direct_kernel",
            "test_macro_defined_kernel",
            "test_macro_decl_kernel",
            "test_templated_kernel",
            "test_record_ptr_copy",
            "test_clamped_write",
            "test_scaled_write",
            "test_bitwise_blend",
            "test_reduce_pair",
            "test_add_kernel",
            "test_mul_kernel",
            "test_clamped_add",
            "test_fused_mul_add",
            "test_ns_add",
            "test_ns_scale",
            "test_cuda_api_stress",
            "test_cublas_host_api_smoke",
        ):
            self.assertIn(fn, text, f"wrapper '{fn}' not declared in fixture_api.cuh")

    def test_fixture_runner_source_exists(self) -> None:
        """fixture_runner.cu must exist."""
        self.assertTrue(
            (FIXTURES_DIR / "fixture_runner.cu").is_file(),
            "fixture_runner.cu missing",
        )

    @unittest.skipUnless(_has_cuda_compiler(), "no CUDA compiler (clang++/nvcc)")
    def test_all_build_contains_fixture_objects(self) -> None:
        """Shared E2E all-build should include fixture object outputs."""
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        build_dir = ctx.fixtures_build_dir
        for obj in ("complex_kernels.o", "math_kernels.o", "ns_kernels.o", "cublas_host_api_smoke.o"):
            artifact = build_dir / obj
            self.assertTrue(
                artifact.is_file(),
                f"Expected artifact {artifact} not found",
            )

    @unittest.skipUnless(_has_cuda_compiler(), "no CUDA compiler (clang++/nvcc)")
    def test_all_build_contains_lib_only_outputs(self) -> None:
        """all target should also produce lib_only .so/.a artifacts."""
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        build_dir = ctx.fixtures_build_dir
        self.assertTrue((build_dir / "libonly_kernels.so").is_file())
        self.assertTrue((build_dir / "libonly_kernels.a").is_file())

    @unittest.skipUnless(_has_cuda_compiler(), "no CUDA compiler (clang++/nvcc)")
    def test_all_build_produces_fixtures_runner(self) -> None:
        """Shared E2E all-build should produce fixtures_runner."""
        try:
            get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        runner = FIXTURES_DIR / "fixtures_runner"
        self.assertTrue(
            runner.is_file(),
            "fixtures_runner binary not produced by make all",
        )

    @unittest.skipUnless(_has_cuda_compiler(), "no CUDA compiler (clang++/nvcc)")
    def test_run_fixtures_runner(self) -> None:
        """Run fixtures_runner built by shared all-build."""
        if reason := cuda_device_skip_reason():
            self.skipTest(reason)
        try:
            get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        runner = FIXTURES_DIR / "fixtures_runner"
        if not runner.is_file():
            self.skipTest("fixtures_runner not built")

        result = subprocess.run(
            [str(runner)],
            capture_output=True,
            timeout=60,
        )
        stdout = result.stdout.decode()
        self.assertEqual(
            result.returncode,
            0,
            f"fixtures_runner exited with {result.returncode}:\n"
            f"stdout={stdout}\nstderr={result.stderr.decode()}",
        )
        self.assertIn(
            "PASS",
            stdout,
            f"stdout does not contain PASS:\n{stdout}",
        )

    @unittest.skipUnless(
        _has_cuda_compiler() and _has_nm(),
        "no CUDA compiler or nm/llvm-nm",
    )
    def test_symbol_check_on_objects(self) -> None:
        """Use nm on objects from shared all-build to confirm wrapper symbols."""
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        nm = _nm_cmd()
        # Check that wrapper symbols appear in object files.
        # extern "C" wrappers should appear unmangled.
        build_dir = ctx.fixtures_build_dir
        complex_obj = build_dir / "complex_kernels.o"
        math_obj = build_dir / "math_kernels.o"

        complex_wrappers = [
            "test_direct_kernel",
            "test_macro_defined_kernel",
            "test_macro_decl_kernel",
            "test_templated_kernel",
            "test_record_ptr_copy",
            "test_clamped_write",
            "test_scaled_write",
            "test_bitwise_blend",
            "test_reduce_pair",
        ]
        math_wrappers = [
            "test_add_kernel",
            "test_mul_kernel",
            "test_clamped_add",
            "test_fused_mul_add",
        ]
        ns_wrappers = [
            "test_ns_add",
            "test_ns_scale",
            "test_cuda_api_stress",
        ]
        cublas_wrappers = [
            "test_cublas_host_api_smoke",
        ]

        checked = 0
        for obj_path, wrappers in [
            (complex_obj, complex_wrappers),
            (math_obj, math_wrappers),
            (build_dir / "ns_kernels.o", ns_wrappers),
            (build_dir / "cublas_host_api_smoke.o", cublas_wrappers),
        ]:
            if not obj_path.is_file():
                continue
            result = subprocess.run(
                [nm, str(obj_path)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                # nm may fail on fat objects; skip gracefully
                continue
            nm_output = result.stdout.decode()
            for sym in wrappers:
                with self.subTest(obj=obj_path.name, symbol=sym):
                    self.assertIn(
                        sym,
                        nm_output,
                        f"Symbol '{sym}' not found in {obj_path.name}",
                    )
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
