import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_KERNEL_DIR = REPO_ROOT / "cuda-kernel"
if str(CUDA_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_KERNEL_DIR))

from builtin_phase_pipeline import (  # noqa: E402
    _capture_compile_command,
    _kernel_dir_from_index_entry,
)
from backend_build import (  # noqa: E402
    build_optimization_contract,
    instrument_feedback_module,
    optimize_device_bitcode,
    select_feedback_device_bitcode,
    validate_phase2_context_abi,
)


class BuiltinPhasePipelineTest(unittest.TestCase):
    def test_release_optimization_contract_is_explicit_o3(self) -> None:
        contract = build_optimization_contract("release")

        self.assertEqual(contract.profile, "release")
        self.assertEqual(contract.device_clang_flags, ("-O3", "-DNDEBUG"))
        self.assertEqual(contract.host_cxx_flags, ("-O3", "-DNDEBUG"))
        self.assertEqual(contract.llvm_opt_pipeline, "default<O3>")
        self.assertEqual(contract.llc_flags, ("-O3",))

    def test_debug_optimization_contract_is_explicit_o0(self) -> None:
        contract = build_optimization_contract("debug")

        self.assertEqual(contract.profile, "debug")
        self.assertEqual(contract.device_clang_flags, ("-O0", "-g"))
        self.assertEqual(contract.host_cxx_flags, ("-O0", "-g"))
        self.assertIsNone(contract.llvm_opt_pipeline)
        self.assertEqual(contract.llc_flags, ("-O0",))

    def test_unknown_optimization_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported build profile"):
            build_optimization_contract("fast")

    def test_release_optimizer_emits_bitcode_and_records_real_command(self) -> None:
        llvm_as = shutil.which("llvm-as-22")
        llvm_dis = shutil.which("llvm-dis-22")
        if llvm_as is None or llvm_dis is None:
            self.skipTest("LLVM 22 assembler/disassembler is unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_ll = root / "input.ll"
            source_ll.write_text(
                "define i32 @identity(i32 %value) {\n"
                "entry:\n"
                "  %copy = add i32 %value, 0\n"
                "  ret i32 %copy\n"
                "}\n",
                encoding="utf-8",
            )
            input_bc = root / "input.bc"
            output_bc = root / "optimized.bc"
            subprocess.run([llvm_as, str(source_ll), "-o", str(input_bc)], check=True)
            commands: list[list[str]] = []

            optimized = optimize_device_bitcode(
                input_bc=input_bc,
                output_bc=output_bc,
                optimization=build_optimization_contract("release"),
                command_log=commands,
            )

            self.assertEqual(optimized, output_bc)
            self.assertTrue(output_bc.is_file())
            self.assertEqual(len(commands), 2)
            self.assertIn("-passes=default<O3>", commands[0])
            self.assertEqual(commands[1][0], llvm_dis)
            output_ll = subprocess.run(
                [llvm_dis, str(output_bc), "-o", "-"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertNotIn(" add i32 %value, 0", output_ll)

    def test_release_optimizer_rejects_optnone_module(self) -> None:
        llvm_as = shutil.which("llvm-as-22")
        if llvm_as is None:
            self.skipTest("llvm-as-22 is unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_ll = root / "optnone.ll"
            source_ll.write_text(
                "; Function Attrs: noinline optnone\n"
                "define void @unoptimized() #0 {\n"
                "entry:\n"
                "  ret void\n"
                "}\n"
                "attributes #0 = { noinline optnone }\n",
                encoding="utf-8",
            )
            input_bc = root / "input.bc"
            subprocess.run([llvm_as, str(source_ll), "-o", str(input_bc)], check=True)

            with self.assertRaisesRegex(RuntimeError, "still contains optnone"):
                optimize_device_bitcode(
                    input_bc=input_bc,
                    output_bc=root / "optimized.bc",
                    optimization=build_optimization_contract("release"),
                    command_log=[],
                )

    def test_capture_command_uses_requested_cuda_source(self) -> None:
        source = REPO_ROOT / "tests" / "feedback_e2e" / "fixtures" / "feedback_kernels.cu"

        command = _capture_compile_command(
            wrapped_compiler="/tmp/clang++",
            source_path=source,
            object_path=Path("/tmp/feedback.o"),
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
        )

        self.assertIn(str(source), command)
        self.assertIn(str(source.parent), command)
        self.assertIn("-O3", command)
        self.assertIn("-DNDEBUG", command)

    def test_feedback_instrumentation_helper_rejects_wrong_entry_abi(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_spec = root / "build_spec.json"
            build_spec.write_text(
                json.dumps(
                    {
                        "entry_abi_version": 0,
                        "entry_context": "unsupported_context",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Phase2 entry ABI"):
                instrument_feedback_module(
                    linked_bc=root / "linked.bc",
                    build_spec_path=build_spec,
                    intermediates_dir=root / "intermediates",
                )

    def test_feedback_instrumentation_helper_emits_instrumented_bc_and_metadata(self) -> None:
        llvm_as = shutil.which("llvm-as-22")
        if llvm_as is None:
            self.skipTest("llvm-as-22 is unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            linked_bc = root / "linked.bc"
            fixture = REPO_ROOT / "tests" / "feedback_instrument" / "fixtures" / "branch_memory.ll"
            subprocess.run([llvm_as, str(fixture), "-o", str(linked_bc)], check=True)
            build_spec = root / "build_spec.json"
            build_spec.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kernel_id": "builder-fixture",
                        "entry_symbol": "__rapid_entry_test",
                        "entry_abi_version": 1,
                        "entry_context": "rapid_kernel_context_v1",
                        "feedback_memory_enabled": True,
                        "feedback_payload_slot_count": 1,
                        "feedback_payload_slots": [
                            {
                                "slot": 0,
                                "arg_index": 0,
                                "field_path": [],
                                "decoded_expr": "decoded.data",
                                "elem_size": 4,
                                "byte_offset": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            artifacts = instrument_feedback_module(
                linked_bc=linked_bc,
                build_spec_path=build_spec,
                intermediates_dir=root / "intermediates",
            )

            self.assertEqual(artifacts.instrumented_bc.name, "instrumented.bc")
            self.assertEqual(artifacts.metadata.name, "feedback_metadata.json")
            self.assertTrue(artifacts.instrumented_bc.is_file())
            metadata = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
            self.assertGreater(metadata["instrumented_cfg_sites"], 0)

    def test_disabled_feedback_instrumentation_selects_linked_bitcode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            linked_bc = root / "linked.bc"
            linked_bc.write_bytes(b"placeholder")
            build_spec = root / "build_spec.json"
            build_spec.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entry_abi_version": 1,
                        "entry_context": "rapid_kernel_context_v1",
                        "feedback_payload_slots": [],
                    }
                ),
                encoding="utf-8",
            )

            selected = select_feedback_device_bitcode(
                linked_bc=linked_bc,
                build_spec_path=build_spec,
                intermediates_dir=root / "intermediates",
                mode="disabled",
            )

            self.assertEqual(selected.mode, "disabled")
            self.assertEqual(selected.selected_bc, linked_bc)
            self.assertIsNone(selected.instrumented_bc)
            self.assertIsNone(selected.metadata)
            self.assertFalse((root / "intermediates" / "feedback_metadata.json").exists())

    def test_phase2_context_abi_rejects_stale_build_spec(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            build_spec_path = Path(td) / "build_spec.json"
            build_spec_path.write_text(
                json.dumps(
                    {
                        "entry_abi_version": 0,
                        "entry_context": "legacy",
                        "feedback_payload_slots": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "rapid_kernel_context_v1",
            ):
                validate_phase2_context_abi(build_spec_path)

    def test_all_backend_builders_support_feedback_instrumentation_mode(self) -> None:
        for backend in ("origin", "rapid", "rapid2"):
            with self.subTest(backend=backend):
                build_script = CUDA_KERNEL_DIR / backend / "build.py"
                source = build_script.read_text(encoding="utf-8")
                self.assertIn("select_feedback_device_bitcode(", source)
                self.assertIn("add_feedback_instrumentation_argument(parser)", source)
                self.assertIn("feedback.selected_bc", source)
                self.assertIn("add_feedback_build_metadata(result, feedback)", source)
                self.assertIn('build_spec.get("target_layout_header"', source)
                self.assertNotIn(
                    '"-include", str(CUDA_KERNEL_DIR / "utils" / "feedback" / "feedback.cuh")',
                    source,
                )
                help_result = subprocess.run(
                    [sys.executable, str(build_script), "--help"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn("--feedback-instrumentation", help_result.stdout)

    def test_backend_sources_include_generated_layout_before_feedback_context(self) -> None:
        for backend in ("origin", "rapid", "rapid2"):
            with self.subTest(backend=backend):
                wrapper = (CUDA_KERNEL_DIR / backend / "wrapper.cu").read_text(
                    encoding="utf-8"
                )
                harness = (CUDA_KERNEL_DIR / backend / "harness.cpp").read_text(
                    encoding="utf-8"
                )
                wrapper_feedback_include = (
                    wrapper.index("feedback/")
                    if "feedback/" in wrapper
                    else wrapper.index("#include FUZZER_INVOKE_HEADER")
                )
                self.assertLess(
                    wrapper.index('#include "rapid_target_layout.v1.h"'),
                    wrapper_feedback_include,
                )
                self.assertLess(
                    harness.index('#include "rapid_target_layout.v1.h"'),
                    harness.index('#include "kernel_backend.cuh"')
                    if backend == "rapid2"
                    else harness.index("feedback/feedback_context.cuh"),
                )

    def test_index_relative_kernel_dir_is_resolved_against_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "out" / "builtin-kernel"
            resolved = _kernel_dir_from_index_entry(
                run_dir,
                {"dir": "kernels/vulnerable_kernel_entry__676e8c0d"},
            )

        self.assertEqual(
            resolved,
            run_dir / "kernels" / "vulnerable_kernel_entry__676e8c0d",
        )

    def test_index_absolute_kernel_dir_is_preserved(self) -> None:
        absolute = Path("/tmp/rapid/kernels/kernel0")

        self.assertEqual(_kernel_dir_from_index_entry(Path("/unused"), {"dir": str(absolute)}), absolute)


if __name__ == "__main__":
    unittest.main()
