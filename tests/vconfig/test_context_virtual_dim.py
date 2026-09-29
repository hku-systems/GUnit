from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_HELPER = REPO_ROOT / "tools/rapid-vconfig-instrument/build.py"


BASE_MODULE = r'''
target triple = "nvptx64-nvidia-cuda"

define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  %logical_block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %logical_grid = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %combined = add i32 %logical_block, %logical_grid
  store i32 %combined, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
'''


class ContextVirtualDimTest(unittest.TestCase):
    _build_dir: tempfile.TemporaryDirectory[str] | None = None
    _tool: Path | None = None
    _plugin: Path | None = None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._build_dir is not None:
            cls._build_dir.cleanup()

    def tool(self) -> Path:
        self.assertTrue(
            BUILD_HELPER.is_file(),
            f"VConfig build helper does not exist: {BUILD_HELPER}",
        )
        if self.__class__._tool is None:
            self.__class__._build_dir = tempfile.TemporaryDirectory(
                prefix="rapid-vconfig-tool-"
            )
            tool = Path(self.__class__._build_dir.name) / "rapid-vconfig-instrument"
            subprocess.run(
                [sys.executable, str(BUILD_HELPER), "--output", str(tool)],
                check=True,
                cwd=REPO_ROOT,
            )
            self.__class__._tool = tool
        return self.__class__._tool

    def plugin(self) -> Path:
        self.assertTrue(
            BUILD_HELPER.is_file(),
            f"VConfig build helper does not exist: {BUILD_HELPER}",
        )
        if self.__class__._plugin is None:
            if self.__class__._build_dir is None:
                self.__class__._build_dir = tempfile.TemporaryDirectory(
                    prefix="rapid-vconfig-tool-"
                )
            root = Path(self.__class__._build_dir.name)
            tool = root / "rapid-vconfig-instrument"
            plugin = root / "rapid-vconfig-pass-plugin.so"
            subprocess.run(
                [
                    sys.executable,
                    str(BUILD_HELPER),
                    "--output",
                    str(tool),
                    "--plugin-output",
                    str(plugin),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            self.__class__._tool = tool
            self.__class__._plugin = plugin
        return self.__class__._plugin

    def assemble(self, directory: Path, name: str, source: str) -> Path:
        llvm_ir = directory / f"{name}.ll"
        bitcode = directory / f"{name}.bc"
        llvm_ir.write_text(source, encoding="utf-8")
        subprocess.run(
            ["llvm-as-22", str(llvm_ir), "-o", str(bitcode)],
            check=True,
            cwd=REPO_ROOT,
        )
        return bitcode

    def disassemble(self, bitcode: Path) -> str:
        completed = subprocess.run(
            ["llvm-dis-22", str(bitcode), "-o", "-"],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    def transform(
        self,
        directory: Path,
        source: str,
        *,
        entry_symbol: str = "__rapid_entry__kernel",
        name: str = "input",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        input_bitcode = self.assemble(directory, name, source)
        output_bitcode = directory / f"{name}.out.bc"
        completed = subprocess.run(
            [
                str(self.tool()),
                "--input",
                str(input_bitcode),
                "--output",
                str(output_bitcode),
                "--entry-symbol",
                entry_symbol,
            ],
            check=False,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return completed, output_bitcode

    def test_rewrites_dimensions_from_rapid_context(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), BASE_MODULE)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            self.assertIn(
                "%vdim.config = select i1", transformed
            )
            self.assertIn("getelementptr i8, ptr %vdim.config, i64 12", transformed)
            self.assertIn("vdim.active", transformed)
            self.assertIn("vdim.inactive", transformed)
            self.assertNotIn(
                "%logical_block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x",
                transformed,
            )

    def test_out_of_tree_plugin_rewrites_rapid_context_function(self) -> None:
        source = BASE_MODULE.replace(
            "define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {",
            'define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) #0 {',
        ).replace(
            "\ndeclare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()",
            '\nattributes #0 = { "rapid.vconfig.context" }\n\n'
            "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()",
        )
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            directory = Path(temp_dir)
            input_bitcode = self.assemble(directory, "plugin_input", source)
            output_bitcode = directory / "plugin_output.bc"

            completed = subprocess.run(
                [
                    "opt-22",
                    "-load-pass-plugin",
                    str(self.plugin()),
                    "-passes=rapid-vconfig",
                    str(input_bitcode),
                    "-o",
                    str(output_bitcode),
                ],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output_bitcode)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            self.assertIn("vdim.active", transformed)

    def test_rewrites_thread_and_block_indices_from_linear_physical_ids(self) -> None:
        index_module = r'''
target triple = "nvptx64-nvidia-cuda"

define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  %tx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %tz = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %bx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %bz = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %sum0 = add i32 %tx, %ty
  %sum1 = add i32 %sum0, %tz
  %sum2 = add i32 %sum1, %bx
  %sum3 = add i32 %sum2, %by
  %sum4 = add i32 %sum3, %bz
  store i32 %sum4, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), index_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn("vdim.thread.y", transformed)
            self.assertIn("vdim.block.y", transformed)
            self.assertIn("vdim.thread.linear", transformed)
            self.assertIn("vdim.block.linear", transformed)
            self.assertNotIn(
                "%ty = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()",
                transformed,
            )
            self.assertNotIn(
                "%by = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()",
                transformed,
            )

    def test_strengthens_shared_global_alignment_for_wide_access(self) -> None:
        shared_module = r'''
target triple = "nvptx64-nvidia-cuda"

@shared_values = external addrspace(3) global [16 x float], align 4

define void @__rapid_entry__kernel(ptr %source, ptr %__rapid_context) {
entry:
  %shared = addrspacecast ptr addrspace(3) @shared_values to ptr
  call void @llvm.memcpy.p0.p0.i64(
      ptr align 16 %shared, ptr align 16 %source, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), shared_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn(
                "@shared_values = external addrspace(3) global [16 x float], align 16",
                transformed,
            )

    def test_second_transform_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            directory = Path(temp_dir)
            first, first_output = self.transform(directory, BASE_MODULE, name="first")
            self.assertEqual(first.returncode, 0, first.stderr)

            second_output = directory / "second.out.bc"
            second = subprocess.run(
                [
                    str(self.tool()),
                    "--input",
                    str(first_output),
                    "--output",
                    str(second_output),
                    "--entry-symbol",
                    "__rapid_entry__kernel",
                ],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(second.returncode, 0, second.stderr)
            first_ir = self.disassemble(first_output).splitlines()[1:]
            second_ir = self.disassemble(second_output).splitlines()[1:]
            self.assertEqual(first_ir, second_ir)

    def test_missing_entry_reports_stable_reason(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(
                Path(temp_dir), BASE_MODULE, entry_symbol="missing_entry"
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_entry_missing", completed.stderr)
            self.assertFalse(output.exists())

    def test_invalid_entry_abi_reports_stable_reason(self) -> None:
        invalid_module = r'''
define void @__rapid_entry__kernel(ptr %out, i32 %not_a_context) {
entry:
  ret void
}
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), invalid_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_entry_abi_invalid", completed.stderr)
            self.assertFalse(output.exists())

    def test_straight_line_barrier_inserts_virtual_sync_before_inactive_return(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  %logical_block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  call void @llvm.nvvm.barrier0()
  store i32 %logical_block, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            inactive_block = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn(
                "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)",
                inactive_block,
            )
            self.assertIn("ret void", inactive_block)

    def test_main_path_barrier_after_unconditional_branch_inserts_virtual_sync(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  br label %body

body:
  %logical_block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  call void @llvm.nvvm.barrier0()
  store i32 %logical_block, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_block = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn(
                "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)",
                inactive_block,
            )

    def test_reconverged_barrier_inserts_virtual_sync(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %does_work = icmp ult i32 %tid, 8
  br i1 %does_work, label %work, label %sync

work:
  store i32 %tid, ptr %out, align 4
  br label %sync

sync:
  call void @llvm.nvvm.barrier0()
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_block = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_block.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                1,
            )

    def test_uniform_early_return_guards_only_the_barrier_region(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i32 %limit, ptr %__rapid_context) {
entry:
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %has_work = icmp ult i32 %block, %limit
  br i1 %has_work, label %body, label %done

body:
  call void @llvm.nvvm.barrier0()
  br label %done

done:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            prefix = transformed.split("vdim.active:")[0]
            self.assertIn("br i1 %has_work", prefix)
            inactive_block = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_block.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                1,
            )

    def test_divergent_barrier_reports_stable_reason(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %is_zero = icmp eq i32 %tid, 0
  br i1 %is_zero, label %needs_barrier, label %done

needs_barrier:
  call void @llvm.nvvm.barrier0()
  br label %done

done:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_barrier_unsupported", completed.stderr)
            self.assertFalse(output.exists())

    def test_fixed_trip_loop_barrier_wraps_inactive_sync_loop(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.nvvm.barrier0()
  %next = add i32 %i, 1
  %keep_going = icmp ult i32 %next, 4
  br i1 %keep_going, label %loop, label %done

done:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            self.assertIn("vdim.inactive.sync.loop", transformed)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn(
                "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)",
                inactive_region,
            )
            self.assertIn("icmp ult i32 %vdim.inactive.sync.next, 4", inactive_region)

    def test_fixed_trip_multiblock_loop_emits_all_inactive_barriers(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next, %sync]
  %does_work = icmp ult i32 %i, 2
  br i1 %does_work, label %work, label %sync

work:
  store i32 %i, ptr %out, align 4
  br label %sync

sync:
  call void @llvm.nvvm.barrier0()
  %next = add i32 %i, 1
  %keep_going = icmp ult i32 %next, 4
  br i1 %keep_going, label %loop, label %done

done:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_block = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_block.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                4,
            )

    def test_fixed_trip_latch_can_compare_current_induction_value(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  br label %sync

sync:
  call void @llvm.nvvm.barrier0()
  br label %latch

latch:
  %next = add i32 %i, 1
  %keep_going = icmp ult i32 %i, 3
  br i1 %keep_going, label %loop, label %done

done:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_block = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_block.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                4,
            )

    def test_fixed_two_trip_boolean_loop_emits_two_inactive_barriers(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  br label %loop

loop:
  %first = phi i1 [true, %entry], [false, %latch]
  br label %sync

sync:
  call void @llvm.nvvm.barrier0()
  br label %latch

latch:
  br i1 %first, label %loop, label %done

done:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_block = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_block.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                2,
            )

    def test_uniform_dynamic_loop_emits_sync_only_inactive_loop(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i32 %rounds, ptr %__rapid_context) {
entry:
  %bound = call i32 @llvm.smin.i32(i32 %rounds, i32 8)
  %has_round = icmp sgt i32 %bound, 0
  br i1 %has_round, label %preheader, label %after

preheader:
  br label %loop

loop:
  %i = phi i32 [0, %preheader], [%next, %loop]
  call void @llvm.nvvm.barrier0()
  %next = add nuw nsw i32 %i, 1
  %keep_going = icmp slt i32 %next, %bound
  br i1 %keep_going, label %loop, label %after

after:
  call void @llvm.nvvm.barrier0()
  ret void
}

declare i32 @llvm.smin.i32(i32, i32)
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn("vdim.inactive.sync.dynamic.header", inactive_region)
            self.assertIn("call i32 @llvm.smin.i32", inactive_region)
            self.assertIn("icmp slt i32", inactive_region)
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                2,
            )

    def test_parameter_start_eq_exit_dynamic_loop_replays_twenty_one_barriers(
        self,
    ) -> None:
        header_barriers = "\n".join(
            "  call void @llvm.nvvm.barrier0()" for _ in range(16)
        )
        latch_barriers = "\n".join(
            "  call void @llvm.nvvm.barrier0()" for _ in range(5)
        )
        barrier_module = rf'''
define void @__rapid_entry__kernel(i32 %start, i32 %rounds, ptr %__rapid_context) {{
entry:
  %bound = add i32 %start, %rounds
  %has_round = icmp ult i32 %start, %bound
  br i1 %has_round, label %preheader, label %after

preheader:
  br label %loop

loop:
  %i = phi i32 [%start, %preheader], [%next, %latch]
{header_barriers}
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %takes_side_path = icmp eq i32 %tid, 0
  br i1 %takes_side_path, label %side, label %latch

side:
  br label %latch

latch:
{latch_barriers}
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %bound
  br i1 %done, label %after, label %loop

after:
  ret void
}}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn("vdim.inactive.sync.dynamic.header", inactive_region)
            self.assertIn(
                "%vdim.inactive.sync.dynamic.i = phi i32 [ %start,",
                inactive_region,
            )
            self.assertIn(
                "%vdim.inactive.sync.dynamic.keep_going = icmp eq i32",
                inactive_region,
            )
            self.assertIn(
                "br i1 %vdim.inactive.sync.dynamic.keep_going, "
                "label %vdim.inactive.sync.dynamic.after, "
                "label %vdim.inactive.sync.dynamic.loop",
                inactive_region,
            )
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                21,
            )

    def test_shift_dynamic_loop_replays_inactive_barriers(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i64 %n, ptr %__rapid_context) {
entry:
  %bound = lshr i64 %n, 1
  %has_round = icmp ugt i64 %n, 5
  br i1 %has_round, label %preheader, label %after

preheader:
  br label %loop

loop:
  %num_groups = phi i64 [2, %preheader], [%next, %loop]
  call void @llvm.nvvm.barrier0()
  %next = shl i64 %num_groups, 1
  %keep_going = icmp ult i64 %next, %bound
  br i1 %keep_going, label %loop, label %after

after:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn("vdim.inactive.sync.dynamic.header", inactive_region)
            self.assertIn(
                "%vdim.inactive.sync.dynamic.next = shl i64 "
                "%vdim.inactive.sync.dynamic.i, 1",
                inactive_region,
            )
            self.assertIn(
                "%vdim.inactive.sync.dynamic.keep_going = icmp ult i64",
                inactive_region,
            )
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                1,
            )

    def test_multiply_dynamic_loop_replays_inactive_barriers(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i64 %n, ptr %__rapid_context) {
entry:
  %bound = lshr i64 %n, 1
  %has_round = icmp ugt i64 %n, 5
  br i1 %has_round, label %preheader, label %after

preheader:
  br label %loop

loop:
  %num_groups = phi i64 [2, %preheader], [%next, %loop]
  call void @llvm.nvvm.barrier0()
  %next = mul i64 %num_groups, 2
  %keep_going = icmp ult i64 %next, %bound
  br i1 %keep_going, label %loop, label %after

after:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn("vdim.inactive.sync.dynamic.header", inactive_region)
            self.assertIn(
                "%vdim.inactive.sync.dynamic.next = mul i64 "
                "%vdim.inactive.sync.dynamic.i, 2",
                inactive_region,
            )
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                1,
            )

    def test_entry_guard_shift_loop_replays_pre_and_loop_barriers(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i64 %n, ptr %__rapid_context) {
entry:
  call void @llvm.nvvm.barrier0()
  %bound = lshr i64 %n, 1
  %has_round = icmp ugt i64 %n, 5
  br i1 %has_round, label %loop, label %after

loop:
  %num_groups = phi i64 [2, %entry], [%next, %loop]
  call void @llvm.nvvm.barrier0()
  %next = shl i64 %num_groups, 1
  %keep_going = icmp ult i64 %next, %bound
  br i1 %keep_going, label %loop, label %after

after:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertIn("vdim.inactive.sync.dynamic.header", inactive_region)
            self.assertIn(
                "%vdim.inactive.sync.dynamic.next = shl i64 "
                "%vdim.inactive.sync.dynamic.i, 1",
                inactive_region,
            )
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                2,
            )

    def test_parameter_start_dynamic_loop_rejects_nonuniform_barrier_path(
        self,
    ) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i32 %start, i32 %rounds, ptr %__rapid_context) {
entry:
  %bound = add i32 %start, %rounds
  %has_round = icmp ult i32 %start, %bound
  br i1 %has_round, label %preheader, label %after

preheader:
  br label %loop

loop:
  %i = phi i32 [%start, %preheader], [%next, %latch]
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %is_zero = icmp eq i32 %tid, 0
  br i1 %is_zero, label %needs_barrier, label %latch

needs_barrier:
  call void @llvm.nvvm.barrier0()
  br label %latch

latch:
  call void @llvm.nvvm.barrier0()
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %bound
  br i1 %done, label %after, label %loop

after:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_barrier_unsupported", completed.stderr)
            self.assertFalse(output.exists())

    def test_optnone_fixed_trip_loop_barrier_wraps_inactive_sync_loop(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  %i.addr = alloca i32, align 4
  store i32 0, ptr %i.addr, align 4
  br label %loop.header

loop.header:
  %i.current = load i32, ptr %i.addr, align 4
  %keep_looping = icmp ult i32 %i.current, 4
  br i1 %keep_looping, label %loop.body, label %after.loop

loop.body:
  call void @llvm.nvvm.barrier0()
  br label %loop.latch

loop.latch:
  %i.before.inc = load i32, ptr %i.addr, align 4
  %i.next = add i32 %i.before.inc, 1
  store i32 %i.next, ptr %i.addr, align 4
  br label %loop.header

after.loop:
  call void @llvm.nvvm.barrier0()
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            self.assertIn("vdim.inactive.sync.loop", transformed)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                2,
            )
            self.assertIn("icmp ult i32 %vdim.inactive.sync.next, 4", inactive_region)

    def test_barrier_in_acyclic_helper_replays_inactive_barrier(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  call void @device_helper()
  ret void
}

define internal void @device_helper() {
entry:
  call void @llvm.nvvm.barrier0()
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            self.assertIn('"rapid.vconfig.warp_aligned"', transformed)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                1,
            )

    def test_nested_acyclic_helpers_compose_barrier_summaries(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  call void @outer_helper()
  ret void
}

define internal void @outer_helper() {
entry:
  call void @inner_helper()
  call void @inner_helper()
  ret void
}

define internal void @inner_helper() {
entry:
  call void @llvm.nvvm.barrier0()
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                2,
            )

    def test_equal_helper_return_paths_share_one_barrier_summary(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i1 %choose_left, ptr %__rapid_context) {
entry:
  call void @device_helper(i1 %choose_left)
  ret void
}

define internal void @device_helper(i1 %choose_left) {
entry:
  br i1 %choose_left, label %left, label %right

left:
  call void @llvm.nvvm.barrier0()
  ret void

right:
  call void @llvm.nvvm.barrier0()
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            inactive_region = transformed.split("vdim.inactive:")[1].split(
                "vdim.active:"
            )[0]
            self.assertEqual(
                inactive_region.count(
                    "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"
                ),
                1,
            )

    def test_unequal_helper_return_paths_report_summary_mismatch(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(i1 %needs_barrier, ptr %__rapid_context) {
entry:
  call void @device_helper(i1 %needs_barrier)
  ret void
}

define internal void @device_helper(i1 %needs_barrier) {
entry:
  br i1 %needs_barrier, label %sync, label %done

sync:
  call void @llvm.nvvm.barrier0()
  ret void

done:
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_barrier_unsupported", completed.stderr)
            self.assertIn("barrier path summary mismatch", completed.stderr)
            self.assertFalse(output.exists())

    def test_recursive_helper_reports_stable_reason(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %__rapid_context) {
entry:
  call void @recursive_helper()
  ret void
}

define internal void @recursive_helper() {
entry:
  call void @recursive_helper()
  call void @llvm.nvvm.barrier0()
  ret void
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_barrier_unsupported", completed.stderr)
            self.assertFalse(output.exists())

    def test_indirect_helper_call_reports_stable_reason(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %helper, ptr %__rapid_context) {
entry:
  call void %helper()
  ret void
}
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_barrier_unsupported", completed.stderr)
            self.assertFalse(output.exists())

    def test_helper_loop_with_barrier_reports_stable_reason(self) -> None:
        barrier_module = r'''
define void @__rapid_entry__kernel(ptr %__rapid_context) {
entry:
  call void @looping_helper()
  ret void
}

define internal void @looping_helper() {
entry:
  br label %loop

loop:
  call void @llvm.nvvm.barrier0()
  br label %loop
}

declare void @llvm.nvvm.barrier0()
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), barrier_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_barrier_unsupported", completed.stderr)
            self.assertFalse(output.exists())

    def test_warp_collective_preserves_multidimensional_logical_shape(self) -> None:
        warp_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  %block_x = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %block_y = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %shape = add i32 %block_x, %block_y
  %value = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 0, i32 %shape, i32 31, i32 -1)
  store i32 %value, ptr %out, align 4
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), warp_module)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            transformed = self.disassemble(output)
            self.assertIn('"rapid.vconfig.processed"', transformed)
            self.assertIn("%vdim.block.x = load i32", transformed)
            self.assertIn("%vdim.block.y = load i32", transformed)
            self.assertNotIn("vdim.block.x.warp", transformed)
            self.assertIn(
                "call i32 @llvm.nvvm.shfl.sync.down.i32",
                transformed,
            )

    def test_inline_assembly_reports_stable_reason(self) -> None:
        inline_asm_module = r'''
define void @__rapid_entry__kernel(ptr %out, ptr %__rapid_context) {
entry:
  call void asm sideeffect "bar.sync 0;", ""()
  ret void
}
'''
        with tempfile.TemporaryDirectory(prefix="rapid-vconfig-test-") as temp_dir:
            completed, output = self.transform(Path(temp_dir), inline_asm_module)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("vconfig_inline_asm_unsupported", completed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
