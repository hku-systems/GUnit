import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = REPO_ROOT / "tools" / "rapid-feedback-instrument"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "branch_memory.ll"
NESTED_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "nested_payload.ll"
SPILLED_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "spilled_payload.ll"
SELECT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "select.ll"


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise unittest.SkipTest(f"{name} is unavailable")
    return path


class FeedbackInstrumentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.llvm_as = require_tool("llvm-as-22")
        cls.llvm_dis = require_tool("llvm-dis-22")
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tempdir.name)
        cls.executable = cls.root / "rapid-feedback-instrument"
        subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(TOOL_DIR / "build.py"),
                "--output",
                str(cls.executable),
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        cls.input_bc = cls.root / "branch_memory.bc"
        subprocess.run(
            [cls.llvm_as, str(FIXTURE), "-o", str(cls.input_bc)],
            check=True,
            cwd=REPO_ROOT,
        )
        cls.build_spec = cls.root / "build_spec.json"
        cls.build_spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kernel_id": "branch-memory-fixture",
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
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        cls.select_input_bc = cls.root / "select.bc"
        subprocess.run(
            [cls.llvm_as, str(SELECT_FIXTURE), "-o", str(cls.select_input_bc)],
            check=True,
            cwd=REPO_ROOT,
        )
        cls.select_build_spec = cls.root / "select_build_spec.json"
        cls.select_build_spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kernel_id": "select-fixture",
                    "entry_symbol": "__rapid_entry_select",
                    "entry_abi_version": 1,
                    "entry_context": "rapid_kernel_context_v1",
                    "feedback_memory_enabled": False,
                    "feedback_payload_slot_count": 0,
                    "feedback_payload_slots": [],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def instrument_paths(
        self,
        *,
        input_bc: Path,
        build_spec: Path,
        stem: str,
        extra_args: tuple[str, ...] = (),
    ) -> tuple[str, dict]:
        output_bc = self.root / f"{stem}.bc"
        metadata_path = self.root / f"{stem}.json"
        instrument_result = subprocess.run(
            [
                str(self.executable),
                "--input",
                str(input_bc),
                "--output",
                str(output_bc),
                "--build-spec",
                str(build_spec),
                "--metadata-out",
                str(metadata_path),
                *extra_args,
            ],
            check=False,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(instrument_result.returncode, 0, instrument_result.stderr)
        result = subprocess.run(
            [self.llvm_dis, "-o", "-", str(output_bc)],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return result.stdout, json.loads(metadata_path.read_text(encoding="utf-8"))

    def instrument(self, stem: str) -> tuple[str, dict]:
        return self.instrument_paths(
            input_bc=self.input_bc,
            build_spec=self.build_spec,
            stem=stem,
        )

    def test_instruments_cfg_and_payload_memory_sites(self) -> None:
        llvm_ir, metadata = self.instrument("instrumented")

        self.assertEqual(llvm_ir.count("call void @__rapid_feedback_bb"), 4)
        self.assertEqual(llvm_ir.count("call void @__rapid_feedback_mem"), 2)
        self.assertIn("ptr %__rapid_context", llvm_ir)
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["feedback_abi_version"], 1)
        self.assertEqual(metadata["memory_metric_version"], "rapid-simt-memcov-v1")
        self.assertEqual(metadata["memory_map_bits"], 61_440)
        self.assertEqual(metadata["memory_sector_bytes"], 32)
        self.assertEqual(metadata["thread_activity_map_bits"], 4_096)
        self.assertEqual(
            metadata["memory_pattern_encodings"],
            {
                "single": 0,
                "full_broadcast": 1,
                "full_contiguous": 2,
                "full_other": 3,
                "partial_broadcast": 4,
                "partial_contiguous": 5,
                "partial_other": 6,
            },
        )
        self.assertEqual(
            metadata["memory_hash_contract_version"],
            "rapid-simt-memcov-hash-v1",
        )
        self.assertEqual(metadata["kernel_id"], "branch-memory-fixture")
        self.assertEqual(metadata["entry_symbol"], "__rapid_entry_test")
        self.assertEqual(metadata["instrumented_cfg_sites"], 4)
        self.assertEqual(metadata["instrumented_memory_sites"], 2)
        self.assertEqual(metadata["unknown_memory_sites"], 1)
        self.assertEqual(
            {site["access_kind"] for site in metadata["memory_sites"]},
            {"read", "write"},
        )
        self.assertEqual(
            {site["arg_slot"] for site in metadata["memory_sites"]},
            {0},
        )

    def test_output_and_metadata_are_deterministic(self) -> None:
        first_ir, first_metadata = self.instrument("deterministic-a")
        second_ir, second_metadata = self.instrument("deterministic-b")

        first_semantic_ir = "\n".join(first_ir.splitlines()[1:])
        second_semantic_ir = "\n".join(second_ir.splitlines()[1:])
        self.assertEqual(first_semantic_ir, second_semantic_ir)
        self.assertEqual(first_metadata, second_metadata)

    def test_select_instrumentation_is_default_off(self) -> None:
        llvm_ir, metadata = self.instrument_paths(
            input_bc=self.select_input_bc,
            build_spec=self.select_build_spec,
            stem="select-default-off",
        )

        self.assertEqual(llvm_ir.count("call void @__rapid_feedback_bb"), 1)
        self.assertEqual(metadata["instrumented_cfg_sites"], 1)
        self.assertNotIn("instrumented_basic_block_sites", metadata)
        self.assertNotIn("instrumented_select_sites", metadata)
        self.assertNotIn("select_sites", metadata)
        self.assertNotIn("site_kind", metadata["cfg_sites"][0])

    def test_select_instrumentation_emits_distinct_pseudo_site_metadata(self) -> None:
        llvm_ir, metadata = self.instrument_paths(
            input_bc=self.select_input_bc,
            build_spec=self.select_build_spec,
            stem="select-enabled",
            extra_args=("--instrument-selects",),
        )

        self.assertEqual(llvm_ir.count("call void @__rapid_feedback_bb"), 2)
        self.assertIn(
            "call void @__rapid_feedback_bb(i32 64695)\n"
            "  %selected = select i1 %condition, i32 %true_value, i32 %false_value",
            llvm_ir,
        )
        self.assertEqual(metadata["instrumented_cfg_sites"], 2)
        self.assertEqual(metadata["instrumented_basic_block_sites"], 1)
        self.assertEqual(len(metadata["cfg_sites"]), 2)
        self.assertEqual(
            {site["site_kind"] for site in metadata["cfg_sites"]},
            {"basic_block", "select_evaluated"},
        )
        self.assertEqual(metadata["instrumented_select_sites"], 1)
        self.assertEqual(
            metadata["select_sites"],
            [
                {
                    "function": "__rapid_entry_select",
                    "instruction_ordinal": 0,
                    "site_id": 64_695,
                    "site_kind": "select_evaluated",
                }
            ],
        )

    def test_attributes_nested_payload_access_to_field_byte_offset(self) -> None:
        input_bc = self.root / "nested_payload.bc"
        subprocess.run(
            [self.llvm_as, str(NESTED_FIXTURE), "-o", str(input_bc)],
            check=True,
            cwd=REPO_ROOT,
        )
        build_spec = self.root / "nested_build_spec.json"
        build_spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kernel_id": "nested-payload-fixture",
                    "entry_symbol": "__rapid_entry_nested",
                    "entry_abi_version": 1,
                    "entry_context": "rapid_kernel_context_v1",
                    "feedback_memory_enabled": True,
                    "feedback_payload_slot_count": 2,
                    "feedback_payload_slots": [
                        {
                            "slot": 0,
                            "arg_index": 0,
                            "field_path": [],
                            "decoded_expr": "decoded.output",
                            "elem_size": 4,
                            "byte_offset": None,
                        },
                        {
                            "slot": 1,
                            "arg_index": 1,
                            "field_path": ["data"],
                            "decoded_expr": "decoded.config.data",
                            "elem_size": 4,
                            "byte_offset": 8,
                        },
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        llvm_ir, metadata = self.instrument_paths(
            input_bc=input_bc,
            build_spec=build_spec,
            stem="nested-instrumented",
        )

        self.assertEqual(llvm_ir.count("call void @__rapid_feedback_mem"), 2)
        self.assertEqual(metadata["instrumented_memory_sites"], 2)
        self.assertEqual(metadata["unknown_memory_sites"], 1)
        self.assertEqual(
            {site["arg_slot"] for site in metadata["memory_sites"]},
            {0, 1},
        )

    def test_attributes_access_through_clang_argument_spill(self) -> None:
        input_bc = self.root / "spilled_payload.bc"
        subprocess.run(
            [self.llvm_as, str(SPILLED_FIXTURE), "-o", str(input_bc)],
            check=True,
            cwd=REPO_ROOT,
        )
        build_spec = self.root / "spilled_build_spec.json"
        build_spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kernel_id": "spilled-payload-fixture",
                    "entry_symbol": "__rapid_entry_spilled",
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
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        llvm_ir, metadata = self.instrument_paths(
            input_bc=input_bc,
            build_spec=build_spec,
            stem="spilled-instrumented",
        )

        self.assertEqual(llvm_ir.count("call void @__rapid_feedback_mem"), 1)
        self.assertEqual(metadata["instrumented_memory_sites"], 1)
        self.assertEqual(metadata["memory_sites"][0]["arg_slot"], 0)


if __name__ == "__main__":
    unittest.main()
