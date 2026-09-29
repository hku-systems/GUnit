import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional dependency in minimal envs
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVERS_DIR = Path(__file__).resolve().parent
if str(DRIVERS_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVERS_DIR))
SCRIPT_ROOT = REPO_ROOT / "scripts" / "kernel-smoke"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from shared_e2e import get_capture_e2e_context  # noqa: E402
from pipeline.manifest import write_manifest  # noqa: E402


def _manifest_kernel_by_display_name(run_dir: Path, display_name: str) -> dict | None:
    for entry in json.loads((run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
        kernel_dir = run_dir / entry["dir"]
        manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["kernels"][0]
        if kernel.get("display_name") == display_name:
            return kernel
    return None


def _kernel_dir_by_display_name(run_dir: Path, display_name: str) -> Path | None:
    for entry in json.loads((run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
        kernel_dir = run_dir / entry["dir"]
        manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["kernels"][0]
        if kernel.get("display_name") == display_name:
            return kernel_dir
    return None


def _fields_by_name(layout_or_node: dict) -> dict[str, dict]:
    return {
        field.get("name"): field
        for field in layout_or_node.get("fields", [])
        if isinstance(field, dict)
    }


def _single_arg(kernel: dict, name: str) -> dict:
    args = {arg.get("name"): arg for arg in kernel.get("args", [])}
    arg = args.get(name)
    if arg is None:
        raise AssertionError(f"{name} arg not found in {kernel.get('display_name')}")
    return arg


def _assert_layout_has_no_offsets(testcase: unittest.TestCase, node: dict) -> None:
    testcase.assertNotIn("offset_bytes", node)
    for field in node.get("fields", []):
        if isinstance(field, dict):
            _assert_layout_has_no_offsets(testcase, field)
    element = node.get("element")
    if isinstance(element, dict):
        _assert_layout_has_no_offsets(testcase, element)


def _scalar_pointee_layout(type_name: str = "uint8_t") -> dict:
    size = 1 if type_name in {"char", "uint8_t"} else 4 if type_name in {"int", "float", "unsigned int"} else 8
    return {
        "name": "$pointee",
        "type": type_name,
        "kind": "scalar",
        "size_bytes": size,
        "align_bytes": size,
    }


class ManifestContractTest(unittest.TestCase):
    def test_manifest_writer_preserves_layout_domain_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            write_manifest(
                {
                    "kernel_id": "k0",
                    "symbol_name": "kernel",
                    "args": [
                        {
                            "index": 0,
                            "name": "buf",
                            "type": "uint8_t *",
                            "kind": "pointer",
                            "pointer_role": "payload_buffer",
                            "size_bytes": 8,
                            "align_bytes": 8,
                            "pointee_layout": _scalar_pointee_layout("uint8_t"),
                            "domain": {"kind": "bytes", "max_len": "16", "elem_size_bytes": 1},
                        },
                        {
                            "index": 1,
                            "name": "cfg",
                            "type": "Config",
                            "kind": "opaque_val",
                            "size_bytes": 8,
                            "align_bytes": 4,
                            "type_layout": {
                                "layout_status": "complete",
                                "fields": [
                                    {
                                        "name": "n",
                                        "type": "int",
                                        "kind": "scalar",
                                        "size_bytes": 4,
                                        "align_bytes": 4,
                                        "domain": {"kind": "int_range", "min": "0", "max": "16"},
                                    }
                                ],
                            },
                        },
                    ],
                    "constraints": [
                        {
                            "kind": "scalar_le_buffer_len",
                            "scalar_arg": 1,
                            "scalar_path": ["n"],
                            "buffer_arg": 0,
                            "unit": "bytes",
                        }
                    ],
                },
                {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                "target",
                {},
            )

            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))

        kernel = manifest["kernels"][0]
        self.assertIn("constraints", kernel)
        self.assertEqual(kernel["args"][0]["pointer_role"], "payload_buffer")
        self.assertEqual(kernel["args"][0]["domain"]["kind"], "bytes")
        self.assertEqual(kernel["args"][1]["kind"], "opaque_val")
        self.assertEqual(kernel["args"][1]["type_layout"]["layout_status"], "complete")

    def test_manifest_writer_rejects_type_info_without_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "type_info missing kind"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "cfg",
                                "type": "Config",
                                "kind": "opaque_val",
                                "size_bytes": 4,
                                "align_bytes": 4,
                                "type_info": {
                                    "qualified_name": "Config",
                                },
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_writer_preserves_nested_pointer_array_layout(self) -> None:
        nested_ptr = {
            "name": "$element",
            "type": "layout_nested_ptr_item",
            "kind": "opaque_with_ptr",
            "size_bytes": 24,
            "align_bytes": 8,
            "fields": [
                {
                    "name": "inner",
                    "type": "layout_ptr_leaf",
                    "kind": "opaque_with_ptr",
                    "size_bytes": 16,
                    "align_bytes": 8,
                    "fields": [
                        {
                            "name": "count",
                            "type": "int",
                            "kind": "scalar",
                            "size_bytes": 4,
                            "align_bytes": 4,
                        },
                        {
                            "name": "data",
                            "type": "float *",
                            "kind": "pointer",
                            "pointer_role": "payload_buffer",
                            "size_bytes": 8,
                            "align_bytes": 8,
                            "pointee_layout": _scalar_pointee_layout("float"),
                        },
                    ],
                },
                {"name": "scale", "type": "int", "kind": "scalar", "size_bytes": 4, "align_bytes": 4},
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            write_manifest(
                {
                    "kernel_id": "k0",
                    "symbol_name": "kernel",
                    "args": [
                        {
                            "index": 0,
                            "name": "array_nested_ptr",
                            "type": "layout_struct_array_nested_ptr",
                            "kind": "opaque_val",
                            "size_bytes": 56,
                            "align_bytes": 8,
                            "type_layout": {
                                "layout_status": "complete",
                                "fields": [
                                    {
                                        "name": "items",
                                        "type": "layout_nested_ptr_item[2]",
                                        "kind": "opaque_with_ptr",
                                        "size_bytes": 48,
                                        "align_bytes": 8,
                                        "element_count": 2,
                                        "element": nested_ptr,
                                    }
                                ],
                            },
                        },
                    ],
                },
                {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                "target",
                {},
            )

            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))

        args = {arg["name"]: arg for arg in manifest["kernels"][0]["args"]}
        array_nested_ptr_items = _fields_by_name(args["array_nested_ptr"]["type_layout"])["items"]
        nested_inner = _fields_by_name(array_nested_ptr_items["element"])["inner"]
        nested_data = _fields_by_name(nested_inner)["data"]
        self.assertEqual(args["array_nested_ptr"]["kind"], "opaque_with_ptr")
        self.assertEqual(array_nested_ptr_items["kind"], "opaque_with_ptr")
        self.assertEqual(nested_inner["kind"], "opaque_with_ptr")
        self.assertEqual(nested_data["pointer_role"], "payload_buffer")
        self.assertEqual(nested_data["pointee_layout"]["type"], "float")
        _assert_layout_has_no_offsets(self, args["array_nested_ptr"]["type_layout"])

    def test_manifest_writer_rejects_layout_offset_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "unsupported offset_bytes"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "cfg",
                                "type": "Config",
                                "kind": "opaque_val",
                                "size_bytes": 4,
                                "align_bytes": 4,
                                "type_layout": {
                                    "layout_status": "complete",
                                    "fields": [
                                        {
                                            "name": "n",
                                            "offset_bytes": 0,
                                            "type": "int",
                                            "kind": "scalar",
                                            "size_bytes": 4,
                                            "align_bytes": 4,
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_writer_rejects_missing_pointer_role(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "missing pointer_role"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "buf",
                                "type": "uint8_t *",
                                "kind": "pointer",
                                "size_bytes": 8,
                                "align_bytes": 8,
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_writer_rejects_pointer_role_on_non_pointer_arg(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "non-pointer arg"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "size",
                                "type": "size_t",
                                "kind": "scalar",
                                "pointer_role": "payload_buffer",
                                "size_bytes": 8,
                                "align_bytes": 8,
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_writer_rejects_missing_pointer_pointee_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "missing pointee_layout"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "buf",
                                "type": "uint8_t *",
                                "kind": "pointer",
                                "pointer_role": "payload_buffer",
                                "size_bytes": 8,
                                "align_bytes": 8,
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_writer_rejects_nested_pointer_missing_pointer_role(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "missing pointer_role"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "cfg",
                                "type": "Config",
                                "kind": "opaque_val",
                                "size_bytes": 8,
                                "align_bytes": 8,
                                "type_layout": {
                                    "layout_status": "complete",
                                    "fields": [
                                        {
                                            "name": "buf",
                                            "type": "uint8_t *",
                                            "kind": "pointer",
                                            "size_bytes": 8,
                                            "align_bytes": 8,
                                            "pointee_layout": _scalar_pointee_layout("uint8_t"),
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_writer_summarizes_nested_materialization_facts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            write_manifest(
                {
                    "kernel_id": "k0",
                    "symbol_name": "kernel",
                    "args": [
                        {
                            "index": 0,
                            "name": "cfg",
                            "type": "Config",
                            "kind": "opaque_val",
                            "size_bytes": 4,
                            "align_bytes": 4,
                            "type_layout": {
                                "layout_status": "complete",
                                "fields": [
                                    {
                                        "index": "cfg.value",
                                        "name": "value",
                                        "type": "const int",
                                        "kind": "scalar",
                                        "size_bytes": 4,
                                        "align_bytes": 4,
                                        "materialization_status": "unsafe",
                                        "materialization_reason_codes": ["const_assignment_blocker"],
                                        "materialization_blockers": [
                                            "cfg.value: const data field cannot be assigned after construction"
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                },
                {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                "target",
                {},
            )

            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            kernel = manifest["kernels"][0]
            others = kernel.get("others", {})
            self.assertEqual(others.get("materialization_status"), "unsafe")
            self.assertEqual(others.get("materialization_reason_codes"), ["const_assignment_blocker"])
            self.assertEqual(
                others.get("materialization_blockers"),
                ["cfg.value: const data field cannot be assigned after construction"],
            )

    def test_manifest_writer_rejects_root_type_layout_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "type_layout root must not set index"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "cfg",
                                "type": "Config",
                                "kind": "opaque_val",
                                "size_bytes": 4,
                                "align_bytes": 4,
                                "type_layout": {
                                    "index": "cfg",
                                    "layout_status": "complete",
                                    "fields": [],
                                },
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_writer_preserves_pointer_role_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            write_manifest(
                {
                    "kernel_id": "k0",
                    "symbol_name": "kernel",
                    "args": [
                        {
                            "index": 0,
                            "name": "base",
                            "type": "uint8_t *",
                            "kind": "pointer",
                            "pointer_role": "payload_buffer",
                            "size_bytes": 8,
                            "align_bytes": 8,
                            "pointee_layout": _scalar_pointee_layout("uint8_t"),
                        },
                        {
                            "index": 1,
                            "name": "view",
                            "type": "uint8_t *",
                            "kind": "pointer",
                            "pointer_role": "derived_pointer",
                            "base_arg": 0,
                            "offset_arg": 2,
                            "size_bytes": 8,
                            "align_bytes": 8,
                            "pointee_layout": _scalar_pointee_layout("uint8_t"),
                        },
                        {
                            "index": 2,
                            "name": "offset",
                            "type": "size_t",
                            "kind": "scalar",
                            "size_bytes": 8,
                            "align_bytes": 8,
                        },
                        {
                            "index": 3,
                            "name": "workspace",
                            "type": "uint8_t *",
                            "kind": "pointer",
                            "pointer_role": "external_device_pointer",
                            "source": "scratch.workspace",
                            "size_bytes": 8,
                            "align_bytes": 8,
                            "pointee_layout": _scalar_pointee_layout("uint8_t"),
                        },
                    ],
                },
                {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                "target",
                {},
            )

            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))

        args = manifest["kernels"][0]["args"]
        self.assertEqual(args[1]["pointer_role"], "derived_pointer")
        self.assertEqual(args[1]["base_arg"], 0)
        self.assertEqual(args[1]["offset_arg"], 2)
        self.assertEqual(args[1]["pointee_layout"]["type"], "uint8_t")
        self.assertEqual(args[3]["pointer_role"], "external_device_pointer")
        self.assertEqual(args[3]["source"], "scratch.workspace")

    def test_manifest_writer_rejects_incomplete_pointer_role_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "missing base_arg"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "view",
                                "type": "uint8_t *",
                                "kind": "pointer",
                                "pointer_role": "derived_pointer",
                                "offset_bytes": 4,
                                "size_bytes": 8,
                                "align_bytes": 8,
                                "pointee_layout": _scalar_pointee_layout("uint8_t"),
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            kernel_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "missing source"):
                write_manifest(
                    {
                        "kernel_id": "k0",
                        "symbol_name": "kernel",
                        "args": [
                            {
                                "index": 0,
                                "name": "workspace",
                                "type": "uint8_t *",
                                "kind": "pointer",
                                "pointer_role": "external_device_pointer",
                                "size_bytes": 8,
                                "align_bytes": 8,
                                "pointee_layout": _scalar_pointee_layout("uint8_t"),
                            }
                        ],
                    },
                    {"kernel_id": "k0", "kernel_dir": str(kernel_dir), "status": "built"},
                    "target",
                    {},
                )

    def test_manifest_and_metadata_minimum_contract(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        discover = json.loads((ctx.run_dir / "discover.json").read_text(encoding="utf-8"))
        provenance = discover.get("provenance", {})
        diagnostics = discover.get("diagnostics", {})
        self.assertEqual(provenance.get("mode"), "artifact")
        self.assertEqual(provenance.get("jobs"), 1)
        # Grouped subsections must exist.
        self.assertIn("discovery", provenance)
        self.assertIn("artifact_diagnostics", provenance)
        self.assertIn("variant_failures_by_variant", diagnostics)
        self.assertIn("ptx_ast_mismatch_by_variant", diagnostics)

        index = json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index.get("capture_dir"), str(ctx.capture_dir))
        self.assertGreaterEqual(len(index["kernels"]), 6)
        has_non_empty_args = False

        for entry in index["kernels"]:
            kernel_dir = ctx.run_dir / entry["dir"]
            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            metadata = json.loads((kernel_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["schema_version"], 1)
            self.assertIn("kernels", manifest)
            self.assertEqual(len(manifest["kernels"]), 1)
            one = manifest["kernels"][0]
            self.assertIn("symbol_name", one)
            self.assertIn("display_name", one)
            self.assertIn("args", one)
            if one["args"]:
                has_non_empty_args = True
            for arg in one["args"]:
                self.assertIn("index", arg)
                self.assertIn("name", arg)
                self.assertIn("type", arg)
                self.assertIn("kind", arg)
                self.assertIn("size_bytes", arg)
                self.assertIn("align_bytes", arg)
                if "type_info" in arg:
                    self.assertIsInstance(arg["type_info"], dict)
                    self.assertIn("kind", arg["type_info"])

            others = one.get("others", {})
            if isinstance(others, dict) and "source_line" in others:
                self.assertGreaterEqual(others["source_line"], 1)

            manifest_text = (kernel_dir / "manifest.json").read_text(encoding="utf-8")
            sym_pos = manifest_text.index('"symbol_name"')
            disp_pos = manifest_text.index('"display_name"')
            between = manifest_text[sym_pos:disp_pos]
            for other_key in ("args", "others", "source_file", "source_line"):
                self.assertNotIn(
                    f'"{other_key}"',
                    between,
                    f'"{other_key}" found between symbol_name and display_name',
                )

            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["target_lib"], "fixtures_all_e2e")
            self.assertEqual(metadata["build_status"], "built")
            self.assertIsNone(metadata["failure_reason"])
            self.assertIn("output_hashes", metadata)
            self.assertIn("kernel.bc", metadata["output_hashes"])
            self.assertIn("manifest.json", metadata["output_hashes"])

        self.assertTrue(has_non_empty_args, "artifact mode should preserve AST-extracted args")

    def test_phase1_emitted_manifests_are_schema_valid(self) -> None:
        if jsonschema is None:
            self.skipTest("jsonschema not available")
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        schema = json.loads((REPO_ROOT / "docs" / "kernel-manifest.schema.json").read_text(encoding="utf-8"))
        validator_cls = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
        validator = validator_cls(schema)

        for entry in json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
            with self.subTest(kernel_id=entry["kernel_id"]):
                kernel_dir = ctx.run_dir / entry["dir"]
                manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
                errors = sorted(validator.iter_errors(manifest), key=lambda error: error.path)
                self.assertEqual(errors, [])

    def test_pointer_to_nonbuiltin_generates_type_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        target_kernel_dir = None
        for entry in json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
            kernel_dir = ctx.run_dir / entry["dir"]
            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            kernel = manifest["kernels"][0]
            if kernel.get("display_name") == "record_ptr_copy":
                target_kernel_dir = kernel_dir
                break

        self.assertIsNotNone(target_kernel_dir, "record_ptr_copy kernel not found in phase1 output")
        manifest = json.loads((target_kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["kernels"][0]
        others = kernel.get("others", {})
        self.assertEqual(others.get("type_shim_status"), "ok")
        shim_header = others.get("type_shim_header")
        self.assertEqual(shim_header, "type_shim.v1.cuh")
        shim_text = (target_kernel_dir / shim_header).read_text(encoding="utf-8")
        self.assertIn("struct record_pair", shim_text)
        arg_types = {arg.get("name"): arg.get("type") for arg in kernel.get("args", [])}
        self.assertEqual(arg_types.get("out"), "record_pair *")
        self.assertEqual(arg_types.get("in"), "const record_pair *")

    def test_typedef_scalar_and_size_t_emit_scalar_args_and_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        kernel = _manifest_kernel_by_display_name(ctx.run_dir, "typedef_scalar_kernel")
        self.assertIsNotNone(kernel, "typedef_scalar_kernel not found in phase1 output")
        args = {arg.get("name"): arg for arg in kernel.get("args", [])}

        value = args.get("value")
        count = args.get("count")
        self.assertIsNotNone(value)
        self.assertIsNotNone(count)
        self.assertEqual(value.get("kind"), "scalar")
        self.assertEqual(count.get("kind"), "scalar")
        self.assertNotIn("type_layout", value)
        self.assertNotIn("type_layout", count)
        self.assertEqual(value.get("size_bytes"), 4)
        self.assertIn(count.get("size_bytes"), (4, 8))

        others = kernel.get("others", {})
        self.assertEqual(others.get("type_shim_status"), "ok")
        shim_header = others.get("type_shim_header")
        self.assertEqual(shim_header, "type_shim.v1.cuh")
        kernel_dir = _kernel_dir_by_display_name(ctx.run_dir, "typedef_scalar_kernel")
        self.assertIsNotNone(kernel_dir)
        shim_text = (kernel_dir / shim_header).read_text(encoding="utf-8")
        self.assertIn("typedef_scalar_int", shim_text)
        self.assertIn("typedef_size_count", shim_text)

    def test_typedef_pointer_aliases_keep_pointer_kind_and_pointee_layout(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        int_ptr_kernel = _manifest_kernel_by_display_name(ctx.run_dir, "typedef_int_ptr_kernel")
        record_ptr_kernel = _manifest_kernel_by_display_name(ctx.run_dir, "typedef_record_ptr_kernel")
        self.assertIsNotNone(int_ptr_kernel, "typedef_int_ptr_kernel not found in phase1 output")
        self.assertIsNotNone(record_ptr_kernel, "typedef_record_ptr_kernel not found in phase1 output")

        values = _single_arg(int_ptr_kernel, "values")
        self.assertEqual(values.get("kind"), "pointer")
        self.assertEqual(values.get("pointer_role"), "payload_buffer")
        self.assertEqual(values.get("pointee_layout", {}).get("kind"), "scalar")
        self.assertEqual(values.get("pointee_layout", {}).get("type"), "int")
        self.assertNotIn("type_layout", values)

        records = _single_arg(record_ptr_kernel, "records")
        self.assertEqual(records.get("kind"), "pointer")
        self.assertEqual(records.get("pointer_role"), "payload_buffer")
        self.assertIn(records.get("pointee_layout", {}).get("kind"), {"opaque_val", "opaque_with_ptr"})
        self.assertEqual(records.get("pointee_layout", {}).get("layout_status"), "complete")
        self.assertIn("fields", records.get("pointee_layout", {}))

    def test_typedef_record_aliases_emit_recursive_layout_and_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        record_kernel = _manifest_kernel_by_display_name(ctx.run_dir, "typedef_record_alias_kernel")
        anonymous_kernel = _manifest_kernel_by_display_name(ctx.run_dir, "typedef_anonymous_alias_kernel")
        self.assertIsNotNone(record_kernel, "typedef_record_alias_kernel not found in phase1 output")
        self.assertIsNotNone(anonymous_kernel, "typedef_anonymous_alias_kernel not found in phase1 output")

        cfg = _single_arg(record_kernel, "cfg")
        self.assertEqual(cfg.get("kind"), "opaque_val")
        fields = _fields_by_name(cfg.get("type_layout", {}))
        self.assertEqual(fields.get("x", {}).get("kind"), "scalar")
        self.assertEqual(fields.get("y", {}).get("kind"), "scalar")

        anon = _single_arg(anonymous_kernel, "cfg")
        self.assertEqual(anon.get("kind"), "opaque_val")
        fields = _fields_by_name(anon.get("type_layout", {}))
        self.assertEqual(fields.get("x", {}).get("kind"), "scalar")
        self.assertEqual(fields.get("y", {}).get("kind"), "scalar")

    def test_static_const_template_dependencies_generate_type_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        target_kernel_dir = None
        for entry in json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
            kernel_dir = ctx.run_dir / entry["dir"]
            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            kernel = manifest["kernels"][0]
            if kernel.get("display_name") == "rank_wrapper_copy":
                target_kernel_dir = kernel_dir
                break

        self.assertIsNotNone(target_kernel_dir, "rank_wrapper_copy kernel not found in phase1 output")
        manifest = json.loads((target_kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["kernels"][0]
        others = kernel.get("others", {})
        self.assertEqual(others.get("type_shim_status"), "ok")
        shim_header = others.get("type_shim_header")
        self.assertEqual(shim_header, "type_shim.v1.cuh")
        shim_text = (target_kernel_dir / shim_header).read_text(encoding="utf-8")
        self.assertIn("struct static_rank_coord", shim_text)
        self.assertIn("struct rank_box", shim_text)
        self.assertIn("struct rank_wrapper", shim_text)
        arg_types = {arg.get("name"): arg.get("type") for arg in kernel.get("args", [])}
        self.assertEqual(arg_types.get("out"), "rank_wrapper *")
        self.assertEqual(arg_types.get("in"), "const rank_wrapper *")

    def test_namespace_type_dependency_generates_type_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        target_kernel_dir = None
        for entry in json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
            kernel_dir = ctx.run_dir / entry["dir"]
            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            kernel = manifest["kernels"][0]
            if str(kernel.get("display_name", "")).endswith("ns_record_ptr_copy"):
                target_kernel_dir = kernel_dir
                break

        self.assertIsNotNone(target_kernel_dir, "ns_record_ptr_copy kernel not found in phase1 output")
        manifest = json.loads((target_kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["kernels"][0]
        others = kernel.get("others", {})
        self.assertEqual(others.get("type_shim_status"), "ok")
        shim_header = others.get("type_shim_header")
        self.assertEqual(shim_header, "type_shim.v1.cuh")
        shim_text = (target_kernel_dir / shim_header).read_text(encoding="utf-8")
        self.assertIn("namespace ns1", shim_text)
        self.assertIn("namespace ns2", shim_text)
        self.assertIn("struct ns_pair", shim_text)
        arg_types = {arg.get("name"): arg.get("type") for arg in kernel.get("args", [])}
        self.assertIn("ns1::ns2::ns_pair", arg_types.get("out", ""))
        self.assertIn("ns1::ns2::ns_pair", arg_types.get("in", ""))

    def test_class_nested_type_dependency_generates_owner_type_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        target_kernel_dir = None
        for entry in json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
            kernel_dir = ctx.run_dir / entry["dir"]
            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            kernel = manifest["kernels"][0]
            if str(kernel.get("display_name", "")).endswith("class_nested_record_ptr_copy"):
                target_kernel_dir = kernel_dir
                break

        self.assertIsNotNone(target_kernel_dir, "class_nested_record_ptr_copy kernel not found in phase1 output")
        manifest = json.loads((target_kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["kernels"][0]
        others = kernel.get("others", {})
        self.assertEqual(others.get("type_shim_status"), "ok")
        self.assertEqual(others.get("type_shim_header"), "type_shim.v1.cuh")
        self.assertNotIn("type_shim_reason_codes", others)
        self.assertNotIn("type_shim_missing_dependencies", others)
        shim_text = (target_kernel_dir / "type_shim.v1.cuh").read_text(encoding="utf-8")
        self.assertIn("namespace ns1", shim_text)
        self.assertIn("namespace ns2", shim_text)
        self.assertIn("struct scoped_owner", shim_text)
        self.assertIn("struct nested_pair", shim_text)
        arg_types = {arg.get("name"): arg.get("type") for arg in kernel.get("args", [])}
        self.assertIn("ns1::ns2::scoped_owner::nested_pair", arg_types.get("out", ""))
        self.assertIn("ns1::ns2::scoped_owner::nested_pair", arg_types.get("in", ""))

    def test_header_backed_nested_type_dependency_generates_type_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        target_kernel_dir = None
        for entry in json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
            kernel_dir = ctx.run_dir / entry["dir"]
            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            kernel = manifest["kernels"][0]
            if str(kernel.get("display_name", "")).endswith("header_nested_params_kernel"):
                target_kernel_dir = kernel_dir
                break

        self.assertIsNotNone(target_kernel_dir, "header_nested_params_kernel kernel not found in phase1 output")
        manifest = json.loads((target_kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        kernel = manifest["kernels"][0]
        others = kernel.get("others", {})
        self.assertEqual(others.get("type_shim_status"), "ok")
        self.assertEqual(others.get("type_shim_header"), "type_shim.v1.cuh")
        self.assertIn("scoped_header_types.cuh", "\n".join(others.get("type_shim_system_includes", [])))
        shim_text = (target_kernel_dir / "type_shim.v1.cuh").read_text(encoding="utf-8")
        self.assertIn('#include "scoped_header_types.cuh"', shim_text)
        self.assertNotIn("struct outer_box", shim_text)
        self.assertNotIn("enum class mode", shim_text)
        args = {arg.get("name"): arg for arg in kernel.get("args", [])}
        params = args.get("params")
        self.assertIsNotNone(params)
        self.assertEqual(params.get("kind"), "opaque_with_ptr")
        self.assertIn("header_scope::outer_box::params", params.get("type", ""))
        fields = _fields_by_name(params.get("type_layout", {}))
        self.assertEqual(fields.get("count", {}).get("kind"), "scalar")
        self.assertEqual(fields.get("data", {}).get("kind"), "pointer")
        self.assertEqual(fields.get("data", {}).get("pointer_role"), "payload_buffer")
        self.assertEqual(fields.get("op", {}).get("kind"), "scalar")
        self.assertEqual(fields.get("op", {}).get("domain", {}).get("kind"), "enum")

    def test_struct_enum_kernel_generates_domain_and_layout(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        target_kernel = None
        for entry in json.loads((ctx.run_dir / "index.json").read_text(encoding="utf-8")).get("kernels", []):
            kernel_dir = ctx.run_dir / entry["dir"]
            manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
            kernel = manifest["kernels"][0]
            if kernel.get("display_name") == "struct_enum_kernel":
                target_kernel = kernel
                break

        self.assertIsNotNone(target_kernel, "struct_enum_kernel not found in phase1 output")
        args = {arg.get("name"): arg for arg in target_kernel.get("args", [])}
        cfg = args.get("cfg")
        self.assertIsNotNone(cfg, "cfg arg not found")
        self.assertEqual(cfg.get("kind"), "opaque_val")
        layout = cfg.get("type_layout")
        self.assertIsInstance(layout, dict)
        self.assertNotIn("index", layout)
        self.assertEqual(layout.get("layout_status"), "complete")

        fields = {field.get("name"): field for field in layout.get("fields", [])}
        mode = fields.get("mode")
        inner = fields.get("inner")
        flags = fields.get("flags")
        self.assertIsNotNone(mode, "mode field not found")
        self.assertIsNotNone(inner, "inner field not found")
        self.assertIsNotNone(flags, "flags field not found")
        self.assertEqual(mode.get("index"), "cfg.mode")
        self.assertEqual(inner.get("index"), "cfg.inner")
        self.assertEqual(flags.get("index"), "cfg.flags")
        self.assertEqual(mode.get("kind"), "scalar")
        self.assertEqual(mode.get("domain", {}).get("kind"), "enum")
        self.assertNotIn("underlying_type", mode.get("domain", {}))
        self.assertNotIn("underlying_size_bytes", mode.get("domain", {}))
        enum_values = {v.get("name"): v.get("value") for v in mode.get("domain", {}).get("values", [])}
        self.assertEqual(enum_values.get("ModeA"), "0")
        self.assertEqual(enum_values.get("ModeB"), "1")

        self.assertEqual(inner.get("kind"), "opaque_val")
        inner_fields = {field.get("name"): field for field in inner.get("fields", [])}
        self.assertEqual(inner_fields.get("x", {}).get("kind"), "scalar")
        self.assertEqual(inner_fields.get("y", {}).get("kind"), "scalar")
        self.assertEqual(inner_fields.get("x", {}).get("index"), "cfg.inner.x")
        self.assertEqual(inner_fields.get("y", {}).get("index"), "cfg.inner.y")

        self.assertEqual(flags.get("kind"), "opaque_val")
        flag_fields = {field.get("name"): field for field in flags.get("fields", [])}
        low = flag_fields.get("low")
        high = flag_fields.get("high")
        self.assertIsNotNone(low, "low bitfield not found")
        self.assertIsNotNone(high, "high bitfield not found")
        self.assertEqual(low.get("bit_width"), 3)
        self.assertEqual(high.get("bit_width"), 5)
        self.assertEqual(low.get("index"), "cfg.flags.low")
        self.assertEqual(high.get("index"), "cfg.flags.high")
        self.assertIn("bit_offset", low)
        self.assertIn("bit_offset", high)

    def test_phase1_generates_recursive_struct_array_layout_matrix(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        kernels = {
            name: _manifest_kernel_by_display_name(ctx.run_dir, name)
            for name in (
                "layout_plain_val_kernel",
                "layout_outer_struct_kernel",
                "layout_struct_array_val_kernel",
                "layout_struct_array_ptr_kernel",
                "layout_struct_array_nested_kernel",
                "layout_struct_array_nested_ptr_kernel",
                "layout_anonymous_ptr_kernel",
                "layout_derived_params_kernel",
            )
        }
        for name, kernel in kernels.items():
            self.assertIsNotNone(kernel, f"{name} not found in phase1 output")

        plain_cfg = _single_arg(kernels["layout_plain_val_kernel"], "cfg")
        self.assertEqual(plain_cfg.get("kind"), "opaque_val")
        plain_fields = _fields_by_name(plain_cfg.get("type_layout", {}))
        self.assertEqual(plain_fields.get("a", {}).get("kind"), "scalar")
        self.assertEqual(plain_fields.get("b", {}).get("kind"), "scalar")
        self.assertEqual(plain_fields.get("c", {}).get("kind"), "scalar")

        outer_cfg = _single_arg(kernels["layout_outer_struct_kernel"], "cfg")
        self.assertEqual(outer_cfg.get("kind"), "opaque_val")
        outer_fields = _fields_by_name(outer_cfg.get("type_layout", {}))
        inner = outer_fields.get("inner")
        self.assertIsNotNone(inner, "outer struct inner field not found")
        self.assertEqual(inner.get("kind"), "opaque_val")
        inner_fields = _fields_by_name(inner)
        self.assertEqual(inner_fields.get("x", {}).get("kind"), "scalar")
        self.assertEqual(inner_fields.get("y", {}).get("kind"), "scalar")

        array_val_cfg = _single_arg(kernels["layout_struct_array_val_kernel"], "cfg")
        self.assertEqual(array_val_cfg.get("kind"), "opaque_val")
        array_val_items = _fields_by_name(array_val_cfg.get("type_layout", {})).get("items")
        self.assertIsNotNone(array_val_items, "value struct array field not found")
        self.assertEqual(array_val_items.get("kind"), "opaque_val")
        self.assertEqual(array_val_items.get("element_count"), 2)
        array_val_element = array_val_items.get("element", {})
        self.assertEqual(array_val_element.get("kind"), "opaque_val")
        self.assertEqual(_fields_by_name(array_val_element).get("x", {}).get("kind"), "scalar")

        array_ptr_cfg = _single_arg(kernels["layout_struct_array_ptr_kernel"], "cfg")
        self.assertEqual(array_ptr_cfg.get("kind"), "opaque_with_ptr")
        array_ptr_items = _fields_by_name(array_ptr_cfg.get("type_layout", {})).get("items")
        self.assertIsNotNone(array_ptr_items, "pointer struct array field not found")
        self.assertEqual(array_ptr_items.get("kind"), "opaque_with_ptr")
        self.assertEqual(array_ptr_items.get("element_count"), 2)
        array_ptr_element_fields = _fields_by_name(array_ptr_items.get("element", {}))
        data = array_ptr_element_fields.get("data")
        self.assertIsNotNone(data, "array element pointer field not found")
        self.assertEqual(data.get("kind"), "pointer")
        self.assertEqual(data.get("pointer_role"), "payload_buffer")

        array_nested_cfg = _single_arg(kernels["layout_struct_array_nested_kernel"], "cfg")
        self.assertEqual(array_nested_cfg.get("kind"), "opaque_val")
        array_nested_items = _fields_by_name(array_nested_cfg.get("type_layout", {})).get("items")
        self.assertIsNotNone(array_nested_items, "nested struct array field not found")
        array_nested_element = array_nested_items.get("element", {})
        self.assertEqual(array_nested_element.get("kind"), "opaque_val")
        nested_inner = _fields_by_name(array_nested_element).get("inner")
        self.assertIsNotNone(nested_inner, "array element nested struct field not found")
        self.assertEqual(nested_inner.get("kind"), "opaque_val")
        self.assertEqual(_fields_by_name(nested_inner).get("x", {}).get("kind"), "scalar")

        array_nested_ptr_cfg = _single_arg(kernels["layout_struct_array_nested_ptr_kernel"], "cfg")
        self.assertEqual(array_nested_ptr_cfg.get("kind"), "opaque_with_ptr")
        array_nested_ptr_items = _fields_by_name(array_nested_ptr_cfg.get("type_layout", {})).get("items")
        self.assertIsNotNone(array_nested_ptr_items, "nested pointer struct array field not found")
        self.assertEqual(array_nested_ptr_items.get("kind"), "opaque_with_ptr")
        array_nested_ptr_element = array_nested_ptr_items.get("element", {})
        nested_ptr_inner = _fields_by_name(array_nested_ptr_element).get("inner")
        self.assertIsNotNone(nested_ptr_inner, "nested pointer inner struct field not found")
        self.assertEqual(nested_ptr_inner.get("kind"), "opaque_with_ptr")
        nested_ptr_data = _fields_by_name(nested_ptr_inner).get("data")
        self.assertIsNotNone(nested_ptr_data, "nested pointer data field not found")
        self.assertEqual(nested_ptr_data.get("kind"), "pointer")
        self.assertEqual(nested_ptr_data.get("pointer_role"), "payload_buffer")

        anonymous_cfg = _single_arg(kernels["layout_anonymous_ptr_kernel"], "cfg")
        self.assertEqual(anonymous_cfg.get("kind"), "opaque_with_ptr")
        anonymous_fields = _fields_by_name(anonymous_cfg.get("type_layout", {}))
        self.assertEqual(anonymous_fields.get("prefix", {}).get("kind"), "scalar")
        self.assertEqual(anonymous_fields.get("tail", {}).get("kind"), "scalar")
        anon = next(
            (field for name, field in anonymous_fields.items() if isinstance(name, str) and name.startswith("anon")),
            None,
        )
        self.assertIsNotNone(anon, "anonymous aggregate field not found")
        self.assertEqual(anon.get("kind"), "opaque_with_ptr")
        self.assertIn("anonymous", anon.get("type", "").lower())
        self.assertTrue(anon.get("index", "").startswith("cfg.anon"))
        anon_fields = _fields_by_name(anon)
        anon_value = anon_fields.get("value")
        anon_ptr = anon_fields.get("ptr")
        self.assertIsNotNone(anon_value, "anonymous aggregate value field not found")
        self.assertIsNotNone(anon_ptr, "anonymous aggregate pointer field not found")
        self.assertEqual(anon_value.get("kind"), "scalar")
        self.assertEqual(anon_ptr.get("kind"), "pointer")
        self.assertEqual(anon_ptr.get("pointer_role"), "payload_buffer")
        self.assertIsInstance(anon_ptr.get("pointee_layout"), dict)
        self.assertEqual(anon_ptr.get("pointee_layout", {}).get("type"), "int")

        derived_cfg = _single_arg(kernels["layout_derived_params_kernel"], "cfg")
        self.assertEqual(derived_cfg.get("kind"), "opaque_with_ptr")
        self.assertNotIn("materialization_status", derived_cfg)
        derived_fields = _fields_by_name(derived_cfg.get("type_layout", {}))
        self.assertEqual(derived_fields.get("base_count", {}).get("kind"), "scalar")
        self.assertEqual(derived_fields.get("derived_tail", {}).get("kind"), "scalar")
        self.assertEqual(derived_fields.get("base_data", {}).get("kind"), "pointer")
        self.assertEqual(derived_fields.get("base_data", {}).get("pointer_role"), "payload_buffer")
        self.assertEqual(derived_fields.get("derived_data", {}).get("kind"), "pointer")
        self.assertEqual(derived_fields.get("derived_data", {}).get("pointer_role"), "payload_buffer")

    def test_phase1_generates_manifest_example_complex_args_shape(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        kernel = _manifest_kernel_by_display_name(ctx.run_dir, "complex_args_kernel")
        self.assertIsNotNone(kernel, "complex_args_kernel not found in phase1 output")

        seed = _single_arg(kernel, "seed")
        params = _single_arg(kernel, "params")
        output = _single_arg(kernel, "output")
        self.assertEqual(seed.get("kind"), "scalar")
        self.assertEqual(seed.get("type"), "int")
        self.assertEqual(output.get("kind"), "pointer")
        self.assertEqual(output.get("pointer_role"), "payload_buffer")
        self.assertEqual(output.get("pointee_layout", {}).get("type"), "float")

        self.assertEqual(params.get("kind"), "opaque_with_ptr")
        self.assertEqual(params.get("type"), "KernelParams")
        params_fields = _fields_by_name(params.get("type_layout", {}))
        self.assertEqual(params_fields.get("count", {}).get("kind"), "scalar")

        weights = params_fields.get("weights")
        self.assertIsNotNone(weights, "KernelParams.weights not found")
        self.assertEqual(weights.get("kind"), "opaque_val")
        self.assertEqual(weights.get("element_count"), 4)
        self.assertEqual(weights.get("element", {}).get("kind"), "scalar")
        self.assertEqual(weights.get("element", {}).get("type"), "float")

        data = params_fields.get("data")
        self.assertIsNotNone(data, "KernelParams.data not found")
        self.assertEqual(data.get("kind"), "pointer")
        self.assertEqual(data.get("pointer_role"), "payload_buffer")
        self.assertEqual(data.get("pointee_layout", {}).get("type"), "float")

        kind = params_fields.get("kind")
        self.assertIsNotNone(kind, "KernelParams.kind not found")
        self.assertEqual(kind.get("kind"), "pointer")
        self.assertEqual(kind.get("pointer_role"), "payload_buffer")
        kind_pointee = kind.get("pointee_layout", {})
        self.assertEqual(kind_pointee.get("type"), "Kind")
        self.assertEqual(kind_pointee.get("kind"), "opaque_val")
        kind_fields = _fields_by_name(kind_pointee)
        self.assertEqual(kind_fields.get("dummy", {}).get("kind"), "scalar")
        self.assertEqual(kind_fields.get("selector", {}).get("kind"), "scalar")

        nested = params_fields.get("nested")
        self.assertIsNotNone(nested, "KernelParams.nested not found")
        self.assertEqual(nested.get("kind"), "opaque_with_ptr")
        self.assertEqual(nested.get("type"), "NestedParams")
        nested_fields = _fields_by_name(nested)
        self.assertEqual(nested_fields.get("inner_count", {}).get("kind"), "scalar")

        lanes = nested_fields.get("lanes")
        self.assertIsNotNone(lanes, "NestedParams.lanes not found")
        self.assertEqual(lanes.get("kind"), "opaque_with_ptr")
        self.assertEqual(lanes.get("element_count"), 4)
        lanes_element = lanes.get("element", {})
        self.assertEqual(lanes_element.get("kind"), "pointer")
        self.assertEqual(lanes_element.get("pointer_role"), "payload_buffer")
        self.assertEqual(lanes_element.get("pointee_layout", {}).get("type"), "float")

        choice = nested_fields.get("choice")
        self.assertIsNotNone(choice, "NestedParams.choice not found")
        self.assertEqual(choice.get("kind"), "opaque_with_ptr")
        self.assertEqual(choice.get("type_info", {}).get("kind"), "union")
        choice_fields = _fields_by_name(choice)
        self.assertEqual(choice_fields.get("int_ptr", {}).get("kind"), "pointer")
        self.assertEqual(choice_fields.get("float_ptr", {}).get("kind"), "pointer")
        self.assertEqual(choice_fields.get("tag", {}).get("kind"), "scalar")

        anon = nested_fields.get("anon3")
        self.assertIsNotNone(anon, "NestedParams anonymous aggregate anon3 not found")
        self.assertEqual(anon.get("kind"), "opaque_with_ptr")
        self.assertIn("anonymous", anon.get("type", "").lower())
        anon_fields = _fields_by_name(anon)
        self.assertEqual(anon_fields.get("value", {}).get("kind"), "scalar")
        anon_ptr = anon_fields.get("ptr")
        self.assertIsNotNone(anon_ptr, "NestedParams anonymous ptr field not found")
        self.assertEqual(anon_ptr.get("kind"), "pointer")
        self.assertEqual(anon_ptr.get("pointer_role"), "payload_buffer")
        self.assertEqual(anon_ptr.get("pointee_layout", {}).get("type"), "int")

    def test_phase1_records_materialization_facts_separately_from_type_shim(self) -> None:
        try:
            ctx = get_capture_e2e_context()
        except RuntimeError as e:
            self.skipTest(str(e))

        public_methods = _manifest_kernel_by_display_name(ctx.run_dir, "material_public_methods_kernel")
        private_field = _manifest_kernel_by_display_name(ctx.run_dir, "material_private_field_kernel")
        virtual_method = _manifest_kernel_by_display_name(ctx.run_dir, "material_virtual_method_kernel")
        reference_field = _manifest_kernel_by_display_name(ctx.run_dir, "material_reference_field_kernel")
        const_field = _manifest_kernel_by_display_name(ctx.run_dir, "material_const_field_kernel")

        for name, kernel in (
            ("material_public_methods_kernel", public_methods),
            ("material_private_field_kernel", private_field),
            ("material_virtual_method_kernel", virtual_method),
            ("material_reference_field_kernel", reference_field),
            ("material_const_field_kernel", const_field),
        ):
            self.assertIsNotNone(kernel, f"{name} not found in phase1 output")
            self.assertEqual(kernel.get("others", {}).get("type_shim_status"), "ok")
            self.assertNotIn("type_shim_reason_codes", kernel.get("others", {}))

        public_cfg = _single_arg(public_methods, "cfg")
        self.assertEqual(public_cfg.get("kind"), "opaque_with_ptr")
        self.assertNotIn("materialization_status", public_cfg)
        public_fields = _fields_by_name(public_cfg.get("type_layout", {}))
        self.assertEqual(public_fields.get("count", {}).get("kind"), "scalar")
        self.assertEqual(public_fields.get("data", {}).get("kind"), "pointer")
        self.assertEqual(public_fields.get("data", {}).get("pointer_role"), "payload_buffer")

        private_cfg = _single_arg(private_field, "cfg")
        private_codes = set(private_cfg.get("materialization_reason_codes", []))
        self.assertEqual(private_cfg.get("materialization_status"), "unsafe")
        self.assertIn("private_data_field", private_codes)
        private_layout_codes = set(private_cfg.get("type_layout", {}).get("materialization_reason_codes", []))
        self.assertIn("private_data_field", private_layout_codes)
        private_fields = _fields_by_name(private_cfg.get("type_layout", {}))
        self.assertEqual(private_fields.get("visible", {}).get("kind"), "scalar")
        self.assertIn(
            "private_data_field",
            private_fields.get("hidden", {}).get("materialization_reason_codes", []),
        )

        virtual_cfg = _single_arg(virtual_method, "cfg")
        virtual_codes = set(virtual_cfg.get("materialization_reason_codes", []))
        self.assertEqual(virtual_cfg.get("materialization_status"), "unsafe")
        self.assertIn("virtual_method_or_vptr", virtual_codes)
        self.assertIn("non_trivially_copyable", virtual_codes)

        reference_cfg = _single_arg(reference_field, "cfg")
        reference_codes = set(reference_cfg.get("materialization_reason_codes", []))
        self.assertEqual(reference_cfg.get("materialization_status"), "unsafe")
        self.assertIn("reference_field", reference_codes)
        ref_field = _fields_by_name(reference_cfg.get("type_layout", {})).get("ref")
        self.assertIsNotNone(ref_field)
        self.assertIn("reference_field", ref_field.get("materialization_reason_codes", []))

        const_cfg = _single_arg(const_field, "cfg")
        const_codes = set(const_cfg.get("materialization_reason_codes", []))
        self.assertEqual(const_cfg.get("materialization_status"), "unsafe")
        self.assertIn("const_assignment_blocker", const_codes)
        value_field = _fields_by_name(const_cfg.get("type_layout", {})).get("value")
        self.assertIsNotNone(value_field)
        self.assertIn("const_assignment_blocker", value_field.get("materialization_reason_codes", []))


if __name__ == "__main__":
    unittest.main()
