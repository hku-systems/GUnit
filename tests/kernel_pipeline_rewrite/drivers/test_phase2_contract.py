import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional developer dependency.
    jsonschema = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "kernel-rewrite"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from contracts.kernel import build_kernel_contract  # noqa: E402
from contracts.kernel import ArgValueExpr, BinaryExpr, ConstExpr, KernelConstraintPredicate, PayloadLenExpr  # noqa: E402
from codegen.decode import DecodeHeaderGenerator, DecodeConstraintEmitContext, _emit_predicate_expr  # noqa: E402
from codegen.invoke import InvokeHeaderGenerator  # noqa: E402
from common import KernelPhase1Artifacts  # noqa: E402
from phase2 import Phase2Runner, _is_phase2_input_invalid  # noqa: E402
from rewrite.executor import VConfigRewriteResult, _parse_functions, _rewrite_module_text  # noqa: E402


def _scalar_pointee_layout(type_name: str = "uint8_t") -> dict:
    size = 1 if type_name in {"char", "uint8_t"} else 4 if type_name in {"int", "float", "unsigned int"} else 8
    return {
        "name": "$pointee",
        "type": type_name,
        "kind": "scalar",
        "size_bytes": size,
        "align_bytes": size,
    }


def _record_pointee_layout(type_name: str = "Record") -> dict:
    return {
        "name": "$pointee",
        "type": type_name,
        "kind": "opaque_val",
        "size_bytes": 8,
        "align_bytes": 4,
        "layout_status": "complete",
        "fields": [
            {
                "index": "$pointee.x",
                "name": "x",
                "type": "int",
                "kind": "scalar",
                "size_bytes": 4,
                "align_bytes": 4,
            },
            {
                "index": "$pointee.y",
                "name": "y",
                "type": "int",
                "kind": "scalar",
                "size_bytes": 4,
                "align_bytes": 4,
            },
        ],
        "type_info": _record_type_info(type_name),
    }


def _manifest(pointer_role: str | None = "payload_buffer") -> dict:
    pointer_arg = {
        "index": 0,
        "name": "input",
        "type": "uint8_t *",
        "kind": "pointer",
        "size_bytes": 8,
        "align_bytes": 8,
        "pointee_layout": _scalar_pointee_layout("uint8_t"),
    }
    if pointer_role is not None:
        pointer_arg["pointer_role"] = pointer_role
    return {
        "schema_version": 1,
        "kernels": [
            {
                "symbol_name": "kernel",
                "display_name": "kernel",
                "args": [
                    pointer_arg,
                    {
                        "index": 1,
                        "name": "size",
                        "type": "size_t",
                        "kind": "scalar",
                        "size_bytes": 8,
                        "align_bytes": 8,
                    },
                ],
            }
        ],
    }


def _manifest_with_scalar_len_constraint(scalar_name: str = "count") -> dict:
    manifest = _manifest("payload_buffer")
    kernel = manifest["kernels"][0]
    kernel["args"][1]["name"] = scalar_name
    kernel["constraints"] = [
        {
            "kind": "scalar_le_buffer_len",
            "scalar_arg": 1,
            "buffer_arg": 0,
            "unit": "bytes",
        }
    ]
    return manifest


def _manifest_with_scalar_only_args() -> dict:
    return {
        "schema_version": 1,
        "kernels": [
            {
                "symbol_name": "kernel",
                "display_name": "kernel",
                "args": [
                    {
                        "index": 0,
                        "name": "count",
                        "type": "int",
                        "kind": "scalar",
                        "size_bytes": 4,
                        "align_bytes": 4,
                    },
                    {
                        "index": 1,
                        "name": "scale",
                        "type": "float",
                        "kind": "scalar",
                        "size_bytes": 4,
                        "align_bytes": 4,
                    },
                ],
            }
        ],
    }


def _manifest_with_top_level_cv_value_args() -> dict:
    return {
        "schema_version": 1,
        "kernels": [
            {
                "symbol_name": "kernel",
                "display_name": "kernel",
                "args": [
                    {
                        "index": 0,
                        "name": "count",
                        "type": "const int64_t",
                        "kind": "scalar",
                        "size_bytes": 8,
                        "align_bytes": 8,
                    },
                    {
                        "index": 1,
                        "name": "dims",
                        "type": "const uint3",
                        "kind": "opaque_val",
                        "size_bytes": 12,
                        "align_bytes": 4,
                        "type_info": {
                            "kind": "struct",
                            "qualified_name": "uint3",
                        },
                    },
                ],
                "others": {
                    "type_shim_header": "type_shim.v1.cuh",
                    "type_shim_status": "ok",
                },
            }
        ],
    }


def _manifest_with_pointer_bearing_struct() -> dict:
    return {
        "schema_version": 1,
        "kernels": [
            {
                "symbol_name": "kernel",
                "display_name": "kernel",
                "args": [
                    {
                        "index": 0,
                        "name": "params",
                        "type": "Params",
                        "kind": "opaque_with_ptr",
                        "size_bytes": 32,
                        "align_bytes": 8,
                        "type_info": {
                            "kind": "struct",
                            "qualified_name": "Params",
                        },
                        "type_layout": {
                            "layout_status": "complete",
                            "fields": [
                                {
                                    "index": "params.n",
                                    "name": "n",
                                    "type": "int",
                                    "kind": "scalar",
                                    "size_bytes": 4,
                                    "align_bytes": 4,
                                },
                                {
                                    "index": "params.shape",
                                    "name": "shape",
                                    "type": "int[2]",
                                    "kind": "opaque_val",
                                    "size_bytes": 8,
                                    "align_bytes": 4,
                                    "layout_status": "complete",
                                    "element_count": 2,
                                    "element": {
                                        "index": "params.shape[]",
                                        "name": "$element",
                                        "type": "int",
                                        "kind": "scalar",
                                        "size_bytes": 4,
                                        "align_bytes": 4,
                                    },
                                },
                                {
                                    "index": "params.data",
                                    "name": "data",
                                    "type": "float *",
                                    "kind": "pointer",
                                    "pointer_role": "payload_buffer",
                                    "size_bytes": 8,
                                    "align_bytes": 4,
                                    "pointee_layout": _scalar_pointee_layout("float"),
                                },
                            ],
                        },
                    }
                ],
                "others": {
                    "type_shim_header": "type_shim.v1.cuh",
                    "type_shim_status": "ok",
                },
            }
        ],
    }


def _record_type_info(type_name: str) -> dict:
    return {
        "kind": "struct",
        "qualified_name": type_name,
    }


def _scalar_layout_node(name: str, type_name: str = "int") -> dict:
    size = 4 if type_name in {"int", "float", "unsigned int"} else 8
    return {
        "name": name,
        "type": type_name,
        "kind": "scalar",
        "size_bytes": size,
        "align_bytes": size,
    }


def _payload_pointer_layout_node(name: str, type_name: str = "float *") -> dict:
    pointee_type = type_name.rsplit("*", 1)[0].strip() if "*" in type_name else "float"
    return {
        "name": name,
        "type": type_name,
        "kind": "pointer",
        "pointer_role": "payload_buffer",
        "size_bytes": 8,
        "align_bytes": 8,
        "pointee_layout": _scalar_pointee_layout(pointee_type),
    }


def _field_pointer_cast(target_expr: str) -> str:
    return (
        f"{target_expr} = reinterpret_cast<typename std::remove_reference<decltype({target_expr})>::type>"
        "(const_cast<uint8_t *>(data + offset));"
    )


def _record_layout_node(name: str, type_name: str, fields: list[dict], *, kind: str) -> dict:
    return {
        "name": name,
        "type": type_name,
        "kind": kind,
        "size_bytes": 24,
        "align_bytes": 8,
        "type_info": _record_type_info(type_name),
        "layout_status": "complete",
        "fields": fields,
    }


def _array_layout_node(name: str, type_name: str, element: dict, *, kind: str) -> dict:
    return {
        "name": name,
        "type": type_name,
        "kind": kind,
        "size_bytes": 48,
        "align_bytes": 8,
        "layout_status": "complete",
        "element_count": 2,
        "element": element,
    }


def _type_shim_others() -> dict:
    return {
        "type_shim_header": "type_shim.v1.cuh",
        "type_shim_status": "ok",
    }


def _manifest_example_expected_manifest() -> dict:
    doc = (REPO_ROOT / "manifest.example.md").read_text(encoding="utf-8")
    section = doc.split("## Expected Manifest", 1)[1]
    json_block = section.split("```json", 1)[1].split("```", 1)[0]
    return json.loads(json_block)


def _manifest_example_supported_projection() -> dict:
    manifest = copy.deepcopy(_manifest_example_expected_manifest())
    kernel = manifest["kernels"][0]
    params = kernel["args"][1]
    nested = next(field for field in params["type_layout"]["fields"] if field["name"] == "nested")
    nested["fields"] = [field for field in nested["fields"] if field["name"] != "choice"]
    kernel["constraints"] = [
        {
            "kind": "scalar_le_buffer_len",
            "scalar_arg": 0,
            "buffer_arg": 2,
            "unit": "bytes",
        },
        {
            "kind": "scalar_compare_const",
            "scalar_arg": 0,
            "op": ">=",
            "value": 0,
        },
        {
            "kind": "scalar_compare_const",
            "scalar_arg": 1,
            "scalar_path": ["nested", "inner_count"],
            "op": ">=",
            "value": 0,
        },
    ]
    return manifest


def _manifest_with_layout_arg(arg_name: str, arg_type: str, arg_kind: str, fields: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "kernels": [
            {
                "symbol_name": "kernel",
                "display_name": "kernel",
                "args": [
                    {
                        "index": 0,
                        "name": arg_name,
                        "type": arg_type,
                        "kind": arg_kind,
                        "size_bytes": 64,
                        "align_bytes": 8,
                        "type_info": _record_type_info(arg_type),
                        "type_layout": {
                            "layout_status": "complete",
                            "fields": fields,
                        },
                    }
                ],
                "others": _type_shim_others(),
            }
        ],
    }


def _manifest_with_pointer_scalar_and_struct() -> dict:
    manifest = _manifest_with_pointer_bearing_struct()
    kernel = manifest["kernels"][0]
    kernel["args"] = [
        {
            "index": 0,
            "name": "input",
            "type": "uint8_t *",
            "kind": "pointer",
            "pointer_role": "payload_buffer",
            "size_bytes": 8,
            "align_bytes": 8,
            "pointee_layout": _scalar_pointee_layout("uint8_t"),
        },
        {
            "index": 1,
            "name": "count",
            "type": "int",
            "kind": "scalar",
            "size_bytes": 4,
            "align_bytes": 4,
        },
        {**kernel["args"][0], "index": 2, "name": "params"},
    ]
    return manifest


def _layout_matrix_manifests() -> dict[str, dict]:
    pointer_leaf = _record_layout_node(
        "$element",
        "layout_ptr_leaf",
        [
            _scalar_layout_node("count", "int"),
            _payload_pointer_layout_node("data"),
        ],
        kind="opaque_with_ptr",
    )
    nested_ptr = _record_layout_node(
        "$element",
        "layout_nested_ptr_item",
        [
            _record_layout_node(
                "inner",
                "layout_ptr_leaf",
                [
                    _scalar_layout_node("count", "int"),
                    _payload_pointer_layout_node("data"),
                ],
                kind="opaque_with_ptr",
            ),
            _scalar_layout_node("scale", "int"),
        ],
        kind="opaque_with_ptr",
    )

    return {
        "struct_array_ptr": _manifest_with_layout_arg(
            "cfg",
            "layout_struct_array_ptr",
            "opaque_with_ptr",
            [
                _array_layout_node("items", "layout_ptr_leaf[2]", pointer_leaf, kind="opaque_with_ptr"),
                _scalar_layout_node("tail", "int"),
            ],
        ),
        "struct_array_nested_ptr": _manifest_with_layout_arg(
            "cfg",
            "layout_struct_array_nested_ptr",
            "opaque_with_ptr",
            [
                _array_layout_node("items", "layout_nested_ptr_item[2]", nested_ptr, kind="opaque_with_ptr"),
                _scalar_layout_node("tail", "int"),
            ],
        ),
    }


def _manifest_with_deep_array_pointer() -> dict:
    pointer_element = _payload_pointer_layout_node("$element")
    inner_array = _array_layout_node(
        "ptrs",
        "float *[2]",
        pointer_element,
        kind="opaque_with_ptr",
    )
    nested_element = _record_layout_node(
        "$element",
        "layout_deep_item",
        [
            _scalar_layout_node("count", "int"),
            inner_array,
        ],
        kind="opaque_with_ptr",
    )
    return _manifest_with_layout_arg(
        "cfg",
        "layout_deep_array_ptr",
        "opaque_with_ptr",
        [
            _array_layout_node("items", "layout_deep_item[2]", nested_element, kind="opaque_with_ptr"),
            _scalar_layout_node("tail", "int"),
        ],
    )


def _manifest_with_opaque_val_array_layout() -> dict:
    manifest = _manifest("payload_buffer")
    manifest["kernels"][0]["args"] = [
        {
            "index": 0,
            "name": "cfg",
            "type": "ArrayConfig",
            "kind": "opaque_val",
            "size_bytes": 16,
            "align_bytes": 4,
            "type_info": _record_type_info("ArrayConfig"),
            "type_layout": {
                "layout_status": "complete",
                "fields": [
                    _array_layout_node(
                        "shape",
                        "int[2]",
                        _scalar_layout_node("$element", "int"),
                        kind="opaque_val",
                    ),
                    _scalar_layout_node("tail", "int"),
                ],
            },
        }
    ]
    manifest["kernels"][0]["others"] = _type_shim_others()
    return manifest


def _manifest_with_anonymous_pointer_struct() -> dict:
    return _manifest_with_layout_arg(
        "cfg",
        "layout_anonymous_ptr",
        "opaque_with_ptr",
        [
            _scalar_layout_node("prefix", "int"),
            _record_layout_node(
                "anon1",
                "layout_anonymous_ptr::(anonymous struct)",
                [
                    _scalar_layout_node("value", "int"),
                    _payload_pointer_layout_node("ptr", "int *"),
                ],
                kind="opaque_with_ptr",
            ),
            _scalar_layout_node("tail", "int"),
        ],
    )


def _manifest_with_nested_scalar_constraints() -> dict:
    return {
        "schema_version": 1,
        "kernels": [
            {
                "symbol_name": "kernel",
                "display_name": "kernel",
                "args": [
                    {
                        "index": 0,
                        "name": "params",
                        "type": "Params",
                        "kind": "opaque_with_ptr",
                        "size_bytes": 32,
                        "align_bytes": 8,
                        "type_info": _record_type_info("Params"),
                        "type_layout": {
                            "layout_status": "complete",
                            "fields": [
                                _scalar_layout_node("count", "int"),
                                _record_layout_node(
                                    "nested",
                                    "Nested",
                                    [
                                        _scalar_layout_node("inner_count", "int"),
                                        _record_layout_node(
                                            "choice",
                                            "Choice",
                                            [_scalar_layout_node("tag", "int")],
                                            kind="opaque_with_ptr",
                                        ),
                                    ],
                                    kind="opaque_with_ptr",
                                ),
                            ],
                        },
                    }
                ],
                "constraints": [
                    {
                        "kind": "scalar_compare_const",
                        "scalar_arg": 0,
                        "scalar_path": ["nested", "inner_count"],
                        "op": ">=",
                        "value": 0,
                    },
                    {
                        "kind": "scalar_compare_scalar",
                        "lhs_arg": 0,
                        "lhs_path": ["nested", "choice", "tag"],
                        "op": "<=",
                        "rhs_arg": 0,
                        "rhs_path": ["count"],
                    },
                ],
                "others": _type_shim_others(),
            }
        ],
    }


def _contract(manifest: dict):
    return build_kernel_contract(
        kernel_id="kernel__test",
        manifest=manifest,
        manifest_path=Path("manifest.json"),
        metadata={"kernel_symbol": "kernel", "build_status": "built"},
    )


def _manifest_example_contract(manifest: dict):
    return build_kernel_contract(
        kernel_id="complex_args_kernel",
        manifest=manifest,
        manifest_path=Path("manifest.json"),
        metadata={"kernel_symbol": "complex_args_kernel", "build_status": "built"},
    )


def _write_synthetic_phase1_run(run_dir: Path, manifest: dict) -> Path:
    kernel_dir = run_dir / "complex_args_kernel"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.bc").write_bytes(b"synthetic-bitcode")
    (kernel_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (kernel_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kernel_id": "complex_args_kernel",
                "kernel_symbol": "complex_args_kernel",
                "build_status": "built",
                "output_hashes": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "synthetic-example",
                "kernels": [
                    {
                        "kernel_id": "complex_args_kernel",
                        "dir": "complex_args_kernel",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return kernel_dir


def _write_llvm_phase1_run(
    run_dir: Path,
    manifest: dict,
    *,
    barrier_mode: str = "none",
) -> Path:
    kernel_dir = _write_synthetic_phase1_run(run_dir, manifest)
    metadata_path = kernel_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["kernel_symbol"] = manifest["kernels"][0]["symbol_name"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    declarations = "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n"
    if barrier_mode == "none":
        body = (
            "  %block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n"
            "  store i32 %block, ptr %input, align 1\n"
        )
    elif barrier_mode == "straight":
        body = (
            "  %block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n"
            "  call void @llvm.nvvm.barrier0()\n"
            "  store i32 %block, ptr %input, align 1\n"
        )
        declarations += "declare void @llvm.nvvm.barrier0()\n"
    elif barrier_mode == "divergent":
        body = (
            "  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
            "  %is_zero = icmp eq i32 %tid, 0\n"
            "  br i1 %is_zero, label %needs_barrier, label %done\n\n"
            "needs_barrier:\n"
            "  call void @llvm.nvvm.barrier0()\n"
            "  br label %done\n\n"
            "done:\n"
        )
        declarations = (
            "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
            "declare void @llvm.nvvm.barrier0()\n"
        )
    elif barrier_mode == "loop":
        body = (
            "  br label %loop\n\n"
            "loop:\n"
            "  %i = phi i32 [0, %entry], [%next, %loop]\n"
            "  call void @llvm.nvvm.barrier0()\n"
            "  %next = add i32 %i, 1\n"
            "  %keep_going = icmp ult i32 %next, 4\n"
            "  br i1 %keep_going, label %loop, label %done\n\n"
            "done:\n"
        )
        declarations = "declare void @llvm.nvvm.barrier0()\n"
    elif barrier_mode == "warp":
        body = (
            "  %value = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 0, i32 1, i32 31, i32 -1)\n"
            "  store i32 %value, ptr %input, align 1\n"
        )
        declarations = (
            "declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)\n"
        )
    else:
        raise ValueError(f"unknown barrier mode: {barrier_mode}")
    llvm_ir = (
        "target triple = \"nvptx64-nvidia-cuda\"\n\n"
        "define ptx_kernel void @kernel(ptr %input, i64 %size) {\n"
        "entry:\n"
        f"{body}"
        "  ret void\n"
        "}\n\n"
        f"{declarations}"
    )
    llvm_ir_path = kernel_dir / "kernel.ll"
    llvm_ir_path.write_text(llvm_ir, encoding="utf-8")
    subprocess.run(
        ["llvm-as-22", str(llvm_ir_path), "-o", str(kernel_dir / "kernel.bc")],
        check=True,
        cwd=REPO_ROOT,
    )
    return kernel_dir


class Phase2ContractTest(unittest.TestCase):
    def test_vconfig_request_defaults_true_and_can_be_disabled(self) -> None:
        default_contract = _contract(_manifest("payload_buffer"))
        disabled_manifest = _manifest("payload_buffer")
        disabled_manifest["kernels"][0]["launch_policy"] = {
            "vconfig_reserved": False
        }
        disabled_contract = _contract(disabled_manifest)

        self.assertTrue(
            hasattr(default_contract, "vconfig_requested"),
            "KernelContract does not expose VConfig eligibility",
        )
        self.assertTrue(default_contract.vconfig_requested)
        self.assertFalse(disabled_contract.vconfig_requested)
        self.assertTrue(default_contract.to_plan_dict()["vconfig_requested"])

    def test_vconfig_request_rejects_non_boolean_manifest_value(self) -> None:
        manifest = _manifest("payload_buffer")
        manifest["kernels"][0]["launch_policy"] = {
            "vconfig_reserved": "enabled"
        }

        with self.assertRaisesRegex(
            ValueError,
            "manifest launch_policy.vconfig_reserved invalid",
        ):
            _contract(manifest)

    def test_phase2_enables_requested_vconfig_and_records_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_llvm_phase1_run(
                run_dir, _manifest("payload_buffer")
            )

            summary = Phase2Runner().run(run_dir)
            phase2_dir = kernel_dir / "phase2"
            build_spec = json.loads(
                (phase2_dir / "build_spec.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (phase2_dir / "metadata.phase2.json").read_text(encoding="utf-8")
            )
            transformed = subprocess.run(
                ["llvm-dis-22", str(phase2_dir / "kernel.device.bc"), "-o", "-"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(summary["counts"]["built"], 1)
        self.assertTrue(build_spec.get("vconfig_requested"))
        self.assertTrue(build_spec.get("vconfig_enabled"))
        self.assertTrue(metadata.get("vconfig_requested"))
        self.assertTrue(metadata.get("vconfig_enabled"))
        self.assertTrue(summary["results"][0].get("vconfig_enabled"))
        self.assertIn('"rapid.vconfig.processed"', transformed)

    def test_phase2_renames_dotted_globals_before_nvptx_lowering(self) -> None:
        llvm_as = shutil.which("llvm-as-22")
        if llvm_as is None:
            self.skipTest("llvm-as-22 is unavailable")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_synthetic_phase1_run(
                run_dir, _manifest("payload_buffer")
            )
            metadata_path = kernel_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["kernel_symbol"] = "kernel"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            llvm_ir = (
                "target triple = \"nvptx64-nvidia-cuda\"\n\n"
                "@.str = external hidden unnamed_addr constant [24 x i8], align 1\n\n"
                "define ptx_kernel void @kernel(ptr %input, i64 %size) {\n"
                "entry:\n"
                "  %printed = call i32 @vprintf(ptr nonnull @.str, ptr null)\n"
                "  store i32 %printed, ptr %input, align 1\n"
                "  ret void\n"
                "}\n\n"
                "declare i32 @vprintf(ptr, ptr) local_unnamed_addr\n"
            )
            llvm_ir_path = kernel_dir / "kernel.ll"
            llvm_ir_path.write_text(llvm_ir, encoding="utf-8")
            subprocess.run(
                [llvm_as, str(llvm_ir_path), "-o", str(kernel_dir / "kernel.bc")],
                check=True,
                cwd=REPO_ROOT,
            )

            summary = Phase2Runner().run(run_dir)
            transformed = subprocess.run(
                [
                    "llvm-dis-22",
                    str(kernel_dir / "phase2/kernel.device.bc"),
                    "-o",
                    "-",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(summary["counts"]["built"], 1)
        self.assertNotIn("@.str", transformed)
        self.assertIn("@rapid_nvptx_global_str", transformed)

    def test_phase2_vconfig_opt_out_keeps_uninstrumented_entry(self) -> None:
        manifest = _manifest("payload_buffer")
        manifest["kernels"][0]["launch_policy"] = {"vconfig_reserved": False}
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_llvm_phase1_run(run_dir, manifest)

            summary = Phase2Runner().run(run_dir)
            phase2_dir = kernel_dir / "phase2"
            build_spec = json.loads(
                (phase2_dir / "build_spec.json").read_text(encoding="utf-8")
            )
            transformed = subprocess.run(
                ["llvm-dis-22", str(phase2_dir / "kernel.device.bc"), "-o", "-"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(summary["counts"]["built"], 1)
        self.assertIs(build_spec.get("vconfig_requested"), False)
        self.assertIs(build_spec.get("vconfig_enabled"), False)
        self.assertNotIn("vconfig_disabled_reason", build_spec)
        self.assertNotIn('"rapid.vconfig.processed"', transformed)

    def test_phase2_enables_straight_line_barrier_vconfig(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_llvm_phase1_run(
                run_dir,
                _manifest("payload_buffer"),
                barrier_mode="straight",
            )

            summary = Phase2Runner().run(run_dir)
            phase2_dir = kernel_dir / "phase2"
            build_spec = json.loads(
                (phase2_dir / "build_spec.json").read_text(encoding="utf-8")
            )
            transformed = subprocess.run(
                ["llvm-dis-22", str(phase2_dir / "kernel.device.bc"), "-o", "-"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(summary["counts"]["built"], 1)
        self.assertTrue(build_spec.get("vconfig_requested"))
        self.assertIs(build_spec.get("vconfig_enabled"), True)
        inactive_block = transformed.split("vdim.inactive:")[1].split(
            "vdim.active:"
        )[0]
        self.assertIn(
            "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)",
            inactive_block,
        )

    def test_phase2_enables_fixed_trip_loop_barrier_vconfig(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_llvm_phase1_run(
                run_dir,
                _manifest("payload_buffer"),
                barrier_mode="loop",
            )

            summary = Phase2Runner().run(run_dir)
            phase2_dir = kernel_dir / "phase2"
            build_spec = json.loads(
                (phase2_dir / "build_spec.json").read_text(encoding="utf-8")
            )
            transformed = subprocess.run(
                ["llvm-dis-22", str(phase2_dir / "kernel.device.bc"), "-o", "-"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(summary["counts"]["built"], 1)
        self.assertTrue(build_spec.get("vconfig_requested"))
        self.assertIs(build_spec.get("vconfig_enabled"), True)
        self.assertIn("vdim.inactive.sync.loop", transformed)
        self.assertIn(
            "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)",
            transformed,
        )

    def test_phase2_preserves_logical_shape_for_warp_collective_vconfig(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_llvm_phase1_run(
                run_dir,
                _manifest("payload_buffer"),
                barrier_mode="warp",
            )

            summary = Phase2Runner().run(run_dir)
            phase2_dir = kernel_dir / "phase2"
            build_spec = json.loads(
                (phase2_dir / "build_spec.json").read_text(encoding="utf-8")
            )
            transformed = subprocess.run(
                ["llvm-dis-22", str(phase2_dir / "kernel.device.bc"), "-o", "-"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        self.assertEqual(summary["counts"]["built"], 1)
        self.assertTrue(build_spec.get("vconfig_requested"))
        self.assertIs(build_spec.get("vconfig_enabled"), True)
        self.assertIs(build_spec.get("vconfig_warp_aligned"), True)
        self.assertIs(summary["results"][0].get("vconfig_warp_aligned"), True)
        self.assertIn("%vdim.block.x = load i32", transformed)
        self.assertIn("urem i32 %vdim.thread.linear, %vdim.block.x", transformed)
        self.assertNotIn("vdim.block.x.warp.rounded", transformed)

    def test_phase2_disables_divergent_barrier_vconfig_without_rejecting_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_llvm_phase1_run(
                run_dir,
                _manifest("payload_buffer"),
                barrier_mode="divergent",
            )

            summary = Phase2Runner().run(run_dir)
            phase2_dir = kernel_dir / "phase2"
            build_spec = json.loads(
                (phase2_dir / "build_spec.json").read_text(encoding="utf-8")
            )
            metadata = json.loads(
                (phase2_dir / "metadata.phase2.json").read_text(encoding="utf-8")
            )

        expected_reason = "vconfig_barrier_unsupported"
        self.assertEqual(summary["counts"]["built"], 1)
        self.assertTrue(build_spec.get("vconfig_requested"))
        self.assertIs(build_spec.get("vconfig_enabled"), False)
        self.assertEqual(build_spec.get("vconfig_disabled_reason"), expected_reason)
        self.assertEqual(metadata.get("vconfig_disabled_reason"), expected_reason)
        self.assertEqual(
            summary["results"][0].get("vconfig_disabled_reason"),
            expected_reason,
        )

    def test_target_layout_codegen_uses_exact_payload_slot_count(self) -> None:
        two_payloads = _manifest("payload_buffer")
        second = copy.deepcopy(two_payloads["kernels"][0]["args"][0])
        second["index"] = 2
        second["name"] = "output"
        two_payloads["kernels"][0]["args"].append(second)
        cases = (
            (_contract(_manifest_with_scalar_only_args()), 0),
            (_contract(_manifest("payload_buffer")), 1),
            (_contract(two_payloads), 2),
        )

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            runner = Phase2Runner()
            for contract, expected_count in cases:
                runner.target_layout_generator.generate(
                    contract=contract,
                    phase2_dir=phase2_dir,
                )
                layout = (
                    phase2_dir / "gen" / "rapid_target_layout.v1.h"
                ).read_text(encoding="utf-8")
                self.assertIn(
                    f"#define RAPID_PAYLOAD_SLOT_COUNT {expected_count}u",
                    layout,
                )
                self.assertIn(
                    "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT \\\n  (RAPID_PAYLOAD_SLOT_COUNT == 0u ? 1u : RAPID_PAYLOAD_SLOT_COUNT)",
                    layout,
                )

    def test_rewrite_appends_context_only_to_generated_entry(self) -> None:
        llvm_ir = """\
define ptx_kernel void @kernel(ptr %input, i32 %count) {
entry:
  ret void
}
"""
        contract = _contract(_manifest_with_scalar_len_constraint("count"))

        rewritten, _ = _rewrite_module_text(
            ll_text=llvm_ir,
            rewrite_plan=contract.to_plan_dict(),
        )
        functions = {function.name: function for function in _parse_functions(rewritten)}

        self.assertEqual(functions["kernel"].param_count, 2)
        self.assertTrue(functions["kernel"].has_ptx_kernel)
        self.assertNotIn("%__rapid_context", functions["kernel"].signature_text)
        entry = functions[contract.entry_symbol]
        self.assertEqual(entry.param_count, 3)
        self.assertFalse(entry.has_ptx_kernel)
        self.assertIn("ptr %__rapid_context", entry.signature_text)

    def test_feedback_codegen_retains_payload_bounds_and_passes_context(self) -> None:
        contract = _contract(_manifest_with_scalar_len_constraint("count"))

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            InvokeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            decode = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")
            invoke = (phase2_dir / "gen" / "fuzzer_invoke.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("uint64_t __rapid_payload_0_len_bytes;", decode)
        self.assertIn("decoded.__rapid_payload_0_len_bytes = input_len_u64;", decode)
        self.assertIn("RapidKernelContext *context", invoke)
        self.assertIn("fuzzer_feedback_prepare_v1", invoke)
        self.assertIn("const RapidVConfig &vconfig", invoke)
        self.assertIn("context->vconfig = vconfig;", invoke)
        self.assertIn(
            "static_assert(RAPID_PAYLOAD_SLOT_COUNT == 1u",
            invoke,
        )
        self.assertIn(
            "context->feedback.bounds_count = RAPID_PAYLOAD_SLOT_COUNT;",
            invoke,
        )
        self.assertIn("reinterpret_cast<uintptr_t>(decoded.input)", invoke)
        self.assertIn("decoded.__rapid_payload_0_len_bytes", invoke)
        self.assertIn(
            "RapidPayloadBounds{\n"
            "        reinterpret_cast<uintptr_t>(decoded.input),\n"
            "        decoded.__rapid_payload_0_len_bytes}",
            invoke,
        )
        self.assertNotIn("4u, 0u, 0u", invoke)
        self.assertIn(f"{contract.entry_symbol}(decoded.input, decoded.count, context);", invoke)
        self.assertIn("fuzzer_invoke_v1(const DecodedKernelArgs &decoded) {", invoke)
        self.assertIn("fuzzer_invoke_v1(decoded, nullptr);", invoke)

    def test_build_spec_records_feedback_entry_abi(self) -> None:
        manifest = _manifest("payload_buffer")
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            kernel_dir = Path(td) / "kernel"
            phase2_dir = kernel_dir / "phase2"
            phase2_dir.mkdir(parents=True)
            artifacts = KernelPhase1Artifacts(
                kernel_id="kernel",
                kernel_dir=kernel_dir,
                phase2_dir=phase2_dir,
                kernel_bc=kernel_dir / "kernel.bc",
                manifest_path=kernel_dir / "manifest.json",
                manifest=manifest,
                metadata={"kernel_symbol": "kernel", "build_status": "built"},
            )
            Phase2Runner()._write_build_spec(
                kernel=artifacts,
                contract=contract,
                vconfig_result=VConfigRewriteResult(requested=True, enabled=True),
            )
            build_spec = json.loads((phase2_dir / "build_spec.json").read_text(encoding="utf-8"))

        self.assertEqual(build_spec["entry_abi_version"], 1)
        self.assertEqual(build_spec["entry_context"], "rapid_kernel_context_v1")
        self.assertEqual(
            build_spec["target_layout_header"],
            "gen/rapid_target_layout.v1.h",
        )
        self.assertEqual(build_spec["feedback_payload_slots"][0]["arg_index"], 0)
        self.assertEqual(build_spec["feedback_payload_slots"][0]["slot"], 0)
        self.assertEqual(
            build_spec["feedback_payload_slots"][0]["elem_size"],
            contract.feedback_payload_slots[0].elem_size,
        )

    def test_memory_feedback_is_disabled_when_payload_slots_exceed_context_capacity(self) -> None:
        manifest = _manifest("payload_buffer")
        template = manifest["kernels"][0]["args"][0]
        manifest["kernels"][0]["args"] = []
        for index in range(33):
            arg = copy.deepcopy(template)
            arg["index"] = index
            arg["name"] = f"input_{index}"
            manifest["kernels"][0]["args"].append(arg)

        contract = _contract(manifest)

        self.assertTrue(contract.supported)
        self.assertEqual(len(contract.payload_slots), 33)
        self.assertFalse(contract.feedback_memory_enabled)
        self.assertEqual(contract.feedback_payload_slots, [])

    def test_manifest_example_expected_manifest_is_schema_valid(self) -> None:
        if jsonschema is None:
            self.skipTest("jsonschema not available")

        schema = json.loads((REPO_ROOT / "docs" / "kernel-manifest.schema.json").read_text(encoding="utf-8"))
        validator_cls = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(_manifest_example_expected_manifest()), key=lambda error: error.path)

        self.assertEqual(errors, [])

    def test_schema_rejects_root_type_layout_index(self) -> None:
        if jsonschema is None:
            self.skipTest("jsonschema not available")

        manifest = _manifest_with_pointer_bearing_struct()
        manifest["kernels"][0]["args"][0]["type_layout"]["index"] = "params"
        schema = json.loads((REPO_ROOT / "docs" / "kernel-manifest.schema.json").read_text(encoding="utf-8"))
        validator_cls = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(manifest), key=lambda error: error.path)

        self.assertNotEqual(errors, [])
        self.assertTrue(any("Additional properties are not allowed" in error.message for error in errors))

    def test_manifest_example_full_manifest_fails_on_nested_union(self) -> None:
        contract = _manifest_example_contract(_manifest_example_expected_manifest())

        self.assertFalse(contract.supported)
        self.assertEqual(contract.support_reason, "params.nested.choice:union_not_supported")
        self.assertTrue(_is_phase2_input_invalid(contract.support_reason))

    def test_manifest_example_without_unsupported_constraints_fails_on_nested_union(self) -> None:
        manifest = _manifest_example_expected_manifest()
        manifest["kernels"][0]["constraints"] = []

        contract = _manifest_example_contract(manifest)

        self.assertFalse(contract.supported)
        self.assertEqual(contract.support_reason, "params.nested.choice:union_not_supported")
        self.assertTrue(_is_phase2_input_invalid(contract.support_reason))

    def test_manifest_example_strict_outcomes_are_written_by_phase2_runner(self) -> None:
        cases = [
            (
                "full_manifest",
                _manifest_example_expected_manifest(),
                "params.nested.choice:union_not_supported",
            ),
            (
                "without_constraints",
                {**_manifest_example_expected_manifest(), "kernels": copy.deepcopy(_manifest_example_expected_manifest()["kernels"])},
                "params.nested.choice:union_not_supported",
            ),
        ]
        cases[1][1]["kernels"][0]["constraints"] = []

        for _case_name, manifest, expected_detail in cases:
            with self.subTest(expected_detail=expected_detail), tempfile.TemporaryDirectory() as td:
                run_dir = Path(td)
                kernel_dir = _write_synthetic_phase1_run(run_dir, manifest)

                summary = Phase2Runner().run(run_dir)
                summary_path = run_dir / "rewrite_summary.json"
                phase2_meta = json.loads((kernel_dir / "phase2" / "metadata.phase2.json").read_text(encoding="utf-8"))

                self.assertTrue(summary_path.is_file())
                self.assertEqual([path.name for path in run_dir.glob("*summary.json")], ["rewrite_summary.json"])
                self.assertEqual(
                    json.loads(summary_path.read_text(encoding="utf-8")),
                    summary,
                )
                self.assertEqual(summary["counts"]["failed"], 1)
                self.assertEqual(phase2_meta["phase2_status"], "failed")
                self.assertEqual(phase2_meta["failure_reason"], "phase2_input_invalid")
                self.assertEqual(phase2_meta["failure_detail"], expected_detail)
                if expected_detail.endswith(":union_not_supported"):
                    self.assertEqual(
                        phase2_meta.get("failure_context"),
                        {
                            "support_path": "params.nested.choice",
                            "reason_code": "union_not_supported",
                        },
                    )

    def test_manifest_example_supported_projection_generates_documented_decode_and_invoke(self) -> None:
        contract = _manifest_example_contract(_manifest_example_supported_projection())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            InvokeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            decode = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")
            invoke = (phase2_dir / "gen" / "fuzzer_invoke.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertEqual([arg.name for arg in contract.args], ["seed", "params", "output"])
        self.assertIn("int seed;", decode)
        self.assertIn("KernelParams params;", decode)
        self.assertIn("float * output;", decode)
        self.assertIn("decoded.params.count = read_scalar_le<int>(data, offset);", decode)
        self.assertIn("decoded.params.weights[0] = read_f32_le(data, offset);", decode)
        self.assertIn("decoded.params.weights[3] = read_f32_le(data, offset);", decode)
        self.assertIn(_field_pointer_cast("decoded.params.data"), decode)
        self.assertIn(_field_pointer_cast("decoded.params.kind"), decode)
        self.assertIn("uint64_t params_nested_lanes_0_len_u64", decode)
        self.assertIn("uint64_t params_nested_lanes_3_len_u64", decode)
        self.assertNotIn("params_nested_params_nested_lanes", decode)
        self.assertIn(_field_pointer_cast("decoded.params.nested.lanes[3]"), decode)
        self.assertIn("decoded.params.nested.value = read_scalar_le<int>(data, offset);", decode)
        self.assertIn(_field_pointer_cast("decoded.params.nested.ptr"), decode)
        self.assertIn("uint64_t params_nested_anon3_ptr_len_u64", decode)
        self.assertNotIn("decoded.params.nested.anon3.value", decode)
        self.assertNotIn("decoded.params.nested.anon3.ptr", decode)
        self.assertIn(
            "decoded.output = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));",
            decode,
        )
        self.assertLess(
            decode.index("RAPID_DECODE_ASSERT(decoded.seed <= output_len_u64);"),
            decode.index("RAPID_DECODE_ASSERT(decoded.seed >= 0);"),
        )
        self.assertLess(
            decode.index("RAPID_DECODE_ASSERT(decoded.seed >= 0);"),
            decode.index("RAPID_DECODE_ASSERT(decoded.params.nested.inner_count >= 0);"),
        )
        self.assertIn(
            "extern \"C\" __device__ void __rapid_entry__complex_args_kernel(int seed, KernelParams params, float * output, RapidKernelContext *context);",
            invoke,
        )
        self.assertIn(
            "__rapid_entry__complex_args_kernel(decoded.seed, decoded.params, decoded.output, context);",
            invoke,
        )
        self.assertIn(
            "context->feedback.bounds_count = RAPID_PAYLOAD_SLOT_COUNT;",
            invoke,
        )

    def test_payload_buffer_pointer_is_supported(self) -> None:
        contract = _contract(_manifest("payload_buffer"))

        self.assertTrue(contract.supported)
        self.assertEqual(contract.args[0].kind, "pointer")
        self.assertEqual(contract.args[0].pointer_role, "payload_buffer")
        self.assertTrue(contract.args[0].is_payload_buffer_pointer)
        self.assertEqual(contract.to_plan_dict()["payload_args"][0]["pointer_role"], "payload_buffer")

    def test_pointer_typedef_alias_uses_manifest_kind_not_type_suffix(self) -> None:
        manifest = _manifest("payload_buffer")
        pointer = manifest["kernels"][0]["args"][0]
        pointer["type"] = "IntPtr"
        pointer["pointee_layout"] = _scalar_pointee_layout("int")
        manifest["kernels"][0]["others"] = _type_shim_others()

        contract = _contract(manifest)

        self.assertTrue(contract.supported)
        self.assertEqual(contract.args[0].kind, "pointer")
        self.assertEqual(contract.args[0].codegen_type, "IntPtr")
        self.assertTrue(contract.args[0].is_payload_buffer_pointer)
        self.assertTrue(contract.needs_shim)

    def test_pointer_to_record_typedef_alias_uses_pointee_layout_for_shim(self) -> None:
        manifest = _manifest("payload_buffer")
        pointer = manifest["kernels"][0]["args"][0]
        pointer["type"] = "RecordPtr"
        pointer["pointee_layout"] = _record_pointee_layout("Record")
        manifest["kernels"][0]["others"] = _type_shim_others()

        contract = _contract(manifest)

        self.assertTrue(contract.supported)
        self.assertEqual(contract.args[0].codegen_type, "RecordPtr")
        self.assertTrue(contract.args[0].is_payload_buffer_pointer)
        self.assertTrue(contract.needs_shim)

    def test_payload_buffer_decode_uses_pointee_alignment_when_wider_than_pointer(self) -> None:
        manifest = _manifest("payload_buffer")
        pointer = manifest["kernels"][0]["args"][0]
        pointer["type"] = "uint4 *"
        pointer["align_bytes"] = 8
        pointer["pointee_layout"] = {
            "name": "$pointee",
            "type": "uint4",
            "kind": "opaque_val",
            "size_bytes": 16,
            "align_bytes": 16,
        }
        manifest["kernels"][0]["others"] = _type_shim_others()
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertIn("offset = rapid_payload_len_offset(offset, 16);", content)
        self.assertLess(
            content.index("offset = rapid_payload_len_offset(offset, 16);"),
            content.index("uint64_t input_len_u64"),
        )

    def test_nested_payload_buffer_decode_uses_wider_pointee_alignment(self) -> None:
        pointer = _payload_pointer_layout_node("data", "uint4 *")
        pointer["align_bytes"] = 8
        pointer["pointee_layout"] = {
            "index": "params.data.*",
            "name": "$pointee",
            "type": "uint4",
            "kind": "opaque_val",
            "size_bytes": 16,
            "align_bytes": 16,
        }
        contract = _contract(
            _manifest_with_layout_arg("params", "Params", "opaque_with_ptr", [pointer])
        )

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertIn("offset = rapid_payload_len_offset(offset, 16);", content)
        self.assertIn(
            "RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.data) % 16) == 0);",
            content,
        )

    def test_scalar_le_buffer_len_constraint_is_parsed_to_predicate(self) -> None:
        contract = _contract(_manifest_with_scalar_len_constraint("count"))

        self.assertEqual(len(contract.constraint), 1)
        predicate = contract.constraint[0]
        self.assertIsInstance(predicate.lhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.arg_index, 1)
        self.assertEqual(predicate.op, "<=")
        self.assertIsInstance(predicate.rhs, PayloadLenExpr)
        self.assertEqual(predicate.rhs.arg_index, 0)
        self.assertEqual(contract.to_plan_dict()["constraint"][0]["op"], "<=")

    def test_nested_scalar_le_buffer_len_constraint_is_parsed_to_predicate(self) -> None:
        manifest = _manifest_with_pointer_bearing_struct()
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "scalar_le_buffer_len",
                "scalar_arg": 0,
                "scalar_path": ["n"],
                "buffer_arg": 0,
                "buffer_path": ["data"],
                "unit": "bytes",
            }
        ]

        contract = _contract(manifest)

        self.assertEqual(len(contract.constraint), 1)
        predicate = contract.constraint[0]
        self.assertIsInstance(predicate.lhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.arg_index, 0)
        self.assertEqual(predicate.lhs.field_path, ("n",))
        self.assertEqual(predicate.op, "<=")
        self.assertIsInstance(predicate.rhs, PayloadLenExpr)
        self.assertEqual(predicate.rhs.arg_index, 0)
        self.assertEqual(predicate.rhs.field_path, ("data",))

    def test_scalar_le_buffer_len_elements_uses_pointee_size(self) -> None:
        manifest = _manifest_with_pointer_bearing_struct()
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "scalar_le_buffer_len",
                "scalar_arg": 0,
                "scalar_path": ["n"],
                "buffer_arg": 0,
                "buffer_path": ["data"],
                "unit": "elements",
            }
        ]

        contract = _contract(manifest)

        predicate = contract.constraint[0]
        self.assertIsInstance(predicate.lhs, BinaryExpr)
        self.assertIsInstance(predicate.lhs.lhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.lhs.field_path, ("n",))
        self.assertEqual(predicate.lhs.op, "*")
        self.assertIsInstance(predicate.lhs.rhs, ConstExpr)
        self.assertEqual(predicate.lhs.rhs.value, 4)
        self.assertEqual(predicate.op, "<=")
        self.assertIsInstance(predicate.rhs, PayloadLenExpr)
        self.assertEqual(predicate.rhs.field_path, ("data",))

    def test_scalar_product_le_const_is_parsed_and_emitted(self) -> None:
        manifest = _manifest_with_scalar_only_args()
        kernel = manifest["kernels"][0]
        kernel["args"][0]["name"] = "batch"
        kernel["args"][1].update({"name": "channels", "type": "uint32_t"})
        kernel["constraints"] = [
            {
                "kind": "scalar_product_le_const",
                "lhs_arg": 0,
                "rhs_arg": 1,
                "value": 2048,
                "repair_arg": 1,
            }
        ]

        if jsonschema is not None:
            schema = json.loads((REPO_ROOT / "docs" / "kernel-manifest.schema.json").read_text())
            jsonschema.validate(instance=manifest, schema=schema)

        contract = _contract(manifest)

        self.assertEqual(len(contract.constraint), 1)
        predicate = contract.constraint[0]
        self.assertIsInstance(predicate.lhs, BinaryExpr)
        self.assertEqual(predicate.lhs.op, "*")
        self.assertIsInstance(predicate.lhs.lhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.lhs.arg_index, 0)
        self.assertIsInstance(predicate.lhs.rhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.rhs.arg_index, 1)
        self.assertEqual(predicate.op, "<=")
        self.assertIsInstance(predicate.rhs, ConstExpr)
        self.assertEqual(predicate.rhs.value, 2048)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(
                encoding="utf-8"
            )

        self.assertIn(
            "RAPID_DECODE_ASSERT((decoded.batch * decoded.channels) <= 2048);",
            content,
        )

    def test_count_fits_buffer_constraint_is_parsed_to_binary_predicate(self) -> None:
        manifest = _manifest_with_pointer_bearing_struct()
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "count_fits_buffer",
                "count_arg": 0,
                "count_path": ["n"],
                "buffer_arg": 0,
                "buffer_path": ["data"],
                "elem_size_bytes": 4,
            }
        ]

        contract = _contract(manifest)

        predicate = contract.constraint[0]
        self.assertIsInstance(predicate.lhs, BinaryExpr)
        self.assertIsInstance(predicate.lhs.lhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.lhs.field_path, ("n",))
        self.assertEqual(predicate.lhs.op, "*")
        self.assertIsInstance(predicate.lhs.rhs, ConstExpr)
        self.assertEqual(predicate.lhs.rhs.value, 4)
        self.assertEqual(predicate.op, "<=")
        self.assertIsInstance(predicate.rhs, PayloadLenExpr)
        self.assertEqual(predicate.rhs.field_path, ("data",))

    def test_buffer_elements_lt_scalar_constraint_is_fuzzer_side_only(self) -> None:
        manifest = _manifest("payload_buffer")
        manifest["kernels"][0]["args"][0]["pointee_layout"]["index"] = "input.*"
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "buffer_elements_lt_scalar",
                "buffer_arg": 0,
                "scalar_arg": 1,
                "elem_size_bytes": 4,
            }
        ]

        if jsonschema is not None:
            schema = json.loads((REPO_ROOT / "docs" / "kernel-manifest.schema.json").read_text())
            jsonschema.validate(instance=manifest, schema=schema)

        contract = _contract(manifest)

        self.assertTrue(contract.supported)
        self.assertEqual(contract.constraint, [])

    def test_scalar_le_logical_block_dim_constraint_is_fuzzer_side_only(self) -> None:
        manifest = _manifest("payload_buffer")
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "scalar_le_logical_block_dim",
                "scalar_arg": 1,
                "dimension": "x",
            }
        ]

        if jsonschema is not None:
            schema = json.loads((REPO_ROOT / "docs" / "kernel-manifest.schema.json").read_text())
            jsonschema.validate(instance=manifest, schema=schema)

        contract = _contract(manifest)

        self.assertTrue(contract.supported)
        self.assertEqual(contract.constraint, [])

    def test_expression_compare_parses_generic_product_and_payload_len(self) -> None:
        manifest = _manifest("payload_buffer")
        kernel = manifest["kernels"][0]
        kernel["args"][0]["pointee_layout"]["index"] = "input.*"
        kernel["args"].append(
            {
                "index": 2,
                "name": "rows",
                "type": "int",
                "kind": "scalar",
                "size_bytes": 4,
                "align_bytes": 4,
            }
        )
        kernel["constraints"] = [
            {
                "kind": "expression_compare",
                "lhs": {
                    "kind": "binary",
                    "op": "*",
                    "lhs": {
                        "kind": "binary",
                        "op": "*",
                        "lhs": {"kind": "arg_value", "arg": 1},
                        "rhs": {"kind": "arg_value", "arg": 2},
                    },
                    "rhs": {"kind": "const", "value": 4},
                },
                "op": "<=",
                "rhs": {"kind": "payload_len", "arg": 0},
                "repair": {"kind": "resize_payload", "arg": 0},
            }
        ]

        if jsonschema is not None:
            schema = json.loads((REPO_ROOT / "docs" / "kernel-manifest.schema.json").read_text())
            jsonschema.validate(instance=manifest, schema=schema)

        contract = _contract(manifest)

        predicate = contract.constraint[0]
        self.assertIsInstance(predicate.lhs, BinaryExpr)
        self.assertEqual(predicate.lhs.op, "*")
        self.assertIsInstance(predicate.lhs.lhs, BinaryExpr)
        self.assertIsInstance(predicate.lhs.lhs.lhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.lhs.lhs.arg_index, 1)
        self.assertIsInstance(predicate.lhs.lhs.rhs, ArgValueExpr)
        self.assertEqual(predicate.lhs.lhs.rhs.arg_index, 2)
        self.assertIsInstance(predicate.lhs.rhs, ConstExpr)
        self.assertEqual(predicate.lhs.rhs.value, 4)
        self.assertIsInstance(predicate.rhs, PayloadLenExpr)
        self.assertEqual(predicate.rhs.arg_index, 0)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertIn(
            "RAPID_DECODE_ASSERT(((decoded.size * decoded.rows) * 4) <= input_len_u64);",
            content,
        )

    def test_decode_codegen_uses_manifest_constraints_not_arg_name(self) -> None:
        contract = _contract(_manifest_with_scalar_len_constraint("count"))

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("uint64_t input_len_u64", content)
        self.assertIn("RAPID_DECODE_ASSERT(decoded.count <= input_len_u64);", content)
        self.assertIn("#if !defined(NDEBUG) && defined(RAPID_DECODE_DEBUG_ASSERT)", content)
        self.assertIn("#define RAPID_DECODE_ASSERT(cond) ((void)0)", content)
        self.assertNotIn("decoded.size > input_len_u64", content)

    def test_decode_codegen_uses_nested_payload_len_constraints(self) -> None:
        manifest = _manifest_with_pointer_bearing_struct()
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "count_fits_buffer",
                "count_arg": 0,
                "count_path": ["n"],
                "buffer_arg": 0,
                "buffer_path": ["data"],
                "elem_size_bytes": 4,
            }
        ]
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("uint64_t params_data_len_u64", content)
        self.assertIn("RAPID_DECODE_ASSERT((decoded.params.n * 4) <= params_data_len_u64);", content)

    def test_decode_constraint_codegen_supports_all_predicate_ops(self) -> None:
        contract = _contract(_manifest_with_scalar_len_constraint("count"))
        ctx = DecodeConstraintEmitContext(
            args_by_index={arg.index: arg for arg in contract.args},
            payload_len_vars={0: "input_len_u64"},
        )
        lhs = ArgValueExpr(1)
        rhs = PayloadLenExpr(0)

        cases = {
            "<=": "decoded.count <= input_len_u64",
            "<": "decoded.count < input_len_u64",
            "==": "decoded.count == input_len_u64",
            "!=": "decoded.count != input_len_u64",
            ">=": "decoded.count >= input_len_u64",
            ">": "decoded.count > input_len_u64",
        }
        for op, expected in cases.items():
            with self.subTest(op=op):
                pred = KernelConstraintPredicate(lhs=lhs, op=op, rhs=rhs)
                self.assertEqual(_emit_predicate_expr(pred, ctx), expected)

    def test_nested_scalar_constraints_parse_to_predicates(self) -> None:
        contract = _contract(_manifest_with_nested_scalar_constraints())

        self.assertTrue(contract.supported)
        self.assertEqual(len(contract.constraint), 2)
        lower_bound = contract.constraint[0]
        self.assertIsInstance(lower_bound.lhs, ArgValueExpr)
        self.assertEqual(lower_bound.lhs.arg_index, 0)
        self.assertEqual(lower_bound.lhs.field_path, ("nested", "inner_count"))
        self.assertEqual(lower_bound.op, ">=")
        scalar_compare = contract.constraint[1]
        self.assertIsInstance(scalar_compare.lhs, ArgValueExpr)
        self.assertEqual(scalar_compare.lhs.field_path, ("nested", "choice", "tag"))
        self.assertIsInstance(scalar_compare.rhs, ArgValueExpr)
        self.assertEqual(scalar_compare.rhs.field_path, ("count",))

    def test_nested_scalar_constraint_codegen_uses_decoded_field_paths(self) -> None:
        contract = _contract(_manifest_with_nested_scalar_constraints())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("RAPID_DECODE_ASSERT(decoded.params.nested.inner_count >= 0);", content)
        self.assertIn("RAPID_DECODE_ASSERT(decoded.params.nested.choice.tag <= decoded.params.count);", content)

    def test_scalar_only_decode_codegen_reads_each_scalar_in_order(self) -> None:
        contract = _contract(_manifest_with_scalar_only_args())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        count_pos = content.index("decoded.count = read_scalar_le<int>(data, offset);")
        scale_pos = content.index("decoded.scale = read_f32_le(data, offset);")
        self.assertLess(count_pos, scale_pos)

    def test_mixed_width_scalar_decode_aligns_each_value(self) -> None:
        manifest = _manifest_with_scalar_only_args()
        manifest["kernels"][0]["args"][0].update(
            type="bool", size_bytes=1, align_bytes=1
        )
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(
                encoding="utf-8"
            )

        reads = [("count", 1, "read_scalar_le<bool>"), ("scale", 4, "read_f32_le")]
        for name, align, read in reads:
            start = content.index(f"// decode value arg `{name}`")
            align_pos = content.index(
                f"offset = rapid_align_up(offset, {align});", start
            )
            read_pos = content.index(read, align_pos)
            self.assertLess(align_pos, read_pos)

    def test_scalar_alias_decode_template_accepts_float_aliases(self) -> None:
        manifest = _manifest_with_scalar_only_args()
        manifest["kernels"][0]["args"][1]["type"] = "OutputOp::ElementCompute"
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertIn(
            "decoded.scale = read_scalar_le<OutputOp::ElementCompute>(data, offset);",
            content,
        )
        self.assertIn(
            "std::is_floating_point<T>::value",
            content,
        )
        self.assertIn(
            "read_scalar_le only supports integral, enum, or floating-point types",
            content,
        )

    def test_decode_storage_drops_top_level_cv_from_value_args(self) -> None:
        contract = _contract(_manifest_with_top_level_cv_value_args())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            InvokeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            decode = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")
            invoke = (phase2_dir / "gen" / "fuzzer_invoke.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("int64_t count;", decode)
        self.assertIn("uint3 dims;", decode)
        self.assertIn("decoded.count = read_scalar_le<int64_t>(data, offset);", decode)
        self.assertIn("decoded.dims = read_object_bytes<uint3>(data, offset);", decode)
        self.assertIn(
            "void __rapid_entry__kernel(const int64_t count, uint3 dims, RapidKernelContext *context);",
            invoke,
        )

    def test_opaque_val_arg_uses_object_copy(self) -> None:
        manifest = _manifest("payload_buffer")
        manifest["kernels"][0]["args"] = [
            {
                "index": 0,
                "name": "cfg",
                "type": "Config",
                "kind": "opaque_val",
                "size_bytes": 16,
                "align_bytes": 8,
                "type_info": {
                    "kind": "struct",
                    "qualified_name": "Config",
                },
            }
        ]
        manifest["kernels"][0]["others"] = {
            "type_shim_header": "type_shim.v1.cuh",
            "type_shim_status": "ok",
        }
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertEqual(contract.args[0].kind, "opaque_val")
        self.assertIn("decoded.cfg = read_object_bytes<Config>(data, offset);", content)

    def test_type_shim_include_dirs_are_written_to_build_spec(self) -> None:
        manifest = _manifest("payload_buffer")
        kernel = manifest["kernels"][0]
        kernel["others"] = {
            "type_shim_header": "type_shim.v1.cuh",
            "type_shim_status": "ok",
            "type_shim_system_headers": [
                "/repo/third_party/cutlass/include/cutlass/tensor_ref.h",
                "/repo/third_party/cutlass/include/cutlass/transform/threadblock/predicated_tile_iterator.h",
                "/usr/include/x86_64-linux-gnu/bits/stdint-intn.h",
            ],
            "type_shim_system_includes": [
                '#include "cutlass/tensor_ref.h"',
                '#include "cutlass/transform/threadblock/predicated_tile_iterator.h"',
            ],
        }
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            kernel_dir = root / "kernel"
            phase2_dir = kernel_dir / "phase2"
            phase2_dir.mkdir(parents=True)
            artifacts = KernelPhase1Artifacts(
                kernel_id="kernel",
                kernel_dir=kernel_dir,
                phase2_dir=phase2_dir,
                kernel_bc=kernel_dir / "kernel.bc",
                manifest_path=kernel_dir / "manifest.json",
                manifest=manifest,
                metadata={"kernel_symbol": "kernel", "build_status": "built"},
            )
            Phase2Runner()._write_build_spec(
                kernel=artifacts,
                contract=contract,
                vconfig_result=VConfigRewriteResult(requested=True, enabled=True),
            )
            build_spec = json.loads((phase2_dir / "build_spec.json").read_text(encoding="utf-8"))

        self.assertEqual(contract.type_shim_include_dirs[0], "/repo/third_party/cutlass/include")
        self.assertIn("/repo/third_party/cutlass/include", build_spec["type_shim_include_dirs"])
        self.assertNotIn("/usr/include/x86_64-linux-gnu/bits", build_spec["type_shim_include_dirs"])

    def test_top_level_opaque_val_with_array_layout_still_uses_object_copy(self) -> None:
        contract = _contract(_manifest_with_opaque_val_array_layout())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertEqual(contract.args[0].kind, "opaque_val")
        self.assertIn("decoded.cfg = read_object_bytes<ArrayConfig>(data, offset);", content)
        self.assertNotIn("decoded.cfg.shape[0]", content)
        self.assertNotIn("decoded.cfg.tail = read_scalar_le<int>(data, offset);", content)

    def test_opaque_with_ptr_decodes_fields_recursively(self) -> None:
        contract = _contract(_manifest_with_pointer_bearing_struct())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertEqual(contract.args[0].kind, "opaque_with_ptr")
        self.assertIn("decoded.params.n = read_scalar_le<int>(data, offset);", content)
        self.assertIn("decoded.params.shape[0] = read_scalar_le<int>(data, offset);", content)
        self.assertIn("decoded.params.shape[1] = read_scalar_le<int>(data, offset);", content)
        self.assertEqual(contract.args[0].layout[0].index, "params.n")
        self.assertEqual(contract.args[0].layout[1].element.index, "params.shape[]")
        self.assertEqual(contract.args[0].layout[2].index, "params.data")
        self.assertIn("uint64_t params_data_len_u64", content)
        self.assertIn(_field_pointer_cast("decoded.params.data"), content)

    def test_nested_pointer_decode_uses_target_field_type_for_cast(self) -> None:
        manifest = _manifest_with_layout_arg(
            "tile_state",
            "ScanTileState",
            "opaque_with_ptr",
            [
                {
                    "index": "tile_state.d_tile_descriptors",
                    "name": "d_tile_descriptors",
                    "type": "TxnWord *",
                    "kind": "pointer",
                    "pointer_role": "payload_buffer",
                    "size_bytes": 8,
                    "align_bytes": 8,
                    "pointee_layout": {
                        "index": "tile_state.d_tile_descriptors.*",
                        "name": "$pointee",
                        "type": "TxnWord",
                        "kind": "opaque_val",
                        "size_bytes": 8,
                        "align_bytes": 8,
                        "layout_status": "complete",
                        "fields": [
                            _scalar_layout_node("x", "int"),
                            _scalar_layout_node("y", "int"),
                        ],
                    },
                }
            ],
        )
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertIn(_field_pointer_cast("decoded.tile_state.d_tile_descriptors"), content)
        self.assertNotIn("reinterpret_cast<TxnWord *>", content)

    def test_nested_payload_length_name_dedupes_repeated_path_segments(self) -> None:
        manifest = _manifest_with_layout_arg(
            "x0",
            "KernelAgent",
            "opaque_with_ptr",
            [
                _record_layout_node(
                    "input",
                    "IteratorWrapper",
                    [
                        _record_layout_node(
                            "m_iterator",
                            "NestedIterator",
                            [
                                _payload_pointer_layout_node("m_iterator", "float *"),
                            ],
                            kind="opaque_with_ptr",
                        )
                    ],
                    kind="opaque_with_ptr",
                ),
                _record_layout_node(
                    "output",
                    "IteratorWrapper",
                    [
                        _record_layout_node(
                            "m_iterator",
                            "NestedIterator",
                            [
                                _payload_pointer_layout_node("m_iterator", "float *"),
                            ],
                            kind="opaque_with_ptr",
                        )
                    ],
                    kind="opaque_with_ptr",
                ),
            ],
        )
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertIn(_field_pointer_cast("decoded.x0.input.m_iterator.m_iterator"), content)
        self.assertIn(_field_pointer_cast("decoded.x0.output.m_iterator.m_iterator"), content)
        self.assertIn("uint64_t x0_input_m_iterator_len_u64", content)
        self.assertIn("uint64_t x0_output_m_iterator_len_u64", content)
        self.assertNotIn("x0_input_m_iterator_m_iterator_len_u64", content)
        self.assertNotIn("x0_output_m_iterator_m_iterator_len_u64", content)

    def test_multi_arg_decode_codegen_preserves_manifest_order_and_names(self) -> None:
        contract = _contract(_manifest_with_pointer_scalar_and_struct())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        input_pos = content.index("uint64_t input_len_u64")
        count_pos = content.index("decoded.count = read_scalar_le<int>(data, offset);")
        params_pos = content.index("decoded.params.n = read_scalar_le<int>(data, offset);")
        self.assertLess(input_pos, count_pos)
        self.assertLess(count_pos, params_pos)
        self.assertIn("uint8_t * input;", content)
        self.assertIn("int count;", content)
        self.assertIn("Params params;", content)
        self.assertIn("uint64_t params_data_len_u64", content)

    def test_recursive_struct_array_layout_matrix_contract(self) -> None:
        manifests = _layout_matrix_manifests()

        array_ptr_contract = _contract(manifests["struct_array_ptr"])
        self.assertTrue(array_ptr_contract.supported)
        self.assertEqual(array_ptr_contract.args[0].kind, "opaque_with_ptr")
        items = array_ptr_contract.args[0].layout[0]
        self.assertEqual(items.kind, "opaque_with_ptr")
        self.assertEqual(items.element_count, 2)
        self.assertIsNotNone(items.element)
        self.assertEqual(items.element.fields[1].name, "data")
        self.assertTrue(items.element.fields[1].is_payload_buffer_pointer)
        self.assertEqual(
            [slot.byte_offset for slot in array_ptr_contract.payload_slots],
            [8, 32],
        )

        nested_ptr_contract = _contract(manifests["struct_array_nested_ptr"])
        self.assertTrue(nested_ptr_contract.supported)
        self.assertEqual(nested_ptr_contract.args[0].kind, "opaque_with_ptr")
        nested_items = nested_ptr_contract.args[0].layout[0]
        nested_element = nested_items.element
        self.assertIsNotNone(nested_element)
        nested_inner = nested_element.fields[0]
        self.assertEqual(nested_inner.name, "inner")
        self.assertEqual(nested_inner.kind, "opaque_with_ptr")
        self.assertTrue(nested_inner.fields[1].is_payload_buffer_pointer)
        self.assertEqual(
            [slot.byte_offset for slot in nested_ptr_contract.payload_slots],
            [8, 32],
        )

    def test_recursive_struct_array_layout_matrix_decode_codegen(self) -> None:
        manifests = _layout_matrix_manifests()

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(
                contract=_contract(manifests["struct_array_ptr"]),
                phase2_dir=phase2_dir,
            )
            array_ptr = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("decoded.cfg.items[0].count = read_scalar_le<int>(data, offset);", array_ptr)
        self.assertIn("decoded.cfg.items[1].count = read_scalar_le<int>(data, offset);", array_ptr)
        self.assertIn(_field_pointer_cast("decoded.cfg.items[0].data"), array_ptr)
        self.assertIn(_field_pointer_cast("decoded.cfg.items[1].data"), array_ptr)
        self.assertIn("uint64_t cfg_items_0_data_len_u64", array_ptr)
        self.assertIn("uint64_t cfg_items_1_data_len_u64", array_ptr)
        self.assertNotIn("cfg_items_cfg_items", array_ptr)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(
                contract=_contract(manifests["struct_array_nested_ptr"]),
                phase2_dir=phase2_dir,
            )
            nested_ptr = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("decoded.cfg.items[0].inner.count = read_scalar_le<int>(data, offset);", nested_ptr)
        self.assertIn(_field_pointer_cast("decoded.cfg.items[1].inner.data"), nested_ptr)
        self.assertIn("uint64_t cfg_items_1_inner_data_len_u64", nested_ptr)
        self.assertNotIn("cfg_items_cfg_items", nested_ptr)
        self.assertIn("decoded.cfg.items[1].scale = read_scalar_le<int>(data, offset);", nested_ptr)

    def test_array_indexed_count_fits_buffer_constraint_codegen(self) -> None:
        manifest = _layout_matrix_manifests()["struct_array_nested_ptr"]
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "count_fits_buffer",
                "count_arg": 0,
                "count_path": ["items", "1", "inner", "count"],
                "buffer_arg": 0,
                "buffer_path": ["items", "1", "inner", "data"],
                "elem_size_bytes": 4,
            }
        ]

        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        predicate = contract.constraint[0]
        self.assertEqual(predicate.lhs.lhs.field_path, ("items", "1", "inner", "count"))
        self.assertEqual(predicate.rhs.field_path, ("items", "1", "inner", "data"))
        self.assertIn("uint64_t cfg_items_1_inner_data_len_u64", content)
        self.assertIn(
            "RAPID_DECODE_ASSERT((decoded.cfg.items[1].inner.count * 4) <= cfg_items_1_inner_data_len_u64);",
            content,
        )

    def test_deep_array_pointer_decode_codegen_recurses_through_multiple_arrays(self) -> None:
        contract = _contract(_manifest_with_deep_array_pointer())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertIn(_field_pointer_cast("decoded.cfg.items[0].ptrs[0]"), content)
        self.assertIn(_field_pointer_cast("decoded.cfg.items[0].ptrs[1]"), content)
        self.assertIn(_field_pointer_cast("decoded.cfg.items[1].ptrs[0]"), content)
        self.assertIn(_field_pointer_cast("decoded.cfg.items[1].ptrs[1]"), content)

    def test_anonymous_aggregate_decode_uses_transparent_cpp_access(self) -> None:
        contract = _contract(_manifest_with_anonymous_pointer_struct())

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertTrue(contract.supported)
        self.assertEqual(contract.args[0].layout[1].name, "anon1")
        self.assertEqual(contract.args[0].layout[1].index, "cfg.anon1")
        self.assertEqual(contract.args[0].layout[1].fields[1].index, "cfg.anon1.ptr")
        self.assertIn("decoded.cfg.prefix = read_scalar_le<int>(data, offset);", content)
        self.assertIn("decoded.cfg.value = read_scalar_le<int>(data, offset);", content)
        self.assertIn(_field_pointer_cast("decoded.cfg.ptr"), content)
        self.assertIn("decoded.cfg.tail = read_scalar_le<int>(data, offset);", content)
        self.assertIn("uint64_t cfg_anon1_ptr_len_u64", content)
        self.assertNotIn("decoded.cfg.anon1.value", content)
        self.assertNotIn("decoded.cfg.anon1.ptr", content)

    def test_anonymous_aggregate_constraint_uses_transparent_cpp_access(self) -> None:
        manifest = _manifest_with_anonymous_pointer_struct()
        manifest["kernels"][0]["constraints"] = [
            {
                "kind": "scalar_compare_const",
                "scalar_arg": 0,
                "scalar_path": ["anon1", "value"],
                "op": ">=",
                "value": 0,
            }
        ]
        contract = _contract(manifest)

        with tempfile.TemporaryDirectory() as td:
            phase2_dir = Path(td)
            DecodeHeaderGenerator().generate(contract=contract, phase2_dir=phase2_dir)
            content = (phase2_dir / "gen" / "fuzzer_decode.v1.cuh").read_text(encoding="utf-8")

        self.assertIn("RAPID_DECODE_ASSERT(decoded.cfg.value >= 0);", content)
        self.assertNotIn("decoded.cfg.anon1.value", content)

    def test_union_opaque_with_ptr_fails_contract(self) -> None:
        manifest = _manifest_with_pointer_bearing_struct()
        arg = manifest["kernels"][0]["args"][0]
        arg["type_info"]["kind"] = "union"

        contract = _contract(manifest)

        self.assertFalse(contract.supported)
        self.assertEqual(contract.support_reason, "union_not_supported")

    def test_nested_union_opaque_with_ptr_fails_contract(self) -> None:
        manifest = _manifest_with_pointer_bearing_struct()
        params = manifest["kernels"][0]["args"][0]
        union_field = params["type_layout"]["fields"][0]
        union_field["kind"] = "opaque_with_ptr"
        union_field["type"] = "Choice"
        union_field["layout_status"] = "complete"
        union_field["type_info"] = {
            "kind": "union",
            "qualified_name": "Choice",
        }
        union_field["fields"] = [
            {
                "name": "value",
                "type": "int",
                "kind": "scalar",
                "size_bytes": 4,
                "align_bytes": 4,
            }
        ]

        contract = _contract(manifest)

        self.assertFalse(contract.supported)
        self.assertEqual(contract.args[0].support_reason, "params.n:union_not_supported")
        self.assertEqual(contract.support_reason, "params.n:union_not_supported")

    def test_deep_nested_union_reason_does_not_duplicate_parent_path(self) -> None:
        manifest = _manifest_with_nested_scalar_constraints()
        params = manifest["kernels"][0]["args"][0]
        nested = params["type_layout"]["fields"][1]
        choice = nested["fields"][1]
        choice["type_info"] = {
            "kind": "union",
            "qualified_name": "Nested::Choice",
        }

        contract = _contract(manifest)

        self.assertFalse(contract.supported)
        self.assertEqual(contract.support_reason, "params.nested.choice:union_not_supported")

    def test_partial_or_opaque_layout_status_fails_contract(self) -> None:
        for layout_status in ("partial", "opaque"):
            with self.subTest(layout_status=layout_status):
                manifest = _manifest_with_pointer_bearing_struct()
                field = manifest["kernels"][0]["args"][0]["type_layout"]["fields"][0]
                field["kind"] = "opaque_with_ptr"
                field["layout_status"] = layout_status

                contract = _contract(manifest)

                self.assertFalse(contract.supported)
                self.assertEqual(
                    contract.support_reason,
                    "params.n:opaque_with_ptr_layout_incomplete",
                )

    def test_scalar_le_buffer_len_with_missing_nested_path_fails_fast(self) -> None:
        manifest = _manifest_with_scalar_len_constraint("count")
        manifest["kernels"][0]["constraints"][0]["scalar_path"] = ["n"]

        with self.assertRaisesRegex(ValueError, "constraint_field_path_not_found"):
            _contract(manifest)

    def test_root_type_layout_index_fails_fast(self) -> None:
        manifest = _manifest_with_pointer_bearing_struct()
        manifest["kernels"][0]["args"][0]["type_layout"]["index"] = "params"

        with self.assertRaisesRegex(ValueError, "type_layout_root_index_not_supported"):
            _contract(manifest)

    def test_missing_pointer_role_rejects_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "pointer_role_missing"):
            _contract(_manifest(None))

    def test_missing_pointer_pointee_layout_rejects_manifest(self) -> None:
        manifest = _manifest("payload_buffer")
        del manifest["kernels"][0]["args"][0]["pointee_layout"]

        with self.assertRaisesRegex(ValueError, "pointer_pointee_layout_missing"):
            _contract(manifest)

    def test_unsupported_type_shim_reason_is_preserved_for_pointer_args(self) -> None:
        manifest = _manifest("payload_buffer")
        kernel = manifest["kernels"][0]
        kernel["args"][0]["type"] = "ScopedType *"
        kernel["args"][0]["pointee_layout"] = {
            "name": "$pointee",
            "type": "ScopedType",
            "kind": "opaque_val",
            "size_bytes": 4,
            "align_bytes": 4,
            "layout_status": "complete",
            "fields": [],
            "type_info": {
                "kind": "struct",
                "qualified_name": "ns::ScopedType",
            },
        }
        kernel["others"] = {
            "type_shim_status": "unsupported",
            "type_shim_reason_codes": ["scoped_type_shim_unsupported"],
            "type_shim_missing_dependencies": ["ns::ScopedType"],
            "type_shim_system_headers": ["/project/include/scoped.h"],
        }

        contract = _contract(manifest)

        self.assertFalse(contract.supported)
        self.assertEqual(contract.support_reason, "scoped_type_shim_unsupported")
        self.assertEqual(contract.args[0].support_reason, "scoped_type_shim_unsupported")

    def test_materialization_unsafe_fails_contract_with_specific_field_reason(self) -> None:
        manifest = _manifest_with_layout_arg(
            "cfg",
            "MaterialRef",
            "opaque_val",
            [
                {
                    "index": "cfg.ref",
                    "name": "ref",
                    "type": "int &",
                    "kind": "opaque_with_ptr",
                    "size_bytes": 8,
                    "align_bytes": 8,
                    "materialization_status": "unsafe",
                    "materialization_reason_codes": ["reference_field"],
                    "materialization_blockers": ["cfg.ref: reference data field cannot be rebound by decode"],
                }
            ],
        )
        cfg = manifest["kernels"][0]["args"][0]
        cfg["materialization_status"] = "unsafe"
        cfg["materialization_reason_codes"] = ["reference_field", "non_trivially_copyable"]
        cfg["materialization_blockers"] = ["cfg.ref: reference data field cannot be rebound by decode"]

        contract = _contract(manifest)

        self.assertFalse(contract.supported)
        self.assertEqual(contract.support_reason, "cfg.ref:reference_field")
        self.assertEqual(contract.args[0].support_reason, "cfg.ref:reference_field")
        self.assertFalse(contract.args[0].layout[0].supported)
        self.assertEqual(contract.args[0].layout[0].support_reason, "cfg.ref:reference_field")

    def test_nonblocking_materialization_note_does_not_fail_phase2(self) -> None:
        manifest = _manifest_with_layout_arg(
            "params",
            "Flash_fwd_params",
            "opaque_with_ptr",
            [_scalar_layout_node("count", "int")],
        )
        params = manifest["kernels"][0]["args"][0]
        params["materialization_status"] = "unsafe"
        params["materialization_reason_codes"] = ["non_standard_layout"]
        params["materialization_blockers"] = ["params: record is not standard-layout"]

        contract = _contract(manifest)

        self.assertTrue(contract.supported)
        self.assertIsNone(contract.support_reason)

    def test_blocking_materialization_context_is_written_by_phase2_runner(self) -> None:
        manifest = _manifest_with_layout_arg(
            "params",
            "MaterialConst",
            "opaque_val",
            [_scalar_layout_node("value", "const int")],
        )
        params = manifest["kernels"][0]["args"][0]
        params["materialization_status"] = "unsafe"
        params["materialization_reason_codes"] = ["const_assignment_blocker"]
        params["materialization_blockers"] = ["params.value: const data field cannot be assigned after construction"]
        params["type_layout"]["fields"][0]["materialization_status"] = "unsafe"
        params["type_layout"]["fields"][0]["materialization_reason_codes"] = ["const_assignment_blocker"]
        params["type_layout"]["fields"][0]["materialization_blockers"] = [
            "params.value: const data field cannot be assigned after construction"
        ]

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_synthetic_phase1_run(run_dir, manifest)

            summary = Phase2Runner().run(run_dir)
            phase2_meta = json.loads((kernel_dir / "phase2" / "metadata.phase2.json").read_text(encoding="utf-8"))

        expected_context = {
            "support_path": "params.value",
            "reason_code": "const_assignment_blocker",
        }
        self.assertEqual(phase2_meta["failure_reason"], "phase2_input_invalid")
        self.assertEqual(phase2_meta["failure_detail"], "params.value:const_assignment_blocker")
        self.assertEqual(phase2_meta.get("failure_context"), expected_context)
        self.assertEqual(summary["results"][0].get("failure_context"), expected_context)

    def test_unsupported_type_shim_context_is_written_by_phase2_runner(self) -> None:
        manifest = _manifest("payload_buffer")
        kernel = manifest["kernels"][0]
        kernel["args"][0]["type"] = "ScopedType *"
        kernel["args"][0]["pointee_layout"] = {
            "name": "$pointee",
            "type": "ScopedType",
            "kind": "opaque_val",
            "size_bytes": 4,
            "align_bytes": 4,
            "layout_status": "complete",
            "fields": [],
            "type_info": {
                "kind": "struct",
                "qualified_name": "ns::ScopedType",
            },
        }
        kernel["others"] = {
            "type_shim_status": "unsupported",
            "type_shim_reason_codes": ["scoped_type_shim_unsupported"],
            "type_shim_missing_dependencies": ["ns::ScopedType"],
            "type_shim_system_headers": ["/project/include/scoped.h"],
            "type_shim_system_includes": ["#include <scoped.h>"],
        }

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            kernel_dir = _write_synthetic_phase1_run(run_dir, manifest)

            summary = Phase2Runner().run(run_dir)
            phase2_meta = json.loads((kernel_dir / "phase2" / "metadata.phase2.json").read_text(encoding="utf-8"))

        expected_context = {
            "reason_code": "scoped_type_shim_unsupported",
            "type_shim_status": "unsupported",
            "type_shim_reason_codes": ["scoped_type_shim_unsupported"],
            "type_shim_missing_dependencies": ["ns::ScopedType"],
            "type_shim_system_headers": ["/project/include/scoped.h"],
            "type_shim_system_includes": ["#include <scoped.h>"],
        }
        self.assertEqual(phase2_meta["failure_reason"], "phase2_input_invalid")
        self.assertEqual(phase2_meta["failure_detail"], "scoped_type_shim_unsupported")
        self.assertEqual(phase2_meta.get("failure_context"), expected_context)
        self.assertEqual(summary["results"][0].get("failure_context"), expected_context)

    def test_path_qualified_input_invalid_reasons_are_classified_by_reason_code(self) -> None:
        self.assertTrue(_is_phase2_input_invalid("params.nested.choice:union_not_supported"))
        self.assertTrue(_is_phase2_input_invalid("params.ptr:scoped_type_shim_unsupported"))
        self.assertTrue(_is_phase2_input_invalid("params:non_trivially_copyable"))
        self.assertTrue(_is_phase2_input_invalid("cfg.ref:reference_field"))
        self.assertTrue(_is_phase2_input_invalid("params:opaque_with_ptr_layout_missing"))
        self.assertTrue(_is_phase2_input_invalid("params.n:opaque_with_ptr_layout_incomplete"))
        self.assertFalse(_is_phase2_input_invalid("params.ptr:unexpected_rewrite_issue"))

    def test_unknown_pointer_role_rejects_manifest(self) -> None:
        with self.assertRaisesRegex(ValueError, "pointer_role_not_supported"):
            _contract(_manifest("unknown_pointer_role"))

    def test_unsupported_pointer_role_fails_contract(self) -> None:
        for pointer_role in ("derived_pointer", "external_device_pointer"):
            with self.subTest(pointer_role=pointer_role):
                contract = _contract(_manifest(pointer_role))

                self.assertFalse(contract.supported)
                self.assertEqual(contract.args[0].pointer_role, pointer_role)
                self.assertEqual(contract.support_reason, "pointer_role_not_supported")
                self.assertFalse(contract.args[0].supported)

    def test_multiple_kernels_rejects_manifest(self) -> None:
        manifest = _manifest("payload_buffer")
        manifest["kernels"].append(manifest["kernels"][0].copy())

        with self.assertRaisesRegex(ValueError, "exactly one kernel"):
            _contract(manifest)


if __name__ == "__main__":
    unittest.main()
