import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "kernel-smoke" / "print_type_defs.py"


class PrintTypeDefsTest(unittest.TestCase):
    def test_builtin_pointer_omits_unresolved(self) -> None:
        manifest = {
            "schema_version": 1,
            "kernels": [
                {
                    "symbol_name": "_Z6kernelv",
                    "display_name": "kernel",
                    "args": [
                        {
                            "index": 0,
                            "name": "ptr",
                            "type": "const float *",
                            "kind": "pointer",
                            "size_bytes": 8,
                            "align_bytes": 8,
                            "pointer_role": "payload_buffer",
                            "pointee_layout": {
                                "name": "$pointee",
                                "type": "float",
                                "kind": "scalar",
                                "size_bytes": 4,
                                "align_bytes": 4,
                            },
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            manifest_path = Path(td) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = subprocess.run(
                ["python3", str(SCRIPT), str(manifest_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("arg[0] ptr: const float *", result.stdout)
        self.assertNotIn("unresolved", result.stdout)

    def test_prints_resolved_definition(self) -> None:
        manifest = {
            "schema_version": 1,
            "kernels": [
                {
                    "symbol_name": "_Z6kernelv",
                    "display_name": "kernel",
                    "args": [
                        {
                            "index": 0,
                            "name": "op",
                            "type": "struct cub::Sum",
                            "kind": "opaque_val",
                            "size_bytes": 1,
                            "align_bytes": 4,
                            "type_info": {
                                "kind": "struct",
                                "qualified_name": "cub::Sum",
                                "usr": "c:@N@cub@S@Sum",
                                "decl_loc": {
                                    "file": "cub/thread/thread_operators.cuh",
                                    "line": 109,
                                    "column": 8,
                                },
                                "definition": {
                                    "status": "available",
                                    "loc": {
                                        "file": "cub/thread/thread_operators.cuh",
                                        "line": 109,
                                        "column": 8,
                                    },
                                },
                            },
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            manifest_path = Path(td) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = subprocess.run(
                ["python3", str(SCRIPT), str(manifest_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("arg[0] op: struct cub::Sum", result.stdout)
        self.assertIn("struct cub::Sum", result.stdout)
        self.assertIn("definition=available", result.stdout)
        self.assertIn("cub/thread/thread_operators.cuh:109:8", result.stdout)

    def test_prints_include_guidance_from_capture_roots(self) -> None:
        manifest = {
            "schema_version": 1,
            "kernels": [
                {
                    "symbol_name": "_Z6kernelv",
                    "display_name": "kernel",
                    "args": [
                        {
                            "index": 0,
                            "name": "op",
                            "type": "struct cub::Sum",
                            "kind": "opaque_val",
                            "size_bytes": 1,
                            "align_bytes": 4,
                            "type_info": {
                                "kind": "struct",
                                "qualified_name": "cub::Sum",
                                "definition": {
                                    "status": "available",
                                    "loc": {
                                        "file": "/tmp/fake/include/cub/thread/thread_operators.cuh",
                                        "line": 109,
                                        "column": 8,
                                    },
                                },
                            },
                        }
                    ],
                    "others": {"source_file": "/tmp/src/kernel.cu", "source_line": 1},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            capture_dir = Path(td) / "capture"
            kernel_dir.mkdir(parents=True)
            capture_dir.mkdir(parents=True)
            manifest_path = kernel_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "index.json").write_text(
                json.dumps({"capture_dir": str(capture_dir), "kernels": [{"dir": str(kernel_dir)}]}),
                encoding="utf-8",
            )
            (capture_dir / "commands.jsonl").write_text(
                json.dumps(
                    {
                        "argv": [
                            "nvcc",
                            "-I/tmp/fake/include",
                            "-isystem",
                            "/tmp/system/include",
                            "-c",
                            "/tmp/src/kernel.cu",
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                ["python3", str(SCRIPT), str(manifest_path)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("add include path: -I/tmp/fake/include", result.stdout)
        self.assertIn('include with: #include "cub/thread/thread_operators.cuh"', result.stdout)


if __name__ == "__main__":
    unittest.main()
