import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(REPO_ROOT / "scripts" / "kernel-smoke"))

from utils.type_resolution import load_manifest, resolve_manifest_arg_types


class TypeResolutionTest(unittest.TestCase):
    def test_resolves_include_guidance_from_manifest_and_capture(self) -> None:
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
                            "-c",
                            "/tmp/src/kernel.cu",
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = resolve_manifest_arg_types(load_manifest(manifest_path), manifest_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["include_root"], "/tmp/fake/include")
        self.assertEqual(rows[0]["include_stmt"], '#include "cub/thread/thread_operators.cuh"')


if __name__ == "__main__":
    unittest.main()
