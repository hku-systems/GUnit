from __future__ import annotations

import ctypes
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from shared_runtime import get_runtime_case, materialize_input_envelope_for_case


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_VCONFIG = (1, 1, 1, 1024, 1, 1)


def _rapid_task_envelope(payload: bytes) -> bytes:
    header = struct.pack("<6IQ", *DEFAULT_TASK_VCONFIG, len(payload))
    return header + payload


def _find_phase2_dir(run_dir: Path, display_name: str) -> Path:
    for manifest_path in (run_dir / "kernels").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kernels = manifest.get("kernels") or []
        kernel_entry = kernels[0] if kernels and isinstance(kernels[0], dict) else {}
        if kernel_entry.get("display_name") == display_name:
            phase2_dir = manifest_path.parent / "phase2"
            if phase2_dir.is_dir():
                return phase2_dir
    raise RuntimeError(f"phase2 dir not found for kernel: {display_name}")


def build_origin_for_case(*, run_dir: Path, case_name: str = "add_kernel") -> dict[str, Any]:
    case = get_runtime_case(case_name)
    cli = REPO_ROOT / "cuda-kernel" / "origin" / "build.py"
    phase2_dir = _find_phase2_dir(run_dir, case["kernel_display_name"])
    out_dir = phase2_dir / "backends" / "origin"
    cmd = [
        shutil.which("python3") or "python3",
        str(cli),
        "--phase2-dir",
        str(phase2_dir),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return json.loads((out_dir / "backend_build.json").read_text(encoding="utf-8"))


def run_origin_case(*, run_dir: Path, case_name: str = "add_kernel") -> dict[str, Any]:
    case = get_runtime_case(case_name)
    build = build_origin_for_case(run_dir=run_dir, case_name=case_name)
    lib = ctypes.CDLL(build["shared_lib"])

    lib.libafl_target.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.libafl_target.restype = None
    lib.libafl_get_last_output_size.argtypes = []
    lib.libafl_get_last_output_size.restype = ctypes.c_size_t
    lib.libafl_copy_last_output.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.libafl_copy_last_output.restype = ctypes.c_size_t

    manifest_path = Path(build["phase2_dir"]).parent / "manifest.json"
    arg_pack = materialize_input_envelope_for_case(
        manifest_path=manifest_path,
        case=case,
        payload_path=Path(case["input_payload"]),
    )
    payload = _rapid_task_envelope(arg_pack)
    expected = materialize_input_envelope_for_case(
        manifest_path=manifest_path,
        case=case,
        payload_path=Path(case["expected_payload"]),
    )
    payload_buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)

    lib.libafl_target(payload_buf, len(payload))
    output_size = lib.libafl_get_last_output_size()
    output_buf = (ctypes.c_uint8 * output_size)()
    copied = lib.libafl_copy_last_output(output_buf, output_size)
    output = bytes(output_buf[:copied])

    cov_map = (ctypes.c_uint8 * 65536).in_dll(lib, "libafl_cov_map")
    nonzero_edges = sum(1 for value in cov_map if value)

    runtime_dir = Path(build["shared_lib"]).parent
    (runtime_dir / "origin_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_name": case_name,
                "kernel_display_name": case["kernel_display_name"],
                "build": build,
                "output": list(output),
                "expected": list(expected),
                "output_size": copied,
                "expected_size": len(expected),
                "nonzero_edges": nonzero_edges,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "case_name": case_name,
        "kernel_display_name": case["kernel_display_name"],
        "build": build,
        "output": output,
        "expected": expected,
        "output_size": copied,
        "expected_size": len(expected),
        "nonzero_edges": nonzero_edges,
    }
