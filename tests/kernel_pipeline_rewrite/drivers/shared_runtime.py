import json
import shutil
import subprocess
import struct
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "kernel_pipeline_rewrite" / "fixtures"
CASES_DIR = FIXTURE_DIR / "cases"


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required tool not found: {name}")
    return path


def _find_supported_kernel_dir(run_dir: Path, display_name: str) -> Path:
    for manifest_path in (run_dir / "kernels").glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kernels = manifest.get("kernels") or []
        if kernels and isinstance(kernels[0], dict) and kernels[0].get("display_name") == display_name:
            return manifest_path.parent
    raise RuntimeError(f"supported kernel not found: {display_name}")


def _load_build_spec(phase2_dir: Path) -> dict[str, Any]:
    path = phase2_dir / "build_spec.json"
    if not path.is_file():
        raise RuntimeError(f"missing build_spec.json for kernel: {phase2_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runtime_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not CASES_DIR.is_dir():
        return cases
    for case_dir in sorted(CASES_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        case_json = case_dir / "case.json"
        input_bin = case_dir / "input_payload.bin"
        expected_bin = case_dir / "expected_payload.bin"
        if not case_json.is_file() or not input_bin.is_file() or not expected_bin.is_file():
            continue
        meta = json.loads(case_json.read_text(encoding="utf-8"))
        cases.append(
            {
                "case_name": case_dir.name,
                "case_dir": case_dir,
                "kernel_display_name": meta["kernel_display_name"],
                "description": meta.get("description", ""),
                "expected_stdout": meta.get("expected_stdout", "payload matched"),
                "expected_exit_code": int(meta.get("expected_exit_code", 0)),
                "input_payload": input_bin,
                "expected_payload": expected_bin,
                "pointer_patches": case_dir / "pointer_patches.txt",
            }
        )
    return cases


def get_runtime_case(case_name: str) -> dict[str, Any]:
    for case in discover_runtime_cases():
        if case["case_name"] == case_name:
            return case
    raise RuntimeError(f"runtime smoke case not configured: {case_name}")


def _align_up(value: int, align: int) -> int:
    if align <= 1:
        return value
    return ((value + align - 1) // align) * align


def _read_pointer_patches(path: Path) -> dict[int, int]:
    patches: dict[int, int] = {}
    if not path.is_file():
        return patches
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pointer_offset, target_offset = (int(part) for part in line.split())
        patches[pointer_offset] = target_offset
    return patches


def _arg_offsets(args: list[dict[str, Any]]) -> dict[int, int]:
    offset = 0
    offsets: dict[int, int] = {}
    for arg in sorted(args, key=lambda item: int(item["index"])):
        align = max(1, int(arg.get("align_bytes", 1)))
        size = int(arg["size_bytes"])
        offset = _align_up(offset, align)
        offsets[int(arg["index"])] = offset
        offset += size
    return offsets


def _pointer_encoded_len(
    *,
    args: list[dict[str, Any]],
    arg_position: int,
    len_pos: int,
    data_len: int,
) -> int:
    data_start = len_pos + 8
    next_arg = args[arg_position + 1] if arg_position + 1 < len(args) else None
    if next_arg is None:
        return data_len
    if next_arg["kind"] == "pointer":
        next_data_unaligned = data_start + data_len + 8
        next_data_align = max(8, int(next_arg.get("align_bytes", 1)))
        next_data = _align_up(next_data_unaligned, next_data_align)
        return next_data - (data_start + 8)
    next_start = _align_up(data_start + data_len, int(next_arg.get("align_bytes", 1)))
    return next_start - data_start


def _old_pointer_segments(
    *,
    payload: bytes,
    patches: dict[int, int],
) -> dict[int, bytes]:
    sorted_patches = sorted(patches.items(), key=lambda item: item[1])
    segments: dict[int, bytes] = {}
    for idx, (pointer_offset, target_offset) in enumerate(sorted_patches):
        end = sorted_patches[idx + 1][1] if idx + 1 < len(sorted_patches) else len(payload)
        segments[pointer_offset] = payload[target_offset:end]
    return segments


def materialize_input_envelope_for_case(
    *,
    manifest_path: Path,
    case: dict[str, Any],
    payload_path: Path,
) -> bytes:
    """Convert checked-in legacy runtime fixture bytes into Input Envelope v1.

    The checked-in fixtures predate generated decode and store real pointer
    placeholders plus `pointer_patches.txt`. Runtime E2E now exercises the
    generated decoder, so the bytes handed to the backend must be the current
    envelope: payload-buffer pointers become `[len:u64][payload+filler]`, while
    scalar/by-value arguments remain inline in manifest argument order.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernels = manifest.get("kernels") or []
    if len(kernels) != 1:
        raise RuntimeError(f"runtime smoke expects one kernel manifest: {manifest_path}")
    args = sorted(kernels[0].get("args") or [], key=lambda item: int(item["index"]))
    legacy_payload = payload_path.read_bytes()
    patches = _read_pointer_patches(Path(case["pointer_patches"]))
    offsets = _arg_offsets(args)
    pointer_segments = _old_pointer_segments(payload=legacy_payload, patches=patches)

    out = bytearray()
    for position, arg in enumerate(args):
        arg_offset = offsets[int(arg["index"])]
        kind = arg["kind"]
        if kind == "pointer":
            if arg.get("pointer_role") != "payload_buffer":
                raise RuntimeError(
                    f"runtime smoke only supports payload_buffer pointers today: {arg.get('name')}"
                )
            if len(out) % 8 != 0:
                raise RuntimeError(f"pointer length field is not 8-byte aligned for {arg.get('name')}")
            segment = pointer_segments.get(arg_offset)
            if segment is None:
                raise RuntimeError(f"missing legacy pointer segment for {arg.get('name')}")
            encoded_len = _pointer_encoded_len(
                args=args,
                arg_position=position,
                len_pos=len(out),
                data_len=len(segment),
            )
            out.extend(struct.pack("<Q", encoded_len))
            out.extend(segment)
            out.extend(b"\x00" * (encoded_len - len(segment)))
        else:
            align = max(1, int(arg.get("align_bytes", 1)))
            aligned = _align_up(len(out), align)
            out.extend(b"\x00" * (aligned - len(out)))
            size = int(arg["size_bytes"])
            out.extend(legacy_payload[arg_offset : arg_offset + size])
    return bytes(out)


class RuntimeSmokeRunner:
    """Compile and run a complete generated wrapper/device runtime example."""

    def __init__(self) -> None:
        self.clang = _require_tool("clang++-22")
        self.llvm_link = _require_tool("llvm-link-22")
        self.llc = _require_tool("llc-22")

    def run(self, *, run_dir: Path, case_name: str = "add_kernel") -> dict[str, Any]:
        case = get_runtime_case(case_name)

        kernel_dir = _find_supported_kernel_dir(run_dir, case["kernel_display_name"])
        phase2_dir = kernel_dir / "phase2"
        if not phase2_dir.is_dir():
            raise RuntimeError(f"missing phase2 dir for kernel: {kernel_dir}")
        build_spec = _load_build_spec(phase2_dir)

        runtime_dir = run_dir / "phase2_runtime" / kernel_dir.name
        runtime_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "make",
                "-C",
                str(FIXTURE_DIR),
                "clean",
                "all",
                f"OUT_DIR={runtime_dir}",
                f"CLANG={self.clang}",
                f"LLVM_LINK={self.llvm_link}",
                f"LLC={self.llc}",
                f'PHASE2_INVOKE_HEADER={phase2_dir / str(build_spec.get("invoke_header", "gen/fuzzer_invoke.v1.cuh"))}',
                f'PHASE2_DEVICE_BC={phase2_dir / str(build_spec.get("device_bc", "kernel.device.bc"))}',
            ],
            check=True,
        )

        wrapper = FIXTURE_DIR / "wrapper.cu"
        wrapper_bc = runtime_dir / "wrapper.bc"
        linked_bc = runtime_dir / "linked.bc"
        ptx = runtime_dir / "linked.ptx"
        host = FIXTURE_DIR / "host.cpp"
        exe = runtime_dir / "host.out"
        manifest_path = kernel_dir / "manifest.json"
        (runtime_dir / "input_payload.bin").write_bytes(
            materialize_input_envelope_for_case(
                manifest_path=manifest_path,
                case=case,
                payload_path=case["input_payload"],
            )
        )
        (runtime_dir / "expected_payload.bin").write_bytes(
            materialize_input_envelope_for_case(
                manifest_path=manifest_path,
                case=case,
                payload_path=case["expected_payload"],
            )
        )
        run = subprocess.run([str(exe)], check=True, capture_output=True, text=True, cwd=runtime_dir)
        result = {
            "schema_version": 1,
            "case_name": case_name,
            "kernel_display_name": case["kernel_display_name"],
            "kernel_dir": str(kernel_dir),
            "phase2_dir": str(phase2_dir),
            "runtime_dir": str(runtime_dir),
            "fixture_dir": str(FIXTURE_DIR),
            "case_dir": str(case["case_dir"]),
            "wrapper_cu": str(wrapper),
            "wrapper_bc": str(wrapper_bc),
            "linked_bc": str(linked_bc),
            "linked_ptx": str(ptx),
            "host_cpp": str(host),
            "host_exe": str(exe),
            "input_payload": str(runtime_dir / "input_payload.bin"),
            "expected_payload": str(runtime_dir / "expected_payload.bin"),
            "stdout": run.stdout,
            "stderr": run.stderr,
            "returncode": run.returncode,
            "expected_stdout": case["expected_stdout"],
            "expected_exit_code": case["expected_exit_code"],
        }
        (runtime_dir / "runtime_smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    def run_all(self, *, run_dir: Path) -> list[dict[str, Any]]:
        return [self.run(run_dir=run_dir, case_name=case["case_name"]) for case in discover_runtime_cases()]
