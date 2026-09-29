#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REWRITE_DIR = REPO_ROOT / "scripts" / "kernel-rewrite"
if str(REWRITE_DIR) not in sys.path:
    sys.path.insert(0, str(REWRITE_DIR))

from common import read_json  # noqa: E402
from contracts.kernel import KernelContract, build_kernel_contract  # noqa: E402


CUDA_KERNEL_DIR = REPO_ROOT / "cuda-kernel"
RAPID_DIR = CUDA_KERNEL_DIR / "rapid"
if str(CUDA_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_KERNEL_DIR))

from backend_build import (  # noqa: E402
    add_build_profile_argument,
    add_canary_argument,
    add_feedback_build_metadata,
    add_feedback_instrumentation_argument,
    add_optimization_build_metadata,
    build_optimization_contract,
    canary_defines,
    kernel_timing_defines,
    optimize_device_bitcode,
    run_build_command,
    select_feedback_device_bitcode,
)
from launch_config import (  # noqa: E402
    launch_config_from_kernel_entry,
    single_kernel_entry_from_manifest,
    write_launch_config_header,
)


def _require_tool(*names: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(f"required tool not found: {'/'.join(names)}")


def _load_contract_from_kernel_dir(kernel_dir: Path) -> tuple[KernelContract, dict]:
    manifest = read_json(kernel_dir / "manifest.json") or {}
    metadata = read_json(kernel_dir / "metadata.json") or {}
    contract = build_kernel_contract(
        kernel_id=kernel_dir.name,
        manifest=manifest,
        manifest_path=kernel_dir / "manifest.json",
        metadata=metadata,
    )
    if not contract.supported:
        raise RuntimeError(f"unsupported kernel for rapid build: {contract.support_reason}")
    build_spec = read_json(kernel_dir / "phase2" / "build_spec.json") or {}
    return contract, build_spec


def _resolve_phase2_inputs(phase2_dir: Path) -> tuple[Path, Path, KernelContract, dict]:
    resolved_phase2_dir = phase2_dir.resolve()
    if not resolved_phase2_dir.is_dir():
        raise RuntimeError(f"missing phase2 dir: {resolved_phase2_dir}")
    kernel_dir = resolved_phase2_dir.parent
    contract, build_spec = _load_contract_from_kernel_dir(kernel_dir)
    return kernel_dir, resolved_phase2_dir, contract, build_spec


def _single_kernel_entry(kernel_dir: Path) -> dict:
    manifest_path = kernel_dir / "manifest.json"
    manifest = read_json(manifest_path) or {}
    return single_kernel_entry_from_manifest(
        manifest,
        manifest_path=manifest_path,
        backend_name="rapid",
    )


def _write_embedded_ptx_source(out_dir: Path, *, ptx_path: Path) -> tuple[Path, int]:
    ptx_bytes = ptx_path.read_bytes()
    if not ptx_bytes.endswith(b"\0"):
        ptx_bytes += b"\0"
    values: list[str] = []
    for idx, value in enumerate(ptx_bytes):
        prefix = "    " if idx % 16 == 0 else ""
        suffix = ",\n" if idx % 16 == 15 or idx == len(ptx_bytes) - 1 else ", "
        values.append(f"{prefix}{value}{suffix}")
    source_path = out_dir / "embedded_ptx.cpp"
    source_path.write_text(
        "#include <cstddef>\n\n"
        "namespace rapid_embedded_ptx {\n"
        "extern const unsigned char kEmbeddedPtx[] = {\n"
        + "".join(values)
        + "};\n"
        f"extern const size_t kEmbeddedPtxSize = {len(ptx_bytes)}u;\n"
        "}  // namespace rapid_embedded_ptx\n",
        encoding="utf-8",
    )
    return source_path, len(ptx_bytes)


def _type_shim_include_args(contract: KernelContract, build_spec: dict) -> list[str]:
    raw = build_spec.get("type_shim_include_dirs")
    include_dirs = [str(x) for x in raw if isinstance(x, str) and x] if isinstance(raw, list) else []
    if not include_dirs:
        include_dirs = contract.type_shim_include_dirs
    args: list[str] = []
    seen: set[str] = set()
    for path in include_dirs:
        if path in seen:
            continue
        seen.add(path)
        args.extend(["-I", path])
    return args


def build_rapid_target(
    *,
    phase2_dir: Path,
    out_dir: Path | None,
    cuda_arch: str,
    cuda_path: str,
    feedback_instrumentation: str = "enabled",
    instrument_selects: bool = False,
    build_profile: str = "release",
    enable_kernel_timing: bool = False,
    enable_canary: bool = True,
) -> dict:
    if feedback_instrumentation not in ("enabled", "disabled"):
        raise RuntimeError(
            f"unsupported feedback instrumentation mode: {feedback_instrumentation}"
        )
    inner_feedback_enabled = 1 if feedback_instrumentation == "enabled" else 0
    kernel_dir, resolved_phase2_dir, contract, build_spec = _resolve_phase2_inputs(phase2_dir)
    invoke_header = resolved_phase2_dir / str(build_spec.get("invoke_header", "gen/fuzzer_invoke.v1.cuh"))
    target_layout_header = resolved_phase2_dir / str(
        build_spec.get("target_layout_header", "gen/rapid_target_layout.v1.h")
    )
    device_bc = resolved_phase2_dir / str(build_spec.get("device_bc", "kernel.device.bc"))
    if not invoke_header.is_file():
        raise RuntimeError(f"missing invoke header: {invoke_header}")
    if not target_layout_header.is_file():
        raise RuntimeError(f"missing target layout header: {target_layout_header}")
    if not device_bc.is_file():
        raise RuntimeError(f"missing device bitcode: {device_bc}")

    output_dir = out_dir.resolve() if out_dir is not None else resolved_phase2_dir / "backends" / "rapid"
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    launch_config = launch_config_from_kernel_entry(
        _single_kernel_entry(kernel_dir),
        backend_name="rapid",
        require_warp_aligned_block=bool(build_spec.get("vconfig_warp_aligned")),
        vconfig_enabled=bool(build_spec.get("vconfig_enabled")),
    )
    launch_config_header = write_launch_config_header(output_dir, launch_config)

    clang = _require_tool("clang++-22", "clang++")
    llvm_link = _require_tool("llvm-link-22", "llvm-link")
    llc = _require_tool("llc-22", "llc")
    cxx = _require_tool("clang++", "g++")

    wrapper_bc = intermediates_dir / "rapid_wrapper.bc"
    linked_bc = intermediates_dir / "rapid_linked.bc"
    optimized_bc = intermediates_dir / "optimized.bc"
    ptx = intermediates_dir / "module.ptx"
    shared_lib = output_dir / "libphase2_rapid_target.so"
    optimization = build_optimization_contract(build_profile)
    command_log: list[list[str]] = []

    type_shim_include_args = _type_shim_include_args(contract, build_spec)
    coverage_defines = (
        ["-DRAPID_DEFINE_DEVICE_COVERAGE_MAP=1"]
        if inner_feedback_enabled
        else []
    )
    timing_defines = kernel_timing_defines(enable_kernel_timing)
    canary_compile_defines = canary_defines(enable_canary)
    timing_link_args = (
        (f"-L{cuda_path}/targets/x86_64-linux/lib", "-lcudart")
        if enable_kernel_timing
        else ()
    )

    run_build_command([
        clang,
        *optimization.device_clang_flags,
        "-x", "cuda",
        f"--cuda-path={cuda_path}",
        f"--cuda-gpu-arch={cuda_arch}",
        "--cuda-device-only",
        "-emit-llvm",
        "-c",
        *coverage_defines,
        *timing_defines,
        *canary_compile_defines,
        f"-DRAPID_ENABLE_INNER_FEEDBACK={inner_feedback_enabled}",
        f'-DFUZZER_INVOKE_HEADER="{invoke_header}"',
        str(RAPID_DIR / "wrapper.cu"),
        "-I", str(CUDA_KERNEL_DIR),
        "-I", str(CUDA_KERNEL_DIR / "utils"),
        "-I", str(target_layout_header.parent),
        *type_shim_include_args,
        "-o", str(wrapper_bc),
    ], command_log=command_log)
    run_build_command(
        [llvm_link, str(wrapper_bc), str(device_bc), "-o", str(linked_bc)],
        command_log=command_log,
    )
    feedback = select_feedback_device_bitcode(
        linked_bc=linked_bc,
        build_spec_path=resolved_phase2_dir / "build_spec.json",
        intermediates_dir=intermediates_dir,
        mode=feedback_instrumentation,
        instrument_selects=instrument_selects,
        command_log=command_log,
    )
    optimize_device_bitcode(
        input_bc=feedback.selected_bc,
        output_bc=optimized_bc,
        optimization=optimization,
        command_log=command_log,
    )
    run_build_command([
        llc,
        *optimization.llc_flags,
        "-march=nvptx64",
        f"-mcpu={cuda_arch}",
        str(optimized_bc),
        "-o",
        str(ptx),
    ], command_log=command_log)

    embedded_ptx_cpp, _ = _write_embedded_ptx_source(output_dir, ptx_path=ptx)

    run_build_command([
        cxx,
        *optimization.host_cxx_flags,
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(RAPID_DIR / "harness.cpp"),
        str(embedded_ptx_cpp),
        str(CUDA_KERNEL_DIR / "utils" / "coverage" / "coverage_globals.cpp"),
        str(CUDA_KERNEL_DIR / "utils" / "feedback" / "feedback_globals.cpp"),
        *timing_defines,
        *canary_compile_defines,
        f"-DRAPID_ENABLE_INNER_FEEDBACK={inner_feedback_enabled}",
        f'-DRAPID_LAUNCH_CONFIG_HEADER="{launch_config_header}"',
        "-I", str(CUDA_KERNEL_DIR),
        "-I", str(CUDA_KERNEL_DIR / "utils"),
        "-I", str(target_layout_header.parent),
        "-I", str(output_dir),
        "-I", str(RAPID_DIR),
        f"-I{cuda_path}/targets/x86_64-linux/include",
        "-L/usr/lib/x86_64-linux-gnu",
        "-lcuda",
        *timing_link_args,
        "-o", str(shared_lib),
    ], command_log=command_log)

    result = {
        "schema_version": 2,
        "kernel_dir": str(kernel_dir),
        "phase2_dir": str(resolved_phase2_dir),
        "kernel_id": contract.kernel_id,
        "display_name": contract.display_name,
        "device_bc": str(device_bc),
        "invoke_header": str(invoke_header),
        "target_layout_header": str(target_layout_header),
        "wrapper_bc": str(wrapper_bc),
        "linked_bc": str(linked_bc),
        "module_ptx": str(ptx),
        "embedded_ptx_cpp": str(embedded_ptx_cpp),
        "launch_config_header": str(launch_config_header),
        "launch_config": launch_config,
        "shared_lib": str(shared_lib),
        "type_shim_include_dirs": type_shim_include_args[1::2],
        "kernel_timing_enabled": enable_kernel_timing,
        "canary_enabled": enable_canary,
    }
    add_feedback_build_metadata(result, feedback)
    add_optimization_build_metadata(
        result,
        optimization=optimization,
        optimized_device_bc=optimized_bc,
        command_log=command_log,
    )
    result["llvm_feedback_instrumentation"] = feedback.mode
    result["inner_cuda_feedback"] = feedback.mode
    result["backend_added_inner_feedback"] = feedback.mode
    result["device_feedback_transport"] = feedback.mode
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output_dir / "backend_build.json").write_text(serialized, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rapid-style Phase2 shared library from a phase2 directory.")
    parser.add_argument("--phase2-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH", "sm_86"))
    parser.add_argument("--enable-kernel-timing", action="store_true")
    add_canary_argument(parser)
    add_build_profile_argument(parser)
    add_feedback_instrumentation_argument(parser)
    args = parser.parse_args()

    result = build_rapid_target(
        phase2_dir=Path(args.phase2_dir).resolve(),
        out_dir=Path(args.out_dir).resolve() if args.out_dir else None,
        cuda_arch=args.cuda_arch,
        cuda_path=args.cuda_path,
        feedback_instrumentation=args.feedback_instrumentation,
        instrument_selects=args.instrument_selects,
        build_profile=args.build_profile,
        enable_kernel_timing=args.enable_kernel_timing,
        enable_canary=args.canary == "enabled",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
