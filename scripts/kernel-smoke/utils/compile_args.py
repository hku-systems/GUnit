"""Compile-command normalization shared by artifact-mode Clang helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any


_AST_KEEP_PREFIXES = (
    "-I",
    "-D",
    "-U",
    "-std=",
    "--std=",
    "-isystem",
    "-include",
    "-isysroot",
    "--cuda-gpu-arch",
    "--cuda-path",
    "-target",
    "--target",
)

_AST_CUDA_ARCH_FALLBACK = "sm_86"

_DROP_FLAGS = frozenset(
    [
        "-shared",
        "-static",
        "-rdynamic",
        "-pie",
        "-no-pie",
        "-Wl,",
        "-s",
        "--as-needed",
        "--no-as-needed",
    ]
)

_DROP_FLAG_PREFIXES = ("-L", "-l", "-Wl,", "-rpath", "--soname")


def _extract_cuda_arches(argv: list[str]) -> list[str]:
    arches: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--cuda-gpu-arch="):
            arch = arg.split("=", 1)[1].strip()
            if arch and arch not in arches:
                arches.append(arch)
            i += 1
            continue
        if arg in ("--cuda-gpu-arch", "-arch") and i + 1 < len(argv):
            arch = argv[i + 1].strip()
            if arch and arch not in arches:
                arches.append(arch)
            i += 2
            continue
        if arg == "-gencode" and i + 1 < len(argv):
            spec = argv[i + 1]
            for part in spec.split(","):
                part = part.strip()
                if part.startswith("code=sm_"):
                    arch = part.split("=", 1)[1]
                    if arch and arch not in arches:
                        arches.append(arch)
            i += 2
            continue
        i += 1
    return arches


def _detect_local_cuda_arch() -> str | None:
    cmd = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?\s*$", lines[0])
    if match is None:
        return None
    return f"sm_{match.group(1)}{match.group(2) or '0'}"


def _select_ast_cuda_arch(argv: list[str]) -> str:
    captured_arches = _extract_cuda_arches(argv)
    local_arch = _detect_local_cuda_arch()
    if local_arch and (not captured_arches or local_arch in captured_arches):
        return local_arch
    if captured_arches:
        return captured_arches[0]
    if local_arch:
        return local_arch
    return _AST_CUDA_ARCH_FALLBACK


def _extract_cuda_path_arg(argv: list[str]) -> str | None:
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--cuda-path="):
            return arg
        if arg == "--cuda-path" and i + 1 < len(argv):
            return f"--cuda-path={argv[i + 1]}"
        i += 1
    return None


def _is_source_file(arg: str) -> bool:
    return arg.endswith((".cu", ".cpp", ".cxx", ".cc", ".c"))


def _is_object_file(arg: str) -> bool:
    return arg.endswith((".o", ".obj", ".so", ".a", ".out"))


def _should_drop_flag(flag: str) -> bool:
    if flag in _DROP_FLAGS:
        return True
    return any(flag.startswith(prefix) for prefix in _DROP_FLAG_PREFIXES)


def _transform_clang_ast(argv: list[str]) -> list[str]:
    compiler = argv[0]
    new_argv = [compiler]
    ast_arch = _select_ast_cuda_arch(argv)
    added_cuda_arch = False
    added_no_ptx = False

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-o":
            i += 2
            continue
        if _is_object_file(arg) and not arg.startswith("-"):
            i += 1
            continue
        if _should_drop_flag(arg):
            if arg in ("-L", "-l", "-rpath", "--soname"):
                i += 2
            else:
                i += 1
            continue
        if _is_source_file(arg) and not arg.startswith("-"):
            i += 1
            continue
        if arg.startswith("--no-cuda-include-ptx="):
            if not added_no_ptx and arg.endswith(ast_arch):
                new_argv.append(arg)
                added_no_ptx = True
            i += 1
            continue
        if arg.startswith("--cuda-gpu-arch="):
            if not added_cuda_arch:
                new_argv.append(f"--cuda-gpu-arch={ast_arch}")
                added_cuda_arch = True
            i += 1
            continue
        if arg in ("-isystem", "-include", "-isysroot", "-target", "--target") and i + 1 < len(argv):
            new_argv.extend([arg, argv[i + 1]])
            i += 2
            continue
        if any(arg.startswith(prefix) for prefix in _AST_KEEP_PREFIXES):
            new_argv.append(arg)
            i += 1
            continue
        if arg in ("-c", "-emit-llvm", "--cuda-device-only", "-S", "-fsyntax-only", "-E"):
            i += 1
            continue
        new_argv.append(arg)
        i += 1

    new_argv = [
        arg
        for arg in new_argv
        if arg not in ("-c", "-S", "-emit-llvm", "--cuda-device-only", "-fsyntax-only", "-E")
    ]
    if not added_cuda_arch:
        new_argv.append(f"--cuda-gpu-arch={ast_arch}")
    if not added_no_ptx:
        new_argv.append(f"--no-cuda-include-ptx={ast_arch}")
    return new_argv


def _map_nvcc_ast(argv: list[str]) -> list[str]:
    clang = shutil.which("clang++") or "clang++"
    new_argv = [clang]
    ast_arch = _select_ast_cuda_arch(argv)

    captured_cuda_path = _extract_cuda_path_arg(argv)
    if captured_cuda_path is not None:
        new_argv.append(captured_cuda_path)
    else:
        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        new_argv.append(f"--cuda-path={cuda_path}")
    new_argv.append(f"--cuda-gpu-arch={ast_arch}")
    new_argv.append(f"--no-cuda-include-ptx={ast_arch}")

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--cuda-path" and i + 1 < len(argv):
            i += 2
            continue
        if arg.startswith("--cuda-path="):
            i += 1
            continue
        if arg == "-o":
            i += 2
            continue
        if _is_source_file(arg) and not arg.startswith("-"):
            i += 1
            continue
        if arg == "-I" and i + 1 < len(argv):
            new_argv.extend(["-I", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("-I"):
            new_argv.append(arg)
            i += 1
            continue
        if arg in ("-isystem", "-include", "-isysroot", "-target", "--target") and i + 1 < len(argv):
            new_argv.extend([arg, argv[i + 1]])
            i += 2
            continue
        if arg.startswith("--target=") or arg.startswith("-target="):
            new_argv.append(arg)
            i += 1
            continue
        if arg == "-D" and i + 1 < len(argv):
            new_argv.extend(["-D", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("-D") or arg.startswith("-U"):
            new_argv.append(arg)
            i += 1
            continue
        if arg.startswith("-std=") or arg.startswith("--std="):
            new_argv.append(arg)
            i += 1
            continue
        if arg in ("-gencode", "-arch") and i + 1 < len(argv):
            i += 2
            continue
        if arg in (
            "-c",
            "-dc",
            "-dw",
            "--device-c",
            "--device-w",
            "-rdc=true",
            "-rdc=false",
            "--relocatable-device-code=true",
            "--relocatable-device-code=false",
        ):
            i += 1
            continue
        if arg in ("-Xcompiler", "-Xlinker", "-Xptxas"):
            i += 2
            continue
        if _should_drop_flag(arg) or _is_object_file(arg):
            i += 1
            continue
        i += 1

    new_argv.extend(["-x", "cuda"])
    return new_argv


def prepare_helper_parse_args(
    capture_entry: dict[str, Any],
) -> tuple[list[str] | None, str, str, str | None, str | None]:
    """Normalize captured compile args for the artifact-mode Clang helper."""
    compiler = os.path.basename(capture_entry.get("compiler", ""))
    argv = capture_entry.get("argv", [])
    source_file = capture_entry.get("source_file", "")
    cwd = capture_entry.get("cwd", ".")

    if not isinstance(argv, list) or not argv:
        return (None, source_file, cwd, "argv_missing", "captured argv is missing")
    if "clang" in compiler:
        normalized_argv = _transform_clang_ast([str(arg) for arg in argv])
    elif "nvcc" in compiler:
        if shutil.which("clang++") is None:
            return (None, source_file, cwd, "clang_not_found", "clang++ not found in PATH")
        normalized_argv = _map_nvcc_ast([str(arg) for arg in argv])
    else:
        return (None, source_file, cwd, "toolchain_mismatch", f"unsupported compiler: {compiler}")

    return (normalized_argv, source_file, cwd, None, None)
