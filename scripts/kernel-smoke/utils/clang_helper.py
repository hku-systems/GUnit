"""Build and invoke the standalone Clang metadata helper."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

from utils.compile_args import prepare_helper_parse_args


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_SRC = REPO_ROOT / "scripts" / "kernel-smoke" / "tooling" / "clang_entry_metadata.cc"
TOOL_BUILD_DIR = REPO_ROOT / "scripts" / "kernel-smoke" / "tooling" / ".build"
TOOL_BIN = TOOL_BUILD_DIR / "clang-entry-metadata"


def _split_flags(text: str) -> list[str]:
    return shlex.split(text.strip()) if text.strip() else []


def _llvm_root_candidates() -> list[Path]:
    seen: set[Path] = set()
    roots: list[Path] = []

    path_llvm_config = shutil.which("llvm-config")
    path_clangxx = shutil.which("clang++")
    if path_llvm_config and path_clangxx:
        llvm_path = Path(path_llvm_config).resolve()
        clang_path = Path(path_clangxx).resolve()
        if llvm_path.parent == clang_path.parent:
            root = llvm_path.parent.parent
            if root not in seen:
                seen.add(root)
                roots.append(root)

    for pattern in (
        "/usr/lib/llvm-*",
        "/opt/homebrew/opt/llvm*",
        "/opt/local/libexec/llvm-*",
    ):
        for match in sorted(glob.glob(pattern), reverse=True):
            root = Path(match)
            if root not in seen:
                seen.add(root)
                roots.append(root)

    return roots


def _find_clang_cpp_lib(libdir: Path) -> Path | None:
    patterns = [
        "libclang-cpp.so",
        "libclang-cpp.so.*",
        "libclang-cpp.dylib",
        "libclang-cpp*.dylib",
    ]
    for pattern in patterns:
        matches = sorted(libdir.glob(pattern), reverse=True)
        if matches:
            return matches[0]
    return None


def _toolchain_from_root(root: Path) -> tuple[Path, Path, Path, Path | None] | None:
    clangxx = root / "bin" / "clang++"
    llvm_config = root / "bin" / "llvm-config"
    if not clangxx.is_file() or not llvm_config.is_file():
        return None

    try:
        libdir_text = subprocess.check_output([str(llvm_config), "--libdir"], text=True)
    except subprocess.CalledProcessError:
        return None

    libdir = Path(libdir_text.strip())
    if not libdir.is_dir():
        return None
    return (clangxx, llvm_config, libdir, _find_clang_cpp_lib(libdir))


def ensure_clang_helper() -> Path:
    env_bin = os.environ.get("KSMOKE_CLANG_HELPER_BIN")
    if env_bin:
        path = Path(env_bin)
        if not path.is_file():
            raise RuntimeError(f"KSMOKE_CLANG_HELPER_BIN not found: {path}")
        return path

    if TOOL_BIN.is_file() and TOOL_BIN.stat().st_mtime >= TOOL_SRC.stat().st_mtime:
        return TOOL_BIN

    TOOL_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for root in _llvm_root_candidates():
        toolchain = _toolchain_from_root(root)
        if toolchain is None:
            continue

        clangxx, llvm_config, libdir, clang_cpp = toolchain
        cxxflags = _split_flags(subprocess.check_output([str(llvm_config), "--cxxflags"], text=True))
        ldflags = _split_flags(subprocess.check_output([str(llvm_config), "--ldflags"], text=True))
        syslibs = _split_flags(subprocess.check_output([str(llvm_config), "--system-libs"], text=True))
        libs = _split_flags(subprocess.check_output([str(llvm_config), "--libs", "support", "option"], text=True))

        cmd = [
            str(clangxx),
            *cxxflags,
            str(TOOL_SRC),
            "-o",
            str(TOOL_BIN),
            f"-L{libdir}",
            f"-Wl,-rpath,{libdir}",
        ]
        if clang_cpp is not None:
            cmd.append(str(clang_cpp))
        else:
            cmd.append("-lclang-cpp")
        cmd.extend([*ldflags, *libs, *syslibs])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return TOOL_BIN

        errors.append(
            f"root={root}\nstdout={result.stdout}\nstderr={result.stderr}",
        )

    raise RuntimeError(
        "failed to build clang helper; tried PATH and common LLVM roots. "
        "Set KSMOKE_CLANG_HELPER_BIN explicitly if needed.\n\n"
        + "\n\n".join(errors[:3])
    )


def run_clang_entry_metadata(
    capture_entry: dict[str, Any],
    *,
    ptx_symbols: set[str],
) -> tuple[dict[str, dict[str, Any]], list[str], str | None, str | None, list[str]]:
    normalized_argv, source_file, cwd, reason, detail = prepare_helper_parse_args(capture_entry)
    if normalized_argv is None:
        return ({}, [], reason, detail, [])

    helper = ensure_clang_helper()
    with tempfile.TemporaryDirectory(prefix="ksmoke-clang-helper-") as td:
        request_path = Path(td) / "request.json"
        request = {
            "source_file": source_file,
            "cwd": cwd,
            "args": normalized_argv[1:] + ["--cuda-device-only", "-fsyntax-only"],
            "ptx_symbols": sorted(ptx_symbols),
        }
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        cmd = [str(helper), str(request_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0 and not result.stdout.strip():
            return ({}, [], "clang_helper_failed", (result.stderr or "").strip() or None, cmd)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return ({}, [], "clang_helper_bad_output", (result.stderr or result.stdout).strip() or None, cmd)

    symbols = payload.get("symbols", {})
    diagnostics = payload.get("diagnostics", [])
    missing = payload.get("missing_symbols", [])
    ok = bool(payload.get("ok", False))

    reason_out = None
    detail_out = None
    if missing:
        reason_out = "ast_match_failed"
        detail_out = "missing PTX symbols in clang helper metadata"
    if not ok and reason_out is None:
        reason_out = "clang_helper_failed"
        detail_out = (result.stderr or "").strip() or "clang helper failed"

    return (symbols, diagnostics, reason_out, detail_out, cmd)


__all__ = ["ensure_clang_helper", "run_clang_entry_metadata"]
