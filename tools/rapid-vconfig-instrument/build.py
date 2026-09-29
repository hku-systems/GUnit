#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
TOOL_SOURCE = TOOL_DIR / "rapid_vconfig_instrument.cpp"
PASS_SOURCE = TOOL_DIR / "virtual_dim_pass.cpp"
PASS_HEADER = TOOL_DIR / "virtual_dim_pass.h"
PLUGIN_SOURCE = TOOL_DIR / "virtual_dim_plugin.cpp"


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required tool not found: {name}")
    return path


def _is_current(output: Path, inputs: tuple[Path, ...]) -> bool:
    if not output.is_file():
        return False
    output_mtime = output.stat().st_mtime_ns
    return all(output_mtime >= path.stat().st_mtime_ns for path in inputs)


def _llvm_flags(*components: str) -> list[str]:
    llvm_config = _require_tool("llvm-config-22")
    return shlex.split(
        subprocess.run(
            [
                llvm_config,
                "--cxxflags",
                "--ldflags",
                "--system-libs",
                "--libs",
                *components,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def build(*, output: Path) -> None:
    inputs = (Path(__file__).resolve(), TOOL_SOURCE, PASS_SOURCE, PASS_HEADER)
    if _is_current(output, inputs):
        return

    compiler = _require_tool("clang++-22")
    flags = _llvm_flags("core", "irreader", "bitwriter", "passes", "support")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            str(TOOL_SOURCE),
            str(PASS_SOURCE),
            "-I",
            str(TOOL_DIR),
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            "-o",
            str(output),
            *flags,
        ],
        check=True,
        cwd=TOOL_DIR,
    )


def build_plugin(*, output: Path) -> None:
    inputs = (Path(__file__).resolve(), PLUGIN_SOURCE, PASS_SOURCE, PASS_HEADER)
    if _is_current(output, inputs):
        return

    compiler = _require_tool("clang++-22")
    flags = _llvm_flags("core", "passes", "support")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            str(PLUGIN_SOURCE),
            str(PASS_SOURCE),
            "-I",
            str(TOOL_DIR),
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            "-fPIC",
            "-shared",
            "-o",
            str(output),
            *flags,
        ],
        check=True,
        cwd=TOOL_DIR,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build rapid-vconfig-instrument against LLVM 22"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--plugin-output",
        help="Optionally build the out-of-tree VirtualDim pass plugin",
    )
    args = parser.parse_args()
    build(output=Path(args.output).resolve())
    if args.plugin_output:
        build_plugin(output=Path(args.plugin_output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
