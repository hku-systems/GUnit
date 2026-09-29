#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required tool not found: {name}")
    return path


def build(*, output: Path) -> None:
    compiler = _require_tool("clang++-22")
    llvm_config = _require_tool("llvm-config-22")
    flags = subprocess.run(
        [
            llvm_config,
            "--cxxflags",
            "--ldflags",
            "--system-libs",
            "--libs",
            "core",
            "irreader",
            "bitwriter",
            "support",
            "analysis",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            str(TOOL_DIR / "rapid_feedback_instrument.cpp"),
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            "-o",
            str(output),
            *shlex.split(flags),
        ],
        check=True,
        cwd=TOOL_DIR,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rapid-feedback-instrument against LLVM 22")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(output=Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
