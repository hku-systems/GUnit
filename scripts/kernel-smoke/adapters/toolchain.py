"""Toolchain version detection for reproducibility metadata."""

import shutil
import subprocess
from typing import Optional


def _run_version(binary: str, flag: str = "--version") -> Optional[str]:
    path = shutil.which(binary)
    if path is None:
        return None
    try:
        result = subprocess.run(
            [path, flag], capture_output=True, text=True, timeout=10
        )
        first_line = result.stdout.strip().split("\n")[0]
        return first_line if first_line else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def detect_toolchain_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    mapping = {
        "clang": "clang",
        "llvm_opt": "opt",
        "nvcc": "nvcc",
    }
    for key, binary in mapping.items():
        ver = _run_version(binary)
        if ver is not None:
            versions[key] = ver
    return versions
