"""Compile command capture wrapper.

Runs a compiler command unchanged and appends a JSON-line record
to ${RAPID_CAPTURE_DIR:-.rapid/capture}/commands.jsonl.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

ENV_KEYS = ("CC", "CXX", "NVCC", "CUDAFLAGS", "CFLAGS", "CXXFLAGS")


def _capture_dir() -> Path:
    return Path(os.environ.get("RAPID_CAPTURE_DIR", ".rapid/capture"))


def _append_jsonl(path: Path, line: str) -> None:
    """Append one JSONL line with a required process-safe lock."""
    if fcntl is None:
        raise RuntimeError("fcntl is required for safe parallel capture logging")

    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def run_and_capture(argv: list[str]) -> int:
    if not argv:
        print("rapid-wrap: no command given", file=sys.stderr)
        return 1

    compiler = argv[0]
    cwd = os.getcwd()
    ts = time.time()

    result = subprocess.run(argv)
    exit_code = result.returncode

    env_subset = {k: os.environ[k] for k in ENV_KEYS if k in os.environ}

    record = {
        "timestamp": ts,
        "cwd": cwd,
        "argv": argv,
        "compiler": compiler,
        "env_subset": env_subset,
        "exit_code": exit_code,
    }

    out = _capture_dir()
    out.mkdir(parents=True, exist_ok=True)
    _append_jsonl(out / "commands.jsonl", json.dumps(record, sort_keys=True) + "\n")

    return exit_code
