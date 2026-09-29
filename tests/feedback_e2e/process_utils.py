from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path


def _cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "RUSTFLAGS",
        "-A function-casts-as-integer -A unstable-name-collisions",
    )
    return env


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _read_log(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _wait_for(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    interval: float = 0.1,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _has_crash_artifact(work_dir: Path) -> bool:
    crashes_dir = work_dir / "crashes"
    return crashes_dir.is_dir() and any(
        path.is_file() and path.stat().st_size > 0 for path in crashes_dir.rglob("*")
    )


def _tcp_port_is_listening(port: int) -> bool:
    expected_port = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) >= 4:
                local_address = fields[1]
                state = fields[3]
                if (
                    local_address.rsplit(":", 1)[-1] == expected_port
                    and state == "0A"
                ):
                    return True
    return False
