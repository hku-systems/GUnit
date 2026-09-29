#!/usr/bin/env python3
"""Replay crash inputs against CUDA fuzzer backends for attribution.

Supports two ABIs:
  - sync (cufuzz/origin): libafl_target + libafl_get_last_run_status
  - async (rapid2): libafl_submit_with_id + libafl_poll_results + libafl_release_tasks

Usage:
  python benchmark/rq4/crash_replay.py --library path/to/lib.so --input crashes/hash
  python benchmark/rq4/crash_replay.py --library path/to/lib.so --input-dir crashes/
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.runtime_abi import RunStatus, TaskResult, stop_library  # noqa: E402


STATUS_NAMES = {0: "OK", 1: "CUDA_ERROR", 2: "TIMEOUT", 3: "INVALID_INPUT", 4: "INTERNAL_ERROR"}


@dataclass(frozen=True)
class ReplayResult:
    input_path: str
    input_size: int
    status_code: int
    status_name: str
    stage: int
    detail: int
    exec_time_ns: int | None
    exit_kind: str

    def as_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "input_size": self.input_size,
            "status_code": self.status_code,
            "status_name": self.status_name,
            "stage": self.stage,
            "detail": f"0x{self.detail:x}",
            "exec_time_ns": self.exec_time_ns,
            "exit_kind": self.exit_kind,
        }


def _classify_exit(code: int) -> str:
    if code == 0:
        return "ok"
    if code == 1:
        return "crash"
    if code == 2:
        return "timeout"
    if code == 3:
        return "invalid_input"
    return "error"


def _detect_abi(target: ctypes.CDLL) -> str:
    try:
        target.libafl_submit_with_id
        return "async"
    except AttributeError:
        return "sync"


def _replay_sync(target: ctypes.CDLL, payload: bytes) -> ReplayResult:
    run = target.libafl_target
    run.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    run.restype = None

    get_status = target.libafl_get_last_run_status
    get_status.argtypes = [ctypes.POINTER(RunStatus)]
    get_status.restype = None

    has_wait = hasattr(target, "libafl_wait")
    if has_wait:
        wait = target.libafl_wait
        wait.argtypes = []
        wait.restype = None

    buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    t0 = time.perf_counter_ns()
    run(buf, len(payload))
    if has_wait:
        target.libafl_wait()
    elapsed = time.perf_counter_ns() - t0

    status = RunStatus()
    get_status(ctypes.byref(status))
    return ReplayResult(
        input_path="",
        input_size=len(payload),
        status_code=status.code,
        status_name=STATUS_NAMES.get(status.code, f"UNKNOWN({status.code})"),
        stage=status.stage,
        detail=status.detail,
        exec_time_ns=elapsed,
        exit_kind=_classify_exit(status.code),
    )


def _replay_async(target: ctypes.CDLL, payload: bytes, timeout_ms: int = 30000) -> ReplayResult:
    submit = target.libafl_submit_with_id
    submit.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    submit.restype = ctypes.c_uint64

    poll = target.libafl_poll_results
    poll.argtypes = [ctypes.POINTER(TaskResult), ctypes.c_size_t]
    poll.restype = ctypes.c_size_t

    release = target.libafl_release_tasks
    release.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    release.restype = ctypes.c_size_t

    set_timeout = target.libafl_set_target_timeout_ms
    set_timeout.argtypes = [ctypes.c_uint64]
    set_timeout.restype = None

    set_timeout(timeout_ms)

    buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    task_id = submit(buf, len(payload))
    if task_id == 0:
        return ReplayResult(
            input_path="",
            input_size=len(payload),
            status_code=3,
            status_name="REJECTED",
            stage=0,
            detail=0,
            exec_time_ns=None,
            exit_kind="invalid_input",
        )

    results = (TaskResult * 1)()
    deadline = time.monotonic() + timeout_ms / 1000.0 + 5.0
    while time.monotonic() < deadline:
        n = poll(results, 1)
        if n > 0:
            break
        time.sleep(0.001)
    else:
        return ReplayResult(
            input_path="",
            input_size=len(payload),
            status_code=2,
            status_name="POLL_TIMEOUT",
            stage=0,
            detail=0,
            exec_time_ns=None,
            exit_kind="timeout",
        )

    r = results[0]
    ids = (ctypes.c_uint64 * 1)(r.task_id)
    release(ids, 1)

    return ReplayResult(
        input_path="",
        input_size=len(payload),
        status_code=r.status.code,
        status_name=STATUS_NAMES.get(r.status.code, f"UNKNOWN({r.status.code})"),
        stage=r.status.stage,
        detail=r.status.detail,
        exec_time_ns=r.exec_time_ns,
        exit_kind=_classify_exit(r.status.code),
    )


def replay_input(library: Path, input_path: Path, *, timeout_ms: int = 30000) -> ReplayResult:
    """Replay a single crash input against the given backend library."""
    payload = input_path.read_bytes()
    target = ctypes.CDLL(str(library.resolve()))
    try:
        abi = _detect_abi(target)
        if abi == "sync":
            result = _replay_sync(target, payload)
        else:
            result = _replay_async(target, payload, timeout_ms=timeout_ms)
    finally:
        stop_library(target)
    return ReplayResult(
        input_path=str(input_path),
        input_size=result.input_size,
        status_code=result.status_code,
        status_name=result.status_name,
        stage=result.stage,
        detail=result.detail,
        exec_time_ns=result.exec_time_ns,
        exit_kind=result.exit_kind,
    )


def collect_crash_inputs(path: Path) -> list[Path]:
    """Collect crash input files from a directory (skip metadata/hidden files)."""
    if path.is_file():
        return [path]
    return sorted(f for f in path.iterdir() if f.is_file() and not f.name.startswith("."))


def replay_all(
    library: Path, inputs: Sequence[Path], *, timeout_ms: int = 30000
) -> list[ReplayResult]:
    """Replay multiple crash inputs, loading the library once per input."""
    results = []
    for inp in inputs:
        results.append(replay_input(library, inp, timeout_ms=timeout_ms))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path, help="Path to backend .so")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single crash input file")
    group.add_argument("--input-dir", type=Path, help="Directory of crash inputs")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Execution timeout")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.library.exists():
        print(f"error: library not found: {args.library}", file=sys.stderr)
        return 1

    path = args.input or args.input_dir
    inputs = collect_crash_inputs(path)
    if not inputs:
        print(f"error: no crash inputs found at {path}", file=sys.stderr)
        return 1

    results = replay_all(args.library, inputs, timeout_ms=args.timeout_ms)

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
    else:
        for r in results:
            detail_hex = f"0x{r.detail:x}"
            timing = f"{r.exec_time_ns/1e6:.2f}ms" if r.exec_time_ns else "n/a"
            print(
                f"{r.input_path}: {r.exit_kind} "
                f"[code={r.status_code} stage={r.stage} detail={detail_hex}] "
                f"size={r.input_size} time={timing}"
            )

    crashes = sum(1 for r in results if r.exit_kind == "crash")
    print(f"\nSummary: {len(results)} inputs, {crashes} crashes", file=sys.stderr)
    return 0 if crashes == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
