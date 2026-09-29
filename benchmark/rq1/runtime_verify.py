#!/usr/bin/env python3
"""Run fixed-seed GPU equivalence checks for every RQ1 backend artifact."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq1.artifacts import dump_seed, envelope_payload_size  # noqa: E402
from benchmark.rq1.build import EXPECTED_BACKENDS  # noqa: E402
from benchmark.rq1.schema import sha256_file  # noqa: E402
from benchmark.runtime_abi import RunStatus, TaskResult, stop_library  # noqa: E402


EDGES_MAP_SIZE = 65_536
SIMT_MEMCOV_STORAGE_SIZE = 8_192


class _AllocationStats(ctypes.Structure):
    _fields_ = [
        ("runs_started", ctypes.c_uint64),
        ("allocation_attempts", ctypes.c_uint64),
        ("allocations_succeeded", ctypes.c_uint64),
        ("frees", ctypes.c_uint64),
        ("runs_completed", ctypes.c_uint64),
    ]


class _OrderedQueueCounts(ctypes.Structure):
    _fields_ = [
        ("pending", ctypes.c_size_t),
        ("completed", ctypes.c_size_t),
        ("outstanding", ctypes.c_size_t),
    ]


def _stats_dict(stats: _AllocationStats) -> dict[str, int]:
    return {
        "runs_started": stats.runs_started,
        "allocation_attempts": stats.allocation_attempts,
        "allocations_succeeded": stats.allocations_succeeded,
        "frees": stats.frees,
        "runs_completed": stats.runs_completed,
    }


def _stats_delta(before: _AllocationStats, after: _AllocationStats) -> dict[str, int]:
    before_dict = _stats_dict(before)
    after_dict = _stats_dict(after)
    return {key: after_dict[key] - before_dict[key] for key in before_dict}


def _classify_status(status: RunStatus, counts: dict[str, int]) -> None:
    if status.code == 0:
        counts["completed"] += 1
    elif status.code == 2:
        counts["timeouts"] += 1
        counts["failed"] += 1
    elif status.code == 3:
        counts["invalid_inputs"] += 1
        counts["failed"] += 1
    else:
        counts["failed"] += 1


def _map_summary(target: ctypes.CDLL) -> tuple[int, int]:
    edge_map = bytes((ctypes.c_uint8 * EDGES_MAP_SIZE).in_dll(target, "libafl_cov_map"))
    mem_map = bytes(
        (ctypes.c_uint8 * SIMT_MEMCOV_STORAGE_SIZE).in_dll(
            target,
            "libafl_simt_memcov_bits",
        )
    )
    return (
        sum(value != 0 for value in edge_map),
        sum(value.bit_count() for value in mem_map),
    )


def _sync_worker(backend: str, library: Path, seed: bytes, runs: int) -> dict[str, Any]:
    target = ctypes.CDLL(str(library.resolve()))
    run = target.libafl_target
    run.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    run.restype = None
    get_status = target.libafl_get_last_run_status
    get_status.argtypes = [ctypes.POINTER(RunStatus)]
    get_status.restype = None
    wait = None
    if backend in ("rapid", "rapid-no-feedback"):
        wait = target.libafl_wait
        wait.argtypes = []
        wait.restype = None

    get_stats = None
    before_stats = None
    if backend == "cufuzz":
        get_stats = target.libafl_get_cufuzz_allocation_stats
        get_stats.argtypes = [ctypes.POINTER(_AllocationStats)]
        get_stats.restype = None
        before_stats = _AllocationStats()
        get_stats(ctypes.byref(before_stats))

    counts = {
        "runs": runs,
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "timeouts": 0,
        "invalid_inputs": 0,
    }
    statuses: list[dict[str, int]] = []
    try:
        for _ in range(runs):
            seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)
            counts["submitted"] += 1
            run(seed_buffer, len(seed))
            if wait is not None:
                wait()
            status = RunStatus()
            get_status(ctypes.byref(status))
            _classify_status(status, counts)
            statuses.append(
                {"code": status.code, "stage": status.stage, "detail": status.detail}
            )
        coverage_nonzero_bytes, simt_memcov_nonzero_bits = _map_summary(target)
        report: dict[str, Any] = {
            **counts,
            "statuses": statuses,
            "coverage_nonzero_bytes": coverage_nonzero_bytes,
            "simt_memcov_nonzero_bits": simt_memcov_nonzero_bits,
        }
        if get_stats is not None and before_stats is not None:
            after_stats = _AllocationStats()
            get_stats(ctypes.byref(after_stats))
            report["allocation_stats_before"] = _stats_dict(before_stats)
            report["allocation_stats_after"] = _stats_dict(after_stats)
            report["allocation_stats_delta"] = _stats_delta(before_stats, after_stats)
        return report
    finally:
        stop_library(target)


def _rapid2_worker(library: Path, seed: bytes, runs: int) -> dict[str, Any]:
    target = ctypes.CDLL(str(library.resolve()))
    set_timeout = target.libafl_set_target_timeout_ms
    set_timeout.argtypes = [ctypes.c_uint64]
    set_timeout.restype = None
    set_timeout(10_000)
    submit = target.libafl_submit_with_id
    submit.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    submit.restype = ctypes.c_uint64
    poll = target.libafl_poll_results
    poll.argtypes = [ctypes.POINTER(TaskResult), ctypes.c_size_t]
    poll.restype = ctypes.c_size_t
    release = target.libafl_release_tasks
    release.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    release.restype = ctypes.c_size_t
    wait = target.libafl_wait
    wait.argtypes = []
    wait.restype = None

    counts = {
        "runs": runs,
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "timeouts": 0,
        "invalid_inputs": 0,
    }
    statuses: list[dict[str, int]] = []
    max_coverage = 0
    max_mem_bits = 0
    try:
        for _ in range(runs):
            seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)
            task_id = submit(seed_buffer, len(seed))
            if task_id == 0:
                raise RuntimeError("rapid2 rejected fixed-seed submission")
            counts["submitted"] += 1
            result_array = (TaskResult * 1)()
            deadline = time.monotonic() + 20.0
            while poll(result_array, 1) != 1:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for rapid2 fixed-seed completion")
                time.sleep(0.001)
            result = result_array[0]
            if result.task_id != task_id:
                raise RuntimeError(
                    f"rapid2 completion mismatch: submitted {task_id}, got {result.task_id}"
                )
            _classify_status(result.status, counts)
            statuses.append(
                {
                    "code": result.status.code,
                    "stage": result.status.stage,
                    "detail": result.status.detail,
                }
            )
            if result.edge_size != EDGES_MAP_SIZE or result.simt_memcov_size != SIMT_MEMCOV_STORAGE_SIZE:
                raise RuntimeError("rapid2 returned incompatible feedback-map sizes")
            edge_map = ctypes.string_at(result.edge_ptr, result.edge_size)
            mem_map = ctypes.string_at(result.simt_memcov_ptr, result.simt_memcov_size)
            max_coverage = max(max_coverage, sum(value != 0 for value in edge_map))
            max_mem_bits = max(max_mem_bits, sum(value.bit_count() for value in mem_map))
            task_ids = (ctypes.c_uint64 * 1)(task_id)
            if release(task_ids, 1) != 1:
                raise RuntimeError(f"rapid2 failed to release task {task_id}")
        wait()
        return {
            **counts,
            "statuses": statuses,
            "coverage_nonzero_bytes": max_coverage,
            "simt_memcov_nonzero_bits": max_mem_bits,
        }
    finally:
        try:
            wait()
        finally:
            stop_library(target)


def _rapid_ordered_worker(
    library: Path,
    seed: bytes,
    runs: int,
    window_size: int,
) -> dict[str, Any]:
    target = ctypes.CDLL(str(library.resolve()))
    set_timeout = target.libafl_set_target_timeout_ms
    set_timeout.argtypes = [ctypes.c_uint64]
    set_timeout.restype = None
    set_timeout(10_000)
    submit = target.libafl_submit_with_id
    submit.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    submit.restype = ctypes.c_uint64
    poll = target.libafl_poll_results
    poll.argtypes = [ctypes.POINTER(TaskResult), ctypes.c_size_t]
    poll.restype = ctypes.c_size_t
    release = target.libafl_release_tasks
    release.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    release.restype = ctypes.c_size_t
    get_counts = target.libafl_get_ordered_queue_counts
    get_counts.argtypes = [ctypes.POINTER(_OrderedQueueCounts)]
    get_counts.restype = None
    wait = target.libafl_wait
    wait.argtypes = []
    wait.restype = None

    counts = {
        "runs": runs,
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "timeouts": 0,
        "invalid_inputs": 0,
    }
    statuses: list[dict[str, int]] = []
    pending: list[tuple[int, Any]] = []
    retired_task_ids: list[int] = []
    max_coverage = 0
    max_mem_bits = 0
    deadline = time.monotonic() + 20.0
    try:
        while counts["completed"] + counts["failed"] < runs:
            while counts["submitted"] < runs and len(pending) < window_size:
                seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)
                task_id = submit(seed_buffer, len(seed))
                if task_id == 0:
                    raise RuntimeError("rapid rejected ordered fixed-seed submission")
                if pending and task_id <= pending[-1][0]:
                    raise RuntimeError("rapid returned non-monotonic submission task IDs")
                pending.append((task_id, seed_buffer))
                counts["submitted"] += 1

            result_array = (TaskResult * max(1, len(pending)))()
            result_count = poll(result_array, len(result_array))
            if result_count == 0:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for rapid ordered completion")
                time.sleep(0.001)
                continue
            deadline = time.monotonic() + 20.0

            for index in range(result_count):
                result = result_array[index]
                if not pending:
                    raise RuntimeError(
                        f"rapid produced unknown completion task_id={result.task_id}"
                    )
                expected_id, seed_buffer = pending.pop(0)
                if result.task_id != expected_id:
                    raise RuntimeError(
                        "rapid completion retirement is not FIFO: "
                        f"expected {expected_id}, got {result.task_id}"
                    )
                if result.input_ptr != ctypes.addressof(seed_buffer):
                    raise RuntimeError(
                        f"rapid completion input attribution mismatch for task_id={result.task_id}"
                    )
                _classify_status(result.status, counts)
                statuses.append(
                    {
                        "code": result.status.code,
                        "stage": result.status.stage,
                        "detail": result.status.detail,
                    }
                )
                if (
                    result.edge_size != EDGES_MAP_SIZE
                    or result.simt_memcov_size != SIMT_MEMCOV_STORAGE_SIZE
                ):
                    raise RuntimeError("rapid returned incompatible feedback-map sizes")
                edge_map = ctypes.string_at(result.edge_ptr, result.edge_size)
                mem_map = ctypes.string_at(result.simt_memcov_ptr, result.simt_memcov_size)
                max_coverage = max(
                    max_coverage, sum(value != 0 for value in edge_map)
                )
                max_mem_bits = max(
                    max_mem_bits, sum(value.bit_count() for value in mem_map)
                )
                task_ids = (ctypes.c_uint64 * 1)(result.task_id)
                if release(task_ids, 1) != 1:
                    raise RuntimeError(
                        f"rapid failed to release ordered task {result.task_id}"
                    )
                retired_task_ids.append(result.task_id)

        wait()
        final_counts = _OrderedQueueCounts()
        get_counts(ctypes.byref(final_counts))
        return {
            **counts,
            "statuses": statuses,
            "coverage_nonzero_bytes": max_coverage,
            "simt_memcov_nonzero_bits": max_mem_bits,
            "window_size": window_size,
            "pending": final_counts.pending,
            "completed_queue": final_counts.completed,
            "outstanding": final_counts.outstanding,
            "retired_task_ids": retired_task_ids,
        }
    finally:
        try:
            wait()
        finally:
            stop_library(target)


def _worker(
    backend: str,
    library: Path,
    seed: bytes,
    runs: int,
    rapid_window_size: int,
) -> dict[str, Any]:
    if backend in ("rapid", "rapid-no-feedback"):
        return _rapid_ordered_worker(library, seed, runs, rapid_window_size)
    if backend in ("rapid2", "rapid2-no-feedback"):
        return _rapid2_worker(library, seed, runs)
    return _sync_worker(backend, library, seed, runs)


def _run_worker(
    backend: str,
    library: Path,
    seed: bytes,
    runs: int,
    *,
    gpu_device: int,
    rapid_window_size: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--library",
        str(library),
        "--seed-hex",
        seed.hex(),
        "--runs",
        str(runs),
        "--rapid-window-size",
        str(rapid_window_size),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=max(120, runs * 20),
    )
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(
        f"runtime worker produced no JSON for {backend}: "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def validate_runtime_report(report: Mapping[str, Any]) -> None:
    runs = report.get("runs")
    backends = report.get("backends")
    if not isinstance(runs, int) or runs <= 0 or not isinstance(backends, dict):
        raise RuntimeError("invalid runtime report header")
    if tuple(backends) != tuple(EXPECTED_BACKENDS):
        raise RuntimeError("runtime report backend order/set differs from RQ1 matrix")

    identity_fields = ("seed_sha256", "manifest_sha256", "payload_size", "vconfig")
    reference = {field: backends["cufuzz"].get(field) for field in identity_fields}
    for name, backend in backends.items():
        if backend.get("runs") != runs or backend.get("submitted") != runs:
            raise RuntimeError(f"{name}: submitted run count mismatch")
        if backend.get("completed") != runs or backend.get("failed") != 0:
            raise RuntimeError(f"{name}: terminal completion accounting mismatch")
        if backend.get("timeouts") != 0 or backend.get("invalid_inputs") != 0:
            raise RuntimeError(f"{name}: unexpected non-OK fixed-seed status")
        for field in identity_fields:
            if backend.get(field) != reference[field]:
                raise RuntimeError(f"{name}: cross-backend {field} drift")

    for name in (
        "cufuzz",
        "origin-no-feedback",
        "rapid-no-feedback",
        "rapid2-no-feedback",
    ):
        backend = backends[name]
        if backend.get("coverage_nonzero_bytes") != 0 or backend.get("simt_memcov_nonzero_bits") != 0:
            raise RuntimeError(f"{name}: expected zero feedback")

    for name in ("origin", "rapid", "rapid2"):
        backend = backends[name]
        if (
            int(backend.get("coverage_nonzero_bytes", 0))
            + int(backend.get("simt_memcov_nonzero_bits", 0))
            <= 0
        ):
            raise RuntimeError(f"{name}: LLVM feedback is unexpectedly empty")

    expected_allocation = {
        "runs_started": runs,
        "allocation_attempts": 2 * runs,
        "allocations_succeeded": 2 * runs,
        "frees": 2 * runs,
        "runs_completed": runs,
    }
    if backends["cufuzz"].get("allocation_stats_delta") != expected_allocation:
        raise RuntimeError(
            "cufuzz allocation lifecycle mismatch: "
            f"expected={expected_allocation} "
            f"actual={backends['cufuzz'].get('allocation_stats_delta')}"
        )

    for name in ("rapid-no-feedback", "rapid"):
        rapid = backends[name]
        window_size = rapid.get("window_size")
        if not isinstance(window_size, int) or not 1 <= window_size <= 32:
            raise RuntimeError(f"{name}: missing or invalid ordered window_size")
        for field in ("pending", "completed_queue", "outstanding"):
            if rapid.get(field) != 0:
                raise RuntimeError(f"{name}: final {field}={rapid.get(field)}")
        retired = rapid.get("retired_task_ids")
        if (
            not isinstance(retired, list)
            or len(retired) != runs
            or any(not isinstance(task_id, int) or task_id <= 0 for task_id in retired)
            or any(left >= right for left, right in zip(retired, retired[1:]))
        ):
            raise RuntimeError(f"{name}: completion retirement is not FIFO")


def verify_workload(
    build_report_path: Path,
    *,
    fuzzer: Path,
    gpu_device: int,
    runs: int,
    rapid_window_size: int,
) -> Path:
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    phase2_dir = Path(build_report["shared_phase2"]["phase2_dir"])
    manifest = phase2_dir.parent / "manifest.json"
    seed = dump_seed(fuzzer, manifest)
    payload_size = envelope_payload_size(seed)
    expected_payload = build_report["execution_contract"]["payload_size"]
    if payload_size != expected_payload:
        raise RuntimeError(
            f"{build_report['workload_id']}: generated payload size {payload_size} "
            f"differs from provenance {expected_payload}"
        )
    seed_sha256 = hashlib.sha256(seed).hexdigest()
    manifest_sha256 = sha256_file(manifest)

    backend_reports: dict[str, Any] = {}
    for name in EXPECTED_BACKENDS:
        backend = build_report["backends"][name]
        shared_library = backend["artifacts"]["shared_library"]["path"]
        library = Path(shared_library)
        if not library.is_absolute():
            report_dir = build_report_path.parent / "backends" / name
            library = report_dir / library
        worker_report = _run_worker(
            name,
            library,
            seed,
            runs,
            gpu_device=gpu_device,
            rapid_window_size=rapid_window_size,
        )
        backend_reports[name] = {
            **worker_report,
            "library": str(library.resolve()),
            "seed_sha256": seed_sha256,
            "manifest_sha256": manifest_sha256,
            "payload_size": payload_size,
            "vconfig": build_report["execution_contract"]["vconfig"],
            "launch_config": backend["launch_config"],
        }

    runtime_report = {
        "schema_version": 1,
        "workload_id": build_report["workload_id"],
        "runs": runs,
        "seed_sha256": seed_sha256,
        "manifest_sha256": manifest_sha256,
        "payload_size": payload_size,
        "rapid_window_size": rapid_window_size,
        "backends": backend_reports,
    }
    validate_runtime_report(runtime_report)
    output_path = build_report_path.parent / "runtime_verification.json"
    output_path.write_text(
        json.dumps(runtime_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--rapid-window-size", type=int, default=2)
    parser.add_argument(
        "--fuzzer",
        type=Path,
        default=REPO_ROOT / "cuda-fuzzer" / "target" / "release" / "fuzzer",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=tuple(EXPECTED_BACKENDS))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--seed-hex")
    args = parser.parse_args()

    if args.worker:
        if args.backend is None or args.library is None or args.seed_hex is None:
            parser.error("worker mode requires --backend, --library, and --seed-hex")
        print(
            json.dumps(
                _worker(
                    args.backend,
                    args.library,
                    bytes.fromhex(args.seed_hex),
                    args.runs,
                    args.rapid_window_size,
                )
            )
        )
        return 0
    if args.build_root is None:
        parser.error("--build-root is required")
    if not 1 <= args.rapid_window_size <= 32:
        parser.error("--rapid-window-size must be in 1..=32")
    if not args.fuzzer.is_file():
        raise RuntimeError(f"missing release fuzzer: {args.fuzzer}")
    reports = sorted(args.build_root.glob("*/build_report.json"))
    if not reports:
        raise RuntimeError(f"no RQ1 build reports under {args.build_root}")
    outputs = [
        str(
            verify_workload(
                report,
                fuzzer=args.fuzzer,
                gpu_device=args.gpu_device,
                runs=args.runs,
                rapid_window_size=args.rapid_window_size,
            )
        )
        for report in reports
    ]
    summary = args.build_root / "runtime_verification_summary.json"
    summary.write_text(
        json.dumps({"schema_version": 1, "reports": outputs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
