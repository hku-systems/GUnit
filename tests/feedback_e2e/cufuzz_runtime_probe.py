from __future__ import annotations

import argparse
import ctypes
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .feedback_runtime import EDGES_MAP_SIZE, SIMT_MEMCOV_STORAGE_SIZE
else:
    from feedback_runtime import EDGES_MAP_SIZE, SIMT_MEMCOV_STORAGE_SIZE


@dataclass(frozen=True)
class AllocationStats:
    runs_started: int
    allocation_attempts: int
    allocations_succeeded: int
    frees: int
    runs_completed: int

    def __sub__(self, other: "AllocationStats") -> "AllocationStats":
        return AllocationStats(
            runs_started=self.runs_started - other.runs_started,
            allocation_attempts=self.allocation_attempts - other.allocation_attempts,
            allocations_succeeded=(
                self.allocations_succeeded - other.allocations_succeeded
            ),
            frees=self.frees - other.frees,
            runs_completed=self.runs_completed - other.runs_completed,
        )


class _AllocationStats(ctypes.Structure):
    _fields_ = [
        ("runs_started", ctypes.c_uint64),
        ("allocation_attempts", ctypes.c_uint64),
        ("allocations_succeeded", ctypes.c_uint64),
        ("frees", ctypes.c_uint64),
        ("runs_completed", ctypes.c_uint64),
    ]


class _RunStatus(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("stage", ctypes.c_uint16),
        ("detail", ctypes.c_uint32),
    ]


def _snapshot_stats(get_stats: Any) -> AllocationStats:
    raw = _AllocationStats()
    get_stats(ctypes.byref(raw))
    return AllocationStats(
        runs_started=raw.runs_started,
        allocation_attempts=raw.allocation_attempts,
        allocations_succeeded=raw.allocations_succeeded,
        frees=raw.frees,
        runs_completed=raw.runs_completed,
    )


def validate_allocation_delta(
    before: AllocationStats,
    after: AllocationStats,
    *,
    expected_runs: int,
) -> AllocationStats:
    if expected_runs <= 0:
        raise ValueError(f"expected_runs must be positive, got {expected_runs}")
    actual = after - before
    expected = AllocationStats(
        runs_started=expected_runs,
        allocation_attempts=expected_runs * 2,
        allocations_succeeded=expected_runs * 2,
        frees=expected_runs * 2,
        runs_completed=expected_runs,
    )
    if actual != expected:
        raise RuntimeError(
            "allocation lifecycle mismatch: "
            f"expected={asdict(expected)} actual={asdict(actual)}"
        )
    return actual


def probe_cufuzz_library(library: Path, seed: bytes, *, runs: int) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError(f"runs must be positive, got {runs}")
    if not seed:
        raise ValueError("seed must not be empty")

    target = ctypes.CDLL(str(library.resolve()))
    run = target.libafl_target
    run.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    run.restype = None
    get_status = target.libafl_get_last_run_status
    get_status.argtypes = [ctypes.POINTER(_RunStatus)]
    get_status.restype = None
    get_output_size = target.libafl_get_last_output_size
    get_output_size.argtypes = []
    get_output_size.restype = ctypes.c_size_t
    get_stats = target.libafl_get_cufuzz_allocation_stats
    get_stats.argtypes = [ctypes.POINTER(_AllocationStats)]
    get_stats.restype = None

    edge_map = (ctypes.c_uint8 * EDGES_MAP_SIZE).in_dll(target, "libafl_cov_map")
    simt_memcov_bits = (ctypes.c_uint8 * SIMT_MEMCOV_STORAGE_SIZE).in_dll(
        target, "libafl_simt_memcov_bits"
    )

    before = _snapshot_stats(get_stats)
    output_sizes: list[int] = []
    for run_index in range(runs):
        seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)
        run(seed_buffer, len(seed))

        status = _RunStatus()
        get_status(ctypes.byref(status))
        if status.code != 0:
            raise RuntimeError(
                "cufuzz target failed: "
                f"run={run_index} code={status.code} "
                f"stage={status.stage} detail={status.detail}"
            )
        if any(edge_map):
            raise RuntimeError(f"coverage map is nonzero after run {run_index}")
        if any(simt_memcov_bits):
            raise RuntimeError(f"memory/index map is nonzero after run {run_index}")
        output_sizes.append(get_output_size())

    after = _snapshot_stats(get_stats)
    delta = validate_allocation_delta(before, after, expected_runs=runs)
    return {
        "library": str(library.resolve()),
        "runs": runs,
        "allocation_stats_before": asdict(before),
        "allocation_stats_after": asdict(after),
        "allocation_stats_delta": asdict(delta),
        "coverage_nonzero_bytes": sum(byte != 0 for byte in edge_map),
        "simt_memcov_nonzero_bits": sum(byte.bit_count() for byte in simt_memcov_bits),
        "output_sizes": output_sizes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Call a CuFuzz-style backend exactly N times in one process and "
            "verify per-run allocations plus zero host feedback."
        )
    )
    parser.add_argument("library", type=Path)
    parser.add_argument("--seed-hex", required=True)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    report = probe_cufuzz_library(
        args.library,
        bytes.fromhex(args.seed_hex),
        runs=args.runs,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
