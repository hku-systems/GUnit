from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EDGES_MAP_SIZE = 65_536
SIMT_MEMCOV_STORAGE_SIZE = 8_192
THREAD_ACTIVITY_BYTE_BASE = 61_440 // 8
DEFAULT_RAPID2_TARGET_TIMEOUT_MS = 10_000


@dataclass(frozen=True)
class FeedbackResult:
    backend: str
    cfg_sites: int
    simt_memcov_bits: int
    logical_thread_bits: int
    simt_memcov_set_bits: tuple[int, ...]
    logical_thread_set_bits: tuple[int, ...]
    edge_sha256: str
    simt_memcov_sha256: str


@dataclass(frozen=True)
class FeedbackArtifact:
    kernel_dir: Path
    manifest: Path
    origin_library: Path
    rapid_library: Path
    rapid2_library: Path


def summarize_maps(backend: str, edge_map: bytes, simt_memcov_bits: bytes) -> FeedbackResult:
    if len(edge_map) != EDGES_MAP_SIZE:
        raise ValueError(
            f"invalid edge map size: expected {EDGES_MAP_SIZE}, got {len(edge_map)}"
        )
    if len(simt_memcov_bits) != SIMT_MEMCOV_STORAGE_SIZE:
        raise ValueError(
            "invalid memory/index bitset size: "
            f"expected {SIMT_MEMCOV_STORAGE_SIZE}, got {len(simt_memcov_bits)}"
        )

    memory_partition = simt_memcov_bits[:THREAD_ACTIVITY_BYTE_BASE]
    thread_partition = simt_memcov_bits[THREAD_ACTIVITY_BYTE_BASE:]

    return FeedbackResult(
        backend=backend,
        cfg_sites=sum(byte != 0 for byte in edge_map),
        simt_memcov_bits=sum(byte.bit_count() for byte in memory_partition),
        logical_thread_bits=sum(byte.bit_count() for byte in thread_partition),
        simt_memcov_set_bits=tuple(
            byte_index * 8 + bit_index
            for byte_index, byte in enumerate(memory_partition)
            for bit_index in range(8)
            if byte & (1 << bit_index)
        ),
        logical_thread_set_bits=tuple(
            byte_index * 8 + bit_index
            for byte_index, byte in enumerate(thread_partition)
            for bit_index in range(8)
            if byte & (1 << bit_index)
        ),
        edge_sha256=hashlib.sha256(edge_map).hexdigest(),
        simt_memcov_sha256=hashlib.sha256(simt_memcov_bits).hexdigest(),
    )


def patch_trailing_u64(seed: bytes, value: int) -> bytes:
    if len(seed) < 8:
        raise ValueError("arg-pack seed is too short for a trailing u64")
    if not 0 <= value <= (1 << 64) - 1:
        raise ValueError(f"u64 value out of range: {value}")
    return seed[:-8] + value.to_bytes(8, "little")


def discover_feedback_artifact(
    run_dir: Path, *, kernel_id: str | None = None
) -> FeedbackArtifact:
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    for entry in index.get("kernels", []):
        if kernel_id is not None and entry.get("kernel_id") != kernel_id:
            continue
        kernel_dir = Path(entry["dir"])
        if not kernel_dir.is_absolute():
            kernel_dir = run_dir / kernel_dir
        artifact = FeedbackArtifact(
            kernel_dir=kernel_dir,
            manifest=kernel_dir / "manifest.json",
            origin_library=kernel_dir
            / "phase2/backends/origin/libphase2_origin_target.so",
            rapid_library=kernel_dir
            / "phase2/backends/rapid/libphase2_rapid_target.so",
            rapid2_library=kernel_dir / "phase2/backends/rapid2/librapid2_target.so",
        )
        if all(
            path.is_file()
            for path in (
                artifact.manifest,
                artifact.origin_library,
                artifact.rapid_library,
                artifact.rapid2_library,
            )
        ):
            return artifact
    if kernel_id is not None:
        raise FileNotFoundError(
            f"no complete three-backend artifact for {kernel_id} under {run_dir}"
        )
    raise FileNotFoundError(f"no complete three-backend artifact under {run_dir}")


def generate_default_seed(fuzzer: Path, manifest: Path) -> bytes:
    result = subprocess.run(
        [
            str(fuzzer),
            "/dev/null",
            "--manifest",
            str(manifest),
            "--dump-seed",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("fuzzer produced no seed")
    return bytes.fromhex(lines[-1])


def run_feedback_worker(
    backend: str,
    library: Path,
    seeds: list[bytes],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "tests.feedback_e2e.feedback_runtime",
        "--worker",
        "--backend",
        backend,
        "--library",
        str(library),
    ]
    for seed in seeds:
        command.extend(["--seed-hex", seed.hex()])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        f"feedback worker produced no JSON: stdout={result.stdout!r}, stderr={result.stderr!r}"
    )


class _RunStatus(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("stage", ctypes.c_uint16),
        ("detail", ctypes.c_uint32),
    ]


class _TaskResult(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("input_ptr", ctypes.c_size_t),
        ("edge_ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("simt_memcov_ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("edge_size", ctypes.c_uint32),
        ("simt_memcov_size", ctypes.c_uint32),
        ("status", _RunStatus),
        ("exec_time_ns", ctypes.c_uint64),
    ]


class _OrderedQueueCounts(ctypes.Structure):
    _fields_ = [
        ("pending", ctypes.c_size_t),
        ("completed", ctypes.c_size_t),
        ("outstanding", ctypes.c_size_t),
    ]


def _sync_worker(backend: str, library: Path, seed: bytes) -> dict[str, Any]:
    target = ctypes.CDLL(str(library))
    try:
        run = target.libafl_target
        run.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        run.restype = None
        seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)
        run(seed_buffer, len(seed))
        if backend == "rapid":
            wait = target.libafl_wait
            wait.argtypes = []
            wait.restype = None
            wait()

        status = _RunStatus()
        get_status = target.libafl_get_last_run_status
        get_status.argtypes = [ctypes.POINTER(_RunStatus)]
        get_status.restype = None
        get_status(ctypes.byref(status))
        edge_map = bytes(
            (ctypes.c_uint8 * EDGES_MAP_SIZE).in_dll(target, "libafl_cov_map")
        )
        simt_memcov_bits = bytes(
            (ctypes.c_uint8 * SIMT_MEMCOV_STORAGE_SIZE).in_dll(
                target, "libafl_simt_memcov_bits"
            )
        )
        return {
            "backend": backend,
            "result": asdict(summarize_maps(backend, edge_map, simt_memcov_bits)),
            "status": {
                "code": status.code,
                "stage": status.stage,
                "detail": status.detail,
            },
        }
    finally:
        try:
            stop = target.libafl_stop
        except AttributeError:
            pass
        else:
            stop.argtypes = []
            stop.restype = None
            stop()


def _poll_one(
    poll: Any,
    *,
    backend: str,
    deadline: float,
) -> _TaskResult:
    output = (_TaskResult * 1)()
    while time.monotonic() < deadline:
        if poll(output, 1) == 1:
            return output[0]
        time.sleep(0.001)
    raise TimeoutError(f"timed out waiting for {backend} completion")


def _ordered_abi_worker(backend: str, library: Path, seeds: list[bytes]) -> dict[str, Any]:
    target = ctypes.CDLL(str(library))
    set_timeout = target.libafl_set_target_timeout_ms
    set_timeout.argtypes = [ctypes.c_uint64]
    set_timeout.restype = None
    set_timeout(DEFAULT_RAPID2_TARGET_TIMEOUT_MS)
    submit = target.libafl_submit_with_id
    submit.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    submit.restype = ctypes.c_uint64
    poll = target.libafl_poll_results
    poll.argtypes = [ctypes.POINTER(_TaskResult), ctypes.c_size_t]
    poll.restype = ctypes.c_size_t
    release = target.libafl_release_tasks
    release.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    release.restype = ctypes.c_size_t
    wait = target.libafl_wait
    wait.argtypes = []
    wait.restype = None
    stop = target.libafl_stop
    stop.argtypes = []
    stop.restype = None
    get_ordered_counts = None
    if backend == "rapid":
        get_ordered_counts = target.libafl_get_ordered_queue_counts
        get_ordered_counts.argtypes = [ctypes.POINTER(_OrderedQueueCounts)]
        get_ordered_counts.restype = None

    acquired: list[_TaskResult] = []
    serialized: list[dict[str, Any]] = []
    submitted_task_ids: list[int] = []
    first_edge_snapshot: bytes | None = None
    first_mem_snapshot: bytes | None = None
    deadline = time.monotonic() + 20.0
    try:
        for seed in seeds:
            seed_buffer = (ctypes.c_uint8 * len(seed)).from_buffer_copy(seed)
            task_id = submit(seed_buffer, len(seed))
            if task_id == 0:
                raise RuntimeError(f"{backend} rejected submission with task_id=0")
            submitted_task_ids.append(task_id)

        for task_id in submitted_task_ids:
            result = _poll_one(poll, backend=backend, deadline=deadline)
            if result.task_id != task_id:
                raise RuntimeError(
                    f"{backend} completion mismatch: submitted {task_id}, got {result.task_id}"
                )
            if result.edge_size != EDGES_MAP_SIZE:
                raise RuntimeError(f"invalid {backend} edge size: {result.edge_size}")
            if result.simt_memcov_size != SIMT_MEMCOV_STORAGE_SIZE:
                raise RuntimeError(
                    f"invalid {backend} memory/index size: {result.simt_memcov_size}"
                )
            if not result.edge_ptr or not result.simt_memcov_ptr:
                raise RuntimeError(f"{backend} returned a null feedback pointer")
            edge_map = ctypes.string_at(result.edge_ptr, result.edge_size)
            simt_memcov_bits = ctypes.string_at(
                result.simt_memcov_ptr, result.simt_memcov_size
            )
            if first_edge_snapshot is None:
                first_edge_snapshot = edge_map
                first_mem_snapshot = simt_memcov_bits
            serialized.append(
                {
                    "task_id": result.task_id,
                    "result": asdict(
                        summarize_maps(backend, edge_map, simt_memcov_bits)
                    ),
                    "status": {
                        "code": result.status.code,
                        "stage": result.status.stage,
                        "detail": result.status.detail,
                    },
                }
            )
            acquired.append(result)

        first_edge_after_reuse = ctypes.string_at(
            acquired[0].edge_ptr, acquired[0].edge_size
        )
        first_mem_after_reuse = ctypes.string_at(
            acquired[0].simt_memcov_ptr, acquired[0].simt_memcov_size
        )
        task_ids = (ctypes.c_uint64 * len(acquired))(
            *(result.task_id for result in acquired)
        )
        released = release(task_ids, len(acquired))
        if released != len(acquired):
            raise RuntimeError(
                f"{backend} released {released} of {len(acquired)} acquired tasks"
            )
        acquired.clear()
        queue_counts: dict[str, int] | None = None
        if get_ordered_counts is not None:
            counts = _OrderedQueueCounts()
            get_ordered_counts(ctypes.byref(counts))
            queue_counts = {
                "pending": counts.pending,
                "completed": counts.completed,
                "outstanding": counts.outstanding,
            }
            if any(queue_counts.values()):
                raise RuntimeError(
                    f"{backend} left ordered queue work after release: {queue_counts}"
                )
        return {
            "backend": backend,
            "submitted_task_ids": submitted_task_ids,
            "results": serialized,
            "first_result_lifetime_preserved": (
                first_edge_snapshot == first_edge_after_reuse
                and first_mem_snapshot == first_mem_after_reuse
            ),
            "queue_counts": queue_counts,
        }
    finally:
        if acquired:
            task_ids = (ctypes.c_uint64 * len(acquired))(
                *(result.task_id for result in acquired)
            )
            release(task_ids, len(acquired))
        wait()
        stop()


def _rapid_ordered_worker(library: Path, seed: bytes) -> dict[str, Any]:
    ordered = _ordered_abi_worker("rapid", library, [seed])
    (result,) = ordered["results"]
    return {
        "backend": "rapid",
        "result": result["result"],
        "status": result["status"],
        "task_id": result["task_id"],
        "first_result_lifetime_preserved": ordered[
            "first_result_lifetime_preserved"
        ],
        "queue_counts": ordered["queue_counts"],
    }


def _rapid2_worker(library: Path, seeds: list[bytes]) -> dict[str, Any]:
    return _ordered_abi_worker("rapid2", library, seeds)


def _worker_main(backend: str, library: Path, seed_hexes: list[str]) -> dict[str, Any]:
    seeds = [bytes.fromhex(seed_hex) for seed_hex in seed_hexes]
    if not seeds:
        raise ValueError("at least one --seed-hex is required")
    if backend == "origin":
        if len(seeds) != 1:
            raise ValueError(f"{backend} worker expects exactly one seed")
        return _sync_worker(backend, library, seeds[0])
    if backend == "rapid":
        if len(seeds) != 1:
            raise ValueError(f"{backend} worker expects exactly one seed")
        return _rapid_ordered_worker(library, seeds[0])
    if backend == "rapid2":
        return _rapid2_worker(library, seeds)
    raise ValueError(f"unsupported backend: {backend}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=("origin", "rapid", "rapid2"))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--seed-hex", action="append", default=[])
    args = parser.parse_args()
    if not args.worker or args.backend is None or args.library is None:
        parser.error("--worker, --backend, and --library are required")
    print(json.dumps(_worker_main(args.backend, args.library, args.seed_hex)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
