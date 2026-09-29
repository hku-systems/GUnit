#!/usr/bin/env python3
"""Extract compact, thread-aware evidence from an Nsight SQLite export."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_TABLES = {
    "StringIds",
    "ThreadNames",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
}


class TraceSchemaError(RuntimeError):
    pass


def cuda_api_category(name: str) -> str:
    lowered = name.lower()
    if "malloc" in lowered or "memalloc" in lowered:
        return "allocation"
    if "free" in lowered or "memfree" in lowered:
        return "free"
    if "memcpy" in lowered:
        return "memcpy"
    if "launch" in lowered:
        return "launch"
    if any(
        token in lowered for token in ("synchronize", "waitevent", "eventquery")
    ):
        return "synchronization"
    return "other"


@dataclass(frozen=True)
class KernelInterval:
    start_ns: int
    end_ns: int
    correlation_id: int
    name: str
    launcher_thread: str

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class TraceEvidence:
    capture_start_ns: int
    capture_end_ns: int
    thread_names: dict[int, str]
    cuda_api_time_ns: dict[str, dict[str, int]]
    cuda_api_calls: dict[str, dict[str, int]]
    kernels: tuple[KernelInterval, ...]
    gpu_memcpy_time_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _duration(start: int, end: int, *, source: str) -> int:
    if end < start:
        raise TraceSchemaError(
            f"{source} interval ends before it starts: start={start}, end={end}"
        )
    return end - start


def extract_trace(path: Path) -> TraceEvidence:
    connection = sqlite3.connect(path)
    try:
        tables = _tables(connection)
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise TraceSchemaError(
                f"Nsight SQLite trace is missing required tables: {', '.join(missing)}"
            )

        thread_names: dict[int, str] = {}
        for global_tid, name, _priority in connection.execute(
            """
            SELECT t.globalTid, s.value, t.priority
            FROM ThreadNames AS t
            JOIN StringIds AS s ON s.id = t.nameId
            ORDER BY t.globalTid, t.priority DESC, t.nameId
            """
        ):
            thread_names.setdefault(int(global_tid), str(name))

        api_time: dict[str, dict[str, int]] = {}
        api_calls: dict[str, dict[str, int]] = {}
        correlation_threads: dict[int, str] = {}
        bounds: list[int] = []
        runtime_rows = connection.execute(
            """
            SELECT r.start, r.end, r.globalTid, r.correlationId, s.value
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON s.id = r.nameId
            ORDER BY r.start, r.end, r.globalTid, r.correlationId
            """
        )
        for start, end, global_tid, correlation_id, api_name in runtime_rows:
            start_ns, end_ns = int(start), int(end)
            duration_ns = _duration(start_ns, end_ns, source="CUDA API")
            bounds.extend((start_ns, end_ns))
            thread = thread_names.get(int(global_tid), f"tid:{int(global_tid)}")
            category = cuda_api_category(str(api_name))
            thread_times = api_time.setdefault(thread, {})
            thread_times[category] = thread_times.get(category, 0) + duration_ns
            thread_calls = api_calls.setdefault(thread, {})
            thread_calls[category] = thread_calls.get(category, 0) + 1
            correlation_threads[int(correlation_id)] = thread

        kernels: list[KernelInterval] = []
        kernel_rows = (
            connection.execute(
                """
                SELECT k.start, k.end, k.correlationId,
                       COALESCE(demangled.value, short.value, '<unknown>')
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                LEFT JOIN StringIds AS short ON short.id = k.shortName
                LEFT JOIN StringIds AS demangled ON demangled.id = k.demangledName
                ORDER BY k.start, k.end, k.correlationId
                """
            )
            if "CUPTI_ACTIVITY_KIND_KERNEL" in tables
            else ()
        )
        for start, end, correlation_id, name in kernel_rows:
            start_ns, end_ns = int(start), int(end)
            _duration(start_ns, end_ns, source="GPU kernel")
            bounds.extend((start_ns, end_ns))
            correlation = int(correlation_id)
            kernels.append(
                KernelInterval(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    correlation_id=correlation,
                    name=str(name),
                    launcher_thread=correlation_threads.get(
                        correlation, "unresolved"
                    ),
                )
            )

        gpu_memcpy_time_ns = 0
        memcpy_rows = (
            connection.execute(
                """
                SELECT start, end
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                ORDER BY start, end, correlationId
                """
            )
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
            else ()
        )
        for start, end in memcpy_rows:
            start_ns, end_ns = int(start), int(end)
            gpu_memcpy_time_ns += _duration(
                start_ns, end_ns, source="GPU memcpy"
            )
            bounds.extend((start_ns, end_ns))

        if not bounds:
            raise TraceSchemaError("Nsight SQLite trace contains no captured intervals")
        return TraceEvidence(
            capture_start_ns=min(bounds),
            capture_end_ns=max(bounds),
            thread_names=thread_names,
            cuda_api_time_ns=api_time,
            cuda_api_calls=api_calls,
            kernels=tuple(kernels),
            gpu_memcpy_time_ns=gpu_memcpy_time_ns,
        )
    finally:
        connection.close()


def write_compact_evidence(
    sqlite_path: Path, output_path: Path, *, keep_sqlite: bool = False
) -> TraceEvidence:
    evidence = extract_trace(sqlite_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    if not keep_sqlite:
        sqlite_path.unlink()
    return evidence
