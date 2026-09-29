"""Shared build-report and benchmark-command helpers for RQ4 profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.rq1 import campaign as rq1
from benchmark.rq1.campaign import benchmark_command


def _load_reports(build_roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for build_root in build_roots:
        for report_path, report in rq1._load_build_reports(build_root.resolve()):
            workload_id = report["workload_id"]
            if workload_id in seen:
                raise RuntimeError(f"duplicate workload across build roots: {workload_id}")
            seen.add(workload_id)
            reports.append((report_path, report))
    return reports
