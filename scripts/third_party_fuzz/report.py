"""Reports for third-party CUDA backend validation campaigns."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


_DEVICE_UNAVAILABLE_MARKERS = (
    "cudaGetDeviceCount failed",
    "no CUDA-capable device is detected",
)


def _log_contains(stage_record: dict[str, Any], needles: Sequence[str]) -> bool:
    """Return true if any stage log contains one of the marker strings."""

    paths = [
        stage_record.get("log"),
        stage_record.get("client_log"),
        stage_record.get("broker_log"),
    ]
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    if any(needle in chunk for needle in needles):
                        return True
        except OSError:
            continue
    return False


def _stage_passed(record: dict[str, Any], stage: str) -> bool:
    stage_record = record.get(stage)
    return isinstance(stage_record, dict) and stage_record.get("status") == "passed"


def _stage_failed(record: dict[str, Any], stage: str) -> bool:
    stage_record = record.get(stage)
    return isinstance(stage_record, dict) and stage_record.get("status") == "failed"


def _stage_status(record: dict[str, Any], stage: str) -> str:
    stage_record = record.get(stage)
    if not isinstance(stage_record, dict):
        return "missing"
    status = stage_record.get("status")
    return str(status) if status else "missing"


def _backend_matrix_status(record: dict[str, Any]) -> str:
    backend_results = record.get("backend_results")
    if not isinstance(backend_results, dict) or not backend_results:
        return _stage_status(record, "backend")
    return ", ".join(
        f"{backend}={_stage_status(backend_results, backend)}"
        for backend in sorted(backend_results)
    )


def _is_runnable(record: dict[str, Any]) -> bool:
    return record.get("phase2_status") == "built" and record.get("support_decision") == "run"


def _stage_failure_kind(record: dict[str, Any], stage: str) -> str:
    stage_record = record.get(stage)
    if not isinstance(stage_record, dict):
        return ""
    if _log_contains(stage_record, _DEVICE_UNAVAILABLE_MARKERS):
        return "cuda_device_unavailable"
    for key in ("failure_kind", "failure_reason"):
        reason = stage_record.get(key)
        if reason:
            return str(reason)
    return ""


def _record_reason(record: dict[str, Any]) -> str:
    for key in ("skip_reason", "runtime_skip_reason"):
        reason = record.get(key)
        if reason:
            return str(reason)
    for stage in ("mutation", "fixed", "backend"):
        reason = _stage_failure_kind(record, stage)
        if reason:
            return reason
    return ""


def _is_target_bug_candidate(record: dict[str, Any]) -> bool:
    reason = _record_reason(record)
    return reason.startswith("target_bug_candidate_") or bool(record.get("target_bug_candidate"))


def _runtime_passed(record: dict[str, Any]) -> bool:
    return _stage_passed(record, "backend") and _stage_passed(record, "fixed") and _stage_passed(record, "mutation")


def _runtime_failed(record: dict[str, Any]) -> bool:
    return _stage_failed(record, "backend") or _stage_failed(record, "fixed") or _stage_failed(record, "mutation")


def _runtime_incomplete(record: dict[str, Any]) -> bool:
    if not _is_runnable(record):
        return False
    if _runtime_passed(record) or _runtime_failed(record):
        return False
    return any(_stage_status(record, stage) in {"missing", "not_attempted", "unknown"} for stage in ("backend", "fixed", "mutation"))


def _coverage_summary(runnable: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matrix_records = [
        record
        for record in runnable
        if isinstance(record.get("coverage_results"), dict)
        and bool(record["coverage_results"])
    ]
    expected_cells = sorted(
        {
            f"{backend}/{vconfig}"
            for record in matrix_records
            for backend, backend_results in (
                record.get("coverage_results", {}).items()
                if isinstance(record.get("coverage_results"), dict)
                else ()
            )
            if isinstance(backend_results, dict)
            for vconfig in backend_results
        }
    )
    if not expected_cells:
        return {
            "enabled": False,
            "expected_cells": [],
            "cells_passed": 0,
            "cells_failed": 0,
            "applicability_skipped": 0,
            "cells_incomplete": 0,
            "final_by_cell": {},
            "normal": True,
        }

    passed = 0
    failed = 0
    applicability_skipped = 0
    incomplete = 0
    final_by_cell: dict[str, dict[str, int]] = {}
    for cell_name in expected_cells:
        backend, vconfig = cell_name.split("/", 1)
        totals = {
            "completed_kernels": 0,
            "executions_completed": 0,
            "cfg_sites": 0,
            "memory_features": 0,
            "thread_activity_features": 0,
        }
        for record in matrix_records:
            coverage_results = record.get("coverage_results")
            backend_results = (
                coverage_results.get(backend)
                if isinstance(coverage_results, dict)
                else None
            )
            cell = (
                backend_results.get(vconfig)
                if isinstance(backend_results, dict)
                else None
            )
            if not isinstance(cell, dict):
                incomplete += 1
                continue
            if cell.get("status") == "passed":
                passed += 1
                totals["completed_kernels"] += 1
                for field in (
                    "executions_completed",
                    "cfg_sites",
                    "memory_features",
                    "thread_activity_features",
                ):
                    value = cell.get(field)
                    if isinstance(value, int) and not isinstance(value, bool):
                        totals[field] += value
                continue
            if (
                cell.get("status") == "skipped"
                and cell.get("skip_kind") == "applicability"
            ):
                applicability_skipped += 1
                continue
            if cell.get("status") == "failed":
                failed += 1
                continue
            incomplete += 1
        final_by_cell[cell_name] = totals

    return {
        "enabled": True,
        "expected_cells": expected_cells,
        "cells_passed": passed,
        "cells_failed": failed,
        "applicability_skipped": applicability_skipped,
        "cells_incomplete": incomplete,
        "final_by_cell": final_by_cell,
        "normal": failed == 0 and incomplete == 0,
    }


def _backend_matrix_summary(runnable: Sequence[dict[str, Any]]) -> dict[str, Any]:
    statuses: list[str] = []
    for record in runnable:
        backend_results = record.get("backend_results")
        if not isinstance(backend_results, dict):
            continue
        statuses.extend(
            _stage_status(backend_results, backend) for backend in backend_results
        )
    counts = Counter(statuses)
    failed = counts.get("failed", 0)
    incomplete = len(statuses) - counts.get("passed", 0) - failed
    return {
        "enabled": bool(statuses),
        "cells_total": len(statuses),
        "cells_passed": counts.get("passed", 0),
        "cells_failed": failed,
        "cells_incomplete": incomplete,
        "normal": failed == 0 and incomplete == 0,
    }


def summarize_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-kernel records into the campaign normality verdict."""

    total = len(records)
    phase2_counts = Counter(str(record.get("phase2_status") or "unknown") for record in records)
    skip_counts = Counter(
        str(record.get("skip_reason") or "unspecified")
        for record in records
        if record.get("support_decision") == "skip"
    )
    runnable = [record for record in records if _is_runnable(record)]
    runtime_passed = [record for record in runnable if _runtime_passed(record)]
    runtime_failed = [record for record in runnable if _runtime_failed(record)]
    runtime_incomplete = [record for record in runnable if _runtime_incomplete(record)]
    fixed_counts = Counter(_stage_status(record, "fixed") for record in runnable)
    mutation_counts = Counter(_stage_status(record, "mutation") for record in runnable)
    runtime_skip_counts = Counter(
        str(record.get("runtime_skip_reason") or "unspecified")
        for record in runnable
        if record.get("runtime_decision") == "skip"
    )
    mutation_failure_counts = Counter(
        _stage_failure_kind(record, "mutation") or "unspecified"
        for record in runnable
        if isinstance(record.get("mutation"), dict) and record["mutation"].get("status") == "failed"
    )
    bug_candidates = [record for record in records if _is_target_bug_candidate(record)]
    bug_candidate_counts = Counter(_record_reason(record) or "unspecified" for record in bug_candidates)
    backend_matrix = _backend_matrix_summary(runnable)
    coverage = _coverage_summary(runnable)
    return {
        "schema_version": 1,
        "total_kernels": total,
        "phase2": dict(sorted(phase2_counts.items())),
        "runnable_kernels": len(runnable),
        "runtime_passed": len(runtime_passed),
        "runtime_failed": len(runtime_failed),
        "runtime_incomplete": len(runtime_incomplete),
        "fixed_passed": fixed_counts.get("passed", 0),
        "fixed_failed": fixed_counts.get("failed", 0),
        "runtime_skipped": sum(runtime_skip_counts.values()),
        "runtime_skip_reasons": dict(sorted(runtime_skip_counts.items())),
        "mutation_passed": mutation_counts.get("passed", 0),
        "mutation_failed": mutation_counts.get("failed", 0),
        "mutation_skipped": mutation_counts.get("skipped", 0),
        "mutation_not_attempted": mutation_counts.get("not_attempted", 0),
        "mutation_failure_kinds": dict(sorted(mutation_failure_counts.items())),
        "skipped_kernels": sum(skip_counts.values()),
        "skip_reasons": dict(sorted(skip_counts.items())),
        "target_bug_candidates": len(bug_candidates),
        "target_bug_candidate_reasons": dict(sorted(bug_candidate_counts.items())),
        "backend_matrix": backend_matrix,
        "coverage": coverage,
        "fuzzer_normal": (
            len(runtime_failed) == 0
            and len(runtime_incomplete) == 0
            and backend_matrix["normal"]
            and coverage["normal"]
        ),
    }


def render_markdown(records: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    """Render a compact per-kernel Markdown report."""

    verdict = "normal" if summary.get("fuzzer_normal") else "not normal"
    coverage = (
        summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
    )
    backend_matrix = (
        summary.get("backend_matrix")
        if isinstance(summary.get("backend_matrix"), dict)
        else {}
    )
    lines = [
        "# Third-party CUDA backend validation",
        "",
        f"- Fuzzer status: `{verdict}`",
        f"- Total kernels: {summary.get('total_kernels', 0)}",
        f"- Runnable kernels: {summary.get('runnable_kernels', 0)}",
        f"- Runtime passed: {summary.get('runtime_passed', 0)}",
        f"- Runtime failed: {summary.get('runtime_failed', 0)}",
        f"- Runtime incomplete: {summary.get('runtime_incomplete', 0)}",
        f"- Fixed passed / failed: {summary.get('fixed_passed', 0)} / {summary.get('fixed_failed', 0)}",
        f"- Runtime skipped after fixed: {summary.get('runtime_skipped', 0)}",
        f"- Mutation passed / failed / skipped: {summary.get('mutation_passed', 0)} / {summary.get('mutation_failed', 0)} / {summary.get('mutation_skipped', 0)}",
        f"- Mutation not attempted: {summary.get('mutation_not_attempted', 0)}",
        f"- Skipped kernels: {summary.get('skipped_kernels', 0)}",
        f"- Target bug candidates: {summary.get('target_bug_candidates', 0)}",
    ]
    if backend_matrix.get("enabled"):
        lines.append(
            "- Backend matrix builds passed: "
            f"{backend_matrix.get('cells_passed', 0)} / "
            f"{backend_matrix.get('cells_total', 0)} "
            f"(failed: {backend_matrix.get('cells_failed', 0)}, "
            f"incomplete: {backend_matrix.get('cells_incomplete', 0)})"
        )
    if coverage.get("enabled"):
        lines.append(
            "- Coverage passed / failed / applicability skipped / incomplete: "
            f"{coverage.get('cells_passed', 0)} / "
            f"{coverage.get('cells_failed', 0)} / "
            f"{coverage.get('applicability_skipped', 0)} / "
            f"{coverage.get('cells_incomplete', 0)}"
        )
    lines.extend(["", "## Skip reasons", ""])
    skip_reasons = summary.get("skip_reasons") if isinstance(summary.get("skip_reasons"), dict) else {}
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Target bug candidate reasons", ""])
    bug_candidate_reasons = (
        summary.get("target_bug_candidate_reasons")
        if isinstance(summary.get("target_bug_candidate_reasons"), dict)
        else {}
    )
    if bug_candidate_reasons:
        for reason, count in sorted(bug_candidate_reasons.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Runtime skip reasons", ""])
    runtime_skip_reasons = (
        summary.get("runtime_skip_reasons") if isinstance(summary.get("runtime_skip_reasons"), dict) else {}
    )
    if runtime_skip_reasons:
        for reason, count in sorted(runtime_skip_reasons.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Mutation failure kinds", ""])
    mutation_failure_kinds = (
        summary.get("mutation_failure_kinds") if isinstance(summary.get("mutation_failure_kinds"), dict) else {}
    )
    if mutation_failure_kinds:
        for reason, count in sorted(mutation_failure_kinds.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    final_by_cell = (
        coverage.get("final_by_cell")
        if isinstance(coverage.get("final_by_cell"), dict)
        else {}
    )
    if coverage.get("enabled"):
        lines.extend(
            [
                "",
                "## Coverage final metrics",
                "",
                "| Cell | Completed kernels | Executions | CFG sites | Memory features | Thread-activity features |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for cell_name, totals in sorted(final_by_cell.items()):
            if not isinstance(totals, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(cell_name),
                        str(totals.get("completed_kernels", 0)),
                        str(totals.get("executions_completed", 0)),
                        str(totals.get("cfg_sites", 0)),
                        str(totals.get("memory_features", 0)),
                        str(totals.get("thread_activity_features", 0)),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Per-kernel results",
            "",
            "| Project | Kernel | Decision | Backend | Fixed | Runtime | Mutation | Reason |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for record in sorted(records, key=lambda item: (str(item.get("project")), str(item.get("kernel_id")))):
        decision = str(record.get("support_decision") or "unknown")
        reason = _record_reason(record)
        runtime = str(record.get("runtime_decision") or "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("project") or ""),
                    str(record.get("kernel_id") or ""),
                    decision,
                    _backend_matrix_status(record),
                    _stage_status(record, "fixed"),
                    runtime,
                    _stage_status(record, "mutation"),
                    reason,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_bug_candidates_markdown(records: Sequence[dict[str, Any]]) -> str:
    """Render target-side bug candidates found or excluded during the campaign."""

    bug_candidates = [record for record in records if _is_target_bug_candidate(record)]
    lines = [
        "# Third-party target bug candidates",
        "",
        "These entries are excluded from fuzzer-normality accounting because the evidence points to target-side behavior that can deadlock or fault independently of fuzzer mutation validity.",
        "",
    ]
    if not bug_candidates:
        lines.append("- none")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Project | Kernel | Symbol | Reason | Detail |",
            "|---|---|---|---|---|",
        ]
    )
    for record in sorted(bug_candidates, key=lambda item: (str(item.get("project")), str(item.get("kernel_id")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("project") or ""),
                    str(record.get("kernel_id") or ""),
                    str(record.get("symbol_name") or ""),
                    _record_reason(record),
                    str(record.get("skip_detail") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(out_dir: Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Persist kernel_results.json, summary.json, and SUMMARY.md."""

    out_dir.mkdir(parents=True, exist_ok=True)
    serializable_records = list(records)
    summary = summarize_results(serializable_records)
    (out_dir / "kernel_results.json").write_text(
        json.dumps({"schema_version": 1, "kernels": serializable_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "SUMMARY.md").write_text(
        render_markdown(serializable_records, summary),
        encoding="utf-8",
    )
    (out_dir / "BUG_CANDIDATES.md").write_text(
        render_bug_candidates_markdown(serializable_records),
        encoding="utf-8",
    )
    return summary
