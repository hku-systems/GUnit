#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "test_e2e_workspace" / "runs" / "phase1_filtering" / "e2e_results.json"

KNOWN_UNSUPPORTED_REASONS = {
    "no_kernels",
    "union_not_supported",
    "pointer_role_not_supported",
    "materialization_status=unsafe",
    "opaque_with_ptr_layout_incomplete",
    "scalar_size_unsupported",
    "const_assignment_blocker",
    "scoped_type_shim_unsupported",
    "anonymous_type_unsupported",
    "local_type_unsupported",
    "missing_input",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _relativize(path: str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def _reason_from_phase1(phase1: dict[str, Any]) -> str:
    reason = phase1.get("failure_reason") or phase1.get("status") or "unknown"
    text = f"{reason}\n{phase1.get('stderr') or ''}\n{phase1.get('stdout') or ''}".lower()
    for known in sorted(KNOWN_UNSUPPORTED_REASONS):
        if known.lower() in text:
            return known
    return str(reason)


def _load_run_metadata(phase1: dict[str, Any]) -> dict[str, Any]:
    run_dir_raw = phase1.get("run_dir")
    if not run_dir_raw:
        return {}
    run_dir = Path(run_dir_raw)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    summary = _read_json(run_dir / "summary.json")
    index = _read_json(run_dir / "index.json")
    return {
        "summary_path": str(run_dir / "summary.json"),
        "index_path": str(run_dir / "index.json"),
        "summary_counts": summary.get("counts", {}) if isinstance(summary.get("counts"), dict) else {},
        "index_kernel_count": len(index.get("kernels", [])) if isinstance(index.get("kernels"), list) else 0,
    }


def _record_line(entry: dict[str, Any]) -> str:
    return "\t".join(
        [
            str(entry["candidate"]),
            str(entry.get("reason", "")),
            f"kernels={entry.get('kernel_count', 0)}",
            str(entry.get("run_dir", "")),
        ]
    ).rstrip()


def categorize_results(results_path: Path, out_dir: Path) -> dict[str, Any]:
    results_path = Path(results_path)
    if not results_path.is_absolute():
        results_path = REPO_ROOT / results_path
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _read_json(results_path)
    records = data.get("results", []) if isinstance(data.get("results"), list) else []
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for record in records:
        candidate = _relativize(str(record.get("candidate", "")))
        phase1 = record.get("stages", {}).get("phase1", {})
        if not isinstance(phase1, dict):
            phase1 = {}
        metadata = _load_run_metadata(phase1)
        kernel_count = int(phase1.get("kernel_count") or metadata.get("index_kernel_count") or 0)
        entry = {
            "candidate": candidate,
            "run_id": phase1.get("run_id"),
            "run_dir": phase1.get("run_dir"),
            "status": phase1.get("status"),
            "kernel_count": kernel_count,
            **metadata,
        }
        if phase1.get("ok"):
            supported.append(entry)
            continue

        reason = _reason_from_phase1(phase1)
        entry["reason"] = reason
        reasons[reason] += 1
        if reason in KNOWN_UNSUPPORTED_REASONS:
            unsupported.append(entry)
        else:
            errors.append(entry)

    (out_dir / "supported_kernels.txt").write_text(
        "".join(f"{entry['candidate']}\n" for entry in supported),
        encoding="utf-8",
    )
    (out_dir / "unsupported_kernels.txt").write_text(
        "".join(f"{_record_line(entry)}\n" for entry in unsupported),
        encoding="utf-8",
    )
    (out_dir / "error_kernels.txt").write_text(
        "".join(f"{_record_line(entry)}\n" for entry in errors),
        encoding="utf-8",
    )

    report = {
        "generated_at": _now_iso(),
        "input_results": str(results_path),
        "candidate_count": len(records),
        "stats": {
            "supported": len(supported),
            "unsupported": len(unsupported),
            "errors": len(errors),
            "failure_reasons": dict(sorted(reasons.items())),
        },
        "supported": supported,
        "unsupported": unsupported,
        "errors": errors,
        "outputs": {
            "supported_kernels": str(out_dir / "supported_kernels.txt"),
            "unsupported_kernels": str(out_dir / "unsupported_kernels.txt"),
            "error_kernels": str(out_dir / "error_kernels.txt"),
            "report": str(out_dir / "phase1_filtering_report.json"),
        },
    }
    (out_dir / "phase1_filtering_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Categorize RAPID Phase 1 filtering results.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to phase1-only e2e_results.json")
    parser.add_argument("--out-dir", default=".", help="Directory for supported/unsupported/error output files")
    args = parser.parse_args()

    report = categorize_results(Path(args.results), Path(args.out_dir))
    print(
        "phase1 categorization: "
        f"supported={report['stats']['supported']} "
        f"unsupported={report['stats']['unsupported']} "
        f"errors={report['stats']['errors']}"
    )
    print(f"report: {report['outputs']['report']}")
    return 1 if report["stats"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
