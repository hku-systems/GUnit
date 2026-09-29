import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_phase2_for_run_dir(run_dir: Path) -> dict[str, Any]:
    cli = REPO_ROOT / "scripts" / "kernel-rewrite" / "cli.py"
    subprocess.run(
        [sys.executable, str(cli), "--run-dir", str(run_dir)],
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads((run_dir / "rewrite_summary.json").read_text(encoding="utf-8"))


def collect_phase2_kernel_results(run_dir: Path) -> list[dict[str, Any]]:
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for entry in index.get("kernels", []):
        kernel_dir = Path(entry["dir"])
        if not kernel_dir.is_absolute():
            kernel_dir = run_dir / kernel_dir
        phase2_dir = kernel_dir / "phase2"
        manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
        metadata = json.loads((kernel_dir / "metadata.json").read_text(encoding="utf-8"))
        phase2_meta = json.loads((phase2_dir / "metadata.phase2.json").read_text(encoding="utf-8"))
        kernels = manifest.get("kernels") or []
        kernel_entry = kernels[0] if kernels and isinstance(kernels[0], dict) else {}
        records.append(
            {
                "kernel_id": entry["kernel_id"],
                "kernel_dir": kernel_dir,
                "phase2_dir": phase2_dir,
                "display_name": kernel_entry.get("display_name"),
                "build_status": metadata.get("build_status"),
                "phase2_status": phase2_meta.get("phase2_status"),
                "failure_reason": phase2_meta.get("failure_reason"),
                "failure_detail": phase2_meta.get("failure_detail"),
                "failure_context": phase2_meta.get("failure_context"),
            }
        )
    return records


def find_phase2_kernel_results_by_display_name(run_dir: Path, display_name: str) -> list[dict[str, Any]]:
    matches = [record for record in collect_phase2_kernel_results(run_dir) if record.get("display_name") == display_name]
    if not matches:
        raise RuntimeError(f"kernel display_name not found in phase2 outputs: {display_name}")
    return matches
