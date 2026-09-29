#!/usr/bin/env python3
"""Build timing-enabled RQ4 artifacts from shared Phase2 inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark.rq4.campaign import _load_reports


REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_KERNEL = REPO_ROOT / "cuda-kernel"
TIMING_BACKENDS = ("origin-no-feedback", "origin", "rapid", "rapid2")


def timing_build_command(
    *,
    backend: str,
    python: Path,
    phase2_dir: Path,
    out_dir: Path,
    cuda_path: str,
    cuda_arch: str,
) -> list[str]:
    if backend not in TIMING_BACKENDS:
        raise ValueError(f"unsupported timing backend: {backend}")
    builder = "origin" if backend.startswith("origin") else backend
    feedback_mode = "disabled" if backend == "origin-no-feedback" else "enabled"
    return [
        str(python),
        str(CUDA_KERNEL / builder / "build.py"),
        "--phase2-dir",
        str(phase2_dir),
        "--out-dir",
        str(out_dir),
        "--cuda-path",
        cuda_path,
        "--cuda-arch",
        cuda_arch,
        "--build-profile",
        "release",
        "--feedback-instrumentation",
        feedback_mode,
        "--enable-kernel-timing",
    ]


def build(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = set(args.workload or ())
    entries: list[dict[str, Any]] = []
    reports = _load_reports(args.build_root)
    known = {report["workload_id"] for _, report in reports}
    missing = selected - known
    if missing:
        raise RuntimeError(f"selected workload missing from build roots: {sorted(missing)}")

    for report_path, report in reports:
        workload_id = report["workload_id"]
        if selected and workload_id not in selected:
            continue
        phase2_dir = Path(report["shared_phase2"]["phase2_dir"]).resolve()
        timing_backends: dict[str, Any] = {}
        for backend in TIMING_BACKENDS:
            out_dir = output / workload_id / backend
            command = timing_build_command(
                backend=backend,
                python=Path(sys.executable),
                phase2_dir=phase2_dir,
                out_dir=out_dir,
                cuda_path=args.cuda_path,
                cuda_arch=args.cuda_arch,
            )
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            backend_report = json.loads(
                (out_dir / "backend_build.json").read_text(encoding="utf-8")
            )
            if backend_report.get("kernel_timing_enabled") is not True:
                raise RuntimeError(f"{backend} timing build did not enable timing")
            timing_backends[backend] = {
                "command": command,
                "shared_library": backend_report["shared_lib"],
                "backend_build": str(out_dir / "backend_build.json"),
            }
        entries.append(
            {
                "workload_id": workload_id,
                "source_build_report": str(report_path),
                "phase2_dir": str(phase2_dir),
                "manifest": str(phase2_dir.parent / "manifest.json"),
                "backends": timing_backends,
            }
        )

    summary = output / "build_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "rq4-persistent-kernel-device-timing",
                "source_build_roots": [str(path.resolve()) for path in args.build_root],
                "workloads": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, action="append", required=True)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-path", default="/usr/local/cuda")
    parser.add_argument("--cuda-arch", default="sm_86")
    return parser


def main() -> None:
    print(build(_parser().parse_args()))


if __name__ == "__main__":
    main()
