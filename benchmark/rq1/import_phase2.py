#!/usr/bin/env python3
"""Import selected shared-corpus Phase2 artifacts and build the RQ1 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq1 import build as rq1_build  # noqa: E402
from benchmark.rq1.artifacts import (  # noqa: E402
    dump_seed,
    envelope_payload_size,
    hash_backend_report,
    hash_shared_phase2,
    verify_shared_phase2_unchanged,
)
from benchmark.workloads.schema import (  # noqa: E402
    Workload,
    load_catalog,
    load_suite,
    sha256_file,
    verify_catalog,
)


WORKLOAD_ROOT = REPO_ROOT / "benchmark" / "workloads"
RQ1_ROOT = Path(__file__).resolve().parent


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _single_kernel(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1 or not isinstance(kernels[0], dict):
        raise RuntimeError(f"imported manifest must contain exactly one kernel: {manifest_path}")
    return kernels[0]


def load_import_map(path: Path, selected: Sequence[str]) -> dict[str, Path]:
    raw = _read_json(path)
    requested = tuple(selected)
    missing = [workload_id for workload_id in requested if workload_id not in raw]
    if missing:
        raise RuntimeError(f"Phase2 import map is missing selected workloads: {missing}")
    result: dict[str, Path] = {}
    for workload_id in requested:
        value = raw[workload_id]
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{workload_id}: Phase2 import path must be a string")
        kernel_dir = Path(value).expanduser().resolve()
        if not (kernel_dir / "manifest.json").is_file() or not (kernel_dir / "phase2").is_dir():
            raise RuntimeError(f"{workload_id}: incomplete imported kernel directory {kernel_dir}")
        result[workload_id] = kernel_dir
    return result


def imported_kernel_destination(work_root: Path, source_kernel_dir: Path) -> Path:
    """Preserve the Phase2 kernel ID encoded by its canonical directory name."""

    return work_root / "resolved" / "kernels" / source_kernel_dir.name


def build_imported_workload(
    workload: Workload,
    *,
    source_kernel_dir: Path,
    out_root: Path,
    fuzzer: Path,
    cuda_path: str,
    cuda_arch: str,
    build_profile: str = "release",
) -> Path:
    if build_profile != "release":
        raise RuntimeError("RQ1 imported Phase2 builds require the release profile")
    work_root = out_root.resolve() / workload.workload_id
    if work_root.exists():
        raise RuntimeError(f"RQ1 imported workload directory already exists: {work_root}")
    imported_kernel_dir = imported_kernel_destination(work_root, source_kernel_dir)
    imported_kernel_dir.parent.mkdir(parents=True)
    shutil.copytree(source_kernel_dir.resolve(), imported_kernel_dir)
    manifest_path = imported_kernel_dir / "manifest.json"
    phase2_dir = imported_kernel_dir / "phase2"
    kernel_entry = _single_kernel(manifest_path)
    build_spec = _read_json(phase2_dir / "build_spec.json")
    imported_kernel_id = build_spec.get("kernel_id")
    if not isinstance(imported_kernel_id, str) or not imported_kernel_id:
        raise RuntimeError(f"imported Phase2 lacks kernel_id: {phase2_dir}")
    if workload.kernel_id is not None and imported_kernel_id != workload.kernel_id:
        raise RuntimeError(
            f"{workload.workload_id}: imported kernel ID {imported_kernel_id!r} "
            f"differs from shared corpus {workload.kernel_id!r}"
        )
    shared_phase2 = hash_shared_phase2(phase2_dir)
    expected_origin_launch = rq1_build._expected_backend_launch_config(
        kernel_entry=kernel_entry,
        backend_directory="origin",
        build_spec=build_spec,
    )

    env = os.environ.copy()
    backend_results: dict[str, dict[str, Any]] = {}
    for name, (directory, feedback) in rq1_build.EXPECTED_BACKENDS.items():
        out_dir = work_root / "backends" / name
        command = rq1_build._backend_command(
            directory=directory,
            feedback=feedback,
            phase2_dir=phase2_dir,
            out_dir=out_dir,
            cuda_path=cuda_path,
            cuda_arch=cuda_arch,
            build_profile=build_profile,
        )
        rq1_build._run(command, env=env)
        verify_shared_phase2_unchanged(phase2_dir, shared_phase2)
        backend_report_path = out_dir / "backend_build.json"
        backend_report = _read_json(backend_report_path)
        expected_launch = rq1_build._expected_backend_launch_config(
            kernel_entry=kernel_entry,
            backend_directory=directory,
            build_spec=build_spec,
        )
        rq1_build._validate_backend_report(
            report=backend_report,
            backend=rq1_build.BackendPlan(
                name=name,
                directory=directory,
                feedback=feedback,
                phase2_dir=phase2_dir,
                out_dir=out_dir,
                command=command,
            ),
            phase2_dir=phase2_dir.resolve(),
            kernel_id=imported_kernel_id,
            expected_launch=expected_launch,
            build_profile=build_profile,
        )
        backend_results[name] = {
            "builder_directory": directory,
            "feedback_instrumentation": feedback,
            "build_profile": build_profile,
            "phase2_dir": str(phase2_dir.resolve()),
            "kernel_id": imported_kernel_id,
            "launch_config": expected_launch,
            "artifacts": hash_backend_report(backend_report_path),
        }

    seed = dump_seed(fuzzer, manifest_path)
    payload_size = envelope_payload_size(seed)
    launch_policy = kernel_entry.get("launch_policy")
    if not isinstance(launch_policy, dict):
        raise RuntimeError(f"{workload.workload_id}: imported manifest lacks launch policy")
    logical_block = launch_policy.get("logical_block")
    if not isinstance(logical_block, list):
        logical_block = [expected_origin_launch["physical_block_max"], 1, 1]
    report = {
        "schema_version": 1,
        "workload_id": workload.workload_id,
        "build_profile": build_profile,
        "optimization": rq1_build.build_optimization_contract(build_profile).as_metadata(),
        "pipeline_commands": [],
        "source_provenance": str((WORKLOAD_ROOT / "catalog.json").resolve()),
        "source_workload_id": workload.workload_id,
        "imported_phase2": {
            "source_kernel_dir": str(source_kernel_dir.resolve()),
            "source_manifest_sha256": sha256_file(source_kernel_dir / "manifest.json"),
        },
        "shared_phase2": {
            "phase2_dir": str(phase2_dir.resolve()),
            "kernel_id": imported_kernel_id,
            "artifacts": shared_phase2,
        },
        "execution_contract": {
            "entry": kernel_entry.get("symbol_name"),
            "grid": expected_origin_launch["grid"],
            "block": logical_block,
            "dynamic_shared_bytes": expected_origin_launch["target_dynamic_shared_bytes"],
            "payload_size": payload_size,
            "canonical_input_sha256": hashlib.sha256(seed).hexdigest(),
            "vconfig": {
                "fixed": True,
                "grid": expected_origin_launch["grid"],
                "block": logical_block,
            },
        },
        "backends": backend_results,
    }
    report_path = work_root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-map", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=RQ1_ROOT / "build")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload", action="append")
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH", "sm_86"))
    parser.add_argument(
        "--fuzzer",
        type=Path,
        default=REPO_ROOT / "cuda-fuzzer" / "target" / "release" / "fuzzer",
    )
    args = parser.parse_args()
    errors = verify_catalog(WORKLOAD_ROOT)
    if errors:
        raise RuntimeError("shared workload verification failed:\n" + "\n".join(errors))
    catalog = load_catalog(WORKLOAD_ROOT)
    rq1_suite = load_suite(RQ1_ROOT / "suite.json", catalog)
    selected = tuple(args.workload or rq1_suite.workload_ids)
    unknown = sorted(set(selected) - set(rq1_suite.workload_ids))
    if unknown:
        raise RuntimeError(f"workloads are not in the RQ1 suite: {unknown}")
    by_id = catalog.by_id()
    imports = load_import_map(args.phase2_map, selected)
    run_root = (args.out_root / args.run_id).resolve()
    if run_root.exists():
        raise RuntimeError(f"RQ1 run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    reports = [
        build_imported_workload(
            by_id[workload_id],
            source_kernel_dir=imports[workload_id],
            out_root=run_root,
            fuzzer=args.fuzzer,
            cuda_path=args.cuda_path,
            cuda_arch=args.cuda_arch,
        )
        for workload_id in selected
    ]
    summary_path = run_root / "build_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "build_profile": "release",
                "optimization": rq1_build.build_optimization_contract("release").as_metadata(),
                "reports": [str(path) for path in reports],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
