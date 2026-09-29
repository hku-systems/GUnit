#!/usr/bin/env python3
"""Build the RQ2 workloads with the four coverage-study backends."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq1 import build as rq1_build  # noqa: E402
from benchmark.rq1.schema import (  # noqa: E402
    Workload as Rq1Workload,
    load_catalog as load_rq1_catalog,
    sha256_file,
)
from benchmark.rq2.schema import (  # noqa: E402
    Workload,
    load_catalog,
    verify_catalog,
)


RQ1_ROOT = REPO_ROOT / "benchmark" / "rq1"
RQ2_ROOT = Path(__file__).resolve().parent
RQ2_BACKENDS = ("cufuzz", "origin", "rapid", "rapid2")
RQ2_VCONFIG_UNSUPPORTED_REASONS = frozenset(
    {
        "vconfig_barrier_unsupported",
        "vconfig_inline_asm_unsupported",
    }
)


def _source_workload(workload: Workload) -> tuple[Rq1Workload, Path]:
    if workload.synthetic_provenance is not None:
        return (
            Rq1Workload(
                workload_id=workload.workload_id,
                label=workload.label,
                provenance_path=workload.synthetic_provenance,
            ),
            RQ2_ROOT,
        )
    by_id = {
        item.workload_id: item for item in load_rq1_catalog(RQ1_ROOT).workloads
    }
    try:
        return by_id[workload.workload_id], RQ1_ROOT
    except KeyError as error:
        raise RuntimeError(
            f"RQ2 workload has no RQ1 provenance contract: {workload.workload_id}"
        ) from error


def _rq1_build_contract(workload: Workload, out_root: Path):
    """Return the RQ1 build inputs for an RQ2 execution-envelope adaptation."""
    rq1_workload, source_root = _source_workload(workload)
    if workload.workload_id != "flashattention_device1xn":
        return rq1_workload, source_root, None

    adapter_root = out_root / "_rq2_execution_contracts"
    adapter_path = adapter_root / rq1_workload.provenance_path
    original_path = RQ1_ROOT / rq1_workload.provenance_path
    contract = json.loads(original_path.read_text(encoding="utf-8"))
    contract["execution"]["block"] = [256, 1, 1]
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rq1_workload, adapter_root, adapter_path


def create_build_plan(
    workload: Workload,
    *,
    out_root: Path,
    cuda_path: str,
    cuda_arch: str,
    build_profile: str = "release",
    instrument_selects: bool = False,
):
    source_workload, source_root = _source_workload(workload)
    return rq1_build.create_build_plan(
        source_workload,
        rq1_root=source_root,
        out_root=out_root,
        cuda_path=cuda_path,
        cuda_arch=cuda_arch,
        build_profile=build_profile,
        source_path_override=workload.kernel_path,
        constraints_path_override=workload.constraints_path,
        backend_names=RQ2_BACKENDS,
        instrument_selects=instrument_selects,
    )


def _prefixed_sha256(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def _applied_constraints_sha256(
    workload: Workload,
    report_path: Path,
    expected_sha256: str,
) -> str:
    constraint_report_path = (
        report_path.parent / "resolved" / "constraint_overrides.report.json"
    )
    constraint_report = json.loads(
        constraint_report_path.read_text(encoding="utf-8")
    )
    if not isinstance(constraint_report, dict):
        raise RuntimeError(
            f"RQ2 constraint report must be an object: {constraint_report_path}"
        )
    applied_sha256 = constraint_report.get("registry_sha256")
    if applied_sha256 != expected_sha256:
        raise RuntimeError(
            f"RQ2 applied constraint digest differs from pre-build input: "
            f"{applied_sha256!r} != {expected_sha256!r}"
        )
    current_sha256 = _prefixed_sha256(workload.constraints_path)
    if current_sha256 != expected_sha256:
        raise RuntimeError(
            f"RQ2 constraints changed during build: "
            f"{current_sha256!r} != {expected_sha256!r}"
        )
    return applied_sha256


def _rq2_report_metadata(
    workload: Workload,
    build_spec: dict,
    constraints_sha256: str,
) -> dict:
    requested = build_spec.get("vconfig_requested")
    effective = build_spec.get("vconfig_enabled")
    if requested is not True:
        raise RuntimeError("RQ2 VConfig must be requested")
    if not isinstance(effective, bool):
        raise RuntimeError("RQ2 VConfig effective state must be boolean")
    disabled_reason = build_spec.get("vconfig_disabled_reason")
    if effective and "vconfig_disabled_reason" in build_spec:
        raise RuntimeError("RQ2 VConfig is enabled but records a disabled reason")
    if not effective and disabled_reason not in RQ2_VCONFIG_UNSUPPORTED_REASONS:
        raise RuntimeError(
            "RQ2 VConfig disabled reason must be a recognized unsupported reason"
        )
    return {
        "constraints_path": str(workload.constraints_path),
        "constraints_sha256": constraints_sha256,
        "source_path": str(workload.kernel_path),
        "source_resolved_path": str(workload.kernel_path.resolve()),
        "vconfig_disabled_reason": disabled_reason,
        "vconfig_effective": effective,
        "vconfig_requested": requested,
    }


def _rq2_execution_contract(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernels = manifest.get("kernels") if isinstance(manifest, dict) else None
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise RuntimeError(
            f"RQ2 resolved manifest must contain exactly one kernel: {manifest_path}"
        )
    kernel = kernels[0]
    if not isinstance(kernel, dict):
        raise RuntimeError(
            f"RQ2 resolved kernel entry must be an object: {manifest_path}"
        )
    arguments = kernel.get("args")
    if not isinstance(arguments, list):
        raise RuntimeError(
            f"RQ2 resolved kernel args must be an array: {manifest_path}"
        )
    return {
        "entry": kernel["symbol_name"],
        "arguments": [
            {
                key: argument[key]
                for key in ("index", "name", "type", "kind", "domain")
            }
            for argument in arguments
        ],
        "constraints": kernel["constraints"],
        "launch_policy": kernel["launch_policy"],
    }


def build_workload(
    workload: Workload,
    *,
    out_root: Path,
    cuda_path: str,
    cuda_arch: str,
    build_profile: str = "release",
    instrument_selects: bool = False,
) -> Path:
    expected_constraints_sha256 = _prefixed_sha256(workload.constraints_path)
    rq1_workload, rq1_root, adapter_path = _rq1_build_contract(workload, out_root)
    report_path = rq1_build.build_workload(
        rq1_workload,
        out_root=out_root,
        cuda_path=cuda_path,
        cuda_arch=cuda_arch,
        rq1_root=rq1_root,
        build_profile=build_profile,
        source_path_override=workload.kernel_path,
        constraints_path_override=workload.constraints_path,
        backend_names=RQ2_BACKENDS,
        instrument_selects=instrument_selects,
    )
    applied_constraints_sha256 = _applied_constraints_sha256(
        workload,
        report_path,
        expected_constraints_sha256,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_provenance = json.loads(
        workload.source_provenance_path.read_text(encoding="utf-8")
    )
    rq1_execution = source_provenance.get("execution")
    if not isinstance(rq1_execution, dict):
        raise RuntimeError(
            f"RQ1 provenance execution contract must be an object: "
            f"{workload.source_provenance_path}"
        )
    report["source_provenance"] = str(workload.source_provenance_path.resolve())
    report["execution_contract"] = rq1_execution
    shared_phase2 = report.get("shared_phase2")
    if not isinstance(shared_phase2, dict):
        raise RuntimeError(f"RQ2 build report missing shared_phase2: {report_path}")
    raw_phase2_dir = shared_phase2.get("phase2_dir")
    if not isinstance(raw_phase2_dir, str) or not raw_phase2_dir:
        raise RuntimeError(f"RQ2 build report missing Phase2 directory: {report_path}")
    build_spec_path = Path(raw_phase2_dir) / "build_spec.json"
    build_spec = json.loads(build_spec_path.read_text(encoding="utf-8"))
    if not isinstance(build_spec, dict):
        raise RuntimeError(f"RQ2 Phase2 build spec must be an object: {build_spec_path}")
    rq2_metadata = _rq2_report_metadata(
        workload,
        build_spec,
        applied_constraints_sha256,
    )
    if adapter_path is not None:
        rq2_metadata["execution_contract"] = _rq2_execution_contract(
            Path(raw_phase2_dir).parent / "manifest.json"
        )
        adapted_provenance = json.loads(adapter_path.read_text(encoding="utf-8"))
        rq2_metadata["build_contract_adapter"] = {
            "path": str(adapter_path.resolve()),
            "sha256": _prefixed_sha256(adapter_path),
            "field": "execution.block",
            "rq1_value": rq1_execution["block"],
            "rq2_value": adapted_provenance["execution"]["block"],
        }
    report["rq2"] = rq2_metadata
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def write_build_summary(
    run_root: Path,
    reports: Sequence[Path],
    build_profile: str,
) -> Path:
    workloads = load_catalog(RQ2_ROOT).workloads
    report_paths = tuple(Path(path) for path in reports)
    if len(report_paths) != len(workloads):
        raise RuntimeError(
            f"RQ2 summary requires exactly {len(workloads)} build reports"
        )
    if len({path.resolve() for path in report_paths}) != len(report_paths):
        raise RuntimeError("RQ2 summary requires distinct build reports")
    for workload, report_path in zip(workloads, report_paths):
        if not report_path.is_file():
            raise RuntimeError(f"RQ2 summary build report does not exist: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise RuntimeError(f"RQ2 summary build report must be an object: {report_path}")
        if report.get("workload_id") != workload.workload_id:
            raise RuntimeError(
                f"RQ2 summary workload order or ID mismatch: "
                f"expected {workload.workload_id!r} in {report_path}"
            )
    summary_path = run_root / "build_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "build_profile": build_profile,
                "optimization": rq1_build.build_optimization_contract(
                    build_profile
                ).as_metadata(),
                "reports": [str(path) for path in report_paths],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=RQ2_ROOT / "build")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda")
    )
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH", "sm_86"))
    parser.add_argument("--build-profile", choices=("release",), default="release")
    parser.add_argument(
        "--instrument-selects",
        action="store_true",
        help="Instrument evaluated LLVM select instructions as pseudo-sites.",
    )
    args = parser.parse_args()

    errors = verify_catalog(RQ2_ROOT)
    if errors:
        raise RuntimeError("RQ2 source verification failed:\n" + "\n".join(errors))
    run_root = (args.out_root / args.run_id).resolve()
    if run_root.exists():
        raise RuntimeError(f"RQ2 run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    reports = tuple(
        build_workload(
            workload,
            out_root=run_root,
            cuda_path=args.cuda_path,
            cuda_arch=args.cuda_arch,
            build_profile=args.build_profile,
            instrument_selects=args.instrument_selects,
        )
        for workload in load_catalog(RQ2_ROOT).workloads
    )
    summary_path = write_build_summary(run_root, reports, args.build_profile)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
