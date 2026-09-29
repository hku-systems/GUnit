#!/usr/bin/env python3
"""Build all RQ1 workloads once through Phase1/Phase2 and seven artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq1.artifacts import (  # noqa: E402
    hash_backend_report,
    hash_shared_phase2,
    verify_shared_phase2_unchanged,
)
from benchmark.rq1.schema import (  # noqa: E402
    Workload,
    load_catalog,
    load_provenance,
    verify_catalog,
)


RQ1_ROOT = Path(__file__).resolve().parent
CUDA_KERNEL_DIR = REPO_ROOT / "cuda-kernel"
if str(CUDA_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_KERNEL_DIR))

from backend_build import build_optimization_contract  # noqa: E402
from launch_config import launch_config_from_kernel_entry  # noqa: E402

KERNEL_SMOKE_CLI = REPO_ROOT / "scripts" / "kernel-smoke" / "cli.py"
KERNEL_REWRITE_CLI = REPO_ROOT / "scripts" / "kernel-rewrite" / "cli.py"
WRAPPER_DIR = REPO_ROOT / "scripts" / "kernel-smoke"

EXPECTED_BACKENDS: Mapping[str, tuple[str, str]] = {
    "cufuzz": ("cufuzz", "disabled"),
    "origin-no-feedback": ("origin", "disabled"),
    "origin": ("origin", "enabled"),
    "rapid-no-feedback": ("rapid", "disabled"),
    "rapid": ("rapid", "enabled"),
    "rapid2-no-feedback": ("rapid2", "disabled"),
    "rapid2": ("rapid2", "enabled"),
}
CONTRACT_LAUNCH_KEYS = (
    "grid",
    "block_candidates",
    "physical_block_max",
    "target_dynamic_shared_bytes",
    "coverage_memory",
    "vconfig_reserved",
)
VCONFIG_EXTENSION_KEYS = (
    "logical_grid",
    "logical_block",
    "has_logical_vconfig_bounds",
    "vconfig_enabled",
    "vconfig_mutation",
    "vconfig_warp_aligned",
)
KNOWN_LAUNCH_KEYS = frozenset((*CONTRACT_LAUNCH_KEYS, *VCONFIG_EXTENSION_KEYS))


@dataclass(frozen=True)
class BackendPlan:
    name: str
    directory: str
    feedback: str
    phase2_dir: Path
    out_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class BuildPlan:
    workload: Workload
    build_profile: str
    work_root: Path
    raw_run_dir: Path
    resolved_run_dir: Path
    phase2_dir: Path
    capture_commands: tuple[tuple[str, ...], ...]
    phase1_commands: tuple[tuple[str, ...], ...]
    constraint_commands: tuple[tuple[str, ...], ...]
    phase2_commands: tuple[tuple[str, ...], ...]
    backends: tuple[BackendPlan, ...]


def _python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.is_file() else Path(sys.executable))


def _wrapped_compiler() -> Path:
    for name in ("clang++", "nvcc"):
        wrapper = WRAPPER_DIR / name
        if shutil.which(name) and wrapper.is_file():
            return wrapper
    raise RuntimeError("no wrapped CUDA compiler available (clang++/nvcc)")


def _backend_command(
    *,
    directory: str,
    feedback: str,
    phase2_dir: Path,
    out_dir: Path,
    cuda_path: str,
    cuda_arch: str,
    build_profile: str,
    instrument_selects: bool = False,
) -> tuple[str, ...]:
    command = [
        _python(),
        str(CUDA_KERNEL_DIR / directory / "build.py"),
        "--phase2-dir",
        str(phase2_dir),
        "--out-dir",
        str(out_dir),
        "--cuda-path",
        cuda_path,
        "--cuda-arch",
        cuda_arch,
        "--build-profile",
        build_profile,
    ]
    if directory != "cufuzz":
        command.extend(["--feedback-instrumentation", feedback])
        if feedback == "enabled" and instrument_selects:
            command.append("--instrument-selects")
    return tuple(command)


def create_build_plan(
    workload: Workload,
    *,
    rq1_root: Path,
    out_root: Path,
    cuda_path: str,
    cuda_arch: str,
    build_profile: str = "release",
    source_path_override: Path | None = None,
    constraints_path_override: Path | None = None,
    backend_names: Sequence[str] | None = None,
    instrument_selects: bool = False,
) -> BuildPlan:
    if build_profile != "release":
        raise RuntimeError(f"RQ1 requires release build profile, got: {build_profile}")
    optimization = build_optimization_contract(build_profile)
    rq1_root = rq1_root.resolve()
    out_root = out_root.resolve()
    provenance = load_provenance(rq1_root, workload)
    workload_source_root = (rq1_root / workload.provenance_path).resolve().parent
    work_root = out_root / workload.workload_id
    capture_dir = work_root / "capture"
    object_path = work_root / "capture-build" / "kernel.o"
    phase1_out = work_root / "phase1"
    raw_run_dir = phase1_out / "raw"
    resolved_run_dir = work_root / "resolved"
    logical_phase2_dir = resolved_run_dir / "selected-kernel" / "phase2"
    source_path = (
        workload_source_root / provenance.extraction.kernel_file
        if source_path_override is None
        else Path(source_path_override).absolute()
    )
    constraints_path = (
        workload_source_root / provenance.extraction.constraints_file
        if constraints_path_override is None
        else Path(constraints_path_override).absolute()
    )
    selected_backends = (
        tuple(EXPECTED_BACKENDS) if backend_names is None else tuple(backend_names)
    )
    if not selected_backends:
        raise RuntimeError("must select at least one backend")
    if len(selected_backends) != len(set(selected_backends)):
        raise RuntimeError("backend names must be unique")
    unknown_backends = tuple(
        name for name in selected_backends if name not in EXPECTED_BACKENDS
    )
    if unknown_backends:
        raise RuntimeError(f"unknown backend name(s): {', '.join(unknown_backends)}")

    capture_command = (
        str(_wrapped_compiler()),
        "-x",
        "cuda",
        "-c",
        str(source_path),
        "-o",
        str(object_path),
        f"--cuda-path={cuda_path}",
        f"--cuda-gpu-arch={cuda_arch}",
        *optimization.device_clang_flags,
        "-I",
        str(source_path.parent),
        "-I",
        str(CUDA_KERNEL_DIR),
        "-I",
        str(CUDA_KERNEL_DIR / "utils"),
    )
    phase1_command = (
        _python(),
        str(KERNEL_SMOKE_CLI),
        "run",
        "--capture-dir",
        str(capture_dir),
        "--out-root",
        str(phase1_out),
        "--run-id",
        "raw",
        "--target-lib",
        f"rq1_{workload.workload_id}",
        "--mode",
        "artifact",
        "--jobs",
        "1",
    )
    constraint_command = (
        _python(),
        "-m",
        "scripts.kernel_constraints.cli",
        "--run-dir",
        str(raw_run_dir),
        "--out-dir",
        str(resolved_run_dir),
        "--overrides",
        str(constraints_path),
    )
    phase2_command = (
        _python(),
        str(KERNEL_REWRITE_CLI),
        "--run-dir",
        str(resolved_run_dir),
    )
    backend_plans = tuple(
        BackendPlan(
            name=name,
            directory=directory,
            feedback=feedback,
            phase2_dir=logical_phase2_dir,
            out_dir=work_root / "backends" / name,
            command=_backend_command(
                directory=directory,
                feedback=feedback,
                phase2_dir=logical_phase2_dir,
                out_dir=work_root / "backends" / name,
                cuda_path=cuda_path,
                cuda_arch=cuda_arch,
                build_profile=build_profile,
                instrument_selects=instrument_selects,
            ),
        )
        for name in selected_backends
        for directory, feedback in (EXPECTED_BACKENDS[name],)
    )
    return BuildPlan(
        workload=workload,
        build_profile=build_profile,
        work_root=work_root,
        raw_run_dir=raw_run_dir,
        resolved_run_dir=resolved_run_dir,
        phase2_dir=logical_phase2_dir,
        capture_commands=(capture_command,),
        phase1_commands=(phase1_command,),
        constraint_commands=(constraint_command,),
        phase2_commands=(phase2_command,),
        backends=backend_plans,
    )


def _run(command: Sequence[str], *, env: Mapping[str, str]) -> None:
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=dict(env))


def _kernel_dir_from_index(run_dir: Path) -> tuple[str, Path]:
    index_path = run_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    kernels = index.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise RuntimeError(
            f"RQ1 workload must resolve to exactly one kernel, got {len(kernels or [])}: {index_path}"
        )
    entry = kernels[0]
    if not isinstance(entry, dict):
        raise RuntimeError(f"invalid kernel index entry: {index_path}")
    kernel_id = entry.get("kernel_id")
    raw_dir = entry.get("dir")
    if not isinstance(kernel_id, str) or not kernel_id:
        raise RuntimeError(f"kernel index entry missing kernel_id: {index_path}")
    if not isinstance(raw_dir, str) or not raw_dir:
        raise RuntimeError(f"kernel index entry missing dir: {index_path}")
    kernel_dir = Path(raw_dir)
    if not kernel_dir.is_absolute():
        kernel_dir = run_dir / kernel_dir
    return kernel_id, kernel_dir.resolve()


def _validate_backend_report(
    *,
    report: dict,
    backend: BackendPlan,
    phase2_dir: Path,
    kernel_id: str,
    expected_launch: dict,
    build_profile: str,
) -> None:
    reported_phase2 = report.get("phase2_dir")
    if not isinstance(reported_phase2, str) or Path(reported_phase2).resolve() != phase2_dir:
        raise RuntimeError(f"{backend.name}: backend report phase2_dir drift")
    if report.get("kernel_id") != kernel_id:
        raise RuntimeError(f"{backend.name}: backend report kernel_id drift")
    if report.get("feedback_instrumentation") != backend.feedback:
        raise RuntimeError(f"{backend.name}: backend report feedback mode drift")
    reported_launch = report.get("launch_config")
    if not isinstance(reported_launch, dict):
        raise RuntimeError(f"{backend.name}: backend report launch config drift")
    unknown_launch_keys = sorted(set(reported_launch) - KNOWN_LAUNCH_KEYS)
    if unknown_launch_keys:
        raise RuntimeError(
            f"{backend.name}: unknown launch config key(s): "
            + ", ".join(unknown_launch_keys)
        )
    for key in CONTRACT_LAUNCH_KEYS:
        if key not in reported_launch or reported_launch[key] != expected_launch.get(key):
            raise RuntimeError(f"{backend.name}: backend report launch config drift")
    for key in VCONFIG_EXTENSION_KEYS:
        if key in reported_launch and reported_launch[key] != expected_launch.get(key):
            raise RuntimeError(f"{backend.name}: backend report launch config drift")
    if report.get("build_profile") != build_profile:
        raise RuntimeError(f"{backend.name}: backend report build profile drift")
    optimization = report.get("optimization")
    if not isinstance(optimization, dict) or optimization.get("profile") != build_profile:
        raise RuntimeError(f"{backend.name}: backend optimization metadata drift")
    if optimization.get("device_clang_flags") != ["-O3", "-DNDEBUG"]:
        raise RuntimeError(f"{backend.name}: backend device optimization drift")
    if optimization.get("host_cxx_flags") != ["-O3", "-DNDEBUG"]:
        raise RuntimeError(f"{backend.name}: backend host optimization drift")
    if optimization.get("llvm_opt_pipeline") != "default<O3>":
        raise RuntimeError(f"{backend.name}: backend LLVM optimization drift")
    if optimization.get("llc_flags") != ["-O3"]:
        raise RuntimeError(f"{backend.name}: backend llc optimization drift")
    commands = report.get("commands")
    if not isinstance(commands, list) or not commands:
        raise RuntimeError(f"{backend.name}: backend command log missing")


def _expected_backend_launch_config(
    *,
    kernel_entry: dict,
    backend_directory: str,
    build_spec: dict,
) -> dict:
    if backend_directory == "cufuzz":
        return launch_config_from_kernel_entry(
            kernel_entry,
            backend_name=backend_directory,
        )
    return launch_config_from_kernel_entry(
        kernel_entry,
        backend_name=backend_directory,
        require_warp_aligned_block=bool(build_spec.get("vconfig_warp_aligned")),
        vconfig_enabled=bool(build_spec.get("vconfig_enabled")),
    )


def build_workload(
    workload: Workload,
    *,
    out_root: Path,
    cuda_path: str,
    cuda_arch: str,
    rq1_root: Path = RQ1_ROOT,
    build_profile: str = "release",
    source_path_override: Path | None = None,
    constraints_path_override: Path | None = None,
    backend_names: Sequence[str] | None = None,
    instrument_selects: bool = False,
) -> Path:
    plan = create_build_plan(
        workload,
        rq1_root=rq1_root,
        out_root=out_root,
        cuda_path=cuda_path,
        cuda_arch=cuda_arch,
        build_profile=build_profile,
        source_path_override=source_path_override,
        constraints_path_override=constraints_path_override,
        backend_names=backend_names,
        instrument_selects=instrument_selects,
    )
    if plan.work_root.exists():
        raise RuntimeError(f"RQ1 workload build directory already exists: {plan.work_root}")
    (plan.work_root / "capture").mkdir(parents=True)
    (plan.work_root / "capture-build").mkdir(parents=True)
    env = os.environ.copy()
    env["RAPID_CAPTURE_DIR"] = str(plan.work_root / "capture")

    for command in (
        *plan.capture_commands,
        *plan.phase1_commands,
        *plan.constraint_commands,
        *plan.phase2_commands,
    ):
        _run(command, env=env)

    kernel_id, kernel_dir = _kernel_dir_from_index(plan.resolved_run_dir)
    phase2_dir = kernel_dir / "phase2"
    if not phase2_dir.is_dir():
        raise RuntimeError(f"Phase2 did not build selected kernel: {phase2_dir}")
    shared_phase2 = hash_shared_phase2(phase2_dir)
    manifest = json.loads((kernel_dir / "manifest.json").read_text(encoding="utf-8"))
    kernel_entry = manifest["kernels"][0]
    build_spec = json.loads((phase2_dir / "build_spec.json").read_text(encoding="utf-8"))
    expected_launch = _expected_backend_launch_config(
        kernel_entry=kernel_entry,
        backend_directory="origin",
        build_spec=build_spec,
    )
    provenance = load_provenance(rq1_root, workload)
    if expected_launch["grid"] != list(provenance.execution.grid):
        raise RuntimeError(f"{workload.workload_id}: manifest grid differs from provenance")
    if expected_launch["physical_block_max"] != provenance.execution.block[0]:
        raise RuntimeError(f"{workload.workload_id}: manifest block differs from provenance")
    if (
        expected_launch["target_dynamic_shared_bytes"]
        != provenance.execution.dynamic_shared_bytes
    ):
        raise RuntimeError(f"{workload.workload_id}: manifest shared memory differs from provenance")

    backend_results: dict[str, dict] = {}
    for template in plan.backends:
        backend = replace(
            template,
            phase2_dir=phase2_dir,
            command=_backend_command(
                directory=template.directory,
                feedback=template.feedback,
                phase2_dir=phase2_dir,
                out_dir=template.out_dir,
                cuda_path=cuda_path,
                cuda_arch=cuda_arch,
                build_profile=build_profile,
                instrument_selects=instrument_selects,
            ),
        )
        _run(backend.command, env=env)
        verify_shared_phase2_unchanged(phase2_dir, shared_phase2)
        report_path = backend.out_dir / "backend_build.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_backend_launch = _expected_backend_launch_config(
            kernel_entry=kernel_entry,
            backend_directory=backend.directory,
            build_spec=build_spec,
        )
        _validate_backend_report(
            report=report,
            backend=backend,
            phase2_dir=phase2_dir.resolve(),
            kernel_id=kernel_id,
            expected_launch=expected_backend_launch,
            build_profile=build_profile,
        )
        backend_results[backend.name] = {
            "builder_directory": backend.directory,
            "feedback_instrumentation": backend.feedback,
            "build_profile": build_profile,
            "phase2_dir": str(phase2_dir.resolve()),
            "kernel_id": kernel_id,
            "launch_config": expected_backend_launch,
            "artifacts": hash_backend_report(report_path),
        }

    build_report = {
        "schema_version": 1,
        "workload_id": workload.workload_id,
        "build_profile": build_profile,
        "optimization": build_optimization_contract(build_profile).as_metadata(),
        "pipeline_commands": [
            list(command)
            for command in (
                *plan.capture_commands,
                *plan.phase1_commands,
                *plan.constraint_commands,
                *plan.phase2_commands,
            )
        ],
        "source_provenance": str((rq1_root / workload.provenance_path).resolve()),
        "shared_phase2": {
            "phase2_dir": str(phase2_dir.resolve()),
            "kernel_id": kernel_id,
            "artifacts": shared_phase2,
        },
        "execution_contract": {
            "entry": provenance.execution.entry,
            "grid": list(provenance.execution.grid),
            "block": list(provenance.execution.block),
            "dynamic_shared_bytes": provenance.execution.dynamic_shared_bytes,
            "payload_size": provenance.execution.payload_size,
            "vconfig": provenance.execution.vconfig,
        },
        "backends": backend_results,
    }
    report_path = plan.work_root / "build_report.json"
    report_path.write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=RQ1_ROOT / "build")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH", "sm_86"))
    parser.add_argument("--build-profile", choices=("release",), default="release")
    parser.add_argument("--workload", action="append", choices=tuple(load_catalog(RQ1_ROOT).workloads[i].workload_id for i in range(len(load_catalog(RQ1_ROOT).workloads))))
    args = parser.parse_args()

    errors = verify_catalog(RQ1_ROOT)
    if errors:
        raise RuntimeError("RQ1 source verification failed:\n" + "\n".join(errors))
    run_root = (args.out_root / args.run_id).resolve()
    if run_root.exists():
        raise RuntimeError(f"RQ1 run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    selected = set(args.workload or [])
    reports: list[str] = []
    for workload in load_catalog(RQ1_ROOT).workloads:
        if selected and workload.workload_id not in selected:
            continue
        report = build_workload(
            workload,
            out_root=run_root,
            cuda_path=args.cuda_path,
            cuda_arch=args.cuda_arch,
            build_profile=args.build_profile,
        )
        reports.append(str(report))
    summary_path = run_root / "build_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "build_profile": args.build_profile,
                "optimization": build_optimization_contract(args.build_profile).as_metadata(),
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
