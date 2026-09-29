"""Command-line entry point for third-party CUDA backend campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.third_party_fuzz.report import write_reports
from scripts.third_party_fuzz.runner import (
    BACKEND_SPECS,
    CAMPAIGN_ROOT,
    RUNS,
    collect_kernel_records,
    load_kernel_results,
    project_runs_for_campaign,
    run_build_stage,
    run_backend_matrix_stage,
    run_coverage_matrix_stage,
    run_fuzz_stage,
    run_phase2_stage,
    save_kernel_results,
    select_kernel_records,
)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _is_applicability_skip(cell: object) -> bool:
    return (
        isinstance(cell, dict)
        and cell.get("status") == "skipped"
        and cell.get("skip_kind") == "applicability"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run third-party CUDA backend validation stages."
    )
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument(
        "--project",
        action="append",
        choices=tuple(RUNS),
        dest="projects",
        help="project to run; repeat to select multiple (default: all)",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        dest="kernels",
        help="exact kernel ID or display name for matrix stages; repeat to select multiple",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase2 = subparsers.add_parser("phase2", help="rerun non-strict Phase2 for all resolved runs")
    phase2.add_argument("--timeout-seconds", type=int, default=900)

    subparsers.add_parser("collect", help="collect Phase2/support records into kernel_results.json")

    build = subparsers.add_parser("build", help="build fresh RAPID2 backends for every runnable kernel")
    build.add_argument("--cuda-path", default="/usr/local/cuda")
    build.add_argument("--cuda-arch", default="sm_86")
    build.add_argument("--timeout-seconds", type=int, default=900)

    backend_matrix = subparsers.add_parser(
        "backend-matrix",
        help="build selected origin/RAPID/RAPID2 backends for every runnable kernel",
    )
    backend_matrix.add_argument(
        "--backend",
        action="append",
        choices=tuple(BACKEND_SPECS),
        dest="backends",
        help="backend to build; repeat to select multiple (default: all)",
    )
    backend_matrix.add_argument("--cuda-path", default="/usr/local/cuda")
    backend_matrix.add_argument("--cuda-arch", default="sm_86")
    backend_matrix.add_argument("--timeout-seconds", type=int, default=900)

    coverage_matrix = subparsers.add_parser(
        "coverage-matrix",
        help="run completion-driven CFG and SIMT coverage for selected backend/VConfig cells",
    )
    coverage_matrix.add_argument(
        "--backend",
        action="append",
        choices=tuple(BACKEND_SPECS),
        dest="backends",
        help="backend to run; repeat to select multiple (default: all)",
    )
    coverage_matrix.add_argument(
        "--vconfig",
        action="append",
        choices=("off", "on"),
        dest="vconfigs",
        help="VConfig cell to run; repeat to select both (default: off and on)",
    )
    coverage_matrix.add_argument("--coverage-seconds", type=int, default=10)
    coverage_matrix.add_argument("--timeout-seconds", type=int, default=900)
    coverage_matrix.add_argument("--seed", type=int, default=1)
    coverage_matrix.add_argument("--gpu-device")
    coverage_matrix.add_argument("--fuzzer", type=Path)
    coverage_matrix.add_argument("--fuzzer-async", type=Path)

    fuzz = subparsers.add_parser("fuzz", help="run bounded fixed or mutation fuzz for runnable kernels")
    fuzz.add_argument("--mode", choices=("fixed", "mutation"), required=True)
    fuzz.add_argument("--fixed-runs", type=int, default=100)
    fuzz.add_argument("--mutation-runs", type=int, default=1000)
    fuzz.add_argument("--timeout-seconds", type=int, default=30)
    fuzz.add_argument("--fuzzer", type=Path)

    subparsers.add_parser("report", help="write reports from kernel_results.json")

    all_cmd = subparsers.add_parser("all", help="run phase2, collect, build, fixed fuzz, mutation fuzz, and report")
    all_cmd.add_argument("--cuda-path", default="/usr/local/cuda")
    all_cmd.add_argument("--cuda-arch", default="sm_86")
    all_cmd.add_argument("--phase2-timeout-seconds", type=int, default=900)
    all_cmd.add_argument("--build-timeout-seconds", type=int, default=900)
    all_cmd.add_argument("--fixed-runs", type=int, default=100)
    all_cmd.add_argument("--mutation-runs", type=int, default=1000)
    all_cmd.add_argument("--fuzz-timeout-seconds", type=int, default=30)
    all_cmd.add_argument("--fuzzer", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    campaign_dir: Path = args.campaign_dir
    available_runs = project_runs_for_campaign(campaign_dir)
    selected_projects = set(args.projects or available_runs)
    kernel_selectors = tuple(args.kernels) if args.kernels else None
    project_runs = {
        project: run
        for project, run in available_runs.items()
        if project in selected_projects
    }

    if args.command == "phase2":
        results = run_phase2_stage(runs=project_runs, timeout_seconds=args.timeout_seconds)
        _print_json({"phase2": results})
        return 0 if all(result.get("passed") for result in results) else 1

    if args.command == "collect":
        records = collect_kernel_records(runs=project_runs)
        save_kernel_results(campaign_dir, records)
        _print_json({"kernels": len(records), "path": str(campaign_dir / "kernel_results.json")})
        return 0

    if args.command == "build":
        records = run_build_stage(
            campaign_dir=campaign_dir,
            runs=project_runs,
            cuda_path=args.cuda_path,
            cuda_arch=args.cuda_arch,
            timeout_seconds=args.timeout_seconds,
        )
        failed = [
            record
            for record in records
            if record.get("support_decision") == "run" and record.get("backend", {}).get("status") != "passed"
        ]
        _print_json({"kernels": len(records), "backend_failures": len(failed)})
        return 0 if not failed else 1

    if args.command == "backend-matrix":
        backends = tuple(args.backends or BACKEND_SPECS)
        records = run_backend_matrix_stage(
            campaign_dir=campaign_dir,
            runs=project_runs,
            backends=backends,
            kernel_selectors=kernel_selectors,
            cuda_path=args.cuda_path,
            cuda_arch=args.cuda_arch,
            timeout_seconds=args.timeout_seconds,
        )
        evaluated_records = (
            select_kernel_records(records, kernel_selectors)
            if kernel_selectors is not None
            else records
        )
        failed_cells = [
            (record, backend)
            for record in evaluated_records
            if record.get("support_decision") == "run"
            for backend in backends
            if record.get("backend_results", {}).get(backend, {}).get("status")
            != "passed"
        ]
        _print_json(
            {
                "kernels": len(records),
                "backends": list(backends),
                "backend_failures": len(failed_cells),
            }
        )
        return 0 if not failed_cells else 1

    if args.command == "coverage-matrix":
        backends = tuple(args.backends or BACKEND_SPECS)
        vconfigs = tuple(args.vconfigs or ("off", "on"))
        records = run_coverage_matrix_stage(
            campaign_dir=campaign_dir,
            runs=project_runs,
            backends=backends,
            kernel_selectors=kernel_selectors,
            vconfigs=vconfigs,
            coverage_seconds=args.coverage_seconds,
            timeout_seconds=args.timeout_seconds,
            seed=args.seed,
            fuzzer=args.fuzzer,
            fuzzer_async=args.fuzzer_async,
            gpu_device=args.gpu_device,
        )
        evaluated_records = (
            select_kernel_records(records, kernel_selectors)
            if kernel_selectors is not None
            else records
        )
        cells = [
            record.get("coverage_results", {}).get(backend, {}).get(vconfig, {})
            for record in evaluated_records
            if record.get("support_decision") == "run"
            for backend in backends
            for vconfig in vconfigs
        ]
        applicability_skips = [cell for cell in cells if _is_applicability_skip(cell)]
        failures = [
            cell
            for cell in cells
            if cell.get("status") != "passed" and not _is_applicability_skip(cell)
        ]
        _print_json(
            {
                "kernels": len(records),
                "backends": list(backends),
                "vconfigs": list(vconfigs),
                "coverage_failures": len(failures),
                "applicability_skips": len(applicability_skips),
            }
        )
        return 0 if not failures else 1

    if args.command == "fuzz":
        runs = args.fixed_runs if args.mode == "fixed" else args.mutation_runs
        records = run_fuzz_stage(
            campaign_dir=campaign_dir,
            runs_profiles=project_runs,
            mode=args.mode,
            runs=runs,
            timeout_seconds=args.timeout_seconds,
            fuzzer=args.fuzzer,
        )
        failures = [
            record
            for record in records
            if record.get("support_decision") == "run"
            and record.get(args.mode, {}).get("status") == "failed"
        ]
        _print_json({"mode": args.mode, "runtime_failures": len(failures)})
        return 0 if not failures else 1

    if args.command == "report":
        records = load_kernel_results(campaign_dir, runs=project_runs)
        summary = write_reports(campaign_dir, records)
        _print_json(summary)
        return 0 if summary.get("fuzzer_normal") else 1

    if args.command == "all":
        phase2_results = run_phase2_stage(runs=project_runs, timeout_seconds=args.phase2_timeout_seconds)
        if not all(result.get("passed") for result in phase2_results):
            _print_json({"phase2": phase2_results})
            return 1
        run_build_stage(
            campaign_dir=campaign_dir,
            runs=project_runs,
            cuda_path=args.cuda_path,
            cuda_arch=args.cuda_arch,
            timeout_seconds=args.build_timeout_seconds,
        )
        run_fuzz_stage(
            campaign_dir=campaign_dir,
            runs_profiles=project_runs,
            mode="fixed",
            runs=args.fixed_runs,
            timeout_seconds=args.fuzz_timeout_seconds,
            fuzzer=args.fuzzer,
        )
        records = run_fuzz_stage(
            campaign_dir=campaign_dir,
            runs_profiles=project_runs,
            mode="mutation",
            runs=args.mutation_runs,
            timeout_seconds=args.fuzz_timeout_seconds,
            fuzzer=args.fuzzer,
        )
        summary = write_reports(campaign_dir, records)
        _print_json(summary)
        return 0 if summary.get("fuzzer_normal") else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
