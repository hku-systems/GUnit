"""Campaign helpers for real third-party CUDA backend validation."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from benchmark.rq2.coverage_delta import validate_delta_records
from benchmark.rq2.verify_results import load_raw_telemetry
from scripts.third_party_fuzz.support import classify_kernel, load_support_registry, validate_support_coverage


REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_KERNEL_DIR = REPO_ROOT / "cuda-kernel"
if str(CUDA_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_KERNEL_DIR))

from launch_config import launch_config_from_kernel_entry, single_kernel_entry_from_manifest  # noqa: E402

CAMPAIGN_ROOT = Path("build/e2e/third-party-fuzz/20260719-final")
RQ2_WAVE3_CAMPAIGN_ROOT = Path("build/e2e/third-party-fuzz/20260802-rq2-wave3")
RESULTS_FILE = "kernel_results.json"
BACKEND_FRESHNESS_SCHEMA_VERSION = 1
BACKEND_BUILD_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".h",
    ".hpp",
    ".py",
}


@dataclass(frozen=True)
class BackendSpec:
    """Build and loader contract for one supported CUDA backend."""

    name: str
    build_script: str
    library_name: str
    fuzzer_binary: str
    required_exports: tuple[str, ...]


_TASK_BACKEND_EXPORTS = (
    "libafl_cov_map",
    "libafl_submit_with_id",
    "libafl_poll_results",
    "libafl_release_tasks",
    "libafl_set_target_timeout_ms",
    "libafl_wait",
    "libafl_stop",
)

BACKEND_SPECS: dict[str, BackendSpec] = {
    "origin": BackendSpec(
        name="origin",
        build_script="cuda-kernel/origin/build.py",
        library_name="libphase2_origin_target.so",
        fuzzer_binary="fuzzer",
        required_exports=(
            "libafl_cov_map",
            "libafl_simt_memcov_bits",
            "libafl_target",
            "libafl_get_last_run_status",
        ),
    ),
    "rapid": BackendSpec(
        name="rapid",
        build_script="cuda-kernel/rapid/build.py",
        library_name="libphase2_rapid_target.so",
        fuzzer_binary="fuzzer",
        required_exports=(
            *_TASK_BACKEND_EXPORTS,
            "libafl_get_ordered_queue_counts",
        ),
    ),
    "rapid2": BackendSpec(
        name="rapid2",
        build_script="cuda-kernel/rapid2/build.py",
        library_name="librapid2_target.so",
        fuzzer_binary="fuzzer_async",
        required_exports=(
            *_TASK_BACKEND_EXPORTS,
            "libafl_get_queue_counts",
            "libafl_wait_for_completion",
        ),
    ),
}

# Preserve the public constant used by existing RAPID2-only callers while the
# campaign record schema is migrated to per-backend results.
EXPECTED_RAPID2_EXPORTS = BACKEND_SPECS["rapid2"].required_exports


def _backend_spec(backend: str) -> BackendSpec:
    try:
        return BACKEND_SPECS[backend]
    except KeyError as error:
        raise ValueError(f"unsupported backend: {backend}") from error


@dataclass(frozen=True)
class ProjectRun:
    """Filesystem layout for one resolved third-party Phase 1 run."""

    project_id: str
    filesystem_project: str
    resolved_run: Path
    support_registry: Path
    constraint_registry: Path | None = None


RUNS: dict[str, ProjectRun] = {
    "darknet": ProjectRun(
        project_id="darknet",
        filesystem_project="darknet",
        resolved_run=Path(
            "build/e2e/third-party-fuzz/20260809-darknet-crop/"
            "darknet/phase1/resolved/darknet"
        ),
        support_registry=Path("third_party/fuzz/support/darknet.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/darknet.json"),
    ),
    "caffe": ProjectRun(
        project_id="caffe",
        filesystem_project="caffe",
        resolved_run=Path(
            "build/e2e/third-party-fuzz/20260809-caffe-cll-backward/"
            "caffe/phase1/resolved/caffe"
        ),
        support_registry=Path("third_party/fuzz/support/caffe.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/caffe.json"),
    ),
    "apex": ProjectRun(
        project_id="apex",
        filesystem_project="apex",
        resolved_run=RQ2_WAVE3_CAMPAIGN_ROOT / "apex/phase1/resolved/apex",
        support_registry=Path("third_party/fuzz/support/apex.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/apex.json"),
    ),
    "deepspeed": ProjectRun(
        project_id="deepspeed",
        filesystem_project="deepspeed",
        resolved_run=RQ2_WAVE3_CAMPAIGN_ROOT / "deepspeed/phase1/resolved/deepspeed",
        support_registry=Path("third_party/fuzz/support/deepspeed.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/deepspeed.json"),
    ),
    "flash_attention": ProjectRun(
        project_id="flash_attention",
        filesystem_project="flash_attention",
        resolved_run=RQ2_WAVE3_CAMPAIGN_ROOT / "flash_attention/phase1/resolved/flash_attention",
        support_registry=Path("third_party/fuzz/support/flash_attention.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/flash_attention.json"),
    ),
    "heongpu": ProjectRun(
        project_id="heongpu",
        filesystem_project="heongpu",
        resolved_run=RQ2_WAVE3_CAMPAIGN_ROOT / "heongpu/phase1/resolved/heongpu",
        support_registry=Path("third_party/fuzz/support/heongpu.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/heongpu.json"),
    ),
    "gpurir": ProjectRun(
        project_id="gpurir",
        filesystem_project="gpurir",
        resolved_run=CAMPAIGN_ROOT / "gpurir/phase1/resolved/gpurir",
        support_registry=Path("third_party/fuzz/support/gpurir.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/gpurir.json"),
    ),
    "phantom_fhe": ProjectRun(
        project_id="phantom_fhe",
        filesystem_project="phantom-fhe",
        resolved_run=CAMPAIGN_ROOT / "phantom-fhe/phase1/resolved/phantom_fhe",
        support_registry=Path("third_party/fuzz/support/phantom_fhe.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/phantom_fhe.json"),
    ),
    "cudasift": ProjectRun(
        project_id="cudasift",
        filesystem_project="cudasift",
        resolved_run=CAMPAIGN_ROOT / "cudasift/phase1/resolved/cudasift",
        support_registry=Path("third_party/fuzz/support/cudasift.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/cudasift.json"),
    ),
    "lietorch": ProjectRun(
        project_id="lietorch",
        filesystem_project="lietorch",
        resolved_run=CAMPAIGN_ROOT / "lietorch/phase1/resolved/lietorch",
        support_registry=Path("third_party/fuzz/support/lietorch.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/lietorch.json"),
    ),
    "llama_cpp": ProjectRun(
        project_id="llama_cpp",
        filesystem_project="llama_cpp",
        resolved_run=CAMPAIGN_ROOT / "llama_cpp/phase1/resolved/llama_cpp",
        support_registry=Path("third_party/fuzz/support/llama_cpp.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/llama_cpp.json"),
    ),
    "kaldi": ProjectRun(
        project_id="kaldi",
        filesystem_project="kaldi",
        resolved_run=CAMPAIGN_ROOT / "kaldi/phase1/resolved/kaldi",
        support_registry=Path("third_party/fuzz/support/kaldi.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/kaldi.json"),
    ),
    "cuda_samples": ProjectRun(
        project_id="cuda_samples",
        filesystem_project="cuda_samples",
        resolved_run=CAMPAIGN_ROOT / "cuda_samples/phase1/resolved/cuda_samples",
        support_registry=Path("third_party/fuzz/support/cuda_samples.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/cuda_samples.json"),
    ),
    "gpujpeg": ProjectRun(
        project_id="gpujpeg",
        filesystem_project="gpujpeg",
        resolved_run=CAMPAIGN_ROOT / "gpujpeg/phase1/resolved/gpujpeg",
        support_registry=Path("third_party/fuzz/support/gpujpeg.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/gpujpeg.json"),
    ),
    "tensorrt_clip": ProjectRun(
        project_id="tensorrt_clip",
        filesystem_project="tensorrt_clip",
        resolved_run=CAMPAIGN_ROOT / "tensorrt_clip/phase1/resolved/existing_capture",
        support_registry=Path("third_party/fuzz/support/tensorrt_clip.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/tensorrt_clip.json"),
    ),
    "cutlass_basic": ProjectRun(
        project_id="cutlass_basic",
        filesystem_project="cutlass_basic",
        resolved_run=CAMPAIGN_ROOT / "cutlass_basic/phase1/resolved/cutlass_basic",
        support_registry=Path("third_party/fuzz/support/cutlass_basic.json"),
        constraint_registry=Path("third_party/fuzz/kernel_constraints/cutlass_basic.json"),
    ),
}


def project_runs_for_campaign(campaign_dir: Path) -> dict[str, ProjectRun]:
    """Return project profiles rebased under a campaign directory.

    The default RUNS constant documents the canonical 20260719 layout, but a
    restarted campaign must read Phase1/Phase2 artifacts from its own
    `phase1/resolved/...` directories.  Keeping support registry paths stable
    while rebasing only the resolved run prevents a fresh campaign from
    accidentally collecting or rebuilding against stale artifacts.
    """

    campaign_dir = Path(campaign_dir)
    return {
        key: ProjectRun(
            project_id=run.project_id,
            filesystem_project=run.filesystem_project,
            resolved_run=campaign_dir / run.filesystem_project / "phase1/resolved" / run.resolved_run.name,
            support_registry=run.support_registry,
            constraint_registry=run.constraint_registry,
        )
        for key, run in RUNS.items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _decode_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _timeout_output_text(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    return _decode_subprocess_output(stdout) + _decode_subprocess_output(stderr)


def build_backend_command(
    *,
    backend: str = "rapid2",
    phase2_dir: Path,
    out_dir: Path,
    cuda_path: str,
    cuda_arch: str,
) -> list[str]:
    """Return a fresh backend build command for one Phase2 directory."""

    spec = _backend_spec(backend)

    return [
        sys.executable,
        str(REPO_ROOT / spec.build_script),
        "--phase2-dir",
        str(phase2_dir),
        "--out-dir",
        str(out_dir),
        "--cuda-path",
        cuda_path,
        "--cuda-arch",
        cuda_arch,
    ]


def build_fuzzer_command(
    *,
    fuzzer: Path,
    backend: Path,
    manifest: Path,
    mode: Literal["fixed", "mutation"],
    runs: int,
) -> list[str]:
    """Return the fuzzer command for fixed/no-mutate or mutation mode."""

    if runs <= 0:
        raise ValueError(f"runs must be positive, got {runs}")
    command = [
        str(fuzzer.resolve()),
        str(backend.resolve()),
        "--manifest",
        str(manifest.resolve()),
    ]
    if mode == "fixed":
        command.append("--no-mutate")
    elif mode != "mutation":
        raise ValueError(f"unsupported fuzz mode: {mode}")
    command.extend(["--runs", str(runs)])
    return command


def build_coverage_command(
    *,
    backend: str,
    fuzzer: Path,
    library: Path,
    manifest: Path,
    coverage_seconds: int,
    coverage_log: Path,
    vconfig: Literal["on", "off"] | str,
    window_size: int | None,
) -> list[str]:
    """Return one completion-driven CFG plus SIMT MemCov campaign command."""

    spec = _backend_spec(backend)
    if coverage_seconds <= 0:
        raise ValueError("coverage_seconds must be positive")
    if vconfig not in ("on", "off"):
        raise ValueError(f"vconfig must be 'on' or 'off', got {vconfig!r}")
    if backend in ("rapid", "rapid2") and (
        window_size is None or window_size <= 0
    ):
        raise ValueError(f"{backend} coverage requires a positive window_size")

    command = [
        str(fuzzer.resolve()),
        str(library.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--coverage-seconds",
        str(coverage_seconds),
        "--coverage-log",
        str(coverage_log.resolve()),
        "--vconfig",
        vconfig,
    ]
    if backend in ("rapid", "rapid2"):
        command.extend(["--window-size", str(window_size)])
    if Path(command[0]).name != spec.fuzzer_binary:
        raise ValueError(
            f"{backend} coverage requires {spec.fuzzer_binary}, got {command[0]}"
        )
    return command


def evaluate_coverage_log(path: Path) -> dict[str, Any]:
    """Validate one completion-driven coverage trace and summarize its terminal state."""

    records = load_raw_telemetry(path)
    validate_delta_records(records)
    final = records[-1]
    if final["executions_completed"] <= 0:
        raise ValueError("coverage campaign completed with zero executions")
    return {
        "status": "passed",
        "samples": len(records),
        "executions_submitted": final["executions_submitted"],
        "executions_completed": final["executions_completed"],
        "cfg_sites": final["cfg_sites"],
        "memory_features": final["memory_features"],
        "thread_activity_features": final["thread_activity_features"],
        "feedback_features_total": final["feedback_features_total"],
        "final_timestamp_s": final["timestamp_s"],
    }


def kernel_campaign_dir(campaign_dir: Path, record: dict[str, Any]) -> Path:
    return campaign_dir / str(record["filesystem_project"]) / "kernels" / str(record["kernel_id"])


def backend_dir_for(
    campaign_dir: Path,
    record: dict[str, Any],
    *,
    backend: str | None = None,
) -> Path:
    if backend is not None:
        _backend_spec(backend)
        return kernel_campaign_dir(campaign_dir, record) / "backends" / backend
    return kernel_campaign_dir(campaign_dir, record) / "backend"


def backend_library_for(
    campaign_dir: Path,
    record: dict[str, Any],
    *,
    backend: str | None = None,
) -> Path:
    if backend is None:
        return backend_dir_for(campaign_dir, record) / "librapid2_target.so"
    spec = _backend_spec(backend)
    return backend_dir_for(campaign_dir, record, backend=backend) / spec.library_name


def load_kernel_results(
    campaign_dir: Path,
    *,
    runs: dict[str, ProjectRun] | None = None,
) -> list[dict[str, Any]]:
    path = campaign_dir / RESULTS_FILE
    if not path.exists():
        return collect_kernel_records(runs=runs)
    data = _read_json(path)
    kernels = data.get("kernels")
    if not isinstance(kernels, list):
        raise ValueError(f"kernel_results.json kernels must be a list: {path}")
    return [dict(record) for record in kernels if isinstance(record, dict)]


def save_kernel_results(campaign_dir: Path, records: list[dict[str, Any]]) -> None:
    from scripts.third_party_fuzz.report import write_reports

    write_reports(campaign_dir, records)


def select_kernel_records(
    records: list[dict[str, Any]], selectors: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Select records by exact kernel ID or display name in campaign order."""

    if not selectors or len(set(selectors)) != len(selectors):
        raise ValueError("kernel selectors must be nonempty and unique")
    requested = set(selectors)
    matched: set[str] = set()
    selected: list[dict[str, Any]] = []
    for record in records:
        identities = {record.get("kernel_id"), record.get("display_name")}
        record_matches = requested.intersection(
            identity for identity in identities if isinstance(identity, str)
        )
        if record_matches:
            matched.update(record_matches)
            selected.append(record)
    unknown = sorted(requested - matched)
    if unknown:
        raise ValueError(f"unknown kernel selector(s): {', '.join(unknown)}")
    return selected


def _index_by_kernel_id(run_dir: Path) -> dict[str, dict[str, Any]]:
    index = _read_json(run_dir / "index.json")
    kernels = index.get("kernels")
    if not isinstance(kernels, list):
        raise ValueError(f"Phase 1 index kernels must be a list: {run_dir / 'index.json'}")
    by_kernel_id: dict[str, dict[str, Any]] = {}
    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        kernel_id = kernel.get("kernel_id")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError(f"Phase 1 index entry missing kernel_id: {run_dir / 'index.json'}")
        if kernel_id in by_kernel_id:
            raise ValueError(f"duplicate kernel_id in Phase 1 index: {kernel_id}")
        by_kernel_id[kernel_id] = kernel
    return by_kernel_id


def _phase2_results(run_dir: Path) -> list[dict[str, Any]]:
    rewrite_summary = _read_json(run_dir / "rewrite_summary.json")
    results = rewrite_summary.get("results")
    if not isinstance(results, list):
        raise ValueError(f"rewrite summary results must be a list: {run_dir / 'rewrite_summary.json'}")
    return [result for result in results if isinstance(result, dict)]


def iter_phase2_built(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield Phase2-built kernels using the authoritative Phase1 kernel_id records."""

    by_kernel_id = _index_by_kernel_id(run_dir)
    for result in _phase2_results(run_dir):
        status = result.get("status")
        if status != "built":
            continue
        kernel_id = result.get("kernel_id")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("Phase2 built entry missing kernel_id")
        if kernel_id not in by_kernel_id:
            raise ValueError(f"Phase2 built entry references unknown kernel_id: {kernel_id}")
        index_record = by_kernel_id[kernel_id]
        kernel_dir = run_dir / str(index_record.get("dir") or f"kernels/{kernel_id}")
        phase2_dir = Path(str(result.get("phase2_dir"))) if result.get("phase2_dir") else kernel_dir / "phase2"
        yield {
            **result,
            "kernel_id": kernel_id,
            "symbol_name": index_record.get("symbol_name"),
            "display_name": index_record.get("display_name"),
            "kernel_dir": str(kernel_dir),
            "manifest": str(kernel_dir / "manifest.json"),
            "phase2_dir": str(phase2_dir),
        }


def _rewrite_summary_for_support(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "rewrite_summary.json")


def validate_resolved_constraint_registry(run_dir: Path, registry_path: Path | None) -> None:
    """Ensure a resolved Phase1 run was generated from the current constraints."""

    if registry_path is None:
        return
    report_path = run_dir / "constraint_overrides.report.json"
    if not report_path.exists():
        raise ValueError(f"resolved run missing constraint overrides report: {report_path}")
    if not registry_path.exists():
        raise ValueError(f"constraint registry missing: {registry_path}")
    report = _read_json(report_path)
    expected_hash = _sha256(registry_path)
    actual_hash = report.get("registry_sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            f"constraint overrides for {run_dir} are stale: "
            f"{actual_hash!r} != current {expected_hash!r}"
        )


def _vconfig_metadata_for_phase2(phase2_dir: Path) -> dict[str, Any]:
    build_spec_path = phase2_dir / "build_spec.json"
    if not build_spec_path.is_file():
        raise ValueError(f"built Phase2 kernel missing build_spec.json: {phase2_dir}")
    build_spec = _read_json(build_spec_path)
    metadata: dict[str, Any] = {}
    for field in ("vconfig_requested", "vconfig_enabled", "vconfig_warp_aligned"):
        value = build_spec.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{build_spec_path} missing boolean {field}")
        metadata[field] = value
    disabled_reason = build_spec.get("vconfig_disabled_reason")
    if disabled_reason is not None and not isinstance(disabled_reason, str):
        raise ValueError(
            f"{build_spec_path} vconfig_disabled_reason must be a string when present"
        )
    metadata["vconfig_disabled_reason"] = disabled_reason
    return metadata


def _display_name_for_kernel(
    index_record: dict[str, Any], *, kernel_dir: Path, symbol_name: str
) -> str:
    display_name = index_record.get("display_name")
    if isinstance(display_name, str) and display_name:
        return display_name
    manifest_path = kernel_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    kernel_entry = single_kernel_entry_from_manifest(
        manifest,
        manifest_path=manifest_path,
        backend_name="third-party campaign",
    )
    if kernel_entry.get("symbol_name") != symbol_name:
        raise ValueError(f"manifest symbol does not match Phase1 index: {manifest_path}")
    display_name = kernel_entry.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        raise ValueError(f"manifest kernel missing display_name: {manifest_path}")
    return display_name


def collect_kernel_records(
    *,
    runs: dict[str, ProjectRun] | None = None,
) -> list[dict[str, Any]]:
    """Collect per-kernel campaign records from resolved Phase1/Phase2 artifacts."""

    selected_runs = runs or RUNS
    records: list[dict[str, Any]] = []
    for project_id, run in selected_runs.items():
        run_dir = run.resolved_run
        validate_resolved_constraint_registry(run_dir, run.constraint_registry)
        by_kernel_id = _index_by_kernel_id(run_dir)
        rewrite_summary = _rewrite_summary_for_support(run_dir)
        decisions = load_support_registry(run.support_registry)
        validate_support_coverage(run_dir, rewrite_summary, decisions)
        for result in _phase2_results(run_dir):
            kernel_id = result.get("kernel_id")
            if not isinstance(kernel_id, str) or kernel_id not in by_kernel_id:
                raise ValueError(f"Phase2 entry references unknown kernel_id: {kernel_id}")
            index_record = by_kernel_id[kernel_id]
            symbol_name = index_record.get("symbol_name")
            if not isinstance(symbol_name, str) or not symbol_name:
                raise ValueError(f"Phase1 index entry missing symbol_name: {kernel_id}")
            kernel_dir = run_dir / str(index_record.get("dir") or f"kernels/{kernel_id}")
            display_name = _display_name_for_kernel(
                index_record,
                kernel_dir=kernel_dir,
                symbol_name=symbol_name,
            )
            phase2_status = str(result.get("status") or "unknown")
            record: dict[str, Any] = {
                "project": project_id,
                "filesystem_project": run.filesystem_project,
                "kernel_id": kernel_id,
                "symbol_name": symbol_name,
                "display_name": display_name,
                "kernel_dir": str(kernel_dir),
                "manifest": str(kernel_dir / "manifest.json"),
                "phase2_status": phase2_status,
                "phase2_dir": str(Path(str(result.get("phase2_dir"))) if result.get("phase2_dir") else kernel_dir / "phase2"),
                "support_decision": None,
                "skip_reason": None,
                "skip_detail": None,
                "support_evidence": [],
                "backend": {"status": "not_attempted"},
                "fixed": {"status": "not_attempted"},
                "mutation": {"status": "not_attempted"},
            }
            if phase2_status == "built":
                record.update(
                    _vconfig_metadata_for_phase2(Path(str(record["phase2_dir"])))
                )
                decision = classify_kernel(symbol_name, phase2_status, decisions, kernel_id=kernel_id)
                record["support_decision"] = decision.decision
                record["support_evidence"] = list(decision.evidence)
                if decision.decision == "skip":
                    record["skip_reason"] = decision.reason_code
                    record["skip_detail"] = decision.detail
            else:
                failure_reason = result.get("failure_reason")
                if not isinstance(failure_reason, str) or not failure_reason:
                    raise ValueError(f"rewrite failed entry missing failure_reason: {kernel_id}")
                failure_detail = result.get("failure_detail")
                if failure_detail is not None and not isinstance(failure_detail, str):
                    raise ValueError(f"rewrite failed entry failure_detail must be a string when present: {kernel_id}")
                record["support_decision"] = "skip"
                record["skip_reason"] = failure_reason
                record["skip_detail"] = failure_detail
            records.append(record)
    return records


_FINAL_STATS_RE = re.compile(
    r"Final statistics:\s*"
    r"(?:\n|\r\n)\s*Corpus size:\s*(?P<corpus_size>\d+)\s*"
    r"(?:\n|\r\n)\s*Crashes found:\s*(?P<crashes_found>\d+)\s*"
    r"(?:\n|\r\n)\s*Total executions:\s*(?P<total_executions>\d+)",
    re.MULTILINE,
)
_ARG_PACK_RE = re.compile(r"Arg-pack stats:\s*(?P<body>[^\n\r]+)")
_ARG_PACK_COUNTER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(\d+)\b")
_OBSERVED_COUNTER_RE = re.compile(r"\b(objectives|executions|submitted|evaluated):\s*(\d+)\b")


def _observed_progress_stats(output: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    for name, raw_value in _OBSERVED_COUNTER_RE.findall(output):
        key = f"observed_{name}"
        value = int(raw_value)
        stats[key] = max(stats.get(key, 0), value)
    return stats


def _has_cuda_or_objective_failure(output: str, stats: dict[str, int]) -> bool:
    if stats.get("observed_objectives", 0) > 0:
        return True
    failure_tokens = (
        "CUDA Error:",
        "CUDA_ERROR_",
        "Error code: 700",
        "Error code: 716",
        "RAPID2 NON-RECOVERABLE COMPLETION",
        "objective discovered",
    )
    return any(token in output for token in failure_tokens)


def parse_fuzzer_stats(output: str) -> dict[str, int]:
    """Parse bounded fuzzer output and require both final and arg-pack counters."""

    final_match = _FINAL_STATS_RE.search(output)
    if not final_match:
        raise ValueError("missing Final statistics block")
    arg_pack_match = _ARG_PACK_RE.search(output)
    if not arg_pack_match:
        raise ValueError("missing Arg-pack stats block")

    stats = {name: int(value) for name, value in final_match.groupdict().items()}
    for name, value in _ARG_PACK_COUNTER_RE.findall(arg_pack_match.group("body")):
        stats[name] = int(value)
    return stats


def evaluate_fixed_fuzz_output(output: str, *, expected_runs: int) -> dict[str, Any]:
    """Validate fixed/no-mutate fuzz output against the bounded-run gate."""

    try:
        stats = parse_fuzzer_stats(output)
    except ValueError as exc:
        return {"passed": False, "failure_reason": str(exc), "stats": {}}
    failures: list[str] = []
    if stats.get("total_executions", 0) < expected_runs:
        failures.append(
            f"executions below expected fixed runs: {stats.get('total_executions', 0)} < {expected_runs}"
        )
    if stats.get("mutation_calls", 0) != 0:
        failures.append(f"mutation calls in fixed mode: {stats.get('mutation_calls', 0)}")
    if stats.get("normalize_calls", 0) <= 0:
        failures.append("normalize calls did not increase")
    if stats.get("seed_generation_count", 0) <= 0:
        failures.append("seed generation did not run")
    return {
        "passed": not failures,
        "failure_reason": "; ".join(failures) if failures else None,
        "stats": stats,
    }


def verify_backend_exports(
    shared_lib: Path, *, backend: str = "rapid2"
) -> tuple[bool, list[str], str]:
    required_exports = _backend_spec(backend).required_exports
    if not shared_lib.exists():
        return False, list(required_exports), f"missing backend library: {shared_lib}"
    try:
        proc = subprocess.run(
            ["nm", "-D", str(shared_lib)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, list(required_exports), f"nm failed: {exc}"
    output = proc.stdout or ""
    missing = [name for name in required_exports if name not in output]
    return not missing and proc.returncode == 0, missing, output


def _completed_stage(
    *,
    status: str,
    command: list[str] | None = None,
    log: Path | None = None,
    passed: bool | None = None,
    failure_reason: str | None = None,
    stats: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"status": status}
    if passed is not None:
        record["passed"] = passed
    if command is not None:
        record["command"] = command
    if log is not None:
        record["log"] = str(log)
    if failure_reason:
        record["failure_reason"] = failure_reason
    if stats is not None:
        record["stats"] = stats
    if extra:
        record.update(extra)
    return record


def run_backend_build(
    record: dict[str, Any],
    *,
    backend: str | None = None,
    campaign_dir: Path,
    cuda_path: str,
    cuda_arch: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if record.get("phase2_status") != "built":
        return _completed_stage(status="skipped", passed=True, failure_reason="phase2_not_built")
    backend_name = backend or "rapid2"
    spec = _backend_spec(backend_name)
    out_dir = backend_dir_for(campaign_dir, record, backend=backend) if backend else backend_dir_for(campaign_dir, record)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "build.log"
    command = build_backend_command(
        backend=backend_name,
        phase2_dir=Path(str(record["phase2_dir"])),
        out_dir=out_dir,
        cuda_path=cuda_path,
        cuda_arch=cuda_arch,
    )
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        log.write_text(proc.stdout or "", encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        log.write_text(_timeout_output_text(exc.stdout, exc.stderr), encoding="utf-8")
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=log,
            failure_reason=f"backend build timed out after {timeout_seconds}s",
        )
    except OSError as exc:
        log.write_text(str(exc), encoding="utf-8")
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=log,
            failure_reason=f"backend build failed to start: {exc}",
        )

    if proc.returncode != 0:
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=log,
            failure_reason=f"backend build exited with {proc.returncode}",
        )
    shared_lib = out_dir / spec.library_name
    exports_ok, missing, nm_output = verify_backend_exports(
        shared_lib, backend=backend_name
    )
    (out_dir / "exports.nm").write_text(nm_output, encoding="utf-8", errors="replace")
    if not exports_ok:
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=log,
            failure_reason=f"backend missing RAPID2 exports: {missing}",
            extra={"missing_exports": missing},
        )
    if not _record_backend_freshness(
        record,
        backend=backend_name,
        out_dir=out_dir,
        cuda_path=cuda_path,
        cuda_arch=cuda_arch,
    ):
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=log,
            failure_reason="failed to record backend build fingerprint",
        )
    return _completed_stage(
        status="passed",
        passed=True,
        command=command,
        log=log,
        extra={"shared_lib": str(shared_lib), "backend": backend_name},
    )


def _rewrite_summary_for_record(record: dict[str, Any]) -> Path | None:
    """Return the project-level rewrite summary that makes a backend stale."""

    phase2_dir_raw = record.get("phase2_dir")
    if not isinstance(phase2_dir_raw, str) or not phase2_dir_raw:
        return None
    phase2_dir = Path(phase2_dir_raw)
    if len(phase2_dir.parents) < 3:
        return None
    return phase2_dir.parents[2] / "rewrite_summary.json"


def _expected_launch_config_for_record(
    record: dict[str, Any], manifest_path: Path, *, backend: str = "rapid2"
) -> dict[str, Any]:
    _backend_spec(backend)
    manifest = _read_json(manifest_path)
    kernel_entry = single_kernel_entry_from_manifest(
        manifest,
        manifest_path=manifest_path,
        backend_name=backend,
    )
    phase2_dir = Path(str(record["phase2_dir"]))
    build_spec = _read_json(phase2_dir / "build_spec.json")
    return launch_config_from_kernel_entry(
        kernel_entry,
        backend_name=backend,
        require_warp_aligned_block=bool(build_spec.get("vconfig_warp_aligned")),
        vconfig_enabled=bool(build_spec.get("vconfig_enabled")),
    )


def _backend_source_inputs(*, backend: str = "rapid2") -> list[Path]:
    spec = _backend_spec(backend)
    inputs = [
        REPO_ROOT / "cuda-kernel/backend_build.py",
        REPO_ROOT / "cuda-kernel/launch_config.py",
        REPO_ROOT / "scripts/kernel-rewrite/common.py",
        REPO_ROOT / "scripts/kernel-rewrite/contracts/kernel.py",
    ]
    for root in (
        REPO_ROOT / "cuda-kernel" / spec.name,
        REPO_ROOT / "cuda-kernel/utils",
        REPO_ROOT / "tools/rapid-feedback-instrument",
        REPO_ROOT / "tools/rapid-vconfig-instrument",
    ):
        if not root.is_dir():
            continue
        inputs.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.suffix in BACKEND_BUILD_SOURCE_SUFFIXES
                or path.name == "CMakeLists.txt"
            )
        )
    return sorted(set(inputs), key=lambda path: str(path))


def _backend_tool_identities() -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    tool_candidates = {
        "clang": ("clang++-22", "clang++"),
        "llvm_link": ("llvm-link-22", "llvm-link"),
        "llc": ("llc-22", "llc"),
        "cxx": ("clang++", "g++"),
    }
    for role, candidates in tool_candidates.items():
        resolved = None
        for name in candidates:
            resolved = shutil.which(name)
            if resolved is not None:
                break
        if resolved is None:
            identities[role] = {"path": None}
            continue
        path = Path(resolved).resolve()
        stat = path.stat()
        identities[role] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return identities


def _backend_dynamic_inputs(
    record: dict[str, Any], build_record: dict[str, Any]
) -> list[Path]:
    paths: list[Path] = []
    rewrite_summary = _rewrite_summary_for_record(record)
    if rewrite_summary is not None:
        paths.append(rewrite_summary)
    for raw in (
        record.get("manifest"),
        build_record.get("device_bc"),
        build_record.get("invoke_header"),
        build_record.get("target_layout_header"),
    ):
        if isinstance(raw, str) and raw:
            paths.append(Path(raw))
    phase2_dir_raw = build_record.get("phase2_dir") or record.get("phase2_dir")
    if isinstance(phase2_dir_raw, str) and phase2_dir_raw:
        phase2_dir = Path(phase2_dir_raw)
        paths.append(phase2_dir / "build_spec.json")
        paths.append(phase2_dir / "kernel.device.bc")
    type_shim_dirs = build_record.get("type_shim_include_dirs")
    if isinstance(type_shim_dirs, list):
        for raw_dir in type_shim_dirs:
            if not isinstance(raw_dir, str) or not raw_dir:
                continue
            directory = Path(raw_dir)
            if directory.is_dir():
                paths.extend(
                    path
                    for path in directory.rglob("*")
                    if path.is_file()
                    and path.suffix in BACKEND_BUILD_SOURCE_SUFFIXES
                )
    return sorted(set(paths), key=lambda path: str(path))


def _fingerprint_path(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": _sha256(resolved)}


def _backend_freshness_record(
    record: dict[str, Any],
    build_record: dict[str, Any],
    *,
    backend: str = "rapid2",
    cuda_path: str,
    cuda_arch: str,
) -> dict[str, Any]:
    import json

    inputs = _backend_source_inputs(backend=backend) + _backend_dynamic_inputs(
        record, build_record
    )
    if not inputs or any(not path.is_file() for path in inputs):
        missing = [str(path) for path in inputs if not path.is_file()]
        raise FileNotFoundError(f"missing backend fingerprint inputs: {missing}")
    files = [
        _fingerprint_path(path)
        for path in sorted(set(inputs), key=lambda path: str(path))
    ]
    payload = {
        "schema_version": BACKEND_FRESHNESS_SCHEMA_VERSION,
        "backend": backend,
        "build_options": {
            "cuda_arch": cuda_arch,
            "cuda_path": str(Path(cuda_path).resolve()),
            "feedback_instrumentation": build_record.get(
                "feedback_instrumentation", "enabled"
            ),
        },
        "files": files,
        "tools": _backend_tool_identities(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": BACKEND_FRESHNESS_SCHEMA_VERSION,
        "backend": backend,
        "sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "build_options": payload["build_options"],
        "inputs": [entry["path"] for entry in files],
        "tools": payload["tools"],
    }


def _record_backend_freshness(
    record: dict[str, Any],
    *,
    backend: str = "rapid2",
    out_dir: Path,
    cuda_path: str,
    cuda_arch: str,
) -> bool:
    spec = _backend_spec(backend)
    build_metadata = out_dir / "backend_build.json"
    shared_lib = out_dir / spec.library_name
    try:
        build_record = _read_json(build_metadata)
        freshness = _backend_freshness_record(
            record,
            build_record,
            backend=backend,
            cuda_path=cuda_path,
            cuda_arch=cuda_arch,
        )
        freshness["artifact_sha256"] = _sha256(shared_lib)
        build_record["campaign_freshness"] = freshness
        _write_json(build_metadata, build_record)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def backend_stage_if_fresh(
    record: dict[str, Any],
    *,
    backend: str | None = None,
    campaign_dir: Path,
    cuda_path: str = "/usr/local/cuda",
    cuda_arch: str = "sm_86",
) -> dict[str, Any] | None:
    """Return a passed backend stage when an existing backend is current.

    Freshness is content-addressed across generated Phase2 artifacts, the selected
    runtime and feedback instrumenter sources, build options, and tool identities.
    Existing metadata without this fingerprint is stale by construction.
    """

    backend_name = backend or "rapid2"
    spec = _backend_spec(backend_name)
    out_dir = (
        backend_dir_for(campaign_dir, record, backend=backend_name)
        if backend is not None
        else backend_dir_for(campaign_dir, record)
    )
    build_metadata = out_dir / "backend_build.json"
    shared_lib = out_dir / spec.library_name
    rewrite_summary = _rewrite_summary_for_record(record)
    manifest_raw = record.get("manifest")
    if not isinstance(manifest_raw, str) or not manifest_raw:
        return None
    manifest_path = Path(manifest_raw)
    if (
        not build_metadata.exists()
        or not shared_lib.exists()
        or rewrite_summary is None
        or not rewrite_summary.exists()
        or not manifest_path.exists()
    ):
        return None
    try:
        build_record = _read_json(build_metadata)
        expected_launch_config = _expected_launch_config_for_record(
            record, manifest_path, backend=backend_name
        )
        recorded_freshness = build_record.get("campaign_freshness")
        if not isinstance(recorded_freshness, dict):
            return None
        current_freshness = _backend_freshness_record(
            record,
            build_record,
            backend=backend_name,
            cuda_path=cuda_path,
            cuda_arch=cuda_arch,
        )
        current_artifact_sha256 = _sha256(shared_lib)
    except (OSError, RuntimeError, ValueError):
        return None
    if (
        recorded_freshness.get("schema_version")
        != BACKEND_FRESHNESS_SCHEMA_VERSION
        or recorded_freshness.get("backend") != backend_name
        or recorded_freshness.get("sha256") != current_freshness["sha256"]
        or recorded_freshness.get("artifact_sha256")
        != current_artifact_sha256
    ):
        return None
    if build_record.get("launch_config") != expected_launch_config:
        return None
    exports_ok, missing, nm_output = verify_backend_exports(
        shared_lib, backend=backend_name
    )
    (out_dir / "exports.nm").write_text(nm_output, encoding="utf-8", errors="replace")
    if not exports_ok:
        return None
    return _completed_stage(
        status="passed",
        passed=True,
        log=out_dir / "build.log",
        extra={
            "shared_lib": str(shared_lib),
            "backend": backend_name,
            "reused_fresh_backend": True,
            "freshness_fingerprint": current_freshness["sha256"],
            "freshness_basis": current_freshness["inputs"],
        },
    )


def _fuzzer_async_build_inputs() -> list[Path]:
    fuzzer_root = REPO_ROOT / "cuda-fuzzer"
    inputs = [fuzzer_root / "Cargo.toml", fuzzer_root / "Cargo.lock"]
    inputs.extend((fuzzer_root / "src").rglob("*.rs"))
    return [path for path in inputs if path.exists()]


def _needs_fuzzer_rebuild(target: Path) -> bool:
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    return any(path.stat().st_mtime > target_mtime for path in _fuzzer_async_build_inputs())


def ensure_fuzzer_async(*, release: bool = True) -> Path:
    target = REPO_ROOT / "cuda-fuzzer/target" / ("release" if release else "debug") / "fuzzer_async"
    if not _needs_fuzzer_rebuild(target):
        return target
    command = ["cargo", "build", "--manifest-path", "cuda-fuzzer/Cargo.toml"]
    if release:
        command.append("--release")
    env = os.environ.copy()
    env.setdefault("RUSTFLAGS", "-A function-casts-as-integer -A unstable-name-collisions")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return target


def run_fixed_fuzz(
    record: dict[str, Any],
    *,
    campaign_dir: Path,
    fuzzer: Path,
    runs: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    if record.get("support_decision") != "run":
        return _completed_stage(status="skipped", passed=True, failure_reason=str(record.get("skip_reason") or "not_runnable"))
    if record.get("backend", {}).get("status") != "passed":
        return _completed_stage(status="failed", passed=False, failure_reason="backend_not_passed")
    workdir = kernel_campaign_dir(campaign_dir, record) / "fixed"
    workdir.mkdir(parents=True, exist_ok=True)
    log = workdir / "fixed.log"
    command = build_fuzzer_command(
        fuzzer=fuzzer,
        backend=backend_library_for(campaign_dir, record),
        manifest=Path(str(record["manifest"])),
        mode="fixed",
        runs=runs,
    )
    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        output = proc.stdout or ""
        log.write_text(output, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        output = _timeout_output_text(exc.stdout, exc.stderr)
        log.write_text(output, encoding="utf-8")
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=log,
            failure_reason=f"fixed fuzz timed out after {timeout_seconds}s",
        )

    evaluation = evaluate_fixed_fuzz_output(output, expected_runs=runs)
    if proc.returncode != 0:
        evaluation["passed"] = False
        evaluation["failure_reason"] = "; ".join(
            reason for reason in [evaluation.get("failure_reason"), f"fuzzer exited with {proc.returncode}"] if reason
        )
    classification = None
    if not evaluation["passed"]:
        classification = classify_fuzz_failure(
            evaluation.get("failure_reason") if isinstance(evaluation.get("failure_reason"), str) else None,
            output,
        )
    return _completed_stage(
        status="passed" if evaluation["passed"] else "failed",
        passed=bool(evaluation["passed"]),
        command=command,
        log=log,
        failure_reason=evaluation.get("failure_reason"),
        stats=evaluation.get("stats") or {},
        extra={"failure_kind": classification["kind"]} if classification else None,
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_mutation_fuzz(
    record: dict[str, Any],
    *,
    campaign_dir: Path,
    fuzzer: Path,
    runs: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    if record.get("support_decision") != "run":
        return _completed_stage(status="skipped", passed=True, failure_reason=str(record.get("skip_reason") or "not_runnable"))
    if record.get("backend", {}).get("status") != "passed":
        return _completed_stage(status="failed", passed=False, failure_reason="backend_not_passed")
    if record.get("fixed", {}).get("status") != "passed":
        return _completed_stage(
            status="skipped",
            passed=True,
            failure_reason="fixed_not_passed",
            extra={"skip_reason": str(record.get("runtime_skip_reason") or "fixed_not_passed")},
        )

    workdir = kernel_campaign_dir(campaign_dir, record) / "mutation"
    workdir.mkdir(parents=True, exist_ok=True)
    broker_log = workdir / "broker.log"
    client_log = workdir / "client.log"
    command = build_fuzzer_command(
        fuzzer=fuzzer,
        backend=backend_library_for(campaign_dir, record),
        manifest=Path(str(record["manifest"])),
        mode="mutation",
        runs=runs,
    )
    env = os.environ.copy()
    env.setdefault("RUST_LOG", "info")
    broker: subprocess.Popen[str] | None = None
    client: subprocess.Popen[str] | None = None
    client_joined = False
    timed_out = False
    try:
        with broker_log.open("w", encoding="utf-8") as broker_output, client_log.open("w", encoding="utf-8") as client_output:
            broker = subprocess.Popen(
                command,
                cwd=workdir,
                env=env,
                stdout=broker_output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            time.sleep(1.0)
            if broker.poll() is not None:
                return _completed_stage(
                    status="failed",
                    passed=False,
                    command=command,
                    log=broker_log,
                    failure_reason=f"broker exited before client startup with {broker.returncode}",
                    extra={"broker_log": str(broker_log), "client_log": str(client_log), "client_joined": False},
                )
            client = subprocess.Popen(
                command,
                cwd=workdir,
                env=env,
                stdout=client_output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            client_joined = True
            try:
                client.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        if client is not None:
            _terminate_process_group(client)
        if broker is not None:
            _terminate_process_group(broker)

    output = _read_text(broker_log) + "\n" + _read_text(client_log)
    evaluation = evaluate_mutation_fuzz_output(output)
    decision = mutation_stage_passes(
        evaluation,
        client_joined=client_joined,
        timed_out=timed_out,
        client_returncode=client.returncode if client is not None else None,
    )
    passed = bool(decision["passed"])
    classification = None
    if not passed:
        classification = classify_fuzz_failure(
            decision.get("failure_reason") if isinstance(decision.get("failure_reason"), str) else None,
            output,
        )
    return _completed_stage(
        status="passed" if passed else "failed",
        passed=passed,
        command=command,
        log=client_log,
        failure_reason=decision.get("failure_reason") if isinstance(decision.get("failure_reason"), str) else None,
        stats=evaluation.get("stats") or {},
        extra={
            "broker_log": str(broker_log),
            "client_log": str(client_log),
            "client_joined": client_joined,
            "timed_out": timed_out,
            **({"failure_kind": classification["kind"]} if classification else {}),
        },
    )


def run_build_stage(
    *,
    campaign_dir: Path = CAMPAIGN_ROOT,
    runs: dict[str, ProjectRun] | None = None,
    cuda_path: str,
    cuda_arch: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    selected_runs = runs or project_runs_for_campaign(campaign_dir)
    records = collect_kernel_records(runs=selected_runs)
    for record in records:
        if record.get("phase2_status") != "built":
            record["backend"] = _completed_stage(
                status="skipped",
                passed=True,
                failure_reason=str(record.get("skip_reason") or "phase2_not_built"),
            )
        elif record.get("support_decision") != "run":
            record["backend"] = _completed_stage(
                status="skipped",
                passed=True,
                failure_reason=str(record.get("skip_reason") or "not_runnable"),
            )
    for record in records:
        if record.get("phase2_status") != "built" or record.get("support_decision") != "run":
            save_kernel_results(campaign_dir, records)
            continue
        record["backend"] = backend_stage_if_fresh(
            record,
            campaign_dir=campaign_dir,
            cuda_path=cuda_path,
            cuda_arch=cuda_arch,
        ) or run_backend_build(
            record,
            campaign_dir=campaign_dir,
            cuda_path=cuda_path,
            cuda_arch=cuda_arch,
            timeout_seconds=timeout_seconds,
        )
        save_kernel_results(campaign_dir, records)
    return records


def run_backend_matrix_stage(
    *,
    campaign_dir: Path = CAMPAIGN_ROOT,
    runs: dict[str, ProjectRun] | None = None,
    backends: tuple[str, ...] = ("origin", "rapid", "rapid2"),
    kernel_selectors: tuple[str, ...] | None = None,
    cuda_path: str,
    cuda_arch: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Build backend-specific artifacts for each structurally runnable kernel."""

    if not backends or len(set(backends)) != len(backends):
        raise ValueError("backends must be a nonempty tuple without duplicates")
    for backend in backends:
        _backend_spec(backend)

    selected_runs = runs or project_runs_for_campaign(campaign_dir)
    records = collect_kernel_records(runs=selected_runs)
    selected_records = (
        select_kernel_records(records, kernel_selectors)
        if kernel_selectors is not None
        else records
    )
    for record in selected_records:
        if (
            record.get("phase2_status") != "built"
            or record.get("support_decision") != "run"
        ):
            save_kernel_results(campaign_dir, records)
            continue
        backend_results = record.setdefault("backend_results", {})
        if not isinstance(backend_results, dict):
            raise ValueError("backend_results must be an object")
        for backend in backends:
            backend_results[backend] = backend_stage_if_fresh(
                record,
                backend=backend,
                campaign_dir=campaign_dir,
                cuda_path=cuda_path,
                cuda_arch=cuda_arch,
            ) or run_backend_build(
                record,
                backend=backend,
                campaign_dir=campaign_dir,
                cuda_path=cuda_path,
                cuda_arch=cuda_arch,
                timeout_seconds=timeout_seconds,
            )
            save_kernel_results(campaign_dir, records)
    return records


def run_coverage_cell(
    record: dict[str, Any],
    *,
    backend: str,
    vconfig: Literal["on", "off"] | str,
    campaign_dir: Path,
    fuzzer: Path,
    fuzzer_async: Path,
    coverage_seconds: int,
    timeout_seconds: int,
    window_size: int | None,
    seed: int,
    gpu_device: str | None = None,
) -> dict[str, Any]:
    """Run one backend/VConfig coverage cell or record why it is inapplicable."""

    _backend_spec(backend)
    if vconfig not in ("on", "off"):
        raise ValueError(f"vconfig must be 'on' or 'off', got {vconfig!r}")
    if vconfig == "on" and record.get("vconfig_enabled") is not True:
        return _completed_stage(
            status="skipped",
            passed=True,
            failure_reason=str(
                record.get("vconfig_disabled_reason") or "vconfig_not_enabled"
            ),
            extra={
                "backend": backend,
                "vconfig": vconfig,
                "skip_kind": "applicability",
                "seed": seed,
            },
        )
    backend_stage = record.get("backend_results", {}).get(backend)
    if not isinstance(backend_stage, dict) or backend_stage.get("status") != "passed":
        return _completed_stage(
            status="failed",
            passed=False,
            failure_reason="backend_not_passed",
            extra={"backend": backend, "vconfig": vconfig, "seed": seed},
        )

    spec = _backend_spec(backend)
    executable = fuzzer_async if spec.fuzzer_binary == "fuzzer_async" else fuzzer
    workdir = kernel_campaign_dir(campaign_dir, record) / "coverage" / backend / vconfig
    workdir.mkdir(parents=True, exist_ok=True)
    coverage_log = workdir / "coverage.jsonl"
    coverage_log.unlink(missing_ok=True)
    stdout_log = workdir / "stdout.log"
    stderr_log = workdir / "stderr.log"
    command = build_coverage_command(
        backend=backend,
        fuzzer=executable,
        library=backend_library_for(campaign_dir, record, backend=backend),
        manifest=Path(str(record["manifest"])),
        coverage_seconds=coverage_seconds,
        coverage_log=coverage_log,
        vconfig=vconfig,
        window_size=window_size,
    )
    env = os.environ.copy()
    env["RAPID_FIXED_SEED"] = str(seed)
    if gpu_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_device
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_output_text(error.stdout, None)
        stderr = _timeout_output_text(None, error.stderr)
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=stdout_log,
            failure_reason=f"coverage timed out after {timeout_seconds}s",
            extra={
                "backend": backend,
                "vconfig": vconfig,
                "seed": seed,
                "coverage_log": str(coverage_log),
                "stderr_log": str(stderr_log),
            },
        )

    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")
    common = {
        "backend": backend,
        "vconfig": vconfig,
        "seed": seed,
        "coverage_log": str(coverage_log),
        "stderr_log": str(stderr_log),
    }
    if completed.returncode != 0:
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=stdout_log,
            failure_reason=f"coverage exited with {completed.returncode}",
            extra=common,
        )
    try:
        evaluation = evaluate_coverage_log(coverage_log)
    except ValueError as error:
        return _completed_stage(
            status="failed",
            passed=False,
            command=command,
            log=stdout_log,
            failure_reason=f"invalid coverage telemetry: {error}",
            extra=common,
        )
    return _completed_stage(
        status="passed",
        passed=True,
        command=command,
        log=stdout_log,
        extra={**common, **evaluation},
    )


def run_coverage_matrix_stage(
    *,
    campaign_dir: Path = CAMPAIGN_ROOT,
    runs: dict[str, ProjectRun] | None = None,
    backends: tuple[str, ...] = ("origin", "rapid", "rapid2"),
    kernel_selectors: tuple[str, ...] | None = None,
    vconfigs: tuple[Literal["on", "off"], ...] = ("off", "on"),
    coverage_seconds: int,
    timeout_seconds: int,
    seed: int,
    fuzzer: Path | None = None,
    fuzzer_async: Path | None = None,
    gpu_device: str | None = None,
) -> list[dict[str, Any]]:
    """Run the completion-driven coverage matrix for structurally runnable kernels."""

    if not backends or len(set(backends)) != len(backends):
        raise ValueError("backends must be a nonempty tuple without duplicates")
    for backend in backends:
        _backend_spec(backend)
    if not vconfigs or len(set(vconfigs)) != len(vconfigs):
        raise ValueError("vconfigs must be a nonempty tuple without duplicates")
    if any(vconfig not in ("on", "off") for vconfig in vconfigs):
        raise ValueError("vconfigs must contain only 'on' or 'off'")
    if coverage_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("coverage_seconds and timeout_seconds must be positive")

    if fuzzer_async is None:
        fuzzer_async = ensure_fuzzer_async(release=True)
    if fuzzer is None:
        fuzzer = fuzzer_async.with_name("fuzzer")
        if not fuzzer.is_file():
            raise FileNotFoundError(f"missing synchronous fuzzer binary: {fuzzer}")

    selected_runs = (
        runs if runs is not None else project_runs_for_campaign(campaign_dir)
    )
    records = load_kernel_results(campaign_dir, runs=selected_runs)
    selected_records = (
        select_kernel_records(records, kernel_selectors)
        if kernel_selectors is not None
        else records
    )
    windows: dict[str, int | None] = {
        "origin": None,
        "rapid": 1,
        "rapid2": 32,
    }
    for record in selected_records:
        if (
            record.get("phase2_status") != "built"
            or record.get("support_decision") != "run"
        ):
            save_kernel_results(campaign_dir, records)
            continue
        coverage_results = record.setdefault("coverage_results", {})
        if not isinstance(coverage_results, dict):
            raise ValueError("coverage_results must be an object")
        for backend in backends:
            backend_results = coverage_results.setdefault(backend, {})
            if not isinstance(backend_results, dict):
                raise ValueError(f"coverage_results.{backend} must be an object")
            for vconfig in vconfigs:
                backend_results[vconfig] = run_coverage_cell(
                    record,
                    backend=backend,
                    vconfig=vconfig,
                    campaign_dir=campaign_dir,
                    fuzzer=fuzzer,
                    fuzzer_async=fuzzer_async,
                    coverage_seconds=coverage_seconds,
                    timeout_seconds=timeout_seconds,
                    window_size=windows[backend],
                    seed=seed,
                    gpu_device=gpu_device,
                )
                save_kernel_results(campaign_dir, records)
    return records


def run_phase2_stage(
    *,
    runs: dict[str, ProjectRun] | None = None,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    selected_runs = runs or RUNS
    stage_results: list[dict[str, Any]] = []
    for project_id, run in selected_runs.items():
        log = run.resolved_run / "phase2.rerun.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/kernel-rewrite/cli.py"),
            "--run-dir",
            str(run.resolved_run),
        ]
        try:
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
            )
            log.write_text(proc.stdout or "", encoding="utf-8")
            stage_results.append(
                {
                    "project": project_id,
                    "command": command,
                    "log": str(log),
                    "returncode": proc.returncode,
                    "passed": proc.returncode == 0,
                }
            )
        except subprocess.TimeoutExpired as exc:
            log.write_text(_timeout_output_text(exc.stdout, exc.stderr), encoding="utf-8")
            stage_results.append(
                {
                    "project": project_id,
                    "command": command,
                    "log": str(log),
                    "returncode": None,
                    "passed": False,
                    "failure_reason": f"Phase2 timed out after {timeout_seconds}s",
                }
            )
    return stage_results


def run_fuzz_stage(
    *,
    campaign_dir: Path = CAMPAIGN_ROOT,
    runs_profiles: dict[str, ProjectRun] | None = None,
    mode: Literal["fixed", "mutation"],
    runs: int,
    timeout_seconds: int,
    fuzzer: Path | None = None,
) -> list[dict[str, Any]]:
    records = load_kernel_results(campaign_dir, runs=runs_profiles)
    enrich_runtime_classifications(records)
    fuzzer_binary = fuzzer or ensure_fuzzer_async(release=True)
    for record in records:
        if record.get("support_decision") != "run":
            continue
        if mode == "fixed":
            record["fixed"] = run_fixed_fuzz(
                record,
                campaign_dir=campaign_dir,
                fuzzer=fuzzer_binary,
                runs=runs,
                timeout_seconds=timeout_seconds,
            )
        elif mode == "mutation":
            record["mutation"] = run_mutation_fuzz(
                record,
                campaign_dir=campaign_dir,
                fuzzer=fuzzer_binary,
                runs=runs,
                timeout_seconds=timeout_seconds,
            )
        else:
            raise ValueError(f"unsupported fuzz mode: {mode}")
        enrich_runtime_classifications(records)
        save_kernel_results(campaign_dir, records)
    enrich_runtime_classifications(records)
    save_kernel_results(campaign_dir, records)
    return records


def evaluate_mutation_fuzz_output(output: str) -> dict[str, Any]:
    """Validate broker/client mutation fuzz output against the liveness gate."""

    try:
        stats = parse_fuzzer_stats(output)
    except ValueError as exc:
        observed_stats = _observed_progress_stats(output)
        if str(exc) == "missing Final statistics block":
            if _has_cuda_or_objective_failure(output, observed_stats):
                return {
                    "passed": False,
                    "failure_reason": "missing Final statistics block after CUDA/objective failure",
                    "stats": observed_stats,
                }
            if observed_stats.get("observed_executions", 0) > 0 and observed_stats.get("observed_evaluated", 0) > 0:
                return {
                    "passed": True,
                    "failure_reason": None,
                    "stats": {**observed_stats, "progress_only": 1},
                }
        return {"passed": False, "failure_reason": str(exc), "stats": observed_stats}
    failures: list[str] = []
    if stats.get("total_executions", 0) <= 0:
        failures.append("no executions completed")
    if stats.get("mutation_calls", 0) <= 0:
        failures.append("mutation calls did not increase")
    if stats.get("normalize_calls", 0) <= 0:
        failures.append("normalize calls did not increase")
    return {
        "passed": not failures,
        "failure_reason": "; ".join(failures) if failures else None,
        "stats": stats,
    }


def mutation_stage_passes(
    evaluation: dict[str, Any],
    *,
    client_joined: bool,
    timed_out: bool,
    client_returncode: int | None,
) -> dict[str, Any]:
    """Return the mutation gate result for the bounded broker/client run.

    The campaign intentionally observes broker/client fuzzing for a bounded
    wall-clock window. Hitting that observation timeout only means the runner
    stopped the process group; it is not a failure when the logs already prove
    that the client joined, executed inputs, and performed mutations.
    """

    failures = [evaluation.get("failure_reason")]
    if not client_joined:
        failures.append("client did not start")
    if not timed_out and client_returncode not in (None, 0):
        failures.append(f"client exited with {client_returncode}")
    filtered = [str(reason) for reason in failures if reason]
    return {
        "passed": bool(evaluation.get("passed")) and not filtered,
        "failure_reason": "; ".join(filtered) if filtered else None,
    }


def classify_fuzz_failure(failure_reason: str | None, log_text: str) -> dict[str, str]:
    """Classify a failed fuzz stage using the recorded process reason and log evidence."""

    reason = failure_reason or ""
    evidence = f"{reason}\n{log_text}"
    if "cudaGetDeviceCount failed" in evidence or "no CUDA-capable device is detected" in evidence:
        return {
            "kind": "cuda_device_unavailable",
            "detail": "CUDA device enumeration failed while running the bounded fuzz stage",
        }
    if "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" in evidence or "too many resources requested for launch" in evidence:
        return {
            "kind": "launch_out_of_resources",
            "detail": "CUDA launch exceeded device resource limits",
        }
    if "illegal memory access" in evidence or "Error code: 700" in evidence:
        return {
            "kind": "cuda_illegal_memory_access",
            "detail": "fixed input reached CUDA illegal memory access in the RAPID2 execution path",
        }
    if "misaligned address" in evidence or "Error code: 716" in evidence:
        return {
            "kind": "cuda_misaligned_address",
            "detail": "fixed or mutated input reached a misaligned CUDA memory access in the RAPID2 execution path",
        }
    if "timed out" in reason:
        return {
            "kind": "fixed_timeout" if "fixed" in reason else "mutation_timeout",
            "detail": reason,
        }
    if "fuzzer exited with -6" in reason:
        return {
            "kind": "process_aborted",
            "detail": "fuzzer process aborted before emitting final statistics",
        }
    if "Final statistics" in reason:
        return {
            "kind": "missing_final_statistics",
            "detail": "fuzzer exited without the bounded-run statistics block",
        }
    if reason:
        return {
            "kind": "fuzz_stage_failed",
            "detail": reason,
        }
    return {
        "kind": "unknown_fuzz_failure",
        "detail": "fuzz stage failed without a classified reason",
    }


def enrich_runtime_classifications(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate records with runtime skip reasons discovered during fixed fuzz."""

    for record in records:
        if record.get("support_decision") != "run":
            record.setdefault("runtime_decision", "skip")
            continue
        fixed = record.get("fixed")
        if not isinstance(fixed, dict):
            record["runtime_decision"] = "unknown"
            continue
        if fixed.get("status") == "passed":
            record["runtime_decision"] = "run"
            record.pop("runtime_skip_reason", None)
            record.pop("runtime_skip_detail", None)
            continue
        if fixed.get("status") == "failed":
            log_text = _read_text(Path(str(fixed.get("log")))) if fixed.get("log") else ""
            classification = classify_fuzz_failure(
                fixed.get("failure_reason") if isinstance(fixed.get("failure_reason"), str) else None,
                log_text,
            )
            fixed["failure_kind"] = classification["kind"]
            record["runtime_decision"] = "skip"
            record["runtime_skip_reason"] = classification["kind"]
            record["runtime_skip_detail"] = classification["detail"]
            mutation = record.get("mutation")
            if not isinstance(mutation, dict) or mutation.get("status") == "not_attempted":
                record["mutation"] = {
                    "status": "skipped",
                    "passed": True,
                    "failure_reason": "fixed_not_passed",
                    "skip_reason": classification["kind"],
                }
    return records
