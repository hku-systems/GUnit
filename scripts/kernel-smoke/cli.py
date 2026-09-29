#!/usr/bin/env python3
"""Kernel-smoke pipeline CLI.

Usage:
    python3 scripts/kernel-smoke/cli.py run --capture-dir .rapid/capture
    python3 scripts/kernel-smoke/cli.py wrap <compiler> <args...>
"""

import argparse
import concurrent.futures
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.demangle import (
    HAS_DEMANGLER,
    demangle_symbol,
    qualified_name_from_demangled,
)
from utils.clang_helper import ensure_clang_helper, run_clang_entry_metadata
from utils.kernel_id import make_kernel_id
from utils.profiling import Profiler, VariantProfiler
from pipeline.emit_bc import (
    emit_all,
    reconcile_variant,
    enumerate_module_symbols,
    enumerate_module_ptx_entries,
)
from pipeline.manifest import write_all_manifests
from adapters.toolchain import detect_toolchain_versions


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _variant_progress_path(run_dir: Path) -> Path:
    return run_dir / "variant_progress.json"


def _read_variant_progress(run_dir: Path) -> dict[str, Any]:
    path = _variant_progress_path(run_dir)
    if not path.exists():
        return {"completed_variants": [], "variant_order": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"completed_variants": [], "variant_order": []}


def _write_variant_progress(
    run_dir: Path,
    variant_order: list[str],
    completed_variants: set[str],
) -> None:
    _write_json(_variant_progress_path(run_dir), {
        "variant_order": variant_order,
        "completed_variants": sorted(completed_variants),
    })


def _source_hash(source_path: str, variant_id: str | None = None) -> str:
    """Compute module cache key (must match emit_bc._source_hash)."""
    key = source_path
    if variant_id:
        key = f"{source_path}#{variant_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _build_artifact_kernel(
    symbol: str,
    source_file: str,
    variant_id: str,
    line: int = 0,
    args: list[dict[str, Any]] | None = None,
    display_name: str | None = None,
    type_shim: str | None = None,
    type_shim_status: str | None = None,
    type_shim_reason_codes: list[str] | None = None,
    type_shim_missing_dependencies: list[str] | None = None,
    type_shim_system_headers: list[str] | None = None,
    type_shim_system_includes: list[str] | None = None,
) -> dict[str, Any]:
    kernel_args = args or []
    kernel = {
        "kernel_id": make_kernel_id(
            symbol,
            kernel_args,
            line,
            source_path=source_file,
            source_label=source_file,
            variant_id=variant_id,
        ),
        "symbol_name": symbol,
        "display_name": display_name or symbol,
        "args": kernel_args,
        "source_file": source_file,
        "line": line,
        "origin": "artifact",
        "variant_id": variant_id,
    }
    if type_shim is not None:
        kernel["type_shim"] = type_shim
    if type_shim_status is not None:
        kernel["type_shim_status"] = type_shim_status
    if type_shim_reason_codes is not None:
        kernel["type_shim_reason_codes"] = type_shim_reason_codes
    if type_shim_missing_dependencies is not None:
        kernel["type_shim_missing_dependencies"] = type_shim_missing_dependencies
    if type_shim_system_headers is not None:
        kernel["type_shim_system_headers"] = type_shim_system_headers
    if type_shim_system_includes is not None:
        kernel["type_shim_system_includes"] = type_shim_system_includes
    return kernel


# ---------------------------------------------------------------------------
# Run-config: parsed CLI arguments for cmd_run
# ---------------------------------------------------------------------------

class _RunConfig(NamedTuple):
    capture_dir: Path
    out_root: Path
    target_lib: str
    run_id: str
    resume: bool
    mode: str          # artifact-only: PTX/IR entries plus Clang helper metadata
    jobs: int
    source_label: str
    profile: bool

    @property
    def scan_ptx(self) -> bool:
        return self.mode == "artifact"


def _parse_run_args(args: argparse.Namespace) -> _RunConfig | str:
    """Parse and validate cmd_run arguments.

    Returns a _RunConfig on success, or an error-message string on failure.
    """
    capture_dir = Path(args.capture_dir).resolve()
    if not capture_dir.exists():
        return f"Error: capture dir not found: {capture_dir}"

    if args.resume and not args.run_id:
        return "Error: --resume requires --run-id"

    out_root = Path(args.out_root)
    target_lib = args.target_lib
    run_id = args.run_id or uuid.uuid4().hex[:12]
    resume = args.resume
    mode = getattr(args, "mode", "artifact")
    if mode != "artifact":
        return "Error: only artifact mode is supported; Phase 1 uses PTX/IR entries plus Clang helper metadata"
    jobs = max(1, int(getattr(args, "jobs", 1)))
    source_label = str(capture_dir)
    profile = bool(getattr(args, "profile", False))

    return _RunConfig(
        capture_dir=capture_dir,
        out_root=out_root,
        target_lib=target_lib,
        run_id=run_id,
        resume=resume,
        mode=mode,
        jobs=jobs,
        source_label=source_label,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Discover-stage state container
# ---------------------------------------------------------------------------

class _DiscoverState:
    """Mutable accumulator used during the discover stage.

    Only holds data that is genuinely accumulated across variants.
    Static inputs / indexes (capture_map, variant_order) are kept as
    local variables in ``_run_discover_stage``.
    """

    def __init__(self) -> None:
        self.all_kernels: list[dict] = []
        self.capture_stats: dict = {}
        self.preprocess_log: list[dict] = []
        self.ast_log: list[dict] = []
        self.replay_log: list[dict] = []
        self.emit_results: list[dict] = []
        self.manifest_results: list[dict] = []

        # AST discovery provenance counters
        self.ast_kernels_count = 0
        self.fallback_kernels_count = 0
        self.ast_failures_count = 0
        self.ir_symbol_count = 0
        self.ptx_entry_count = 0
        self.variant_failures_by_variant: dict[str, dict[str, Any]] = {}
        self.ptx_ast_mismatch_by_variant: dict[str, dict[str, Any]] = {}

        self.completed_variants: set[str] = set()


def _load_resume_discover_state(
    run_dir: Path,
    ds: _DiscoverState,
) -> None:
    """Load prior discover state when resuming a run.

    Mutates *ds* in place with data from variant_progress.json
    and discover.json if they exist.

    Safety: if variant_progress.json lists completed variants but
    discover.json is missing or unreadable, we treat the run as
    non-resumable (clear completed set so all variants are rerun)
    rather than silently skipping variants without prior kernel data.
    """
    if not _variant_progress_path(run_dir).exists():
        return

    prior = _read_variant_progress(run_dir)
    prior_completed = set(prior.get("completed_variants", []))
    if not prior_completed:
        return

    discover_path = run_dir / "discover.json"
    if not discover_path.exists():
        print(
            "  [resume] WARNING: variant_progress.json exists but discover.json "
            "is missing; treating as non-resumable (all variants will rerun)",
            file=sys.stderr,
        )
        return

    try:
        prior_discover = json.loads(discover_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(
            f"  [resume] WARNING: discover.json unreadable ({exc}); "
            "treating as non-resumable (all variants will rerun)",
            file=sys.stderr,
        )
        return

    # discover.json loaded successfully – apply prior state.
    ds.completed_variants = prior_completed
    ds.all_kernels = prior_discover.get("kernels", [])
    prior_prov = prior_discover.get("provenance", {})

    disc = prior_prov.get("discovery", {})
    diag = prior_prov.get("artifact_diagnostics", {})
    ds.ast_kernels_count = int(disc.get("ast_kernels", 0))
    ds.fallback_kernels_count = int(disc.get("fallback_kernels", 0))
    ds.ast_failures_count = int(disc.get("ast_failures", 0))
    ds.ir_symbol_count = int(diag.get("ir_symbol_count", 0))
    ds.ptx_entry_count = int(diag.get("ptx_entry_count", 0))
    prior_diagnostics = prior_discover.get("diagnostics", {})
    prior_missing = prior_diagnostics.get("variant_failures_by_variant", {})
    if isinstance(prior_missing, dict):
        ds.variant_failures_by_variant = dict(prior_missing)
    prior_mismatch = prior_diagnostics.get("ptx_ast_mismatch_by_variant", {})
    if isinstance(prior_mismatch, dict):
        ds.ptx_ast_mismatch_by_variant = dict(prior_mismatch)
    # Reload emit/manifest results for summary reconciliation.
    ds.emit_results = prior_discover.get("emit_results", [])
    ds.manifest_results = prior_discover.get("manifest_results", [])


def _persist_discover_progress(
    run_dir: Path,
    ds: _DiscoverState,
    variant_order: list[str],
    mode: str,
    jobs: int,
) -> None:
    """Write discover.json and variant_progress.json."""
    discover_payload: dict = {"kernels": ds.all_kernels}
    if ds.capture_stats:
        discover_payload["capture_stats"] = ds.capture_stats
    discover_payload["provenance"] = {
        "mode": mode,
        "jobs": jobs,
        "discovery": {
            "ast_kernels": ds.ast_kernels_count,
            "fallback_kernels": ds.fallback_kernels_count,
            "ast_failures": ds.ast_failures_count,
        },
        "artifact_diagnostics": {
            "ir_symbol_count": ds.ir_symbol_count,
            "ptx_entry_count": ds.ptx_entry_count,
            "variant_failure_count": len(ds.variant_failures_by_variant),
            "ptx_ast_mismatch_count": len(ds.ptx_ast_mismatch_by_variant),
        },
    }
    discover_payload["diagnostics"] = {
        "variant_failures_by_variant": ds.variant_failures_by_variant,
        "ptx_ast_mismatch_by_variant": ds.ptx_ast_mismatch_by_variant,
    }
    # Persist emit/manifest results so resume can reconstruct summary counts.
    discover_payload["emit_results"] = ds.emit_results
    discover_payload["manifest_results"] = ds.manifest_results
    _write_json(run_dir / "discover.json", discover_payload)
    _write_variant_progress(run_dir, variant_order, ds.completed_variants)


# ---------------------------------------------------------------------------
# Per-variant processing
# ---------------------------------------------------------------------------

def _discover_artifact(
    entry: dict[str, Any],
    *,
    modules_dir: Path,
    ptx_dir: Path | None,
    variant_profiler: VariantProfiler,
) -> dict[str, Any]:
    """Artifact (PTX/IR-first) kernel discovery for a single variant.

    PTX entries are source-of-truth for kernel existence; AST is used
    for signature extraction (args / line / display_name).  Returns a
    discovery-result dict consumed by ``_process_variant``.
    """
    from adapters.replay import build_device_bc

    vid = entry["variant_id"]
    sp = entry["source_file"]
    ast_log: list[dict] = []
    ast_failures = 0
    ir_symbol_count = 0
    ptx_entry_count = 0
    ptx_missing_or_empty = False
    ptx_diag_reason: str | None = None
    ptx_diag_detail: str | None = None
    ptx_diag_phase: str | None = None
    ptx_diag_command: list[str] = []
    ptx_ast_mismatch: dict[str, Any] | None = None

    module_bc: Path | None = modules_dir / f"{_source_hash(sp, vid)}.bc"
    ir_symbols: set[str] = set()
    ptx_symbols: set[str] = set()

    if module_bc is not None and not module_bc.exists():
        with variant_profiler.span("build_device_bc"):
            ok, bc_reason, bc_cmd, bc_detail = build_device_bc(entry, module_bc)
        if not ok:
            module_bc = None
            ptx_missing_or_empty = True
            ptx_diag_phase = "device_bc"
            ptx_diag_reason = bc_reason
            ptx_diag_detail = bc_detail
            ptx_diag_command = [str(c) for c in bc_cmd]

    if module_bc is not None and module_bc.exists():
        with variant_profiler.span("enumerate_module_symbols"):
            ir_symbols = set(enumerate_module_symbols(module_bc))
        ir_symbol_count = len(ir_symbols)
        if ptx_dir is not None:
            ptx_path = ptx_dir / f"{module_bc.stem}.ptx"
            with variant_profiler.span("enumerate_module_ptx_entries"):
                ptx_list, ptx_reason, ptx_detail = enumerate_module_ptx_entries(
                    module_bc, ptx_path,
                )
            ptx_symbols = set(ptx_list)
            ptx_entry_count = len(ptx_symbols)
            if ptx_reason is not None:
                ptx_missing_or_empty = True
                ptx_diag_phase = "ptx_scan"
                ptx_diag_reason = ptx_reason
                ptx_diag_detail = ptx_detail
        else:
            ptx_missing_or_empty = True
            ptx_diag_phase = "ptx_scan"
            ptx_diag_reason = "ptx_scan_disabled"
            ptx_diag_detail = "PTX scan disabled because no PTX output directory was configured"

    demangled_by_symbol: dict[str, str] = {}
    demangle_diags: list[str] = []
    for sym in sorted(ptx_symbols):
        try:
            demangled_by_symbol[sym] = demangle_symbol(sym)
        except RuntimeError as e:
            msg = str(e)
            if not msg.startswith("demangle_failed:"):
                raise
            demangled_by_symbol[sym] = sym
            demangle_diags.append(msg)
    matched_metadata: dict[str, dict[str, Any]] = {}
    helper_diags: list[str] = []
    ast_reason: str | None = None
    ast_detail: str | None = None
    ast_cmd: list[str] = []
    if ptx_symbols:
        with variant_profiler.span("artifact_clang_helper"):
            matched_metadata, helper_diags, ast_reason, ast_detail, ast_cmd = run_clang_entry_metadata(
                entry,
                ptx_symbols=ptx_symbols,
            )

    matched_ptx_symbols = set(matched_metadata.keys())
    missing_ptx_in_ast = sorted(sym for sym in ptx_symbols if sym not in matched_ptx_symbols)

    selected = sorted(
        [
            _build_artifact_kernel(
                sym,
                source_file=sp,
                variant_id=vid,
                line=matched_metadata[sym].get("line", 0),
                args=matched_metadata[sym].get("args", []),
                display_name=qualified_name_from_demangled(demangled_by_symbol[sym]),
                type_shim=matched_metadata[sym].get("type_shim"),
                type_shim_status=matched_metadata[sym].get("type_shim_status"),
                type_shim_reason_codes=matched_metadata[sym].get("type_shim_reason_codes"),
                type_shim_missing_dependencies=matched_metadata[sym].get("type_shim_missing_dependencies"),
                type_shim_system_headers=matched_metadata[sym].get("type_shim_system_headers"),
                type_shim_system_includes=matched_metadata[sym].get("type_shim_system_includes"),
            )
            for sym in matched_ptx_symbols
        ],
        key=lambda k: k["symbol_name"],
    )

    ast_log.append({
        "variant_id": vid,
        "source_file": sp,
        "ok": bool(matched_metadata) or not ptx_symbols,
        "reason": ast_reason,
        "detail": ast_detail,
        "command": [str(c) for c in ast_cmd],
        "kernel_count": len(selected),
        "artifact_filter": False,
        "compiler": entry.get("compiler"),
        "selection_reason": entry.get("selection_reason"),
        "diagnostics": (demangle_diags + helper_diags)[:20],
    })

    if ast_reason is not None and not matched_metadata:
        ast_failures += 1

    if missing_ptx_in_ast:
        ptx_ast_mismatch = {
            "source_file": sp,
            "ptx_entry_count": len(ptx_symbols),
            "matched_ptx_symbols": sorted(matched_ptx_symbols),
            "missing_ptx_symbols": sorted(missing_ptx_in_ast),
            "ast_reason": ast_reason,
            "ast_detail": ast_detail,
            "ast_diagnostics": (demangle_diags + helper_diags)[:20],
        }

    return {
        "discovered_kernels": selected,
        "preprocess": [],
        "ast_log": ast_log,
        "ast_kernels": len(selected),
        "fallback_kernels": 0,
        "ast_failures": ast_failures,
        "ir_symbol_count": ir_symbol_count,
        "ptx_entry_count": ptx_entry_count,
        "ptx_missing_or_empty": ptx_missing_or_empty,
        "ptx_diag_reason": ptx_diag_reason,
        "ptx_diag_detail": ptx_diag_detail,
        "ptx_diag_phase": ptx_diag_phase,
        "ptx_diag_command": ptx_diag_command,
        "ptx_ast_mismatch": ptx_ast_mismatch,
    }


def _process_variant(
    entry: dict[str, Any],
    *,
    modules_dir: Path,
    ptx_dir: Path | None,
    kernels_dir: Path,
    target_lib: str,
    toolchain: dict,
    profile_enabled: bool,
) -> dict[str, Any]:
    """Process a single capture variant: discover kernels, emit BC, write manifest.

    Uses artifact discovery, then runs emit + manifest on the discovered kernels.
    """
    vid = entry["variant_id"]
    sp = entry["source_file"]
    variant_replay_log: list[dict] = []
    variant_profiler = VariantProfiler(profile_enabled, variant_id=vid, source_file=sp)

    with variant_profiler.total_span():
        disc = _discover_artifact(
            entry,
            modules_dir=modules_dir,
            ptx_dir=ptx_dir,
            variant_profiler=variant_profiler,
        )

        selected = disc["discovered_kernels"]
        for kernel in selected:
            kernel.setdefault("selection_reason", entry.get("selection_reason"))
            kernel.setdefault("compiler", entry.get("compiler"))

        with variant_profiler.span("emit_all"):
            variant_emit = emit_all(
                selected,
                kernels_dir,
                capture_map={vid: entry},
                modules_dir=modules_dir,
                replay_log=variant_replay_log,
            )
        with variant_profiler.span("write_all_manifests"):
            variant_manifest = write_all_manifests(
                selected,
                variant_emit,
                target_lib,
                toolchain,
            )

    return {
        "variant_id": vid,
        "source_file": sp,
        "discovered_kernels": selected,
        "preprocess": disc["preprocess"],
        "ast_log": disc["ast_log"],
        "replay_log": variant_replay_log,
        "emit_results": variant_emit,
        "manifest_results": variant_manifest,
        "ast_kernels": disc["ast_kernels"],
        "fallback_kernels": disc["fallback_kernels"],
        "ast_failures": disc["ast_failures"],
        "ir_symbol_count": disc["ir_symbol_count"],
        "ptx_entry_count": disc["ptx_entry_count"],
        "ptx_missing_or_empty": disc["ptx_missing_or_empty"],
        "ptx_diag_reason": disc["ptx_diag_reason"],
        "ptx_diag_detail": disc.get("ptx_diag_detail"),
        "ptx_diag_phase": disc.get("ptx_diag_phase"),
        "ptx_diag_command": disc.get("ptx_diag_command", []),
        "ptx_ast_mismatch": disc["ptx_ast_mismatch"],
        "selection_reason": entry.get("selection_reason"),
        "compiler": entry.get("compiler"),
        "record_id": entry.get("record_id"),
        "profile": variant_profiler.as_dict(),
    }


def _accumulate_variant_result(
    ds: _DiscoverState,
    result_obj: dict[str, Any],
) -> None:
    """Merge a single variant result into the discover-stage accumulators."""
    vid = result_obj["variant_id"]
    sp = result_obj["source_file"]

    ds.all_kernels.extend(result_obj["discovered_kernels"])
    ds.preprocess_log.extend(result_obj["preprocess"])
    ds.ast_log.extend(result_obj["ast_log"])
    ds.replay_log.extend(result_obj["replay_log"])
    ds.ast_kernels_count += result_obj["ast_kernels"]
    ds.fallback_kernels_count += result_obj["fallback_kernels"]
    ds.ast_failures_count += result_obj["ast_failures"]
    ds.ir_symbol_count += result_obj["ir_symbol_count"]
    ds.ptx_entry_count += result_obj["ptx_entry_count"]
    if result_obj.get("ptx_missing_or_empty"):
        ds.variant_failures_by_variant[vid] = {
            "source_file": sp,
            "reason": result_obj.get("ptx_diag_reason"),
            "detail": result_obj.get("ptx_diag_detail"),
            "phase": result_obj.get("ptx_diag_phase"),
            "command": result_obj.get("ptx_diag_command", []),
            "compiler": result_obj.get("compiler"),
            "record_id": result_obj.get("record_id"),
            "selection_reason": result_obj.get("selection_reason"),
            "ir_symbol_count": result_obj.get("ir_symbol_count", 0),
            "ptx_entry_count": result_obj.get("ptx_entry_count", 0),
        }
    mismatch = result_obj.get("ptx_ast_mismatch")
    if isinstance(mismatch, dict):
        ds.ptx_ast_mismatch_by_variant[vid] = mismatch

    ds.emit_results.extend(result_obj["emit_results"])
    ds.manifest_results.extend(result_obj["manifest_results"])
    ds.completed_variants.add(vid)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _run_discover_stage(
    run_dir: Path,
    cfg: _RunConfig,
    ctx: dict,
    profiler: Profiler,
) -> list[dict]:
    """Discover kernels from all capture variants."""
    from pipeline.capture_db import load_capture_records, collect_cuda_sources

    toolchain = detect_toolchain_versions()
    if cfg.mode == "artifact" and not HAS_DEMANGLER:
        raise RuntimeError(
            "artifact mode requires python package `cxxfilt` or system `c++filt`",
        )
    if cfg.mode == "artifact":
        ensure_clang_helper()

    ds = _DiscoverState()

    records = load_capture_records(cfg.capture_dir)
    cuda_sources = collect_cuda_sources(records)
    ds.capture_stats = {
        "record_count": len(records),
        "variant_count": len(cuda_sources),
    }

    modules_dir = run_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    ptx_dir = modules_dir / "ptx" if cfg.scan_ptx else None
    if ptx_dir is not None:
        ptx_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir = run_dir / "kernels"

    # Build capture_map keyed by variant_id for replay/reconciliation.
    capture_map: dict[str, dict] = {}
    variant_order: list[str] = []
    variant_pos: dict[str, int] = {}
    for entry in cuda_sources:
        vid = entry["variant_id"]
        capture_map[vid] = entry
        variant_order.append(vid)
        variant_pos[vid] = len(variant_order)

    if cfg.resume:
        _load_resume_discover_state(run_dir, ds)

    pending_sources = [
        entry for entry in cuda_sources
        if entry["variant_id"] not in ds.completed_variants
    ]
    skipped_resume_count = len(ds.completed_variants)

    total_variants = len(cuda_sources)
    print(f"  [discover] processing {total_variants} variants ...", flush=True)
    if ds.completed_variants:
        print(
            f"  [discover] resume skip: {len(ds.completed_variants)} variants already completed",
            flush=True,
        )

    def _pv(entry: dict[str, Any]) -> dict[str, Any]:
        return _process_variant(
            entry,
            modules_dir=modules_dir,
            ptx_dir=ptx_dir,
            kernels_dir=kernels_dir,
            target_lib=cfg.target_lib,
            toolchain=toolchain,
            profile_enabled=cfg.profile,
        )

    if cfg.jobs > 1 and len(pending_sources) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.jobs) as executor:
            future_to_entry = {
                executor.submit(_pv, entry): entry
                for entry in pending_sources
            }
            completed_now = 0
            for fut in concurrent.futures.as_completed(future_to_entry):
                result_entry = future_to_entry[fut]
                vid = result_entry["variant_id"]
                sp = result_entry["source_file"]
                result_obj = fut.result()
                completed_now += 1
                print(
                    f"  [discover] done {completed_now}/{len(pending_sources)} {Path(sp).name} ({vid})",
                    flush=True,
                )

                _accumulate_variant_result(ds, result_obj)
                profiler.add_variant_profile(result_obj.get("profile"))
                _persist_discover_progress(run_dir, ds, variant_order, cfg.mode, cfg.jobs)
    else:
        for entry in pending_sources:
            vid = entry["variant_id"]
            sp = entry["source_file"]
            idx = variant_pos[vid]
            print(
                f"  [discover] {idx}/{total_variants} {Path(sp).name} ({vid})",
                flush=True,
            )
            result_obj = _pv(entry)

            _accumulate_variant_result(ds, result_obj)
            profiler.add_variant_profile(result_obj.get("profile"))
            _persist_discover_progress(run_dir, ds, variant_order, cfg.mode, cfg.jobs)

    ctx["capture_map"] = capture_map
    ctx["emit_results"] = ds.emit_results
    ctx["manifest_results"] = ds.manifest_results
    ctx["variant_failures_by_variant"] = ds.variant_failures_by_variant
    ctx["ptx_ast_mismatch_by_variant"] = ds.ptx_ast_mismatch_by_variant
    ctx["kernels"] = ds.all_kernels

    # Final discover.json write uses _persist_discover_progress which now
    # includes emit_results and manifest_results.
    _persist_discover_progress(run_dir, ds, variant_order, cfg.mode, cfg.jobs)
    profiler.set_stage_meta(
        "discover",
        record_count=len(records),
        variant_count=total_variants,
        profiled_variants=len(pending_sources),
        skipped_resume=skipped_resume_count,
    )
    return ds.all_kernels


def _run_summary_stage(
    run_dir: Path,
    cfg: _RunConfig,
    ctx: dict,
) -> dict:
    """Build index.json, reconciliation stats, and summary.json."""
    kernels = ctx.get("kernels", [])
    manifest_results = ctx.get("manifest_results", [])
    capture_map = ctx.get("capture_map")
    kernels_dir = run_dir / "kernels"

    discovered = len(kernels)
    succeeded = sum(
        1 for r in manifest_results if r.get("status") == "built"
    )
    failed = sum(
        1 for r in manifest_results if r.get("status") == "failed"
    )
    skipped = discovered - succeeded - failed

    index = {
        "schema_version": 1,
        "run_id": cfg.run_id,
        "target_lib": cfg.target_lib,
        "capture_dir": cfg.source_label,
        "generated_at": _now_iso(),
        "kernels": [
            {
                "kernel_id": k["kernel_id"],
                "symbol_name": k["symbol_name"],
                "dir": str(Path("kernels") / k["kernel_id"]),
            }
            for k in kernels
        ],
    }
    _write_json(run_dir / "index.json", index)

    # Phase B: Build reconciliation stats by variant
    reconciliation: dict[str, Any] = {}
    variant_failures_by_variant = ctx.get("variant_failures_by_variant", {})
    ptx_ast_mismatch_by_variant = ctx.get("ptx_ast_mismatch_by_variant", {})
    if capture_map:
        modules_dir = run_dir / "modules"

        # Group discovered kernel symbols by variant_id
        variant_symbols: dict[str, set[str]] = {}
        for k in kernels:
            vid = k.get("variant_id", "unknown")
            if vid not in variant_symbols:
                variant_symbols[vid] = set()
            variant_symbols[vid].add(k["symbol_name"])

        # Count replay failures by variant from emit results.
        replay_failed_by_variant: dict[str, int] = {}
        emit_results = ctx.get("emit_results", [])
        for entry in emit_results:
            vid = entry.get("variant_id", "unknown")
            if entry.get("status") == "failed":
                replay_failed_by_variant[vid] = \
                    replay_failed_by_variant.get(vid, 0) + 1

        # Reconcile each variant
        per_variant: list[dict] = []
        missing_in_ir_by_variant: dict[str, list[str]] = {}
        extra_in_ir_by_variant: dict[str, list[str]] = {}

        for vid, syms in variant_symbols.items():
            # Find the module BC for this variant
            entry = capture_map.get(vid, {})
            src = entry.get("source_file", "")
            src_hash = _source_hash(src, vid)
            module_bc = modules_dir / f"{src_hash}.bc"

            rec = reconcile_variant(vid, syms, module_bc)
            per_variant.append(rec)

            if rec["missing_in_ir"]:
                missing_in_ir_by_variant[vid] = rec["missing_in_ir"]
            if rec["extra_in_ir"]:
                extra_in_ir_by_variant[vid] = rec["extra_in_ir"]

        reconciliation = {
            "replay_failed_by_variant": replay_failed_by_variant,
            "missing_in_ir_by_variant": missing_in_ir_by_variant,
            "extra_in_ir_by_variant": extra_in_ir_by_variant,
            "per_variant": per_variant,
            "variant_failures_by_variant": variant_failures_by_variant,
            "ptx_ast_mismatch_by_variant": ptx_ast_mismatch_by_variant,
        }

    summary = {
        "schema_version": 1,
        "run_id": cfg.run_id,
        "target_lib": cfg.target_lib,
        "capture_dir": cfg.source_label,
        "generated_at": _now_iso(),
        "counts": {
            "discovered": discovered,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
        },
    }
    if cfg.profile:
        summary["profile"] = {
            "enabled": True,
            "file": "profile.json",
        }
    if reconciliation:
        summary["reconciliation"] = reconciliation

    _write_json(run_dir / "summary.json", summary)
    print(f"  Summary: {discovered} discovered, {succeeded} succeeded, "
          f"{failed} failed, {skipped} skipped")
    if reconciliation:
        total_missing = sum(
            len(v) for v in reconciliation.get("missing_in_ir_by_variant", {}).values()
        )
        total_extra = sum(
            len(v) for v in reconciliation.get("extra_in_ir_by_variant", {}).values()
        )
        total_replay_fail = sum(
            reconciliation.get("replay_failed_by_variant", {}).values()
        )
        total_ast_mismatch = len(
            reconciliation.get("ptx_ast_mismatch_by_variant", {})
        )
        print(f"  Reconciliation: {total_replay_fail} replay failures, "
              f"{total_missing} missing in IR, {total_extra} extra in IR, "
              f"{total_ast_mismatch} AST mismatches")
    return summary


# ---------------------------------------------------------------------------
# cmd_run: orchestration
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    parsed = _parse_run_args(args)
    if isinstance(parsed, str):
        print(parsed, file=sys.stderr)
        return 1
    cfg = parsed

    run_dir = cfg.out_root / cfg.run_id
    profiler = Profiler(cfg.profile)

    print(f"kernel-smoke run: {cfg.run_id}")
    print(f"  capture:    {cfg.capture_dir}")
    print(f"  out_root:   {cfg.out_root}")
    print(f"  target_lib: {cfg.target_lib}")
    print(f"  resume:     {cfg.resume}")
    print(f"  mode:       {cfg.mode}")
    print(f"  jobs:       {cfg.jobs}")
    print(f"  profile:    {cfg.profile}")

    if cfg.resume and (run_dir / "variant_progress.json").exists():
        print("  Resuming from existing variant progress")

    run_dir.mkdir(parents=True, exist_ok=True)

    ctx: dict = {}

    with profiler.run_span():
        # Stage 1: discover (includes emit + manifest per variant)
        with profiler.stage_span("discover"):
            _run_discover_stage(run_dir, cfg, ctx, profiler)
        print(f"  [discover] completed")

        # Stage 2: summary
        with profiler.stage_span("summary"):
            summary = _run_summary_stage(run_dir, cfg, ctx)
        print(f"  [summary] completed")
        profiler.set_stage_meta("summary", **summary.get("counts", {}))

    profile_payload = profiler.as_dict(
        run_id=cfg.run_id,
        resume=cfg.resume,
        jobs=cfg.jobs,
    )
    if profile_payload is not None:
        _write_json(run_dir / "profile.json", profile_payload)
        print(f"  [profile] wrote {run_dir / 'profile.json'}")

    print(f"  Output: {run_dir}")
    return 0


def cmd_wrap(args: argparse.Namespace) -> int:
    from adapters.capture_wrapper import run_and_capture
    return run_and_capture(args.compiler_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="kernel-smoke",
        description="RAPID kernel-smoke pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the kernel-smoke pipeline")
    run_p.add_argument("--capture-dir", required=True,
                       help="Path to capture directory with commands.jsonl")
    run_p.add_argument("--out-root", default="build/kernel-smoke",
                       help="Output root directory (default: build/kernel-smoke)")
    run_p.add_argument("--target-lib", default="smoke_fixture",
                       help="Logical target library name (default: smoke_fixture)")
    run_p.add_argument("--run-id", default=None,
                       help="Run identifier (default: auto-generated)")
    run_p.add_argument("--resume", action="store_true",
                       help="Resume from variant progress")
    run_p.add_argument(
        "--mode",
        choices=["artifact"],
        default="artifact",
        help="Discovery mode: artifact (PTX/IR entries plus Clang helper metadata)",
    )
    run_p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel jobs for capture variant processing (default: 1)",
    )
    run_p.add_argument(
        "--profile",
        action="store_true",
        help="Write wall-clock profiling data to profile.json",
    )

    wrap_p = sub.add_parser("wrap", help="Capture a compiler invocation")
    wrap_p.add_argument("compiler_args", nargs=argparse.REMAINDER,
                        help="Compiler command and arguments")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return cmd_run(args)
    if args.command == "wrap":
        return cmd_wrap(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
