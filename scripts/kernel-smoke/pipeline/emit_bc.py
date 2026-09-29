"""Emit per-kernel LLVM bitcode and PTX via replay+extract or placeholder.

Also provides IR-based reconciliation helpers for Phase B.
"""

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _source_hash(source_path: str, variant_id: str | None = None) -> str:
    """Short hash for caching module BC per source+variant."""
    key = source_path
    if variant_id:
        key = f"{source_path}#{variant_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _detect_clang_version() -> str | None:
    """Detect the major version of the system clang."""
    clang = shutil.which("clang++") or shutil.which("clang")
    if clang:
        try:
            result = subprocess.run(
                [clang, "--version"], capture_output=True, text=True, timeout=5,
            )
            import re
            m = re.search(r"version\s+(\d+)\.", result.stdout)
            if m:
                return m.group(1)
        except (subprocess.TimeoutExpired, OSError):
            pass
    return None


def _find_versioned_tool(tool_name: str) -> str | None:
    """Find an LLVM tool, preferring a version matching the system clang."""
    ver = _detect_clang_version()
    if ver:
        versioned = shutil.which(f"{tool_name}-{ver}")
        if versioned:
            return versioned
    return shutil.which(tool_name)


def _find_llvm_extract() -> str | None:
    """Find llvm-extract, preferring a version matching the system clang."""
    return _find_versioned_tool("llvm-extract")


def _find_llvm_nm() -> str | None:
    """Find llvm-nm, preferring version matching system clang."""
    result = _find_versioned_tool("llvm-nm")
    return result or shutil.which("nm")


def _find_llc() -> str | None:
    """Find llc, preferring version matching system clang."""
    return _find_versioned_tool("llc")


def _map_reason(reason: str | None) -> str:
    """Map internal replay/extract errors to metadata schema reason enum."""
    if not reason:
        return "unsupported_ir"
    lower = reason.lower()
    if "toolchain" in lower or "clang_not_found" in lower or "llvm_extract_not_found" in lower:
        return "toolchain_mismatch"
    if "extract" in lower:
        return "extract_fail"
    return "unsupported_ir"


def _truncate_detail(detail: str | None, limit: int = 4000) -> str | None:
    """Bound failure details to keep stage/metadata payloads manageable."""
    if not detail:
        return None
    clean = detail.strip()
    if not clean:
        return None
    return clean[:limit]


def _generate_ptx(bc_path: Path, ptx_path: Path) -> tuple[bool, str | None, str | None]:
    """Generate PTX from kernel.bc using llc, falling back to clang.

    Returns (success, failure_reason_or_None, failure_detail_or_None).
    """
    # Try llc first
    llc = _find_llc()
    llc_detail: str | None = None
    if llc:
        try:
            result = subprocess.run(
                [llc, "-march=nvptx64", str(bc_path), "-o", str(ptx_path)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and ptx_path.exists() and ptx_path.stat().st_size > 0:
                return (True, None, None)
            llc_detail = (result.stderr or "").strip()[:1500] or None
            log.debug("llc failed (rc=%d): %s", result.returncode, result.stderr[:200])
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            llc_detail = str(e)
            log.debug("llc error: %s", e)

    # Fallback: clang -S -target nvptx64
    clang = shutil.which("clang++") or shutil.which("clang")
    clang_detail: str | None = None
    if clang:
        try:
            result = subprocess.run(
                [clang, "-S", "-target", "nvptx64", str(bc_path), "-o", str(ptx_path)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and ptx_path.exists() and ptx_path.stat().st_size > 0:
                return (True, None, None)
            clang_detail = (result.stderr or "").strip()[:1500] or None
            log.debug("clang ptx fallback failed (rc=%d): %s",
                      result.returncode, result.stderr[:200])
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            clang_detail = str(e)
            log.debug("clang ptx fallback error: %s", e)

    # Both failed
    if not llc and not clang:
        return (False, "toolchain_mismatch", "both llc and clang are unavailable")

    detail_parts: list[str] = []
    if llc_detail:
        detail_parts.append(f"llc: {llc_detail}")
    if clang_detail:
        detail_parts.append(f"clang: {clang_detail}")
    return (False, "unsupported_ir", _truncate_detail("\n".join(detail_parts) or None))


def enumerate_module_ptx_entries(
    module_bc: Path,
    out_ptx: Path,
) -> tuple[list[str], str | None, str | None]:
    """Generate PTX for a module and list `.entry` symbols.

    Returns (entries, reason, detail). `entries` is empty when PTX generation
    fails or when no `.entry` symbols are found.
    """
    ok, reason, detail = _generate_ptx(module_bc, out_ptx)
    if not ok or not out_ptx.exists():
        return ([], reason or "ptx_generation_failed", detail or reason)

    try:
        text = out_ptx.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return ([], "ptx_read_failed", str(e))

    entries = re.findall(r"\.entry\s+([^\s(]+)", text)
    # Keep deterministic ordering and drop duplicates.
    uniq = sorted(set(entries))
    if not uniq:
        return ([], "ptx_no_entry", f"no .entry symbols found in {out_ptx}")
    return (uniq, None, None)


def _resolve_extract_symbol(module_bc: Path, symbol_name: str) -> str | None:
    """Resolve symbol for llvm-extract.

    Require exact symbol match in module BC.
    """
    llvm_nm = _find_llvm_nm()
    if llvm_nm is None:
        return symbol_name

    try:
        result = subprocess.run(
            [llvm_nm, "--defined-only", str(module_bc)],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.TimeoutExpired, OSError):
        return symbol_name

    if result.returncode != 0:
        return symbol_name

    symbols: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        sym = parts[-1]
        if sym and not sym.startswith("."):
            symbols.append(sym)

    if symbol_name in symbols:
        return symbol_name

    return None


def _run_llvm_extract(
    module_bc: Path, symbol_name: str, out_bc: Path,
) -> tuple[bool, str | None, list[str], str | None]:
    """Run llvm-extract to isolate a single kernel function."""
    llvm_extract = _find_llvm_extract()
    if llvm_extract is None:
        return (False, "llvm_extract_not_found", [], "llvm-extract not found in PATH")

    extract_symbol = _resolve_extract_symbol(module_bc, symbol_name)
    if extract_symbol is None:
        return (False, "extract_fail: symbol_not_found", [], f"symbol '{symbol_name}' not found in module")
    cmd = [
        llvm_extract,
        "--recursive",
        "--func",
        extract_symbol,
        str(module_bc),
        "-o",
        str(out_bc),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return (False, f"extract_fail: {e}", cmd, str(e))

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return (False, f"extract_fail: {stderr[:200]}", cmd, _truncate_detail(stderr))

    if not out_bc.exists() or out_bc.stat().st_size == 0:
        return (False, "extract_fail: empty output", cmd, "llvm-extract produced empty output")

    return (True, None, cmd, None)


def _emit_placeholder(kernel: dict[str, Any], kernel_dir: Path) -> dict[str, Any]:
    """Write deterministic placeholder kernel.bc and kernel.ptx (source-only mode)."""
    payload_obj = {
        "placeholder": True,
        "kernel_id": kernel["kernel_id"],
        "symbol_name": kernel["symbol_name"],
        "args": [{"name": a["name"], "type": a["type"]} for a in kernel["args"]],
    }
    payload = json.dumps(payload_obj, sort_keys=True, indent=2).encode("utf-8")

    bc_path = kernel_dir / "kernel.bc"
    bc_path.write_bytes(payload)

    # Generate deterministic placeholder PTX
    ptx_payload_obj = {
        "placeholder": True,
        "format": "ptx",
        "kernel_id": kernel["kernel_id"],
        "symbol_name": kernel["symbol_name"],
    }
    ptx_payload = json.dumps(ptx_payload_obj, sort_keys=True, indent=2).encode("utf-8")
    ptx_path = kernel_dir / "kernel.ptx"
    ptx_path.write_bytes(ptx_payload)

    return {
        "kernel_id": kernel["kernel_id"],
        "bc_path": str(bc_path),
        "bc_hash": _sha256_bytes(payload),
        "input_bc_hash": None,
        "ptx_path": str(ptx_path),
        "ptx_hash": _sha256_bytes(ptx_payload),
        "kernel_dir": str(kernel_dir),
        "status": "built",
        "failure_reason": None,
        "failure_detail": None,
    }


def emit_bc(
    kernel: dict[str, Any],
    kernels_dir: Path,
    capture_entry: dict[str, Any] | None = None,
    modules_dir: Path | None = None,
    replay_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create per-kernel directory and produce kernel.bc and kernel.ptx.

    In capture mode (capture_entry provided): replay compile to get TU .bc,
    then llvm-extract the kernel symbol, then generate PTX.
    In source-only mode: write deterministic placeholders.

    Returns dict with kernel_id, bc_path, bc_hash, input_bc_hash, ptx_path, ptx_hash,
    kernel_dir, status, failure_reason.
    """
    kid = kernel["kernel_id"]
    kernel_dir = kernels_dir / kid
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Source-only mode: no capture entry
    if capture_entry is None:
        return _emit_placeholder(kernel, kernel_dir)

    # Capture mode: try replay + extract
    from adapters.replay import build_device_bc

    source_file = capture_entry.get("source_file", "")
    variant_id = capture_entry.get("variant_id")
    src_hash = _source_hash(source_file, variant_id)

    # Cache module BC per source file + variant
    if modules_dir is None:
        modules_dir = kernels_dir.parent / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    module_bc = modules_dir / f"{src_hash}.bc"

    # Build module BC if not cached
    if not module_bc.exists():
        ok, reason, cmd, detail = build_device_bc(capture_entry, module_bc)
        if replay_log is not None:
            replay_log.append({
                "source_file": source_file,
                "variant_id": variant_id,
                "compiler": capture_entry.get("compiler"),
                "selection_reason": capture_entry.get("selection_reason"),
                "module_bc": str(module_bc),
                "ok": ok,
                "reason": reason,
                "detail": _truncate_detail(detail),
                "command": [str(c) for c in cmd],
            })
        if not ok:
            return {
                "kernel_id": kid,
                "bc_path": None,
                "bc_hash": None,
                "input_bc_hash": None,
                "ptx_path": None,
                "ptx_hash": None,
                "kernel_dir": str(kernel_dir),
                "status": "failed",
                "failure_reason": _map_reason(reason),
                "failure_stage": "device_bc",
                "failure_code": reason,
                "failure_detail": _truncate_detail(detail or reason),
            }
    else:
        log.debug("module BC cache hit: %s", module_bc)

    module_bc_hash = _sha256_file(module_bc)

    # Extract kernel symbol from module BC
    bc_path = kernel_dir / "kernel.bc"
    ok, reason, cmd, detail = _run_llvm_extract(module_bc, kernel["symbol_name"], bc_path)

    if not ok:
        return {
            "kernel_id": kid,
            "bc_path": None,
            "bc_hash": None,
            "input_bc_hash": module_bc_hash,
            "ptx_path": None,
            "ptx_hash": None,
            "kernel_dir": str(kernel_dir),
            "status": "failed",
            "failure_reason": _map_reason(reason),
            "failure_stage": "llvm_extract",
            "failure_code": reason,
            "failure_detail": _truncate_detail(detail or reason),
        }

    bc_bytes = bc_path.read_bytes()
    result_dict: dict[str, Any] = {
        "kernel_id": kid,
        "bc_path": str(bc_path),
        "bc_hash": _sha256_bytes(bc_bytes),
        "input_bc_hash": module_bc_hash,
        "ptx_path": None,
        "ptx_hash": None,
        "kernel_dir": str(kernel_dir),
        "status": "built",
        "failure_reason": None,
        "failure_stage": None,
        "failure_code": None,
        "failure_detail": None,
    }

    # Generate PTX from extracted kernel.bc
    ptx_path = kernel_dir / "kernel.ptx"
    ptx_ok, ptx_reason, ptx_detail = _generate_ptx(bc_path, ptx_path)
    if ptx_ok:
        ptx_bytes = ptx_path.read_bytes()
        result_dict["ptx_path"] = str(ptx_path)
        result_dict["ptx_hash"] = _sha256_bytes(ptx_bytes)
    else:
        log.warning("PTX generation failed for %s: %s", kid, ptx_reason)
        result_dict["status"] = "failed"
        result_dict["failure_reason"] = ptx_reason
        result_dict["failure_stage"] = "kernel_ptx_emit"
        result_dict["failure_code"] = ptx_reason
        result_dict["failure_detail"] = _truncate_detail(ptx_detail or ptx_reason)

    return result_dict


def emit_all(
    kernels: list[dict[str, Any]],
    kernels_dir: Path,
    capture_map: dict[str, dict[str, Any]] | None = None,
    modules_dir: Path | None = None,
    replay_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Emit kernel.bc and kernel.ptx for all discovered kernels.

    Args:
        kernels: list of kernel descriptors from discover stage.
        kernels_dir: output directory for per-kernel subdirs.
        capture_map: mapping for replay mode: variant_id -> capture_entry.
        modules_dir: optional directory for cached module .bc files.
        replay_log: optional list to collect replay log entries.

    Returns list of emit results.
    """
    results = []
    for k in kernels:
        entry = None
        variant_id = None
        if capture_map:
            vid = k.get("variant_id")
            if vid and vid in capture_map:
                entry = capture_map[vid]
                variant_id = vid
        one = emit_bc(
            k, kernels_dir,
            capture_entry=entry,
            modules_dir=modules_dir,
            replay_log=replay_log,
        )
        if variant_id:
            one["variant_id"] = variant_id
        results.append(one)
    return results


# ---------------------------------------------------------------------------
# Phase B: IR-based reconciliation
# ---------------------------------------------------------------------------

def enumerate_module_symbols(module_bc: Path) -> list[str]:
    """List defined function symbols from a module .bc file.

    Uses llvm-nm (or nm fallback) with --defined-only to enumerate symbols.
    Filters out internal/section symbols (those starting with '.').
    Returns a list of symbol name strings.
    """
    llvm_nm = _find_llvm_nm()
    if llvm_nm is None:
        return []

    try:
        result = subprocess.run(
            [llvm_nm, "--defined-only", str(module_bc)],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    if result.returncode != 0:
        return []

    symbols: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        # llvm-nm output: addr type name  -- or just: type name
        sym = parts[-1]
        if sym and not sym.startswith("."):
            # Keep callable symbols only and exclude local-static mangled symbols.
            if sym.startswith("_ZZ"):
                continue

            # Filter to function-like symbol types if type info available.
            if len(parts) >= 2:
                sym_type = parts[-2] if len(parts) >= 2 else ""
                if sym_type in ("T", "t", "W", "w", ""):
                    symbols.append(sym)
                elif len(sym_type) == 1 and sym_type.isalpha():
                    # Other symbol types (D, B, etc.) - skip for functions
                    continue
                else:
                    # No valid type field, treat last token as symbol
                    symbols.append(sym)
            else:
                symbols.append(sym)

    return symbols


def reconcile_variant(
    variant_id: str,
    discovered_symbols: set[str],
    module_bc: Path | None,
) -> dict[str, Any]:
    """Produce reconciliation stats for a single variant.

    Compares discovered kernel symbols (from preprocessed source analysis)
    against function symbols found in the module .bc (from IR).

    Returns dict with:
        - variant_id
        - discovered_count
        - ir_symbols_count
        - missing_in_ir: symbols discovered but not in module IR
        - extra_in_ir: IR function symbols not in discovered set (best-effort)
        - match_count: symbols present in both
    """
    if module_bc is None or not module_bc.exists():
        return {
            "variant_id": variant_id,
            "discovered_count": len(discovered_symbols),
            "ir_symbols_count": 0,
            "missing_in_ir": sorted(discovered_symbols),
            "extra_in_ir": [],
            "match_count": 0,
        }

    ir_symbols = set(enumerate_module_symbols(module_bc))

    matched_discovered: set[str] = set()
    matched_ir: set[str] = set()

    for dsym in discovered_symbols:
        if dsym in ir_symbols:
            matched_discovered.add(dsym)
            matched_ir.add(dsym)

    missing = discovered_symbols - matched_discovered
    # Extra: IR symbols that look like kernel functions but weren't discovered
    # Best-effort: exclude common runtime/helper symbols
    _skip_prefixes = ("__cuda", "__nv", "__omp", "llvm.", "__clang", "_ZZ")
    extra = set()
    for isym in ir_symbols:
        if isym in matched_ir:
            continue
        if any(isym.startswith(p) for p in _skip_prefixes):
            continue
        extra.add(isym)

    return {
        "variant_id": variant_id,
        "discovered_count": len(discovered_symbols),
        "ir_symbols_count": len(ir_symbols),
        "missing_in_ir": sorted(missing),
        "extra_in_ir": sorted(extra),
        "match_count": len(matched_discovered),
    }
