"""Apply reviewed, symbol-keyed constraints to a copied Phase 1 run."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


KERNEL_MANIFEST_FIELDS = {
    "symbol_name",
    "display_name",
    "args",
    "constraints",
    "launch_policy",
    "others",
}
LAUNCH_POLICY_FIELDS = {
    "grid",
    "block_candidates",
    "physical_block_max",
    "logical_grid",
    "logical_block",
    "logical_block_candidates",
    "target_dynamic_shared_bytes",
    "coverage_memory",
    "vconfig_reserved",
    "vconfig_mutation",
}
CONSTRAINT_OVERRIDE_FIELDS = {
    "symbol_name",
    "kernel_id",
    "display_name",
    "evidence",
    "domains",
    "layouts",
    "constraints",
    "launch_policy",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _kernel_dir(run_dir: Path, entry: dict[str, Any]) -> Path:
    raw = entry.get("dir")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Phase 1 index entry missing dir")
    path = Path(raw)
    return path if path.is_absolute() else run_dir / path


def _validate_registry(registry: dict[str, Any]) -> list[dict[str, Any]]:
    if registry.get("schema_version") != 1:
        raise ValueError("constraint override registry schema_version must be 1")
    if not isinstance(registry.get("project"), str) or not registry["project"]:
        raise ValueError("constraint override registry missing project")
    overrides = registry.get("overrides")
    if not isinstance(overrides, list):
        raise ValueError("constraint override registry overrides must be a list")
    seen: set[tuple[str, str | None]] = set()
    for override in overrides:
        if not isinstance(override, dict):
            raise ValueError("constraint override entry must be an object")
        unknown_fields = sorted(set(override) - CONSTRAINT_OVERRIDE_FIELDS)
        if unknown_fields:
            raise ValueError(
                "unsupported constraint override fields: "
                + ", ".join(unknown_fields)
                + "; use canonical launch_policy"
            )
        symbol = override.get("symbol_name")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("constraint override entry missing symbol_name")
        kernel_id = override.get("kernel_id")
        if kernel_id is not None and (not isinstance(kernel_id, str) or not kernel_id):
            raise ValueError(f"constraint override {symbol} kernel_id must be a non-empty string")
        key = (symbol, kernel_id)
        if key in seen:
            raise ValueError(f"duplicate constraint override symbol: {symbol}")
        seen.add(key)
        evidence = override.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"constraint override {symbol} must include evidence")
        if "domains" in override and not isinstance(override["domains"], list):
            raise ValueError(f"constraint override {symbol} domains must be a list")
        if "layouts" in override and not isinstance(override["layouts"], list):
            raise ValueError(f"constraint override {symbol} layouts must be a list")
        if "constraints" in override and not isinstance(override["constraints"], list):
            raise ValueError(f"constraint override {symbol} constraints must be a list")
        if "launch_policy" in override and not isinstance(override["launch_policy"], dict):
            raise ValueError(f"constraint override {symbol} launch_policy must be an object")
        if "launch_policy" in override:
            unknown_launch_fields = sorted(set(override["launch_policy"]) - LAUNCH_POLICY_FIELDS)
            if unknown_launch_fields:
                raise ValueError(
                    f"constraint override {symbol} unsupported launch policy fields: "
                    + ", ".join(f"launch_policy.{field}" for field in unknown_launch_fields)
                )
    return overrides


def _validate_manifest_kernel_fields(symbol: str, kernel: dict[str, Any]) -> None:
    unknown_fields = sorted(set(kernel) - KERNEL_MANIFEST_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"constraint override {symbol} manifest contains unsupported kernel fields: "
            + ", ".join(unknown_fields)
            + "; use canonical launch_policy"
        )


def _validate_domain_target(
    symbol: str,
    args: list[dict[str, Any]],
    domain_patch: dict[str, Any],
) -> None:
    arg_index = domain_patch.get("arg")
    if not isinstance(arg_index, int) or arg_index < 0 or arg_index >= len(args):
        raise ValueError(f"constraint override {symbol} domain references invalid arg {arg_index}")
    arg = args[arg_index]
    expected_name = domain_patch.get("name")
    expected_type = domain_patch.get("type")
    if expected_name != arg.get("name") or expected_type != arg.get("type"):
        raise ValueError(
            f"constraint override {symbol} argument signature mismatch at arg {arg_index}: "
            f"expected {expected_name}:{expected_type}, got {arg.get('name')}:{arg.get('type')}"
        )
    _resolve_domain_target(arg, domain_patch.get("path"), symbol=symbol, arg_index=arg_index)
    if not isinstance(domain_patch.get("domain"), dict):
        raise ValueError(f"constraint override {symbol} domain patch missing domain")


def _validate_layout_target(
    symbol: str,
    args: list[dict[str, Any]],
    layout_patch: dict[str, Any],
) -> None:
    arg_index = layout_patch.get("arg")
    if not isinstance(arg_index, int) or arg_index < 0 or arg_index >= len(args):
        raise ValueError(f"constraint override {symbol} layout references invalid arg {arg_index}")
    arg = args[arg_index]
    expected_name = layout_patch.get("name")
    expected_type = layout_patch.get("type")
    if expected_name != arg.get("name") or expected_type != arg.get("type"):
        raise ValueError(
            f"constraint override {symbol} argument signature mismatch at arg {arg_index}: "
            f"expected {expected_name}:{expected_type}, got {arg.get('name')}:{arg.get('type')}"
        )
    _resolve_domain_target(arg, layout_patch.get("path"), symbol=symbol, arg_index=arg_index)
    align_bytes = layout_patch.get("align_bytes")
    if not isinstance(align_bytes, int) or align_bytes < 1:
        raise ValueError(f"constraint override {symbol} layout patch missing positive align_bytes")
    if align_bytes & (align_bytes - 1):
        raise ValueError(f"constraint override {symbol} align_bytes must be a power of two")


def _resolve_domain_target(
    arg: dict[str, Any],
    raw_path: Any,
    *,
    symbol: str,
    arg_index: int,
) -> dict[str, Any]:
    if raw_path is None:
        return arg
    if not isinstance(raw_path, list) or not all(isinstance(part, str) and part for part in raw_path):
        raise ValueError(f"constraint override {symbol} arg {arg_index} path must be string segments")

    node = arg
    for part in raw_path:
        layout = node.get("type_layout") if isinstance(node.get("type_layout"), dict) else node
        if part.isdecimal() and isinstance(layout.get("element"), dict):
            element_count = layout.get("element_count")
            if isinstance(element_count, int) and int(part) >= element_count:
                raise ValueError(
                    f"constraint override {symbol} arg {arg_index} path index {part} is out of range"
                )
            node = layout["element"]
            continue
        fields = layout.get("fields")
        if not isinstance(fields, list):
            raise ValueError(
                f"constraint override {symbol} arg {arg_index} path not found: {'.'.join(raw_path)}"
            )
        match = next(
            (field for field in fields if isinstance(field, dict) and field.get("name") == part),
            None,
        )
        if match is None:
            raise ValueError(
                f"constraint override {symbol} arg {arg_index} path not found: {'.'.join(raw_path)}"
            )
        node = match
    return node


def _preflight(
    run_dir: Path,
    registry: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    index = _read_json(run_dir / "index.json")
    entries = index.get("kernels")
    if not isinstance(entries, list):
        raise ValueError("Phase 1 index kernels must be a list")
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    by_kernel_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol_name")
        if isinstance(symbol, str) and symbol:
            by_symbol.setdefault(symbol, []).append(entry)
        kernel_id = entry.get("kernel_id")
        if isinstance(kernel_id, str) and kernel_id:
            if kernel_id in by_kernel_id:
                raise ValueError(f"Phase 1 run contains duplicate kernel_id: {kernel_id}")
            by_kernel_id[kernel_id] = entry

    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for override in _validate_registry(registry):
        symbol = str(override["symbol_name"])
        kernel_id = override.get("kernel_id")
        if kernel_id is not None:
            entry = by_kernel_id.get(str(kernel_id))
            if entry is None:
                raise ValueError(f"constraint override kernel_id not found in Phase 1 run: {kernel_id}")
            if entry.get("symbol_name") != symbol:
                raise ValueError(f"constraint override {symbol} kernel_id symbol mismatch: {kernel_id}")
        else:
            candidates = by_symbol.get(symbol, [])
            if not candidates:
                entry = None
            elif len(candidates) > 1:
                raise ValueError(
                    f"constraint override {symbol} is an ambiguous duplicate symbol; provide kernel_id"
                )
            else:
                entry = candidates[0]
        if entry is None:
            raise ValueError(f"constraint override symbol not found in Phase 1 run: {symbol}")
        manifest = _read_json(_kernel_dir(run_dir, entry) / "manifest.json")
        kernels = manifest.get("kernels")
        if not isinstance(kernels, list) or len(kernels) != 1 or not isinstance(kernels[0], dict):
            raise ValueError(f"constraint override {symbol} requires a single-kernel manifest")
        kernel = kernels[0]
        _validate_manifest_kernel_fields(symbol, kernel)
        if kernel.get("symbol_name") != symbol:
            raise ValueError(f"constraint override {symbol} manifest symbol mismatch")
        expected_display = override.get("display_name")
        if expected_display is not None and kernel.get("display_name") != expected_display:
            raise ValueError(f"constraint override {symbol} display_name mismatch")
        args = kernel.get("args")
        if not isinstance(args, list) or not all(isinstance(arg, dict) for arg in args):
            raise ValueError(f"constraint override {symbol} manifest args must be a list")
        for domain_patch in override.get("domains", []):
            if not isinstance(domain_patch, dict):
                raise ValueError(f"constraint override {symbol} domain patch must be an object")
            _validate_domain_target(symbol, args, domain_patch)
        for layout_patch in override.get("layouts", []):
            if not isinstance(layout_patch, dict):
                raise ValueError(f"constraint override {symbol} layout patch must be an object")
            _validate_layout_target(symbol, args, layout_patch)
        matches.append((entry, override))
    return matches


def _copy_phase1_run(run_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        raise ValueError(f"resolved output already exists: {out_dir}")

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name == "phase2"}
        if "rewrite_summary.json" in names:
            ignored.add("rewrite_summary.json")
        return ignored

    shutil.copytree(run_dir, out_dir, ignore=ignore)


def apply_constraint_overrides(
    run_dir: Path,
    out_dir: Path,
    registry_path: Path,
) -> dict[str, Any]:
    """Create a resolved Phase 1 run and apply all reviewed registry entries."""

    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    registry_path = registry_path.resolve()
    registry = _read_json(registry_path)
    matches = _preflight(run_dir, registry)
    _copy_phase1_run(run_dir, out_dir)

    applied: list[dict[str, Any]] = []
    for raw_entry, override in matches:
        symbol = str(override["symbol_name"])
        resolved_entry = dict(raw_entry)
        raw_entry_dir = _kernel_dir(run_dir, raw_entry)
        relative_dir = raw_entry_dir.relative_to(run_dir)
        kernel_dir = out_dir / relative_dir
        manifest_path = kernel_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        kernel = manifest["kernels"][0]
        args = kernel["args"]
        for domain_patch in override.get("domains", []):
            arg_index = int(domain_patch["arg"])
            target = _resolve_domain_target(
                args[arg_index],
                domain_patch.get("path"),
                symbol=symbol,
                arg_index=arg_index,
            )
            target["domain"] = dict(domain_patch["domain"])
        for layout_patch in override.get("layouts", []):
            arg_index = int(layout_patch["arg"])
            target = _resolve_domain_target(
                args[arg_index],
                layout_patch.get("path"),
                symbol=symbol,
                arg_index=arg_index,
            )
            target["align_bytes"] = int(layout_patch["align_bytes"])
        if "constraints" in override:
            kernel["constraints"] = list(override["constraints"])
        if "launch_policy" in override:
            kernel["launch_policy"] = dict(override["launch_policy"])
        _write_json(manifest_path, manifest)

        metadata_path = kernel_dir / "metadata.json"
        metadata = _read_json(metadata_path)
        output_hashes = metadata.setdefault("output_hashes", {})
        if not isinstance(output_hashes, dict):
            raise ValueError(f"constraint override {symbol} metadata output_hashes must be an object")
        output_hashes["manifest.json"] = _sha256(manifest_path)
        _write_json(metadata_path, metadata)

        applied_record = {
            "schema_version": 1,
            "project": registry["project"],
            "symbol_name": symbol,
            "kernel_id": resolved_entry.get("kernel_id"),
            "display_name": kernel.get("display_name"),
            "registry": str(registry_path),
            "registry_sha256": _sha256(registry_path),
            "evidence": list(override["evidence"]),
        }
        _write_json(kernel_dir / "constraint_override.applied.json", applied_record)
        applied.append(
            {
                "kernel_id": resolved_entry.get("kernel_id"),
                "symbol_name": symbol,
                "dir": str(relative_dir),
            }
        )

    report = {
        "schema_version": 1,
        "project": registry["project"],
        "source_run_dir": str(run_dir),
        "resolved_run_dir": str(out_dir),
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "counts": {"applied": len(applied), "unmatched": 0},
        "applied": applied,
    }
    _write_json(out_dir / "constraint_overrides.report.json", report)
    return report
