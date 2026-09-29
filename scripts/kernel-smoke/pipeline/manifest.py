"""Write per-kernel manifest.json and metadata.json."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REQUIRED_ARG_FIELDS = ("index", "name", "type", "kind", "size_bytes", "align_bytes")
_ARG_KINDS = {"pointer", "scalar", "opaque_val", "opaque_with_ptr"}
_POINTER_ROLES = {"payload_buffer", "derived_pointer", "external_device_pointer"}
_DERIVED_POINTER_FIELDS = ("base_arg", "offset_bytes", "offset_arg")


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepend_includes_to_shim(shim_text: str, include_lines: list[str]) -> str:
    if not include_lines:
        return shim_text
    lines = shim_text.splitlines()
    if len(lines) < 2:
        return shim_text
    return "\n".join(lines[:2] + [""] + include_lines + [""] + lines[2:]) + "\n"


def _normalize_type_info(type_info: Any) -> Any:
    if not isinstance(type_info, dict):
        raise ValueError("type_info must be an object")
    if not isinstance(type_info.get("kind"), str) or not type_info.get("kind"):
        raise ValueError("type_info missing kind")
    return dict(type_info)


def _normalize_arg_kind(kind: Any) -> str | None:
    if not isinstance(kind, str) or not kind:
        return None
    return kind


def _normalize_layout_kind(kind: Any) -> str | None:
    if not isinstance(kind, str) or not kind:
        return None
    return kind


def _layout_contains_pointer(node: dict[str, Any]) -> bool:
    kind = _normalize_layout_kind(node.get("kind"))
    if kind in {"pointer", "opaque_with_ptr"}:
        return True
    layout_status = node.get("layout_status")
    if layout_status in {"partial", "opaque"}:
        return True
    fields = node.get("fields")
    if isinstance(fields, list):
        for field in fields:
            if isinstance(field, dict) and _layout_contains_pointer(field):
                return True
    element = node.get("element")
    if isinstance(element, dict) and _layout_contains_pointer(element):
        return True
    return False


def _layout_child_index(parent_index: str, name: str) -> str:
    if name == "$element":
        return f"{parent_index}[]"
    if name == "$pointee":
        return f"{parent_index}.*" if parent_index else "*"
    return f"{parent_index}.{name}" if parent_index else name


def _normalize_layout_node(node: Any, *, parent_index: str = "") -> Any:
    if not isinstance(node, dict):
        return node
    if "offset_bytes" in node:
        name = node.get("name") or parent_index or "<layout>"
        raise ValueError(f"layout node {name} sets unsupported offset_bytes")
    out = dict(node)
    name = str(out.get("name", ""))
    if name:
        out["index"] = _layout_child_index(parent_index, name) if parent_index else str(out.get("index") or name)
    elif parent_index and not isinstance(out.get("index"), str):
        out["index"] = parent_index
    raw_kind = out.get("kind")
    kind = _normalize_layout_kind(raw_kind)
    if kind is not None:
        out["kind"] = kind
    if out.get("type_info") is not None:
        out["type_info"] = _normalize_type_info(out["type_info"])
    _preserve_materialization_facts(node, out)
    fields = out.get("fields")
    if isinstance(fields, list):
        node_index = str(out.get("index") or parent_index)
        out["fields"] = [_normalize_layout_node(field, parent_index=node_index) for field in fields]
    element = out.get("element")
    if isinstance(element, dict):
        node_index = str(out.get("index") or parent_index)
        out["element"] = _normalize_layout_node(element, parent_index=node_index)
    pointee_layout = out.get("pointee_layout")
    if isinstance(pointee_layout, dict):
        node_index = str(out.get("index") or parent_index)
        out["pointee_layout"] = _normalize_layout_node(pointee_layout, parent_index=node_index)
    if out.get("kind") == "pointer":
        pointer_role = out.get("pointer_role")
        if pointer_role is None:
            raise ValueError(f"layout pointer field {out.get('name', '')} missing pointer_role")
        elif pointer_role not in _POINTER_ROLES:
            raise ValueError(f"layout pointer field {out.get('name', '')} has unsupported pointer_role: {pointer_role}")
        if not isinstance(out.get("pointee_layout"), dict):
            raise ValueError(f"layout pointer field {out.get('name', '')} missing pointee_layout")
    elif out.get("pointer_role") is not None:
        raise ValueError(f"layout non-pointer field {out.get('name', '')} sets pointer_role")
    elif out.get("pointee_layout") is not None:
        raise ValueError(f"layout non-pointer field {out.get('name', '')} sets pointee_layout")
    return out


def _normalize_type_layout(type_layout: Any, *, arg_name: str = "") -> Any:
    if not isinstance(type_layout, dict):
        return type_layout
    out = dict(type_layout)
    if "index" in out:
        raise ValueError("type_layout root must not set index")
    _preserve_materialization_facts(type_layout, out)
    root_index = arg_name
    fields = out.get("fields")
    if isinstance(fields, list):
        out["fields"] = [_normalize_layout_node(field, parent_index=root_index) for field in fields]
    element = out.get("element")
    if isinstance(element, dict):
        out["element"] = _normalize_layout_node(element, parent_index=root_index)
    return out


def _manifest_arg(arg: dict[str, Any], kernel_symbol: str) -> dict[str, Any]:
    missing = [key for key in _REQUIRED_ARG_FIELDS if arg.get(key) is None]
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(
            f"kernel {kernel_symbol} arg metadata missing required fields: {missing_fields}"
        )

    kind = _normalize_arg_kind(arg.get("kind"))
    if kind is None:
        raise ValueError(f"kernel {kernel_symbol} arg {arg['name']} missing kind")
    if kind not in _ARG_KINDS:
        raise ValueError(f"kernel {kernel_symbol} arg metadata has unsupported kind: {kind}")
    type_layout = (
        _normalize_type_layout(arg.get("type_layout"), arg_name=str(arg.get("name") or ""))
        if arg.get("type_layout") is not None
        else None
    )
    if kind == "opaque_val" and isinstance(type_layout, dict) and _layout_contains_pointer(type_layout):
        kind = "opaque_with_ptr"

    out: dict[str, Any] = {
        "index": arg["index"],
        "name": arg["name"],
        "type": arg["type"],
        "kind": kind,
        "size_bytes": arg["size_bytes"],
        "align_bytes": arg["align_bytes"],
    }
    pointer_role = arg.get("pointer_role")
    if kind == "pointer":
        if pointer_role is None:
            raise ValueError(f"kernel {kernel_symbol} pointer arg {arg['name']} missing pointer_role")
        if pointer_role not in _POINTER_ROLES:
            raise ValueError(
                f"kernel {kernel_symbol} arg {arg['name']} has unsupported pointer_role: {pointer_role}"
            )
        out["pointer_role"] = pointer_role
        if not isinstance(arg.get("pointee_layout"), dict):
            raise ValueError(f"kernel {kernel_symbol} pointer arg {arg['name']} missing pointee_layout")
        out["pointee_layout"] = _normalize_layout_node(
            arg["pointee_layout"],
            parent_index=str(arg.get("name") or ""),
        )
        if pointer_role == "derived_pointer":
            if arg.get("base_arg") is None:
                raise ValueError(
                    f"kernel {kernel_symbol} derived_pointer arg {arg['name']} missing base_arg"
                )
            if arg.get("offset_bytes") is None and arg.get("offset_arg") is None:
                raise ValueError(
                    f"kernel {kernel_symbol} derived_pointer arg {arg['name']} missing offset_bytes/offset_arg"
                )
            for field in _DERIVED_POINTER_FIELDS:
                if arg.get(field) is not None:
                    out[field] = arg[field]
        elif pointer_role == "external_device_pointer":
            if not arg.get("source"):
                raise ValueError(
                    f"kernel {kernel_symbol} external_device_pointer arg {arg['name']} missing source"
                )
            out["source"] = arg["source"]
    elif pointer_role is not None:
        raise ValueError(f"kernel {kernel_symbol} non-pointer arg {arg['name']} sets pointer_role")
    elif arg.get("pointee_layout") is not None:
        raise ValueError(f"kernel {kernel_symbol} non-pointer arg {arg['name']} sets pointee_layout")
    if arg.get("domain") is not None:
        out["domain"] = arg["domain"]
    if arg.get("type_info"):
        out["type_info"] = _normalize_type_info(arg["type_info"])
    _preserve_materialization_facts(arg, out)
    if type_layout is not None:
        out["type_layout"] = type_layout
    return out


def _preserve_materialization_facts(src: dict[str, Any], dst: dict[str, Any]) -> None:
    status = src.get("materialization_status")
    if isinstance(status, str) and status:
        dst["materialization_status"] = status
    reason_codes = src.get("materialization_reason_codes")
    if isinstance(reason_codes, list) and reason_codes:
        dst["materialization_reason_codes"] = [str(x) for x in reason_codes if isinstance(x, str) and x]
    blockers = src.get("materialization_blockers")
    if isinstance(blockers, list) and blockers:
        dst["materialization_blockers"] = [str(x) for x in blockers if isinstance(x, str) and x]


def _merge_unique_strings(dst: list[str], values: list[Any] | None) -> None:
    if not isinstance(values, list):
        return
    seen = set(dst)
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        dst.append(value)
        seen.add(value)


def _collect_materialization_facts(node: dict[str, Any], facts: dict[str, Any]) -> None:
    if node.get("materialization_status") == "unsafe":
        facts["materialization_status"] = "unsafe"
    _merge_unique_strings(
        facts.setdefault("materialization_reason_codes", []),
        node.get("materialization_reason_codes"),
    )
    _merge_unique_strings(
        facts.setdefault("materialization_blockers", []),
        node.get("materialization_blockers"),
    )
    for key in ("type_layout", "element", "pointee_layout"):
        child = node.get(key)
        if isinstance(child, dict):
            _collect_materialization_facts(child, facts)
    fields = node.get("fields")
    if isinstance(fields, list):
        for field in fields:
            if isinstance(field, dict):
                _collect_materialization_facts(field, facts)


def _preserve_kernel_materialization_summary(args: list[dict[str, Any]], dst: dict[str, Any]) -> None:
    facts: dict[str, Any] = {
        "materialization_reason_codes": [],
        "materialization_blockers": [],
    }
    for arg in args:
        _collect_materialization_facts(arg, facts)

    reason_codes = facts.get("materialization_reason_codes")
    blockers = facts.get("materialization_blockers")
    if facts.get("materialization_status") == "unsafe" or reason_codes or blockers:
        dst["materialization_status"] = "unsafe"
    if reason_codes:
        existing = dst.setdefault("materialization_reason_codes", [])
        _merge_unique_strings(existing, reason_codes)
    if blockers:
        existing = dst.setdefault("materialization_blockers", [])
        _merge_unique_strings(existing, blockers)


def write_manifest(
    kernel: dict[str, Any],
    emit_result: dict[str, Any],
    target_lib: str,
    toolchain_versions: dict[str, str],
) -> dict[str, str | None]:
    """Write manifest.json and metadata.json for a kernel.

    Uses emit_result status/failure_reason to set build_status.
    Failed/skipped kernels still get manifest+metadata with available hashes.

    Returns dict of output file hashes: {filename: "sha256:..." | None}.
    """
    kernel_dir = Path(emit_result["kernel_dir"])

    # -- manifest.json --
    kernel_entry: dict[str, Any] = {
        "symbol_name": kernel["symbol_name"],
        "display_name": kernel.get("display_name", kernel["symbol_name"]),
        "args": [_manifest_arg(a, kernel["symbol_name"]) for a in kernel["args"]],
    }
    constraints = kernel.get("constraints")
    if isinstance(constraints, list) and constraints:
        kernel_entry["constraints"] = constraints
    source_file = kernel.get("source_file")
    source_line = kernel.get("line")
    others: dict[str, Any] = {}
    if isinstance(source_file, str) and source_file:
        others["source_file"] = source_file
    if isinstance(source_line, int) and source_line >= 1:
        others["source_line"] = source_line
    variant_id = kernel.get("variant_id")
    if isinstance(variant_id, str) and variant_id:
        others["variant_id"] = variant_id
    type_shim_text = kernel.get("type_shim")
    type_shim_status = kernel.get("type_shim_status")
    type_shim_reason_codes = kernel.get("type_shim_reason_codes")
    type_shim_missing_dependencies = kernel.get("type_shim_missing_dependencies")
    type_shim_system_headers = kernel.get("type_shim_system_headers")
    type_shim_system_includes = kernel.get("type_shim_system_includes")
    _preserve_materialization_facts(kernel, others)
    _preserve_kernel_materialization_summary(kernel_entry["args"], others)
    if isinstance(type_shim_status, str) and type_shim_status:
        others["type_shim_status"] = type_shim_status
    if isinstance(type_shim_reason_codes, list) and type_shim_reason_codes:
        others["type_shim_reason_codes"] = type_shim_reason_codes
    if isinstance(type_shim_missing_dependencies, list) and type_shim_missing_dependencies:
        others["type_shim_missing_dependencies"] = type_shim_missing_dependencies
    if isinstance(type_shim_system_headers, list) and type_shim_system_headers:
        others["type_shim_system_headers"] = type_shim_system_headers
    if isinstance(type_shim_text, str) and type_shim_status != "unsupported":
        include_lines: list[str] = []
        if isinstance(type_shim_system_includes, list):
            include_lines.extend(str(x) for x in type_shim_system_includes if isinstance(x, str) and x)
        if include_lines:
            others["type_shim_system_includes"] = include_lines
        shim_path = kernel_dir / "type_shim.v1.cuh"
        shim_path.write_text(_prepend_includes_to_shim(type_shim_text, include_lines), encoding="utf-8")
        others["type_shim_header"] = shim_path.name
    if others:
        kernel_entry["others"] = others

    manifest = {
        "schema_version": 1,
        "kernels": [kernel_entry],
    }
    manifest_path = kernel_dir / "manifest.json"
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
    manifest_path.write_bytes(manifest_bytes)

    # -- metadata.json (conforms to docs/artifact-metadata.schema.json) --
    emit_status = emit_result.get("status", "built")
    failure_reason = emit_result.get("failure_reason")
    failure_detail = emit_result.get("failure_detail")
    bc_hash = emit_result.get("bc_hash")
    input_bc_hash = emit_result.get("input_bc_hash")
    ptx_hash = emit_result.get("ptx_hash")
    manifest_hash = _sha256_bytes(manifest_bytes)

    output_hashes: dict[str, str] = {"manifest.json": manifest_hash}
    if bc_hash is not None:
        output_hashes["kernel.bc"] = bc_hash
    if ptx_hash is not None:
        output_hashes["kernel.ptx"] = ptx_hash

    metadata_content = {
        "schema_version": 1,
        "target_lib": target_lib,
        "kernel_id": kernel["kernel_id"],
        "kernel_symbol": kernel["symbol_name"],
        "build_status": emit_status,
        "failure_reason": failure_reason,
        "failure_detail": failure_detail,
        "output_hashes": output_hashes,
        "toolchain_versions": toolchain_versions,
        "generated_at": _now_iso(),
    }
    # Keep schema-valid output for failed/skipped kernels where BC may be absent.
    if isinstance(input_bc_hash, str) and input_bc_hash:
        metadata_content["input_bc_hash"] = input_bc_hash

    metadata_path = kernel_dir / "metadata.json"
    metadata_bytes = json.dumps(metadata_content, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    metadata_path.write_bytes(metadata_bytes)
    metadata_hash = _sha256_bytes(metadata_bytes)

    result: dict[str, str | None] = {
        "kernel.bc": bc_hash,
        "manifest.json": manifest_hash,
        "metadata.json": metadata_hash,
    }
    if ptx_hash is not None:
        result["kernel.ptx"] = ptx_hash
    return result


def write_all_manifests(
    kernels: list[dict[str, Any]],
    emit_results: list[dict[str, Any]],
    target_lib: str,
    toolchain_versions: dict[str, str],
) -> list[dict[str, Any]]:
    """Write manifests for all kernels. Returns list of {kernel_id, hashes}."""
    results = []
    emit_map = {e["kernel_id"]: e for e in emit_results}
    for kernel in kernels:
        kid = kernel["kernel_id"]
        emit_result = emit_map[kid]
        hashes = write_manifest(kernel, emit_result, target_lib, toolchain_versions)
        results.append({
            "kernel_id": kid,
            "hashes": hashes,
            "status": emit_result.get("status", "built"),
            "failure_reason": emit_result.get("failure_reason"),
            "failure_stage": emit_result.get("failure_stage"),
            "failure_code": emit_result.get("failure_code"),
            "failure_detail": emit_result.get("failure_detail"),
        })
    return results
