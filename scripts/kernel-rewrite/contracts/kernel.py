"""Normalized Phase 2 kernel contract and support gating."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias


_POINTER_ROLES = {
    "payload_buffer",
    "derived_pointer",
    "external_device_pointer",
}

_ARG_KINDS = {"pointer", "scalar", "opaque_val", "opaque_with_ptr"}
RAPID_MAX_PAYLOAD_SLOTS = 32
_MATERIALIZATION_BLOCKING_REASONS = {
    "const_assignment_blocker",
    "materialization_unsafe",
    "non_public_base",
    "non_trivially_copyable",
    "private_data_field",
    "protected_data_field",
    "reference_field",
    "reference_type",
    "virtual_base",
    "virtual_method_or_vptr",
}
_QUALIFIER_RE = re.compile(r"\b(const|volatile|restrict|__restrict(?:__)?|constexpr)\b")
_PATH_TOKEN_RE = re.compile(r"[^0-9A-Za-z_]+")
_TOP_LEVEL_CV_RE = re.compile(r"^\s*(?:(?:const|volatile)\s+)+")


def _normalize_type_name(type_name: str | None) -> str:
    if not type_name:
        return ""
    normalized = _QUALIFIER_RE.sub("", type_name)
    normalized = normalized.replace(" &", "&").replace(" *", "*")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _strip_top_level_cv(type_name: str) -> str:
    return _TOP_LEVEL_CV_RE.sub("", type_name).strip() or type_name


def _format_loc(loc: dict[str, Any] | None) -> str | None:
    if not isinstance(loc, dict):
        return None
    file = loc.get("file")
    line = loc.get("line")
    column = loc.get("column")
    if not isinstance(file, str) or not file:
        return None
    if isinstance(line, int) and isinstance(column, int):
        return f"{file}:{line}:{column}"
    if isinstance(line, int):
        return f"{file}:{line}"
    return file


def _normalize_kind(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def _normalize_pointer_role(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def _path_token(raw: str) -> str:
    token = _PATH_TOKEN_RE.sub("_", raw).strip("_")
    return token or "unnamed"


def _make_index(parent: str, name: str) -> str:
    if name == "$element":
        return f"{parent}[]"
    return f"{parent}.{name}" if parent else name


def _make_c_ident(index: str) -> str:
    return _path_token(index)


def _format_node_support_reason(node: "KernelLayoutNode") -> str | None:
    if node.supported or not node.support_reason:
        return None
    if node.support_reason.startswith(f"{node.index}:") or node.support_reason.startswith(f"{node.index}."):
        return node.support_reason
    return f"{node.index}:{node.support_reason}"


def _materialization_reason(raw: dict[str, Any], *, index: str) -> str | None:
    if raw.get("materialization_status") != "unsafe":
        return None
    reason_codes = raw.get("materialization_reason_codes")
    reason_code = None
    if isinstance(reason_codes, list):
        reason_code = next(
            (
                str(code)
                for code in reason_codes
                if isinstance(code, str) and code in _MATERIALIZATION_BLOCKING_REASONS
            ),
            None,
        )
    if reason_code is None:
        return None
    reason_code = reason_code or "materialization_unsafe"
    return f"{index}:{reason_code}" if index else reason_code


def _decl_from_type_info(type_info: dict[str, Any]) -> dict[str, Any]:
    if isinstance(type_info.get("kind"), str):
        return type_info
    return {}


def _extract_type_info(raw: dict[str, Any]) -> tuple[str, str | None, str | None, str | None]:
    type_info = raw.get("type_info") if isinstance(raw.get("type_info"), dict) else {}
    decl = _decl_from_type_info(type_info)
    definition = decl.get("definition") if isinstance(decl.get("definition"), dict) else {}
    canonical_type = ""
    decl_kind = str(decl.get("kind")) if decl.get("kind") else None
    qualified_name = str(decl.get("qualified_name")) if decl.get("qualified_name") else None
    definition_loc = _format_loc(definition.get("loc") if isinstance(definition.get("loc"), dict) else None)
    return (canonical_type, decl_kind, qualified_name, definition_loc)


def _pointer_type_requires_shim(raw: dict[str, Any]) -> tuple[bool, str | None]:
    pointee = raw.get("pointee_layout")
    if not isinstance(pointee, dict):
        return (False, "pointer_pointee_layout_missing")
    pointee_type = _normalize_type_name(str(pointee.get("type", "")))
    if pointee_type == "void":
        return (False, None)
    pointee_kind = _normalize_kind(pointee.get("kind"))
    if pointee_kind in {"scalar", "pointer"}:
        return (False, None)
    if pointee_kind in {"opaque_val", "opaque_with_ptr"}:
        return (True, None)
    return (False, "pointer_pointee_layout_kind_missing")


def _classify_payload_buffer_pointer(
    raw: dict[str, Any],
    *,
    has_type_shim: bool,
    type_shim_failure_reason: str | None = None,
) -> tuple[bool, str | None]:
    requires_shim, type_reason = _pointer_type_requires_shim(raw)
    if type_reason is not None:
        return (False, type_reason)
    if requires_shim and not has_type_shim:
        return (False, type_shim_failure_reason or "type_shim_missing")
    return (True, None)


def _classify_derived_pointer(raw: dict[str, Any]) -> tuple[bool, str | None]:
    del raw
    # TODO: implement relation-aware decode/invoke support for base_arg/base_path + offset.
    return (False, "pointer_role_not_supported")


def _classify_external_device_pointer(raw: dict[str, Any]) -> tuple[bool, str | None]:
    del raw
    # TODO: implement runtime metadata injection for harness-provided device pointers.
    return (False, "pointer_role_not_supported")


def _classify_pointer_node(
    raw: dict[str, Any],
    *,
    has_type_shim: bool,
    type_shim_failure_reason: str | None = None,
) -> tuple[str | None, bool, str | None]:
    pointer_role = _normalize_pointer_role(raw.get("pointer_role"))
    if pointer_role is None:
        raise ValueError("pointer_role_missing")
    if pointer_role not in _POINTER_ROLES:
        raise ValueError("pointer_role_not_supported")
    if not isinstance(raw.get("pointee_layout"), dict):
        raise ValueError("pointer_pointee_layout_missing")
    if pointer_role == "payload_buffer":
        supported, reason = _classify_payload_buffer_pointer(
            raw,
            has_type_shim=has_type_shim,
            type_shim_failure_reason=type_shim_failure_reason,
        )
    elif pointer_role == "derived_pointer":
        supported, reason = _classify_derived_pointer(raw)
    elif pointer_role == "external_device_pointer":
        supported, reason = _classify_external_device_pointer(raw)
    else:
        raise ValueError("pointer_role_not_supported")
    return (pointer_role, supported, reason)


def _pointee_size_bytes(raw: dict[str, Any]) -> int | None:
    pointee = raw.get("pointee_layout")
    if not isinstance(pointee, dict):
        return None
    size = pointee.get("size_bytes")
    if isinstance(size, int) and size > 0:
        return size
    return None


def _pointee_align_bytes(raw: dict[str, Any]) -> int | None:
    pointee = raw.get("pointee_layout")
    if not isinstance(pointee, dict):
        return None
    alignment = pointee.get("align_bytes")
    if isinstance(alignment, int) and alignment > 0:
        return alignment
    return None


@dataclass(frozen=True)
class KernelLayoutNode:
    name: str
    index: str
    c_ident: str
    type_name: str
    canonical_type: str
    size_bytes: int
    align_bytes: int
    kind: str
    pointer_role: str | None = None
    layout_status: str | None = None
    decl_kind: str | None = None
    fields: tuple["KernelLayoutNode", ...] = field(default_factory=tuple)
    element: "KernelLayoutNode | None" = None
    element_count: int | None = None
    qualified_name: str | None = None
    definition_loc: str | None = None
    pointee_size_bytes: int | None = None
    pointee_align_bytes: int | None = None
    supported: bool = True
    support_reason: str | None = None

    @property
    def codegen_type(self) -> str:
        if self.kind in {"opaque_val", "opaque_with_ptr"}:
            return _normalize_type_name(self.canonical_type or self.type_name) or self.type_name
        return self.type_name

    @property
    def decoded_storage_type(self) -> str:
        if self.kind == "pointer":
            return self.codegen_type
        return _strip_top_level_cv(self.codegen_type)

    @property
    def is_payload_buffer_pointer(self) -> bool:
        return self.kind == "pointer" and self.pointer_role == "payload_buffer"

    @property
    def payload_align_bytes(self) -> int:
        return max(1, self.align_bytes, self.pointee_align_bytes or 1)

    @property
    def has_children(self) -> bool:
        return bool(self.fields) or self.element is not None

    def iter_leaves(self) -> list["KernelLayoutNode"]:
        if self.kind in {"opaque_val", "opaque_with_ptr"} and self.fields:
            leaves: list[KernelLayoutNode] = []
            for child in self.fields:
                leaves.extend(child.iter_leaves())
            return leaves
        if self.kind in {"opaque_val", "opaque_with_ptr"} and self.element is not None and self.element_count is not None:
            leaves = []
            for _ in range(self.element_count):
                leaves.extend(self.element.iter_leaves())
            return leaves
        return [self]

    def to_plan_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "index": self.index,
            "type": self.codegen_type,
            "size_bytes": self.size_bytes,
            "align_bytes": self.align_bytes,
            "kind": self.kind,
            "supported": self.supported,
        }
        if self.pointer_role:
            data["pointer_role"] = self.pointer_role
        if self.support_reason:
            data["support_reason"] = self.support_reason
        if self.qualified_name:
            data["qualified_name"] = self.qualified_name
        if self.definition_loc:
            data["definition_loc"] = self.definition_loc
        if self.decl_kind:
            data["decl_kind"] = self.decl_kind
        if self.fields:
            data["fields"] = [child.to_plan_dict() for child in self.fields]
        if self.element is not None:
            data["element_count"] = self.element_count
            data["element"] = self.element.to_plan_dict()
        return data


@dataclass(frozen=True)
class KernelArgContract:
    index: int
    name: str
    type_name: str
    canonical_type: str
    size_bytes: int
    align_bytes: int
    kind: str
    pointer_role: str | None
    supported: bool
    support_reason: str | None
    qualified_name: str | None
    definition_loc: str | None
    decl_kind: str | None = None
    pointee_size_bytes: int | None = None
    pointee_align_bytes: int | None = None
    layout: tuple[KernelLayoutNode, ...] = field(default_factory=tuple)

    @property
    def path(self) -> str:
        return f"args[{self.index}]"

    @property
    def codegen_type(self) -> str:
        if self.kind in {"opaque_val", "opaque_with_ptr"}:
            return _normalize_type_name(self.canonical_type or self.type_name) or self.type_name
        return self.type_name

    @property
    def decoded_storage_type(self) -> str:
        if self.kind == "pointer":
            return self.codegen_type
        return _strip_top_level_cv(self.codegen_type)

    @property
    def is_payload_buffer_pointer(self) -> bool:
        return self.kind == "pointer" and self.pointer_role == "payload_buffer"

    @property
    def payload_align_bytes(self) -> int:
        return max(1, self.align_bytes, self.pointee_align_bytes or 1)

    @property
    def has_field_materialization(self) -> bool:
        return self.kind == "opaque_with_ptr" and bool(self.layout)

    def to_plan_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "type": self.codegen_type,
            "size_bytes": self.size_bytes,
            "align_bytes": self.align_bytes,
            "kind": self.kind,
            "supported": self.supported,
        }
        if self.pointer_role:
            data["pointer_role"] = self.pointer_role
        if self.support_reason:
            data["support_reason"] = self.support_reason
        if self.qualified_name:
            data["qualified_name"] = self.qualified_name
        if self.definition_loc:
            data["definition_loc"] = self.definition_loc
        if self.decl_kind:
            data["decl_kind"] = self.decl_kind
        if self.layout:
            data["layout"] = [node.to_plan_dict() for node in self.layout]
        return data


@dataclass(frozen=True)
class KernelPayloadSlot:
    slot: int
    arg_index: int
    field_path: tuple[str, ...]
    decoded_expr: str
    elem_size: int
    byte_offset: int | None

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "arg_index": self.arg_index,
            "field_path": list(self.field_path),
            "decoded_expr": self.decoded_expr,
            "elem_size": self.elem_size,
            "byte_offset": self.byte_offset,
        }


def _is_transparent_anonymous_layout(node: KernelLayoutNode) -> bool:
    return (
        node.kind == "opaque_with_ptr"
        and node.name.startswith("anon")
        and "anonymous" in node.type_name.lower()
    )


def _append_layout_payload_slots(
    slots: list[KernelPayloadSlot],
    *,
    arg_index: int,
    node: KernelLayoutNode,
    field_path: tuple[str, ...],
    decoded_expr: str,
    byte_offset: int,
) -> None:
    if node.is_payload_buffer_pointer:
        slots.append(
            KernelPayloadSlot(
                slot=len(slots),
                arg_index=arg_index,
                field_path=field_path,
                decoded_expr=decoded_expr,
                elem_size=node.pointee_size_bytes or 1,
                byte_offset=byte_offset,
            )
        )
        return
    if node.element is not None and node.element_count is not None:
        element_stride = _align_up(node.element.size_bytes, node.element.align_bytes)
        for index in range(node.element_count):
            _append_layout_payload_slots(
                slots,
                arg_index=arg_index,
                node=node.element,
                field_path=(*field_path, str(index)),
                decoded_expr=f"{decoded_expr}[{index}]",
                byte_offset=byte_offset + index * element_stride,
            )
        return
    child_offset = 0
    for child in node.fields:
        child_offset = _align_up(child_offset, child.align_bytes)
        child_expr = (
            decoded_expr
            if _is_transparent_anonymous_layout(child)
            else f"{decoded_expr}.{child.name}"
        )
        _append_layout_payload_slots(
            slots,
            arg_index=arg_index,
            node=child,
            field_path=(*field_path, child.name),
            decoded_expr=child_expr,
            byte_offset=byte_offset + child_offset,
        )
        child_offset += child.size_bytes


def _align_up(value: int, alignment: int) -> int:
    effective_alignment = max(1, alignment)
    return ((value + effective_alignment - 1) // effective_alignment) * effective_alignment


def _kernel_payload_slots(args: list[KernelArgContract]) -> list[KernelPayloadSlot]:
    slots: list[KernelPayloadSlot] = []
    for arg in args:
        if arg.is_payload_buffer_pointer:
            slots.append(
                KernelPayloadSlot(
                    slot=len(slots),
                    arg_index=arg.index,
                    field_path=(),
                    decoded_expr=f"decoded.{arg.name}",
                    elem_size=arg.pointee_size_bytes or 1,
                    byte_offset=None,
                )
            )
            continue
        node_offset = 0
        for node in arg.layout:
            node_offset = _align_up(node_offset, node.align_bytes)
            node_expr = (
                f"decoded.{arg.name}"
                if _is_transparent_anonymous_layout(node)
                else f"decoded.{arg.name}.{node.name}"
            )
            _append_layout_payload_slots(
                slots,
                arg_index=arg.index,
                node=node,
                field_path=(node.name,),
                decoded_expr=node_expr,
                byte_offset=node_offset,
            )
            node_offset += node.size_bytes
    return slots


def _parse_layout_node(
    raw: dict[str, Any],
    *,
    index: str,
    has_type_shim: bool,
    type_shim_failure_reason: str | None = None,
) -> KernelLayoutNode:
    raw_kind = _normalize_kind(raw.get("kind"))
    if raw_kind not in _ARG_KINDS:
        raise ValueError(f"manifest layout node {index} has unsupported kind: {raw.get('kind') or 'missing'}")

    type_name = str(raw.get("type", ""))
    canonical_type, decl_kind, qualified_name, definition_loc = _extract_type_info(raw)
    pointer_role = _normalize_pointer_role(raw.get("pointer_role"))
    supported = True
    reason: str | None = None
    materialization_reason = _materialization_reason(raw, index=index)

    if raw_kind == "pointer":
        pointer_role, supported, reason = _classify_pointer_node(
            raw,
            has_type_shim=has_type_shim,
            type_shim_failure_reason=type_shim_failure_reason,
        )
    elif pointer_role is not None:
        supported = False
        reason = "pointer_role_not_supported"
    elif raw_kind == "scalar" and int(raw.get("size_bytes", 0) or 0) not in {1, 2, 4, 8}:
        supported = False
        reason = "scalar_size_unsupported"
    elif raw_kind == "opaque_with_ptr":
        layout_status = raw.get("layout_status")
        if decl_kind == "union":
            supported = False
            reason = "union_not_supported"
        elif layout_status in {"partial", "opaque"}:
            supported = False
            reason = "opaque_with_ptr_layout_incomplete"

    fields: tuple[KernelLayoutNode, ...] = ()
    raw_fields = raw.get("fields")
    if isinstance(raw_fields, list):
        parsed_fields = []
        for child in raw_fields:
            if not isinstance(child, dict):
                raise ValueError("manifest layout field entry invalid")
            child_name = str(child.get("name", "unnamed"))
            if child_name == "$element":
                child_name = f"{raw.get('name', 'element')}_element_{len(parsed_fields)}"
            parsed_fields.append(
                _parse_layout_node(
                    child,
                    index=str(child.get("index") or _make_index(index, child_name)),
                    has_type_shim=has_type_shim,
                    type_shim_failure_reason=type_shim_failure_reason,
                )
            )
        fields = tuple(parsed_fields)

    element = None
    raw_element = raw.get("element")
    if isinstance(raw_element, dict):
        element = _parse_layout_node(
            raw_element,
            index=str(raw_element.get("index") or _make_index(index, "$element")),
            has_type_shim=has_type_shim,
            type_shim_failure_reason=type_shim_failure_reason,
        )

    child_reason = None
    for child in fields:
        candidate = _format_node_support_reason(child)
        if candidate is not None:
            child_reason = candidate
            break
    if child_reason is None and element is not None and not element.supported:
        child_reason = _format_node_support_reason(element)
    if supported and child_reason is not None:
        supported = False
        reason = child_reason
    if materialization_reason is not None:
        supported = False
        reason = materialization_reason

    return KernelLayoutNode(
        name=str(raw.get("name", "")),
        index=index,
        c_ident=_make_c_ident(index),
        type_name=type_name,
        canonical_type=canonical_type,
        size_bytes=int(raw.get("size_bytes", 0) or 0),
        align_bytes=int(raw.get("align_bytes", 1) or 1),
        kind=raw_kind,
        pointer_role=pointer_role,
        layout_status=str(raw.get("layout_status")) if raw.get("layout_status") else None,
        decl_kind=decl_kind,
        fields=fields,
        element=element,
        element_count=int(raw.get("element_count")) if isinstance(raw.get("element_count"), int) else None,
        qualified_name=qualified_name,
        definition_loc=definition_loc,
        pointee_size_bytes=_pointee_size_bytes(raw),
        pointee_align_bytes=_pointee_align_bytes(raw),
        supported=supported,
        support_reason=reason,
    )


def _parse_arg_layout(
    raw: dict[str, Any],
    *,
    arg_index: int,
    has_type_shim: bool,
    type_shim_failure_reason: str | None = None,
) -> tuple[KernelLayoutNode, ...]:
    type_layout = raw.get("type_layout")
    if not isinstance(type_layout, dict):
        return ()
    if "index" in type_layout:
        raise ValueError("type_layout_root_index_not_supported")
    parsed: list[KernelLayoutNode] = []
    root_index = str(raw.get("name") or f"args[{arg_index}]")
    fields = type_layout.get("fields")
    if isinstance(fields, list):
        for child in fields:
            if not isinstance(child, dict):
                raise ValueError("manifest layout field entry invalid")
            child_name = str(child.get("name", "unnamed"))
            if child_name == "$element":
                child_name = f"element_{len(parsed)}"
            parsed.append(
                _parse_layout_node(
                    child,
                    index=str(child.get("index") or _make_index(root_index, child_name)),
                    has_type_shim=has_type_shim,
                    type_shim_failure_reason=type_shim_failure_reason,
                )
            )
    element = type_layout.get("element")
    if isinstance(element, dict):
        parsed.append(
            _parse_layout_node(
                element,
                index=str(element.get("index") or _make_index(root_index, "$element")),
                has_type_shim=has_type_shim,
                type_shim_failure_reason=type_shim_failure_reason,
            )
        )
    return tuple(parsed)


ConstraintPredicateOp: TypeAlias = Literal["<=", "<", "==", "!=", ">=", ">"]
BinaryExprOp: TypeAlias = Literal["+", "-", "*", "/"]


class ConstraintExpr:
    pass


@dataclass(frozen=True)
class ArgValueExpr(ConstraintExpr):
    arg_index: int
    field_path: tuple[str, ...] = ()

    def to_plan_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": "arg_value", "arg_index": self.arg_index}
        if self.field_path:
            data["field_path"] = list(self.field_path)
        return data


@dataclass(frozen=True)
class PayloadLenExpr(ConstraintExpr):
    arg_index: int
    field_path: tuple[str, ...] = ()

    def to_plan_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": "payload_len", "arg_index": self.arg_index}
        if self.field_path:
            data["field_path"] = list(self.field_path)
        return data


@dataclass(frozen=True)
class ConstExpr(ConstraintExpr):
    value: int

    def to_plan_dict(self) -> dict[str, Any]:
        return {"kind": "const", "value": self.value}


@dataclass(frozen=True)
class BinaryExpr(ConstraintExpr):
    op: BinaryExprOp
    lhs: ConstraintExpr
    rhs: ConstraintExpr

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "kind": "binary",
            "op": self.op,
            "lhs": _constraint_expr_to_plan_dict(self.lhs),
            "rhs": _constraint_expr_to_plan_dict(self.rhs),
        }


def _constraint_expr_to_plan_dict(expr: ConstraintExpr) -> dict[str, Any]:
    if isinstance(expr, (ArgValueExpr, PayloadLenExpr, ConstExpr, BinaryExpr)):
        return expr.to_plan_dict()
    raise TypeError(f"unsupported constraint expr: {expr!r}")


@dataclass(frozen=True)
class KernelConstraintPredicate:
    lhs: ConstraintExpr
    op: ConstraintPredicateOp
    rhs: ConstraintExpr

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "lhs": _constraint_expr_to_plan_dict(self.lhs),
            "op": self.op,
            "rhs": _constraint_expr_to_plan_dict(self.rhs),
        }


@dataclass(frozen=True)
class ConstraintParseContext:
    args: list[KernelArgContract]
    args_by_index: dict[int, KernelArgContract]

    @classmethod
    def from_args(cls, args: list[KernelArgContract]) -> "ConstraintParseContext":
        return cls(args=args, args_by_index={arg.index: arg for arg in args})

    def require_arg(self, index: int) -> KernelArgContract:
        arg = self.args_by_index.get(index)
        if arg is None:
            raise ValueError("constraint_arg_missing")
        return arg

    def require_scalar(self, index: int) -> KernelArgContract:
        arg = self.require_arg(index)
        if arg.kind != "scalar":
            raise ValueError("constraint_scalar_arg_not_scalar")
        return arg

    def require_scalar_ref(self, index: int, path: tuple[str, ...] = ()) -> ArgValueExpr:
        arg = self.require_arg(index)
        if not path:
            if arg.kind != "scalar":
                raise ValueError("constraint_scalar_arg_not_scalar")
            return ArgValueExpr(arg.index)
        node = _find_layout_path(arg.layout, path)
        if node is None:
            raise ValueError("constraint_field_path_not_found")
        if node.kind != "scalar":
            raise ValueError("constraint_scalar_arg_not_scalar")
        return ArgValueExpr(arg.index, path)

    def require_payload_buffer(self, index: int) -> KernelArgContract:
        arg = self.require_arg(index)
        if not arg.is_payload_buffer_pointer:
            raise ValueError("constraint_buffer_arg_not_payload_buffer")
        return arg

    def require_payload_buffer_ref(self, index: int, path: tuple[str, ...] = ()) -> PayloadLenExpr:
        arg = self.require_arg(index)
        if not path:
            if not arg.is_payload_buffer_pointer:
                raise ValueError("constraint_buffer_arg_not_payload_buffer")
            return PayloadLenExpr(arg.index)
        node = _find_layout_path(arg.layout, path)
        if node is None:
            raise ValueError("constraint_field_path_not_found")
        if not node.is_payload_buffer_pointer:
            raise ValueError("constraint_buffer_arg_not_payload_buffer")
        return PayloadLenExpr(arg.index, path)

    def require_payload_buffer_node(self, index: int, path: tuple[str, ...] = ()) -> KernelArgContract | KernelLayoutNode:
        arg = self.require_arg(index)
        if not path:
            if not arg.is_payload_buffer_pointer:
                raise ValueError("constraint_buffer_arg_not_payload_buffer")
            return arg
        node = _find_layout_path(arg.layout, path)
        if node is None:
            raise ValueError("constraint_field_path_not_found")
        if not node.is_payload_buffer_pointer:
            raise ValueError("constraint_buffer_arg_not_payload_buffer")
        return node


@dataclass(frozen=True)
class KernelContract:
    kernel_id: str
    display_name: str
    input_symbol: str
    entry_symbol: str
    args: list[KernelArgContract]
    type_shim_header: str
    type_shim_status: str
    type_shim_reason_codes: list[str]
    type_shim_missing_dependencies: list[str]
    type_shim_include_dirs: list[str]
    vconfig_requested: bool
    constraint: list[KernelConstraintPredicate]
    supported: bool
    support_reason: str | None

    @property
    def needs_shim(self) -> bool:
        return bool(self.type_shim_header)

    @property
    def payload_slots(self) -> list[KernelPayloadSlot]:
        return _kernel_payload_slots(self.args)

    @property
    def feedback_memory_enabled(self) -> bool:
        return len(self.payload_slots) <= RAPID_MAX_PAYLOAD_SLOTS

    @property
    def feedback_payload_slots(self) -> list[KernelPayloadSlot]:
        if not self.feedback_memory_enabled:
            return []
        return self.payload_slots

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            "input_symbol": self.input_symbol,
            "entry_symbol": self.entry_symbol,
            "entry_abi_version": 1,
            "entry_context": "rapid_kernel_context_v1",
            "vconfig_requested": self.vconfig_requested,
            "payload_args": [arg.to_plan_dict() for arg in self.args],
            "feedback_memory_enabled": self.feedback_memory_enabled,
            "feedback_payload_slot_count": len(self.payload_slots),
            "feedback_payload_slots": [
                slot.to_plan_dict() for slot in self.feedback_payload_slots
            ],
            "constraint": [predicate.to_plan_dict() for predicate in self.constraint],
        }


def _parse_constraints(raw_constraints: Any, args: list[KernelArgContract]) -> list[KernelConstraintPredicate]:
    if raw_constraints is None:
        return []
    if not isinstance(raw_constraints, list):
        raise ValueError("manifest constraints invalid")
    predicates: list[KernelConstraintPredicate] = []
    ctx = ConstraintParseContext.from_args(args)
    for raw in raw_constraints:
        if not isinstance(raw, dict):
            raise ValueError("manifest constraint entry invalid")
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("constraint_kind_missing")
        parser = _CONSTRAINT_PARSERS.get(kind)
        if parser is None:
            raise ValueError("constraint_not_supported")
        predicates.extend(parser(raw, ctx))
    return predicates


def _require_int_field(raw: dict[str, Any], field: str) -> int:
    value = raw.get(field)
    if not isinstance(value, int):
        raise ValueError("constraint_arg_invalid")
    return value


def _require_op_field(raw: dict[str, Any]) -> ConstraintPredicateOp:
    op = raw.get("op")
    if op not in {"<=", "<", "==", "!=", ">=", ">"}:
        raise ValueError("constraint_op_invalid")
    return op


def _read_field_path(raw: dict[str, Any], field: str) -> tuple[str, ...]:
    value = raw.get(field)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(part, str) and part for part in value):
        raise ValueError("constraint_field_path_invalid")
    return tuple(value)


def _include_dir_from_include_stmt(include_stmt: str, header_paths: list[str]) -> str | None:
    match = re.search(r"[<\"]([^>\"]+)[>\"]", include_stmt)
    if not match:
        return None
    suffix = match.group(1)
    suffix_parts = Path(suffix).parts
    for header in header_paths:
        path = Path(header)
        try:
            if path.as_posix().endswith("/" + suffix) or path.as_posix() == suffix:
                root = path
                for _ in suffix_parts:
                    root = root.parent
                return str(root)
        except OSError:
            continue
    return None


def _derive_type_shim_include_dirs(others: dict[str, Any] | None) -> list[str]:
    if not isinstance(others, dict):
        return []
    raw_headers = others.get("type_shim_system_headers")
    raw_includes = others.get("type_shim_system_includes")
    headers = [str(x) for x in raw_headers if isinstance(x, str) and x] if isinstance(raw_headers, list) else []
    includes = [str(x) for x in raw_includes if isinstance(x, str) and x] if isinstance(raw_includes, list) else []

    dirs: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path or path in seen:
            return
        seen.add(path)
        dirs.append(path)

    for include_stmt in includes:
        add(_include_dir_from_include_stmt(include_stmt, headers))
    return dirs


def _find_layout_path(nodes: tuple[KernelLayoutNode, ...], path: tuple[str, ...]) -> KernelLayoutNode | None:
    if not path:
        return None
    current_nodes = nodes
    current: KernelLayoutNode | None = None
    for part in path:
        matched = next((node for node in current_nodes if node.name == part), None)
        if matched is None:
            if current is None or current.element is None:
                return None
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= (current.element_count or 0):
                return None
            matched = current.element
        current = matched
        current_nodes = current.fields
    return current


def _parse_scalar_le_buffer_len(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    scalar_arg = _require_int_field(raw, "scalar_arg")
    buffer_arg = _require_int_field(raw, "buffer_arg")
    scalar_path = _read_field_path(raw, "scalar_path")
    buffer_path = _read_field_path(raw, "buffer_path")
    unit = raw.get("unit")
    if unit not in {"bytes", "elements"}:
        raise ValueError("constraint_unit_not_supported")

    scalar_expr = ctx.require_scalar_ref(scalar_arg, scalar_path)
    buffer_node = ctx.require_payload_buffer_node(buffer_arg, buffer_path)
    buffer_expr = ctx.require_payload_buffer_ref(buffer_arg, buffer_path)
    lhs: ConstraintExpr = scalar_expr
    if unit == "elements":
        elem_size = getattr(buffer_node, "pointee_size_bytes", None)
        if not isinstance(elem_size, int) or elem_size <= 0:
            raise ValueError("constraint_elem_size_invalid")
        lhs = BinaryExpr("*", scalar_expr, ConstExpr(elem_size))

    return [
        KernelConstraintPredicate(
            lhs=lhs,
            op="<=",
            rhs=buffer_expr,
        )
    ]


def _parse_count_fits_buffer(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    count_arg = _require_int_field(raw, "count_arg")
    buffer_arg = _require_int_field(raw, "buffer_arg")
    count_path = _read_field_path(raw, "count_path")
    buffer_path = _read_field_path(raw, "buffer_path")
    elem_size_bytes = _require_int_field(raw, "elem_size_bytes")
    if elem_size_bytes <= 0:
        raise ValueError("constraint_elem_size_invalid")

    count_expr = ctx.require_scalar_ref(count_arg, count_path)
    buffer_expr = ctx.require_payload_buffer_ref(buffer_arg, buffer_path)

    return [
        KernelConstraintPredicate(
            lhs=BinaryExpr("*", count_expr, ConstExpr(elem_size_bytes)),
            op="<=",
            rhs=buffer_expr,
        )
    ]


def _parse_buffer_elements_lt_scalar(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    buffer_arg = _require_int_field(raw, "buffer_arg")
    scalar_arg = _require_int_field(raw, "scalar_arg")
    buffer_path = _read_field_path(raw, "buffer_path")
    scalar_path = _read_field_path(raw, "scalar_path")
    ctx.require_payload_buffer_ref(buffer_arg, buffer_path)
    ctx.require_scalar_ref(scalar_arg, scalar_path)
    elem_size_bytes = raw.get("elem_size_bytes")
    if elem_size_bytes is not None and (
        not isinstance(elem_size_bytes, int) or elem_size_bytes <= 0
    ):
        raise ValueError("constraint_elem_size_invalid")

    # This is a fuzzer-side value-domain constraint over buffer contents. Phase 2
    # only validates that the referenced payload/scalar arguments exist; generated
    # decode code currently has no loop IR for per-element payload predicates.
    return []


def _parse_scalar_le_logical_block_dim(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    scalar_arg = _require_int_field(raw, "scalar_arg")
    ctx.require_scalar_ref(scalar_arg, None)
    if raw.get("dimension") not in {"x", "y", "z"}:
        raise ValueError("constraint_logical_block_dimension_invalid")

    # The fuzzer repairs this relation after normalizing the envelope VConfig.
    # Generated payload decode code has no standalone launch-dimension value.
    return []


def _parse_scalar_compare_const(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    scalar_arg = _require_int_field(raw, "scalar_arg")
    scalar_path = _read_field_path(raw, "scalar_path")
    op = _require_op_field(raw)
    value = _require_int_field(raw, "value")

    return [
        KernelConstraintPredicate(
            lhs=ctx.require_scalar_ref(scalar_arg, scalar_path),
            op=op,
            rhs=ConstExpr(value),
        )
    ]


def _parse_scalar_compare_scalar(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    lhs_arg = _require_int_field(raw, "lhs_arg")
    rhs_arg = _require_int_field(raw, "rhs_arg")
    lhs_path = _read_field_path(raw, "lhs_path")
    rhs_path = _read_field_path(raw, "rhs_path")
    op = _require_op_field(raw)

    return [
        KernelConstraintPredicate(
            lhs=ctx.require_scalar_ref(lhs_arg, lhs_path),
            op=op,
            rhs=ctx.require_scalar_ref(rhs_arg, rhs_path),
        )
    ]


def _parse_scalar_product_le_const(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    lhs_arg = _require_int_field(raw, "lhs_arg")
    rhs_arg = _require_int_field(raw, "rhs_arg")
    repair_arg = _require_int_field(raw, "repair_arg")
    lhs_path = _read_field_path(raw, "lhs_path")
    rhs_path = _read_field_path(raw, "rhs_path")
    repair_path = _read_field_path(raw, "repair_path")
    value = _require_int_field(raw, "value")
    if value < 0:
        raise ValueError("constraint_value_invalid")

    lhs = ctx.require_scalar_ref(lhs_arg, lhs_path)
    rhs = ctx.require_scalar_ref(rhs_arg, rhs_path)
    repair_ref = (repair_arg, repair_path)
    if repair_ref not in {(lhs_arg, lhs_path), (rhs_arg, rhs_path)}:
        raise ValueError("constraint_repair_target_invalid")

    return [
        KernelConstraintPredicate(
            lhs=BinaryExpr("*", lhs, rhs),
            op="<=",
            rhs=ConstExpr(value),
        )
    ]


def _parse_constraint_expr(
    raw: Any,
    ctx: ConstraintParseContext,
    *,
    depth: int = 0,
) -> ConstraintExpr:
    if depth > 32:
        raise ValueError("constraint_expr_too_deep")
    if not isinstance(raw, dict):
        raise ValueError("constraint_expr_invalid")
    kind = raw.get("kind")
    if kind == "arg_value":
        arg = _require_int_field(raw, "arg")
        path = _read_field_path(raw, "path")
        return ctx.require_scalar_ref(arg, path)
    if kind == "payload_len":
        arg = _require_int_field(raw, "arg")
        path = _read_field_path(raw, "path")
        return ctx.require_payload_buffer_ref(arg, path)
    if kind == "const":
        return ConstExpr(_require_int_field(raw, "value"))
    if kind == "binary":
        op = raw.get("op")
        if op not in {"+", "-", "*", "/"}:
            raise ValueError("constraint_expr_op_invalid")
        return BinaryExpr(
            op,
            _parse_constraint_expr(raw.get("lhs"), ctx, depth=depth + 1),
            _parse_constraint_expr(raw.get("rhs"), ctx, depth=depth + 1),
        )
    raise ValueError("constraint_expr_not_supported")


def _parse_expression_compare(
    raw: dict[str, Any],
    ctx: ConstraintParseContext,
) -> list[KernelConstraintPredicate]:
    op = _require_op_field(raw)
    repair = raw.get("repair")
    if not isinstance(repair, dict) or repair.get("kind") != "resize_payload":
        raise ValueError("constraint_repair_not_supported")
    repair_arg = _require_int_field(repair, "arg")
    repair_path = _read_field_path(repair, "path")
    ctx.require_payload_buffer_ref(repair_arg, repair_path)
    return [
        KernelConstraintPredicate(
            lhs=_parse_constraint_expr(raw.get("lhs"), ctx),
            op=op,
            rhs=_parse_constraint_expr(raw.get("rhs"), ctx),
        )
    ]


ConstraintParser = Callable[[dict[str, Any], ConstraintParseContext], list[KernelConstraintPredicate]]


_CONSTRAINT_PARSERS: dict[str, ConstraintParser] = {
    "scalar_le_logical_block_dim": _parse_scalar_le_logical_block_dim,
    "scalar_eq_logical_block_dim": _parse_scalar_le_logical_block_dim,
    "scalar_le_buffer_len": _parse_scalar_le_buffer_len,
    "scalar_compare_const": _parse_scalar_compare_const,
    "scalar_compare_scalar": _parse_scalar_compare_scalar,
    "scalar_product_le_const": _parse_scalar_product_le_const,
    "count_fits_buffer": _parse_count_fits_buffer,
    "buffer_elements_lt_scalar": _parse_buffer_elements_lt_scalar,
    "expression_compare": _parse_expression_compare,
}


def build_kernel_contract(
    *,
    kernel_id: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    metadata: dict[str, Any],
) -> KernelContract:
    del manifest_path  # Phase 2 now relies on Phase 1-persisted shim metadata.

    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or not kernels or not isinstance(kernels[0], dict):
        raise ValueError("manifest kernels missing")
    if len(kernels) != 1:
        raise ValueError("manifest must contain exactly one kernel")

    kernel_entry = kernels[0]
    launch_policy = kernel_entry.get("launch_policy")
    if launch_policy is None:
        vconfig_requested = True
    elif not isinstance(launch_policy, dict):
        raise ValueError("manifest launch_policy invalid")
    else:
        vconfig_requested = launch_policy.get("vconfig_reserved", True)
        if not isinstance(vconfig_requested, bool):
            raise ValueError("manifest launch_policy.vconfig_reserved invalid")
    others = kernel_entry.get("others") if isinstance(kernel_entry, dict) else None
    input_symbol = metadata.get("kernel_symbol") or kernel_entry.get("symbol_name")
    if not isinstance(input_symbol, str) or not input_symbol:
        raise ValueError("kernel symbol missing")

    raw_args = kernel_entry.get("args", [])
    if not isinstance(raw_args, list):
        raise ValueError("manifest args invalid")

    type_shim_header = ""
    type_shim_status = "ok"
    type_shim_reason_codes: list[str] = []
    type_shim_missing_dependencies: list[str] = []
    type_shim_include_dirs = _derive_type_shim_include_dirs(others)
    if isinstance(others, dict) and isinstance(others.get("type_shim_header"), str):
        type_shim_header = others["type_shim_header"]
    if isinstance(others, dict) and isinstance(others.get("type_shim_status"), str):
        type_shim_status = others["type_shim_status"]
    if isinstance(others, dict) and isinstance(others.get("type_shim_reason_codes"), list):
        type_shim_reason_codes = [str(x) for x in others["type_shim_reason_codes"] if isinstance(x, str)]
    if isinstance(others, dict) and isinstance(others.get("type_shim_missing_dependencies"), list):
        type_shim_missing_dependencies = [str(x) for x in others["type_shim_missing_dependencies"] if isinstance(x, str)]

    has_type_shim = bool(type_shim_header) and type_shim_status == "ok"
    type_shim_failure_reason = None if type_shim_status == "ok" else (
        type_shim_reason_codes[0] if type_shim_reason_codes else "type_shim_missing"
    )
    args: list[KernelArgContract] = []
    first_unsupported: str | None = None
    for raw in raw_args:
        if not isinstance(raw, dict):
            raise ValueError("manifest arg entry invalid")
        index = int(raw.get("index", len(args)))
        type_name = str(raw.get("type", ""))
        canonical_type, decl_kind, qualified_name, definition_loc = _extract_type_info(raw)
        raw_kind = _normalize_kind(raw.get("kind"))
        if raw_kind not in _ARG_KINDS:
            raise ValueError(f"manifest arg {raw.get('name', '')} has unsupported kind: {raw.get('kind') or 'missing'}")
        arg_name = str(raw.get("name", ""))
        arg_materialization_reason = _materialization_reason(raw, index=arg_name)
        layout = _parse_arg_layout(
            raw,
            arg_index=index,
            has_type_shim=has_type_shim,
            type_shim_failure_reason=type_shim_failure_reason,
        )
        if raw_kind == "opaque_val" and any(not node.supported or node.kind in {"pointer", "opaque_with_ptr"} for node in layout):
            raw_kind = "opaque_with_ptr"
        raw_pointer_role = _normalize_pointer_role(raw.get("pointer_role"))
        pointer_role = raw_pointer_role
        supported = True
        reason: str | None = None

        if raw_kind == "pointer":
            pointer_role, supported, reason = _classify_pointer_node(
                raw,
                has_type_shim=has_type_shim,
                type_shim_failure_reason=type_shim_failure_reason,
            )
        else:
            if pointer_role is not None:
                supported = False
                reason = "pointer_role_not_supported"
            if raw_kind in {"opaque_val", "opaque_with_ptr"}:
                if raw_kind == "opaque_with_ptr" and decl_kind == "union":
                    supported = False
                    reason = "union_not_supported"
                elif type_shim_status != "ok":
                    supported = False
                    reason = type_shim_reason_codes[0] if type_shim_reason_codes else "type_shim_missing"
                elif raw_kind == "opaque_with_ptr" and not layout:
                    supported = False
                    reason = "opaque_with_ptr_layout_missing"
            elif int(raw.get("size_bytes", 0) or 0) not in {1, 2, 4, 8}:
                supported = False
                reason = "scalar_size_unsupported"

        child_reason = None
        for node in layout:
            candidate = _format_node_support_reason(node)
            if candidate is not None:
                child_reason = candidate
                break
        if supported and child_reason is not None:
            supported = False
            reason = child_reason
        if supported and arg_materialization_reason is not None:
            supported = False
            reason = arg_materialization_reason

        if not supported and first_unsupported is None:
            first_unsupported = reason or "arg_not_supported"
        args.append(
            KernelArgContract(
                index=index,
                name=arg_name,
                type_name=type_name,
                canonical_type=canonical_type,
                size_bytes=int(raw.get("size_bytes", 0) or 0),
                align_bytes=int(raw.get("align_bytes", 1) or 1),
                kind=raw_kind,
                pointer_role=pointer_role,
                supported=supported,
                support_reason=reason,
                qualified_name=qualified_name,
                definition_loc=definition_loc,
                decl_kind=decl_kind,
                pointee_size_bytes=_pointee_size_bytes(raw),
                pointee_align_bytes=_pointee_align_bytes(raw),
                layout=layout,
            )
        )

    constraint = _parse_constraints(kernel_entry.get("constraints", []), args)

    return KernelContract(
        kernel_id=kernel_id,
        display_name=str(kernel_entry.get("display_name") or input_symbol),
        input_symbol=input_symbol,
        entry_symbol=f"__rapid_entry__{input_symbol}",
        args=args,
        type_shim_header=type_shim_header,
        type_shim_status=type_shim_status,
        type_shim_reason_codes=type_shim_reason_codes,
        type_shim_missing_dependencies=type_shim_missing_dependencies,
        type_shim_include_dirs=type_shim_include_dirs,
        vconfig_requested=vconfig_requested,
        constraint=constraint,
        supported=first_unsupported is None,
        support_reason=first_unsupported,
    )
