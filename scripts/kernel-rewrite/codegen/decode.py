"""Manifest-driven decode header generation for Phase 2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from contracts.kernel import (
    ArgValueExpr,
    BinaryExpr,
    ConstExpr,
    ConstraintExpr,
    KernelArgContract,
    KernelConstraintPredicate,
    KernelContract,
    KernelLayoutNode,
    KernelPayloadSlot,
    PayloadLenExpr,
)


def _guard_name(kernel_id: str) -> str:
    return "__PHASE2_DECODE_" + "".join(c if c.isalnum() else "_" for c in kernel_id).upper() + "_CUH__"


def _include_block(contract: KernelContract) -> str:
    return '#include "type_shim.v1.cuh"\n' if contract.needs_shim else ""


def _value_read_expr(type_name: str) -> str:
    if type_name in {"float"}:
        return "read_f32_le(data, offset)"
    if type_name in {"double"}:
        return "read_f64_le(data, offset)"
    return f"read_scalar_le<{type_name}>(data, offset)"


def _dedupe_adjacent_path_segments(index: str) -> str:
    parts = index.split(".")
    deduped: list[str] = []
    for part in parts:
        if part and deduped and part == deduped[-1]:
            continue
        deduped.append(part)
    return ".".join(deduped)


def _decode_local_name(index: str, array_indices: tuple[int, ...] = ()) -> str:
    name = _dedupe_adjacent_path_segments(index)
    for array_index in array_indices:
        name = name.replace("[]", f"_{array_index}", 1)
    name = name.replace("[]", "")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return name or "unnamed"


def _reserve_local_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name
    suffix = 2
    while f"{name}_{suffix}" in used_names:
        suffix += 1
    unique = f"{name}_{suffix}"
    used_names.add(unique)
    return unique


def _is_transparent_anonymous_aggregate(node: KernelLayoutNode) -> bool:
    return (
        node.kind == "opaque_with_ptr"
        and node.name.startswith("anon")
        and "anonymous" in node.type_name.lower()
    )


@dataclass(frozen=True)
class DecodeConstraintEmitContext:
    args_by_index: dict[int, KernelArgContract]
    payload_len_vars: dict[object, str]

    def decoded_arg(self, index: int, field_path: tuple[str, ...] = ()) -> str:
        arg = self.args_by_index[index]
        expr = f"decoded.{arg.name}"
        nodes = arg.layout
        current: KernelLayoutNode | None = None
        for part in field_path:
            node = next((candidate for candidate in nodes if candidate.name == part), None)
            if node is None:
                if current is not None and current.element is not None:
                    try:
                        array_index = int(part)
                    except ValueError:
                        array_index = -1
                    if 0 <= array_index < (current.element_count or 0):
                        expr += f"[{array_index}]"
                        current = current.element
                        nodes = current.fields
                        continue
                expr += f".{part}"
                current = None
                nodes = ()
                continue
            if not _is_transparent_anonymous_aggregate(node):
                expr += f".{part}"
            current = node
            nodes = node.fields
        return expr

    def payload_len(self, index: int, field_path: tuple[str, ...] = ()) -> str:
        key = (index, field_path)
        if key in self.payload_len_vars:
            return self.payload_len_vars[key]
        if not field_path and index in self.payload_len_vars:
            return self.payload_len_vars[index]
        return self.payload_len_vars[key]


def _emit_constraint_expr(expr: ConstraintExpr, ctx: DecodeConstraintEmitContext) -> str:
    if isinstance(expr, ArgValueExpr):
        return ctx.decoded_arg(expr.arg_index, expr.field_path)
    if isinstance(expr, PayloadLenExpr):
        return ctx.payload_len(expr.arg_index, expr.field_path)
    if isinstance(expr, ConstExpr):
        return str(expr.value)
    if isinstance(expr, BinaryExpr):
        lhs = _emit_constraint_expr(expr.lhs, ctx)
        rhs = _emit_constraint_expr(expr.rhs, ctx)
        return f"({lhs} {expr.op} {rhs})"
    raise TypeError(f"unsupported constraint expr: {expr!r}")


def _emit_predicate_expr(pred: KernelConstraintPredicate, ctx: DecodeConstraintEmitContext) -> str:
    lhs = _emit_constraint_expr(pred.lhs, ctx)
    rhs = _emit_constraint_expr(pred.rhs, ctx)
    if pred.op not in {"<", "<=", "==", "!=", ">=", ">"}:
        raise ValueError(f"unsupported predicate op: {pred.op}")
    return f"{lhs} {pred.op} {rhs}"


def _emit_invalid_check(pred: KernelConstraintPredicate, ctx: DecodeConstraintEmitContext) -> str:
    return f"  RAPID_DECODE_ASSERT({_emit_predicate_expr(pred, ctx)});"


class DecodeHeaderGenerator:
    """Generate decode headers for currently supported arg-pack-v1 kernels."""

    def _struct_fields(self, contract: KernelContract) -> str:
        lines: list[str] = []
        for arg in contract.args:
            lines.append(f"  {arg.decoded_storage_type} {arg.name};")
        for slot in contract.feedback_payload_slots:
            lines.append(f"  uint64_t __rapid_payload_{slot.slot}_len_bytes;")
        return "\n".join(lines)

    def _emit_payload_pointer_decode(
        self,
        *,
        lines: list[str],
        name: str,
        target_expr: str,
        type_name: str,
        align_bytes: int,
        cast_type: str | None = None,
        payload_len_vars: dict[object, str] | None = None,
        arg_index: int | None = None,
        field_path: tuple[str, ...] = (),
        used_payload_names: set[str] | None = None,
        payload_slots: dict[tuple[int, tuple[str, ...]], KernelPayloadSlot] | None = None,
    ) -> None:
        align = max(1, align_bytes)
        if used_payload_names is not None:
            name = _reserve_local_name(name, used_payload_names)
        len_var = f"{name}_len_u64"
        pointer_cast_type = cast_type or type_name
        payload_slot = None
        if payload_slots is not None and arg_index is not None:
            payload_slot = payload_slots[(arg_index, field_path)]
        if payload_len_vars is not None and arg_index is not None:
            payload_len_vars[(arg_index, field_path)] = len_var
            if not field_path:
                payload_len_vars[arg_index] = len_var
        lines.extend(
            [
                f"  // decode payload-buffer `{name}`",
                f"  offset = rapid_payload_len_offset(offset, {align});",
                "  RAPID_DECODE_ASSERT((offset % 8) == 0);",
                "  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));",
                f"  uint64_t {len_var} = read_scalar_le<uint64_t>(data, offset);",
                *(
                    [f"  decoded.__rapid_payload_{payload_slot.slot}_len_bytes = {len_var};"]
                    if payload_slot is not None
                    else []
                ),
                f"  RAPID_DECODE_ASSERT((offset % {align}) == 0);",
                f"  RAPID_DECODE_ASSERT({len_var} <= full_size && offset <= full_size - {len_var});",
                f"  {target_expr} = reinterpret_cast<{pointer_cast_type}>(const_cast<uint8_t *>(data + offset));",
                f"  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>({target_expr}) % {align}) == 0);",
                f"  offset += static_cast<size_t>({len_var});",
            ]
        )

    def _emit_node_decode(
        self,
        *,
        lines: list[str],
        node: KernelLayoutNode,
        target_expr: str,
        array_indices: tuple[int, ...] = (),
        payload_len_vars: dict[object, str] | None = None,
        used_payload_names: set[str] | None = None,
        arg_index: int | None = None,
        field_path: tuple[str, ...] = (),
        payload_slots: dict[tuple[int, tuple[str, ...]], KernelPayloadSlot] | None = None,
    ) -> None:
        local_name = _decode_local_name(node.index, array_indices)
        align = max(1, node.align_bytes)
        if node.is_payload_buffer_pointer:
            self._emit_payload_pointer_decode(
                lines=lines,
                name=local_name,
                target_expr=target_expr,
                type_name=node.codegen_type,
                cast_type=f"typename std::remove_reference<decltype({target_expr})>::type",
                align_bytes=node.payload_align_bytes,
                payload_len_vars=payload_len_vars,
                arg_index=arg_index,
                field_path=field_path,
                used_payload_names=used_payload_names,
                payload_slots=payload_slots,
            )
            return
        if node.kind == "pointer":
            raise ValueError("pointer_role_not_supported")
        if node.element is not None and node.element_count is not None:
            for index in range(node.element_count):
                self._emit_node_decode(
                    lines=lines,
                    node=node.element,
                    target_expr=f"{target_expr}[{index}]",
                    array_indices=(*array_indices, index),
                    payload_len_vars=payload_len_vars,
                    used_payload_names=used_payload_names,
                    arg_index=arg_index,
                    field_path=(*field_path, str(index)),
                    payload_slots=payload_slots,
                )
            return
        if node.kind == "scalar":
            lines.extend(
                [
                    f"  // decode scalar field `{node.index}`",
                    f"  offset = rapid_align_up(offset, {align});",
                    f"  RAPID_DECODE_ASSERT((offset % {align}) == 0);",
                    f"  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof({node.decoded_storage_type}));",
                    f"  {target_expr} = {_value_read_expr(node.decoded_storage_type)};",
                ]
            )
            return
        if node.kind == "opaque_val":
            lines.extend(
                [
                    f"  // decode copy-safe aggregate field `{node.index}`",
                    f"  offset = rapid_align_up(offset, {align});",
                    f"  RAPID_DECODE_ASSERT((offset % {align}) == 0);",
                    f"  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof({node.decoded_storage_type}));",
                    f"  {target_expr} = read_object_bytes<{node.decoded_storage_type}>(data, offset);",
                ]
            )
            return
        if node.kind == "opaque_with_ptr":
            if not node.fields:
                raise ValueError("opaque_with_ptr_layout_missing")
            for child in node.fields:
                child_target = target_expr if _is_transparent_anonymous_aggregate(child) else f"{target_expr}.{child.name}"
                self._emit_node_decode(
                    lines=lines,
                    node=child,
                    target_expr=child_target,
                    array_indices=array_indices,
                    payload_len_vars=payload_len_vars,
                    used_payload_names=used_payload_names,
                    arg_index=arg_index,
                    field_path=(*field_path, child.name),
                    payload_slots=payload_slots,
                )
            return
        raise ValueError(f"unsupported_layout_kind:{node.kind}")

    def _emit_value_arg_decode(
        self,
        *,
        lines: list[str],
        arg: KernelArgContract,
        payload_len_vars: dict[object, str],
        used_payload_names: set[str],
        payload_slots: dict[tuple[int, tuple[str, ...]], KernelPayloadSlot],
    ) -> None:
        align = max(1, arg.align_bytes)
        if arg.is_payload_buffer_pointer:
            self._emit_payload_pointer_decode(
                lines=lines,
                name=arg.name,
                target_expr=f"decoded.{arg.name}",
                type_name=arg.codegen_type,
                align_bytes=arg.payload_align_bytes,
                payload_len_vars=payload_len_vars,
                arg_index=arg.index,
                field_path=(),
                used_payload_names=used_payload_names,
                payload_slots=payload_slots,
            )
            return
        if arg.kind == "pointer":
            raise ValueError("pointer_role_not_supported")
        if arg.kind == "opaque_with_ptr":
            if not arg.layout:
                raise ValueError("opaque_with_ptr_layout_missing")
            lines.extend([f"  // decode field-aware aggregate arg `{arg.name}`"])
            for node in arg.layout:
                target_expr = (
                    f"decoded.{arg.name}"
                    if _is_transparent_anonymous_aggregate(node)
                    else f"decoded.{arg.name}.{node.name}"
                )
                self._emit_node_decode(
                    lines=lines,
                    node=node,
                    target_expr=target_expr,
                    payload_len_vars=payload_len_vars,
                    used_payload_names=used_payload_names,
                    arg_index=arg.index,
                    field_path=(node.name,),
                    payload_slots=payload_slots,
                )
            return
        if arg.kind == "scalar":
            read_expr = _value_read_expr(arg.decoded_storage_type)
        elif arg.kind == "opaque_val":
            read_expr = f"read_object_bytes<{arg.decoded_storage_type}>(data, offset)"
        else:
            raise ValueError(f"unsupported_arg_kind:{arg.kind}")
        lines.extend(
            [
                f"  // decode value arg `{arg.name}`",
                f"  offset = rapid_align_up(offset, {align});",
                f"  RAPID_DECODE_ASSERT((offset % {align}) == 0);",
                f"  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof({arg.decoded_storage_type}));",
                f"  decoded.{arg.name} = {read_expr};",
            ]
        )

    def _decode_body(self, contract: KernelContract) -> str:
        lines: list[str] = [
            "  RAPID_DECODE_ASSERT(data != nullptr);",
            "",
            "  DecodedKernelArgs decoded{};",
            "  size_t offset = 0;",
        ]
        payload_len_vars: dict[object, str] = {}
        used_payload_names: set[str] = set()
        payload_slots = {
            (slot.arg_index, slot.field_path): slot
            for slot in contract.feedback_payload_slots
        }
        for arg in contract.args:
            self._emit_value_arg_decode(
                lines=lines,
                arg=arg,
                payload_len_vars=payload_len_vars,
                used_payload_names=used_payload_names,
                payload_slots=payload_slots,
            )
        lines.append("  return decoded;")
        ctx = DecodeConstraintEmitContext(
            args_by_index={arg.index: arg for arg in contract.args},
            payload_len_vars=payload_len_vars,
        )
        for pred in contract.constraint:
            lines.insert(-1, _emit_invalid_check(pred, ctx))
        return "\n".join(lines)

    def generate(self, *, contract: KernelContract, phase2_dir: Path) -> None:
        gen_dir = phase2_dir / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        out_path = gen_dir / "fuzzer_decode.v1.cuh"
        guard = _guard_name(contract.kernel_id)
        include_block = _include_block(contract)
        content = f'''#ifndef {guard}
#define {guard}

#include <cstddef>
#include <cstdint>
#include <assert.h>
#include <type_traits>
{include_block}

struct DecodedKernelArgs {{
{self._struct_fields(contract)}
}};

#ifndef RAPID_DECODE_ASSERT
#if !defined(NDEBUG) && defined(RAPID_DECODE_DEBUG_ASSERT)
#define RAPID_DECODE_ASSERT(cond) assert(cond)
#else
#define RAPID_DECODE_ASSERT(cond) ((void)0)
#endif
#endif

template <typename T, bool IsEnum = std::is_enum<T>::value, bool IsFloat = std::is_floating_point<T>::value>
struct RapidScalarStorage {{
  using type = T;
}};

template <typename T>
struct RapidScalarStorage<T, true, false> {{
  using type = typename std::underlying_type<T>::type;
}};

template <typename T>
struct RapidScalarStorage<T, false, true> {{
  using type = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
}};

template <typename T>
__device__ inline T read_scalar_le(volatile uint8_t *data, size_t &offset) {{
  static_assert(std::is_integral<T>::value || std::is_enum<T>::value || std::is_floating_point<T>::value, "read_scalar_le only supports integral, enum, or floating-point types");
  static_assert(!std::is_floating_point<T>::value || sizeof(T) == 4 || sizeof(T) == 8, "read_scalar_le only supports 32-bit or 64-bit floating-point types");
  using Storage = typename RapidScalarStorage<T>::type;
  Storage value = 0;
  for (size_t i = 0; i < sizeof(Storage); ++i) {{
    value |= static_cast<Storage>(static_cast<unsigned long long>(data[offset + i]) << (8 * i));
  }}
  offset += sizeof(Storage);
  if (std::is_floating_point<T>::value) {{
    T out;
    uint8_t *dst = reinterpret_cast<uint8_t *>(&out);
    for (size_t i = 0; i < sizeof(T); ++i) {{
      dst[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xff);
    }}
    return out;
  }}
  return static_cast<T>(value);
}}

__device__ inline float read_f32_le(volatile uint8_t *data, size_t &offset) {{
  union {{ uint32_t u; float f; }} bits{{read_scalar_le<uint32_t>(data, offset)}};
  return bits.f;
}}

template <typename T>
__device__ inline T read_object_bytes(volatile uint8_t *data, size_t &offset) {{
  static_assert(std::is_trivially_copyable<T>::value, "read_object_bytes requires trivially copyable type");
  static_assert(std::is_standard_layout<T>::value, "read_object_bytes requires standard layout type");
  T value{{}};
  uint8_t *dst = reinterpret_cast<uint8_t *>(&value);
  for (size_t i = 0; i < sizeof(T); ++i) {{
    dst[i] = data[offset + i];
  }}
  offset += sizeof(T);
  return value;
}}

__device__ inline double read_f64_le(volatile uint8_t *data, size_t &offset) {{
  union {{ uint64_t u; double d; }} bits{{read_scalar_le<uint64_t>(data, offset)}};
  return bits.d;
}}

__device__ inline size_t rapid_align_up(size_t value, size_t alignment) {{
  if (alignment <= 1) {{
    return value;
  }}
  size_t rem = value % alignment;
  return rem == 0 ? value : value + (alignment - rem);
}}

__device__ inline size_t rapid_payload_len_offset(size_t offset, size_t payload_align) {{
  size_t align = payload_align == 0 ? 1 : payload_align;
  size_t candidate = rapid_align_up(offset, sizeof(uint64_t));
  for (size_t i = 0; i <= (align > sizeof(uint64_t) ? align : sizeof(uint64_t)) / sizeof(uint64_t) + 1; ++i) {{
    if (((candidate + sizeof(uint64_t)) % align) == 0) {{
      return candidate;
    }}
    candidate += sizeof(uint64_t);
  }}
  return candidate;
}}

__device__ inline DecodedKernelArgs fuzzer_decode_v1(volatile uint8_t *data, size_t full_size) {{
{self._decode_body(contract)}
}}

#endif // {guard}
'''
        out_path.write_text(content, encoding="utf-8")
