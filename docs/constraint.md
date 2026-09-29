# Kernel Constraint

This document defines how KernelManifest `constraints` are interpreted by Phase 2.

Kernel constraints are decode-time validation rules. They describe relationships between decoded kernel arguments, payload-buffer lengths, constants, and derived expressions. Phase 2 parses each manifest constraint into a common predicate IR, then generates one `RAPID_DECODE_ASSERT(...)` check for each predicate in `fuzzer_decode.v1.cuh`.

The schema source is `docs/kernel-manifest.schema.json`.

## 1. Manifest Layer

KernelManifest stores constraints under each kernel entry:

```json
{
  "kernels": [
    {
      "symbol_name": "kernel",
      "args": [],
      "constraints": []
    }
  ]
}
```

Each constraint entry must contain:

- `kind`: the constraint kind.
- kind-specific fields, such as `scalar_arg`, `buffer_arg`, or `elem_size_bytes`.

The manifest layer is declarative. It does not decide how to generate C++ checks. It only records the contract that decoded arguments must satisfy.

## 2. Phase 2 Parse Layer

Phase 2 converts manifest constraints into `KernelConstraintPredicate`.

The parser is registry-driven:

```python
_CONSTRAINT_PARSERS: dict[str, ConstraintParser] = {
    "scalar_le_buffer_len": _parse_scalar_le_buffer_len,
    "scalar_compare_const": _parse_scalar_compare_const,
    "scalar_compare_scalar": _parse_scalar_compare_scalar,
    "scalar_product_le_const": _parse_scalar_product_le_const,
    "count_fits_buffer": _parse_count_fits_buffer,
}
```

`_parse_constraints(...)` performs only the generic loop:

1. Validate that `constraints` is a list.
2. Validate that each entry is an object.
3. Read `entry["kind"]`.
4. Look up the parser in `_CONSTRAINT_PARSERS`.
5. Append the predicates returned by that parser.

Each concrete parser owns only one constraint kind. Adding a new constraint kind should not change the generic `_parse_constraints(...)` loop.

## 3. Predicate IR

Phase 2 stores parsed constraints in:

```python
class KernelContract:
    constraint: list[KernelConstraintPredicate]
```

`KernelConstraintPredicate` represents one condition that must be true:

```python
KernelConstraintPredicate(
    lhs=...,
    op="<=",
    rhs=...,
)
```

Supported predicate operators:

| Operator | Meaning |
| --- | --- |
| `<` | left side must be less than right side |
| `<=` | left side must be less than or equal to right side |
| `==` | left side must equal right side |
| `!=` | left side must not equal right side |
| `>=` | left side must be greater than or equal to right side |
| `>` | left side must be greater than right side |

Current `ConstraintExpr` nodes:

| Expr | Meaning | Example codegen input |
| --- | --- | --- |
| `ArgValueExpr(arg_index, field_path=...)` | value of a decoded scalar kernel argument or scalar field | `decoded.size`, `decoded.params.nested.inner_count` |
| `PayloadLenExpr(arg_index, field_path=...)` | decoded payload segment length of a `payload_buffer` pointer argument or pointer field | `input_len_u64`, `params_data_len_u64` |
| `ConstExpr(value)` | integer constant | `4` |
| `BinaryExpr(op, lhs, rhs)` | arithmetic expression over other expressions | `(decoded.count * 4)` |

`ArgValueExpr` is intended for scalar and by-value values that have already been materialized into `DecodedKernelArgs`. `PayloadLenExpr` is valid only for `kind=pointer && pointer_role=payload_buffer` arguments, because only payload-buffer pointers have a decoded payload segment length in Input Envelope v1.

For every `ConstraintExpr` node that references an argument, `arg_index` is the original KernelManifest/kernel argument index. It is not a compressed ordinal within a subset of arguments. For example:

- `ArgValueExpr(1)` refers to the decoded value of original arg 1.
- `ArgValueExpr(1, ("nested", "inner_count"))` refers to `decoded.<arg1-name>.nested.inner_count`.
- `PayloadLenExpr(0)` refers to the payload length decoded for original arg 0.
- `PayloadLenExpr(1, ("data",))` refers to the payload length decoded for `decoded.<arg1-name>.data`.
- `PayloadLenExpr(2)` refers to the payload length decoded for original arg 2.

Manifest field paths are JSON arrays of path segments, not dot-separated strings.
For example, `["nested", "choice", "tag"]` means `.nested.choice.tag`.
For fixed-size arrays in `type_layout`, use a decimal string segment to select
an element; `["items", "1", "inner", "data"]` means
`.items[1].inner.data`. If the path field is absent, the constraint references
the top-level argument itself. If the path field is present, Phase 2 resolves it
by descending through that argument's recursive `type_layout` and requires the
final referenced node to be a scalar for scalar comparison constraints.

If arg 0 and arg 2 are payload buffers and arg 1 is a scalar, then `PayloadLenExpr(1)` is invalid because original arg 1 is not a payload-buffer pointer. The `1` is still the original argument index; it is not interpreted as "the second payload-buffer argument".

A constraint parser must validate this before constructing `PayloadLenExpr(arg_index, field_path=...)`. The Phase 2 contract layer fails fast with `constraint_buffer_arg_not_payload_buffer` if the referenced argument or field is not `kind=pointer && pointer_role=payload_buffer`.

## 4. Decode Lowering

`fuzzer_decode.v1.cuh` generation lowers each predicate into one decode assertion. For a predicate that must be true:

```text
lhs op rhs
```

decode emits the predicate directly, without op inversion or invalid-return control flow:

```cpp
RAPID_DECODE_ASSERT(lhs op rhs);
```

Each predicate produces one `RAPID_DECODE_ASSERT(...)` line. If the caller has not defined `RAPID_DECODE_ASSERT(cond)` before including the generated decode header, the generated header provides this default:

```cpp
#ifndef RAPID_DECODE_ASSERT
#if !defined(NDEBUG) && defined(RAPID_DECODE_DEBUG_ASSERT)
#define RAPID_DECODE_ASSERT(cond) assert(cond)
#else
#define RAPID_DECODE_ASSERT(cond) ((void)0)
#endif
#endif
```

Therefore generated decode checks are opt-in by default: they run only in non-`NDEBUG` builds that also define `RAPID_DECODE_DEBUG_ASSERT`. A caller that needs different behavior, including release-mode checks, must define `RAPID_DECODE_ASSERT(cond)` before including the generated decode header.

The invoke header does not re-check constraints; `fuzzer_invoke_v1(...)` receives already materialized `DecodedKernelArgs`.

## 5. Supported Constraint Kinds

This table separates schema-declared kinds from Phase 2 implemented lowering.

| Constraint kind | Schema status | Phase 2 status | Handling |
| --- | --- | --- | --- |
| `scalar_le_buffer_len` with `unit=bytes` | declared | implemented | Parses to `ArgValueExpr(scalar_arg, scalar_path) <= PayloadLenExpr(buffer_arg, buffer_path)` |
| `scalar_le_buffer_len` with `unit=elements` | declared | implemented | Parses to `(ArgValueExpr(scalar_arg, scalar_path) * pointee_size_bytes) <= PayloadLenExpr(buffer_arg, buffer_path)` |
| `scalar_compare_const` | declared | implemented | Parses to `ArgValueExpr(scalar_arg, scalar_path) op ConstExpr(value)` |
| `scalar_compare_scalar` | declared | implemented | Parses to `ArgValueExpr(lhs_arg, lhs_path) op ArgValueExpr(rhs_arg, rhs_path)` |
| `scalar_product_le_const` | declared | implemented | Phase 3 repairs the selected integer scalar; Phase 2 parses `(lhs * rhs) <= value` |
| `count_fits_buffer` | declared | implemented | Parses to `(ArgValueExpr(count_arg, count_path) * elem_size_bytes) <= PayloadLenExpr(buffer_arg, buffer_path)` |
| `buffer_elements_lt_scalar` | declared | recognized, fuzzer-side repair | Phase 2 validates referenced payload/scalar args but emits no decode predicate; the fuzzer normalizes and mutates payload elements so every element is `< scalar_arg`. |

Unsupported Phase 2 constraint kinds must fail fast during contract parsing. They must not be silently ignored.

## 6. `scalar_le_buffer_len`

Manifest form:

```json
{
  "kind": "scalar_le_buffer_len",
  "scalar_arg": 1,
  "buffer_arg": 0,
  "unit": "bytes"
}
```

Requirements:

- `scalar_arg` must reference an existing scalar argument.
- `buffer_arg` must reference an existing `kind=pointer && pointer_role=payload_buffer` argument.
- If `scalar_path` is present, it must resolve to a scalar field under `scalar_arg`.
- If `buffer_path` is present, it must resolve to a payload-buffer pointer field under `buffer_arg`.
- `unit` may be `bytes` or `elements`.
- For `unit=elements`, the buffer pointer must carry `pointee_layout.size_bytes`; Phase 2 lowers the check to bytes by multiplying the scalar count by that pointee size.

Predicate:

```python
KernelConstraintPredicate(
    lhs=ArgValueExpr(1),
    op="<=",
    rhs=PayloadLenExpr(0),
)
```

Example generated check:

```cpp
RAPID_DECODE_ASSERT(decoded.size <= input_len_u64);
```

The generated variable name depends on the buffer argument name. For a buffer argument named `input`, decode stores its decoded payload length in `input_len_u64`.

Nested buffer paths use the payload length variable emitted for that field. For
example:

```json
{
  "kind": "scalar_le_buffer_len",
  "scalar_arg": 0,
  "scalar_path": ["n"],
  "buffer_arg": 0,
  "buffer_path": ["data"],
  "unit": "elements"
}
```

with `params.data` pointing to `float` lowers to:

```cpp
RAPID_DECODE_ASSERT((decoded.params.n * 4) <= params_data_len_u64);
```

The manifest does not define a separate payload-length equality constraint. A
"length A must be at least scalar B" relationship is expressed as
`scalar_le_buffer_len`; exact equality between two scalar values is expressed as
`scalar_compare_scalar` with `op: "=="`. Payload-buffer-to-payload-buffer
equality is intentionally not a v1 Phase 2 constraint until payload length
expressions can appear on both sides of a generic predicate.

## 7. Scalar Comparisons

`scalar_compare_const` compares one decoded scalar arg or scalar field with an
integer constant:

```json
{
  "kind": "scalar_compare_const",
  "scalar_arg": 1,
  "scalar_path": ["nested", "inner_count"],
  "op": ">=",
  "value": 0
}
```

This lowers to:

```python
ArgValueExpr(1, ("nested", "inner_count")) >= ConstExpr(0)
```

and generates:

```cpp
RAPID_DECODE_ASSERT(decoded.params.nested.inner_count >= 0);
```

If `scalar_path` is absent, the top-level scalar arg is used directly, as in
`seed >= 0`.

`scalar_compare_scalar` compares two decoded scalar args or scalar fields:

```json
{
  "kind": "scalar_compare_scalar",
  "lhs_arg": 1,
  "lhs_path": ["nested", "choice", "tag"],
  "op": "<=",
  "rhs_arg": 1,
  "rhs_path": ["count"]
}
```

This lowers to:

```python
ArgValueExpr(1, ("nested", "choice", "tag")) <= ArgValueExpr(1, ("count",))
```

and generates:

```cpp
RAPID_DECODE_ASSERT(decoded.params.nested.choice.tag <= decoded.params.count);
```

## 8. Count Fits Buffer

Manifest form declared by schema:

```json
{
  "kind": "count_fits_buffer",
  "count_arg": 1,
  "buffer_arg": 0,
  "elem_size_bytes": 4
}
```

Predicate lowering:

```python
BinaryExpr("*", ArgValueExpr(1), ConstExpr(4)) <= PayloadLenExpr(0)
```

This represents:

```text
count * elem_size_bytes <= payload_len_bytes(buffer)
```

`count_path` and `buffer_path` follow the same nested field path rules as
`scalar_le_buffer_len`.

## 8.1 `buffer_elements_lt_scalar`

Manifest form:

```json
{
  "kind": "buffer_elements_lt_scalar",
  "buffer_arg": 2,
  "scalar_arg": 3,
  "elem_size_bytes": 8
}
```

Requirements:

- `buffer_arg` must reference an existing `kind=pointer && pointer_role=payload_buffer` argument.
- `scalar_arg` must reference an existing scalar argument.
- `elem_size_bytes` is optional when the payload buffer has `domain.elem_size_bytes` or pointee layout size metadata; when present it must be 1, 2, 4, or 8 bytes for the current Rust arg-pack runtime.
- The fuzzer interprets complete little-endian elements in the payload and repairs each element to be less than the scalar bound. This is intended for index/permutation tables such as `index_map[i] < slots` or `permutation_table[i] < poly_degree`.

This is not a Phase 2 decode-time predicate today. It is deliberately recognized by Phase 2 so resolved manifests remain accepted, but the enforcement lives in fuzzer normalize/mutate because generated decode code currently has no loop IR for payload-content predicates.

## 8.2 `scalar_product_le_const`

`scalar_product_le_const` bounds the product of two non-negative `int_range`
scalars. `repair_arg` (and optional `repair_path`) must identify one operand;
Phase 3 reduces it to at most `floor(value / other_operand)`. Phase 2 emits the
decode predicate `(ArgValueExpr(lhs) * ArgValueExpr(rhs)) <= ConstExpr(value)`.

## 9. Generation Rules

New constraint support should follow this sequence:

1. Add or update the schema entry in `docs/kernel-manifest.schema.json`.
2. Add a dedicated parser function in the Phase 2 contract layer.
3. Register the parser in `_CONSTRAINT_PARSERS`.
4. Return one or more `KernelConstraintPredicate` values.
5. Add decode codegen support only if the parser needs a new `ConstraintExpr` node or arithmetic operator.
6. Add contract and decode codegen tests.
7. Update this document and `docs/phase2-implementation-plan.md`.

The generic parse loop and generic decode predicate loop should remain unchanged when adding a new constraint kind.
