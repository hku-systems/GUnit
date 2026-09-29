# Input Envelope v1

This document defines the v1 input protocol used by RAPID to encode fuzz cases for a fixed target kernel.

There are two byte layers:

1. the backend-facing task envelope submitted by LibAFL `BytesInput`, and
2. the arg-pack payload inside that envelope, consumed by the generated
   device-side decoder.

The arg-pack payload protocol is about three things only:

1. how the fuzzer encodes a testcase into raw bytes,
2. how the generated device-side decoder materializes those bytes into kernel arguments,
3. how the generated invoke function passes the materialized arguments to the rewritten entry.

It does **not** define runtime scheduling, coverage plumbing, or other execution-control channels. Those channels live in the task envelope header.

---

## 1. KernelManifest

`KernelManifest` is the manifest consumed by the fuzzer and the code generators. The schema is defined in [`docs/kernel-manifest.schema.json`](kernel-manifest.schema.json).

For `input-envelope-v1`, the important parts of a kernel manifest are:

- `kernels[*].symbol_name`
- `kernels[*].display_name`
- `kernels[*].args[*]`
- `kernels[*].constraints`

For each argument or recursive layout field, the schema carries the fields used by the input protocol:

- `kind`
- `pointer_role`
- `pointee_layout`
- `size_bytes`
- `align_bytes`
- `domain`
- `type_info`
- `type_layout`

For `pointer_role=derived_pointer`, relation-specific fields such as `base_arg`, `base_path`, `offset_bytes`, `offset_arg`, and `offset_path` describe `base + offset`. These offset fields are pointer relation metadata; they are not recursive struct-field layout offsets and they do not define input-envelope ordering.

For `pointer_role=external_device_pointer`, `source` describes the runtime or harness provider for the pointer.

The schema reserves more fields for validation and metadata, but the fields above are the ones that matter for encoding, decode, and invoke generation.

### 1.1 Argument kinds

`kind` is the materialization category. The same four values are used for top-level arguments and for recursive `type_layout` field nodes:

- `scalar`
- `pointer`
- `opaque_val`
- `opaque_with_ptr`

The meanings are:

- `scalar`: a fixed-width scalar leaf such as integer, float, double, or enum.
- `pointer`: a real pointer leaf. It must carry `pointer_role` and `pointee_layout`. The pointee layout describes one dereference for sizing and mutation policy; it does not make the pointer itself an inline aggregate.
- `opaque_val`: an inline aggregate/value subtree whose pointer-freeness is known; it is copy-safe as ordinary by-value bytes.
- `opaque_with_ptr`: an inline aggregate/value subtree that contains a pointer field, or whose pointer-freeness cannot be proven; it must be opened and materialized field-by-field.

`pointer_role` further refines `kind=pointer`:

- `payload_buffer`
- `derived_pointer`
- `external_device_pointer`

This document focuses on the input protocol for `payload_buffer` and `derived_pointer`. The other roles are schema-reserved but are not part of the main v1 input path.

The manifest is a fact layer. It records type/layout facts and semantic roles. It must not record current generator support state such as `supported` or `unsupported_reason`; those are Phase 2/3 capability decisions and diagnostics.

---

## 2. Protocol Overview

The complete submitted input is a task envelope:

```text
offset  size  field
0       24    RapidVConfig {grid_x, grid_y, grid_z, block_x, block_y, block_z}
24      8     payload_size, little-endian uint64
32      N     arg-pack-v1 payload bytes
```

The C++ ABI is:

```c++
struct RapidTaskEnvelopeHeader {
  RapidVConfig vconfig;   // 24 bytes
  uint64_t payload_size;  // offset 24, counts arg-pack payload bytes only
};

static_assert(sizeof(RapidTaskEnvelopeHeader) == 32);
static_assert(offsetof(RapidTaskEnvelopeHeader, payload_size) == 24);
```

`payload_size` is the length of the arg-pack payload at offset 32. It does not
include `RapidVConfig`, and it does not include the 8-byte `payload_size` field
itself. Therefore:

```text
submitted BytesInput length = sizeof(RapidTaskEnvelopeHeader) + payload_size
                            = 32 + payload_size

H2D copy length             = sizeof(RapidTaskEnvelopeHeader) + payload_size
                            = 32 + payload_size
```

Backends may repair the header before this H2D copy. In the current launch
policy implementation, origin clamps `RapidVConfig` and uses it as the actual
per-input CUDA launch shape. RAPID and RAPID2 clamp `RapidVConfig` to the
physical persistent launch selected at backend startup.

Do not compute the copy length as
`sizeof(RapidTaskEnvelopeHeader) + payload_size + sizeof(payload_size)`. The
`payload_size` field is already part of `RapidTaskEnvelopeHeader`, so adding it
again double-counts bytes 24..31.

The arg-pack payload protocol is:

1. The fuzzer builds raw testcase bytes from a `KernelManifest`.
2. The fuzzer wraps those bytes in the full task envelope above.
3. The full envelope is copied to device memory.
4. The backend passes `rapid_task_payload(envelope)` and
   `envelope->payload_size` to the generated device-side decoder
   (`fuzzer_decode_v1`).
5. The decoder reads only the arg-pack payload bytes and materializes a
   `DecodedKernelArgs` value.
6. The generated invoke function (`fuzzer_invoke_v1`) passes that materialized
   value to the rewritten entry symbol.

The protocol is therefore a mapping from testcase bytes to kernel arguments, not a direct mirror of the original C/C++ signature on the launch ABI.

The generated functions are part of the build artifacts:

- `fuzzer_decode_v1(...)` is generated from the manifest and rewrite plan.
- `fuzzer_invoke_v1(...)` is generated from the manifest and rewrite plan.

They are not interpreted at runtime from the manifest.

---

## 3. Encoding Order

The generated decoder walks the manifest in ABI order:

1. top-level `KernelManifest.args` order,
2. then recursive `type_layout` field order for any `opaque_with_ptr` argument,
3. then array element order when a recursive layout node has `element` and `element_count`.

Each leaf is encoded by the same rule regardless of depth. A nested field `kind=pointer && pointer_role=payload_buffer` uses the same canonical padding plus `[len:u64][payload+filler]` segment as a top-level payload pointer.

Recursive layout nodes carry a string `index` for diagnostics, constraints, and
generated-code naming. It is a displayable manifest index, not the top-level
numeric ABI `args[*].index`. Examples:

- `params.n`
- `params.inner.buf`
- `params.inner[].buf`
- `params[].inner.buf`

Array element templates keep `"name": "$element"` in JSON and use `[]` in
`index`.

### 3.1 Alignment

- `align_bytes` is the encoding and decoding alignment for the top-level argument.
- The start offset of each argument must satisfy `align_bytes`.
- When needed, the previous segment grows by filler bytes so the next argument starts at a valid offset.
- For a `payload_buffer` pointer, `align_bytes` applies to the payload bytes
  after the `u64` length field. The length field is placed at the first
  8-byte-aligned offset where `offset + sizeof(uint64_t)` satisfies the
  pointer payload alignment.

### 3.2 Length fields

- All length fields are `u64` little-endian.
- Every length field is placed at an 8-byte aligned offset.
- Filler bytes before a payload length field are canonical padding. For a
  preceding payload pointer, that padding is included in the preceding encoded
  `len`; otherwise it is skipped by the generated decoder before reading the
  length field.

### 3.3 Canonical layout

The protocol expects a canonical layout:

- one semantic argument set should have one stable byte representation,
- parse should be followed by canonical repack,
- normalize should repair bytes into that canonical layout.

---

## 4. Payload Buffer Materialization

For any `kind=pointer && pointer_role=payload_buffer` leaf, the pointer represents a fuzzer-owned device allocation whose contents are carried in the testcase bytes.

The wire format is:

```text
[padding][len:u64][payload+filler]
```

`padding` is present only when needed to satisfy both length-field alignment
and payload alignment.

The `len` value is the encoded length of (sizeof, bytewise) the following `[payload+filler]` segment. It does **not** include the 8-byte length field itself.

So for this argument:

- bytes consumed by the segment = `8 + len`
- `len` = encoded payload segment length, including filler bytes

The generated decoder reads the length, then materializes the pointer value from the following segment. The pointer value points at the payload bytes inside the testcase buffer, while the filler bytes exist only to satisfy alignment and canonical layout requirements.

This role uses the `bytes` domain in the manifest for length-related bounds.

---

## 5. Derived Pointer Materialization

For any `kind=pointer && pointer_role=derived_pointer` leaf, the pointer is not an independent payload buffer. It is a pointer derived from another decoded pointer.

The manifest should describe this relation with:

- `base_arg`: the argument index of the base pointer,
- an explicit offset description,
- an explicit offset constraint,
- `offset_bytes` or `offset_arg` as the current schema fields for that offset source.

The protocol for a derived pointer is:

1. decode the base pointer argument first,
2. read the offset value,
3. interpret the offset according to the manifest constraint,
4. compute the final pointer value as `base + offset`,
5. perform alignment and bounds checks in the generated decoder,
6. pass the final pointer value to the generated invoke function.

For the current v1 protocol, the encoded form can be kept simple:

```text
[offset:u64]
```

The offset is not payload bytes. It is a value that the decoder combines with the already materialized base pointer.

### 5.1 Offset semantics

The manifest must make the offset meaning explicit.

Examples:

- `offset` measured in bytes,
- `offset` measured in elements,
- `offset` restricted to a specific range,
- `offset` required to preserve alignment.

If the offset is measured in elements, the generated decoder must convert it to a byte offset before materialization.

---

## 6. Scalar and Aggregate Materialization

### 6.1 Scalars

`kind=scalar` leaves are encoded as little-endian values of `size_bytes`.

Examples:

- `size_t size` -> 8-byte little-endian integer,
- `int n` -> 4-byte little-endian integer.

### 6.2 Copy-safe aggregates

`kind=opaque_val` leaves are encoded as fixed-length opaque bytes of `size_bytes`.

The decoder materializes them as local values, and the generated invoke function passes them by value to the rewritten entry.

Fixed-size inline arrays are inline aggregate values. They are not modeled as payload pointers merely because they contain multiple elements. If the array subtree is pointer-free it is `opaque_val`; if the element subtree contains pointers it is `opaque_with_ptr`.

### 6.3 Pointer-bearing aggregates

`kind=opaque_with_ptr` is not encoded as one black-box byte blob. The decoder opens the aggregate according to `type_layout` and recursively materializes its children:

- scalar fields follow the scalar rule,
- nested `opaque_val` fields follow the fixed-byte copy rule,
- nested payload pointer fields follow canonical padding plus `[len:u64][payload+filler]`,
- nested `opaque_with_ptr` fields recurse again.

The final generated `DecodedKernelArgs` still has the original top-level C/C++ type. Field-aware decode only controls how bytes become that value.

---

## 7. Generated Decode and Invoke

`fuzzer_decode_v1(...)` and `fuzzer_invoke_v1(...)` are generated device-side functions.

### 7.1 Decode

The generated decoder is responsible for:

- checking input bounds,
- reading arguments and recursive fields in manifest layout order,
- applying the materialization rules for `scalar`, `opaque_val`, `opaque_with_ptr`, `payload_buffer`, and `derived_pointer`,
- validating alignment and length constraints,
- returning `DecodedKernelArgs`.

### 7.2 Invoke

The generated invoke function is responsible for:

- taking `DecodedKernelArgs`,
- passing the materialized arguments to the rewritten entry symbol,
- doing nothing else.

It does not re-parse the bytes and does not re-interpret the manifest.

---

## 8. Fuzzer-Side Encode, Normalize, and Seed

The fuzzer consumes the manifest and maintains the same argument semantics as
the generated decoder. Its LibAFL corpus input is the complete task envelope
from Section 2. Arg-pack encode/normalize/seed logic operates on the payload
slice after `RapidTaskEnvelopeHeader`, while VConfig normalize/mutate logic
operates on the header fields.

### 8.1 Encode

The fuzzer first emits canonical arg-pack payload bytes according to the
manifest:

- `payload_buffer` -> canonical padding plus `[len:u64][payload+filler]`
- `derived_pointer` -> `[offset:u64]`
- `scalar` -> little-endian scalar bytes
- `opaque_val` -> fixed-length opaque bytes
- `opaque_with_ptr` -> recursive field encoding

It then wraps those payload bytes as:

```text
RapidTaskEnvelopeHeader { vconfig, payload_size = arg_pack_payload.len() }
+ arg-pack payload
```

### 8.2 Normalize

Normalize reparses a candidate `BytesInput` as a complete task envelope,
repairs/clamps the VConfig header, repairs the arg-pack payload according to
the manifest constraints, updates `payload_size`, and repacks the result into
canonical envelope form.

### 8.3 Seed

The default seed is also generated from the manifest and then normalized, so the
initial testcase is a valid full task envelope whose payload is valid under the
arg-pack protocol.

### 8.4 Constraint consumption

`scalar_le_buffer_len` is not a separate mutator. It is a manifest constraint that affects encode, normalize, and seed generation.

---

## 9. Constraints

The manifest-level constraints that matter for this protocol are cross-argument rules such as:

- `scalar_le_buffer_len`
- `scalar_compare_const`
- `scalar_compare_scalar`
- `count_fits_buffer`

These constraints are consumed by the fuzzer when it generates or repairs the
canonical arg-pack payload bytes inside the task envelope.
Field paths inside constraints are JSON segment arrays relative to the referenced
argument. For example, `["nested", "choice", "tag"]` means
`arg.nested.choice.tag`; when the path field is absent, the constraint references
the top-level argument itself.

For the current builtin kernel example, the important rule is:

- `size <= input.len`
- `size <= output.len`

That rule is expressed as manifest constraints, not by special-casing parameter names inside the protocol document.

---

## 10. Error Semantics

The protocol distinguishes two kinds of failure:

### 10.1 Invalid input

Examples:

- boundary check failure,
- alignment check failure,
- constraint violation,
- unsupported role in a manifest that is expected to be handled by the current implementation.

### 10.2 Unimplemented role

Examples:

- a role is present in the schema but not yet supported by the current decoder,
- the manifest requests a relation that the current materialization logic does not yet implement.

These cases should fail fast rather than silently falling back to another role.

---

## 11. Current v1 Boundary

The current v1 input protocol supports:

- `scalar`
- `opaque_val`
- `opaque_with_ptr` with complete recursive layout for currently supported child roles
- `pointer_role=payload_buffer`

Union layout nodes are not part of the current supported envelope. A manifest may
preserve the fact with `type_info.kind=union`, but Phase 2/3 must fail fast until
an active-member selector/discriminator encoding is defined.

The protocol also defines how to model `derived_pointer` in the manifest and how to materialize it in the generated decoder and fuzzer, but that support still depends on future implementation work.

`external_device_pointer` remains a reserved schema role and is outside the main v1 input path.

---

## 12. Summary

`Input Envelope v1` is a protocol for converting testcase bytes into kernel
arguments. In current backends, the submitted testcase bytes are a full
`RapidTaskEnvelopeHeader + arg-pack payload` envelope.

- `KernelManifest` defines the argument list and the constraints.
- `payload_buffer` carries an encoded payload segment whose `len` is the length of `[payload+filler]`.
- `derived_pointer` is a relation-based pointer, materialized as `base + offset`.
- `opaque_val` is copy-safe inline value storage.
- `opaque_with_ptr` is opened recursively; nested pointer leaves use the same pointer-role envelope rules.
- `fuzzer_decode_v1(...)` and `fuzzer_invoke_v1(...)` are generated device-side functions.
- The fuzzer encodes, normalizes, and seeds a full task envelope; the payload
  portion is canonical arg-pack bytes derived from the same manifest.
