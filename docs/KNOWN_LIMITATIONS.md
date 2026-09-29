# Known Limitations

This document records current Phase 1 and Phase 2 support boundaries for the
CUDA kernel pipeline. These are intentional fail-fast limits, not silent
best-effort cases.

## Backend/Fuzzer Deferred Findings

The following backend/fuzzer findings are known and intentionally deferred from
the current hardening pass:

- `cuda-kernel/rapid` previously reported delayed cumulative feedback with
  approximate attribution; the ordered completion ABI now returns task-owned
  feedback with exact task-ID attribution, matching `origin` and `rapid2`.
- Imported async events are not yet correlated with the original RAPID2 task ID.
- `AsyncBatchFuzzer` pending-task lookup is currently linear per completion,
  which can become O(n^2) when draining large pending queues.
- `UserStats` updates in the async fuzzer are not throttled and may become noisy
  at high completion rates.
- `scheduled_count` is currently updated when completions are evaluated rather
  than when inputs are submitted.
- RAPID2 timeout handling has focused runtime tests, but still lacks a full
  broker/client timeout E2E that proves restart-manager persistence through the
  same path used for long fuzzing campaigns.
- Malformed ArgPack repair policy still uses the current fallback repair path;
  a later pass should decide whether malformed inputs should fail fast or be
  repaired under explicit accounting.

## Phase 1 Fact Collection Limits

Phase 1 records type and layout facts in the manifest. It does not decide the
full current generator support policy, and it must not write generic
`supported` or `unsupported_reason` fields into the manifest.

Known Phase 1 materialization risk facts:

- Private or protected data fields are recorded with
  `materialization_status="unsafe"`.
- Virtual methods or vptr-bearing records are recorded with
  `materialization_status="unsafe"`.
- Non-trivially-copyable records are recorded with
  `materialization_status="unsafe"`.
- Reference fields are recorded with `materialization_status="unsafe"`.
- Const assignment blockers are recorded with
  `materialization_status="unsafe"` and a `const_assignment_blocker` reason.
- Anonymous, local, or otherwise unshimmed scoped types may be recorded as
  `scoped_type_shim_unsupported`.
- Union facts are preserved as `opaque_with_ptr` with `type_info.kind=union`.
- Incomplete or opaque layouts are preserved as `opaque_with_ptr` with
  `layout_status=partial` or `layout_status=opaque`.

Reference: [docs/phase1-implementation-plan.md](phase1-implementation-plan.md)
describes these manifest facts and the distinction between layout discovery and
generator support.

## Phase 2 Fail-Fast Limits

Phase 2 consumes a per-kernel manifest and builds decode/invoke artifacts. It
currently rejects the following structural features:

- Unions: rejected as `union_not_supported`.
- Device or external pointers: rejected as `pointer_role_not_supported`.
- Derived pointers: rejected as `pointer_role_not_supported` until
  `base_arg + offset` materialization is implemented.
- Pointer-like roles on non-pointer layout nodes: rejected as
  `pointer_role_not_supported`.
- Unsafe materialization records, including private/protected fields, virtual
  methods or vptrs, references, non-trivially-copyable records, and const
  assignment blockers: rejected before decode/invoke generation.
- Incomplete or opaque `opaque_with_ptr` layouts: rejected as
  `opaque_with_ptr_layout_incomplete`.
- Non-standard scalar widths outside 1, 2, 4, or 8 bytes: rejected as
  `scalar_size_unsupported`.
- Unsupported type shim boundaries such as anonymous or local scoped types:
  rejected as `scoped_type_shim_unsupported`.
- Per-run manifests with anything other than exactly one kernel are rejected.
- Pointer arguments or nested pointer layout nodes missing `pointer_role` or
  `pointee_layout` are rejected.

Phase 2 VConfig is requested by default through
`launch_policy.vconfig_reserved=true`. The base decode/invoke backend remains
buildable when VConfig is conservatively disabled for synchronization-sensitive
IR. Current VConfig disabled reasons are:

- `vconfig_barrier_unsupported`: data-dependent, divergent,
  helper-contained, non-`barrier0`, or otherwise non-representable
  block-wide barrier operations such as `__syncthreads()` are not rewritten.
  Fixed main-path `barrier0` sites are supported by inserting matching virtual
  sync points on the inactive slicing path before return. This includes simple
  reconverged CFG regions and block-uniform early-return guards. Fixed-trip
  loop barriers are supported for the pass's simple PHI self-loop form and
  Clang `optnone` header/body/latch form. A canonical top-level dynamic loop is
  also supported when its entry condition and bound are block-uniform and its
  induction step is a positive constant. Inactive threads execute only the
  matching sync schedule; they do not execute original memory, atomic, or
  global-lock operations.
- `vconfig_inline_asm_unsupported`: inline assembly is not analyzed or
  rewritten.

NVVM warp intrinsics such as `shfl.sync` are supported only through
warp-aligned adaptation: the pass preserves the logical multidimensional block
shape, VConfig mutation keeps the total logical thread count at a multiple of
32, and Phase 2 backend builds require physical `blockDim.x` candidates that
are multiples of 32. Inline assembly that may hide warp/barrier operations
still disables VConfig.

Set `launch_policy.vconfig_reserved=false` to opt a kernel out explicitly.

Reference: [docs/phase2-implementation-plan.md](phase2-implementation-plan.md)
defines the fail-fast policy for unsupported layout, pointer, union, and
materialization cases.

## Enforcement Points

The current support gates are enforced in:

- [scripts/kernel-rewrite/contracts/kernel.py](../scripts/kernel-rewrite/contracts/kernel.py):
  layout node classification rejects unsupported pointer roles, scalar widths,
  unions, and incomplete `opaque_with_ptr` layouts.
- [scripts/kernel-rewrite/phase2.py](../scripts/kernel-rewrite/phase2.py):
  Phase 2 maps rejected contract reasons into `metadata.phase2.json` and
  `rewrite_summary.json` diagnostics.
- [cuda-kernel/origin/build.py](../cuda-kernel/origin/build.py),
  [cuda-kernel/rapid/build.py](../cuda-kernel/rapid/build.py), and
  [cuda-kernel/rapid2/build.py](../cuda-kernel/rapid2/build.py):
  backend builders reject kernels whose Phase 2 contract is not supported.

## Supported Baseline

The current baseline supports:

- Scalar integer, enum, and floating-point leaves with 1, 2, 4, or 8 byte
  storage.
- Top-level payload buffer pointers with `pointer_role=payload_buffer`.
- Nested `pointer_role=payload_buffer` fields when carried by complete
  field-aware `opaque_with_ptr` layouts and safe materialization facts.
- Pointer-free inline aggregates represented as `opaque_val`.
- Simple public-field structs/classes with complete layouts.
- Fixed-size arrays when their element layout is supported.

All unsupported cases should produce explicit metadata or command diagnostics so
test reports can categorize them without treating the rejection as an
unexpected crash.
