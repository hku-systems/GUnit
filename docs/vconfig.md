# RAPID VCONFIG Design

This document defines how RAPID should add Virtual Configuration, or VCONFIG, to the persistent RAPID2 execution model.

## Scope Boundary

VCONFIG mutation metadata and virtual-dimension rewriting are staged
separately in the implementation, but both are now wired into the Phase 2
pipeline. The current implementation serializes a VConfig in the LibAFL
`BytesInput` task envelope, mutates/clamps that metadata from manifest
`launch_policy` constraints, and runs the Phase 2 virtual-dimension transform
when `launch_policy.vconfig_reserved` is true or omitted.

The policy is consumed by origin, RAPID, and RAPID2 builders. When VConfig is
enabled, all three backends use the selected physical launch shape and pass the
clamped per-input VConfig through the RAPID context. Persistent RAPID and
RAPID2 already keep their physical CUDA launch fixed; origin uses the same
physical/logical split for VConfig-enabled targets. Rewritten targets read
logical `gridDim` and `blockDim` from
`RapidKernelContext::vconfig` and use an active-thread guard for logical thread
ranges.

The feedback runtime may report buckets derived from the actual CUDA launch
shape for actual CUDA execution. The fuzzer-owned task envelope carries one
`RapidVConfig` next to the arg-pack payload. Backends clamp it before H2D
copying. VConfig-enabled builds launch the wrapper/persistent kernel with the
selected physical shape, while the rewritten target observes the clamped
logical dimensions through the RAPID context. VConfig-disabled origin builds
keep the legacy per-input relaunch behavior. Generated GPU prepare code copies
the resulting VConfig into `RapidKernelContext::vconfig` for every task. The
field contains only the six dimensions used by the current design; additional
future controls require a new internal ABI revision.

## Motivation

RAPID2 launches a persistent physical kernel once and then repeatedly invokes the rewritten target kernel for many fuzz inputs. This removes launch overhead, but it also freezes the physical launch configuration. Many CUDA bugs depend on launch configuration, especially:

- `threadIdx` / `blockIdx` index arithmetic,
- `blockDim` / `gridDim` dependent bounds checks,
- near-boundary thread ranges,
- warp-aligned assumptions,
- and hidden out-of-bounds behavior masked by conservative host launch dimensions.

VCONFIG solves this by keeping the physical persistent launch fixed while giving each task a logical launch configuration.

## Current State

The repository has both a standalone virtual-dimension test path and Phase 2
integration:

- `tests/vconfig/` documents and validates runtime virtual dims.
- `tools/rapid-vconfig-instrument/virtual_dim_pass.cpp` recognizes `vdim` / `vdim-args` annotations.
- The pass can append virtual grid/block arguments, replace `gridDim` and `blockDim` builtin uses, and insert an active-thread guard.
- `tools/rapid-vconfig-instrument/` builds a repository-local tool that runs the shared pass over Phase 2 `kernel.device.bc`.
- `scripts/kernel-rewrite/rewrite/executor.py` invokes that tool after the base `__global__ -> __device__` rewrite and verifies the resulting entry ABI.
- `tests/feedback_e2e/test_vconfig_e2e.py` builds a real fixture through capture, Phase 1, Phase 2, and all three backends, then proves that two seeds differing only in envelope `blockDim.x` produce different feedback.

The backend-neutral transport foundation remains unchanged:
`RapidTaskEnvelopeHeader` carries the fixed VConfig and Phase 2 generated
prepare code publishes it into `RapidKernelContext`.

Phase 2 records VConfig status in `build_spec.json`,
`metadata.phase2.json`, and `rewrite_summary.json`:

```json
{
  "vconfig_requested": true,
  "vconfig_enabled": true,
  "vconfig_warp_aligned": false
}
```

Backend build reports carry the full launch configuration, which extends the
manifest `launch_policy` contract keys (`grid`, `block_candidates`,
`physical_block_max`, `target_dynamic_shared_bytes`, `coverage_memory`,
`vconfig_reserved`) with these VConfig extension keys: `logical_grid`,
`logical_block`, `has_logical_vconfig_bounds`, `vconfig_enabled`,
`vconfig_mutation`, and `vconfig_warp_aligned`. Validation treats the report
as a closed schema: contract keys must match exactly, only these extension
keys are permitted, and any other key is a drift error.

When the transform is requested but conservatively disabled, the artifacts keep
the kernel buildable and include a stable reason:

```json
{
  "vconfig_requested": true,
  "vconfig_enabled": false,
  "vconfig_disabled_reason": "vconfig_barrier_unsupported"
}
```

Current stable disabled reasons are:

- `vconfig_barrier_unsupported` for data-dependent, divergent,
  non-`barrier0`, or otherwise non-representable block-wide barriers,
  including barriers inside cyclic or otherwise non-representable helper
  control flow (barriers in acyclic helpers are supported).
- `vconfig_inline_asm_unsupported`

When NVVM warp intrinsics are present and the transform succeeds, Phase 2 also
records `"vconfig_warp_aligned": true`. Backend builds then filter physical
block candidates to multiples of 32. The pass preserves the logical x/y/z
shape, while VConfig normalization keeps the total logical thread count at a
whole-warp boundary before active-thread slicing.

Setting `launch_policy.vconfig_reserved` to `false` opts a kernel out of the
Phase 2 VConfig transform and records `vconfig_requested=false`.

## Design Rule

VCONFIG is runtime metadata in the task envelope, not part of the raw arg-pack
payload.

The raw `arg-pack-v1` payload remains the source for original kernel arguments only. VCONFIG must travel through a separate metadata channel so that:

- existing decode semantics remain stable,
- corpus inputs are not tied to a backend-specific control channel,
- origin, RAPID, and RAPID2 can share the same decoded kernel arguments,
- and VCONFIG can be enabled, disabled, clamped, or replayed independently.

The current LibAFL `BytesInput` is the separate transport channel. It is a full
task envelope followed by the unchanged arg-pack payload:

```text
offset  size  field
0       24    RapidVConfig {grid_x, grid_y, grid_z, block_x, block_y, block_z}
24      8     payload_size, little-endian uint64
32      N     arg-pack-v1 payload bytes
```

`payload_size` is the byte length of the arg-pack payload only. It is already
inside `RapidTaskEnvelopeHeader`, so the H2D transfer size is:

```text
sizeof(RapidTaskEnvelopeHeader) + payload_size
= 32 + payload_size
```

Do not add another `sizeof(payload_size)` term; that would double-count the
8-byte field at offset 24.

## VCONFIG Structure

The first structure should be small and POD-compatible across host/device code:

```c++
struct RapidVConfig {
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
};
```

Required invariants:

- each dimension is at least 1,
- logical dims are less than or equal to physical dims supported by the backend,
- `block_x * block_y * block_z` fits the physical block capacity,
- when VConfig mutation is enabled, `block_x * block_y * block_z` is aligned to
  the warp size so slicing never removes only part of a warp.

## Phase A: Single-Block VCONFIG

The first implementation should support only one physical block:

```text
physical grid = (1, 1, 1)
physical block = current RAPID2 fixed block, currently 1024 threads
logical grid = (1, 1, 1)
logical block = mutated/clamped VCONFIG block dimensions
```

Reason:

- current RAPID2 uses per-block shared state in the persistent loop,
- `iteration_count` is `__shared__`,
- ready/done signaling is designed around a single persistent execution participant,
- expanding physical grid requires a new multi-block synchronization protocol.

This phase is still useful because many kernels depend primarily on `threadIdx` and `blockDim`.

## Phase B: Metadata Channel Integration

The transport foundation of this phase is complete for all three backends:

- `RapidTaskEnvelopeHeader` stores `RapidVConfig` and payload size immediately
  before the unchanged arg-pack bytes;
- LibAFL `BytesInput` stores the complete envelope;
- origin, RAPID, and RAPID2 validate the submitted envelope and clamp its
  VConfig instead of wrapping raw payload bytes again;
- origin uses the selected physical launch shape for VConfig-enabled targets
  and keeps per-input relaunch only when VConfig is disabled;
- persistent RAPID/RAPID2 clamp each input VConfig to the physical launch shape
  selected at backend startup;
- VConfig, payload size, and payload move H2D in one contiguous transfer;
- generated `fuzzer_feedback_prepare_v1` copies VConfig into the current device
  context before `fuzzer_invoke_v1(decoded, context)`.

The fuzzer represents, validates, mutates, serializes, and replays VConfig
metadata independently of arg-pack payload bytes.

## Phase C: Phase 2 Rewrite

Phase 2 makes virtual-dimension semantics part of the generated contract when
VConfig is enabled for a kernel:

- mark rewritten targets that are VCONFIG-enabled,
- read logical dimensions from the existing trailing
  `RapidKernelContext *context` instead of adding another entry parameter,
- replace `gridDim` and `blockDim` reads with values from `context->vconfig`,
- insert active-thread handling for `threadIdx` / `blockIdx` ranges,
- reject or conservatively disable VCONFIG for unsupported synchronization patterns.

The integration target is RAPID's generated Phase 2 device bitcode. The base
callable entry is always produced first. If the VConfig transform succeeds,
`phase2/kernel.device.bc` is replaced with the VConfig-enabled bitcode. If the
transform returns a stable unsupported reason, Phase 2 keeps the verified base
bitcode and records `vconfig_enabled=false` instead of rejecting the whole
kernel.

## Barrier and Warp Limits

Simple early return is not universally safe.

If a kernel contains `__syncthreads()` or equivalent block-wide barriers, inactive physical threads may still need to participate in synchronization. Otherwise, active threads can deadlock waiting at a barrier that inactive threads skipped.

Implemented support is intentionally conservative:

- no barrier: entry guard and early return are acceptable,
- acyclic `barrier0` sites that post-dominate entry, including simple
  reconverged CFG regions, are supported by inserting the same number of
  virtual CTA sync points on the inactive slicing path before return,
- a block-uniform early-return guard may be delayed until after its barrier
  region so inactive physical threads still follow the region's sync schedule,
- fixed-trip loops with `barrier0` sites are supported for the pass's simple
  PHI self-loop form and Clang `optnone` header/body/latch form by emitting an
  inactive sync-only loop with the same trip count and per-iteration barrier
  count,
- one canonical top-level dynamic loop is supported when its entry condition
  and bound are block-uniform, its induction step is a positive constant, and
  every barrier dominates the corresponding latch. Inactive threads clone only
  the uniform loop control and execute the same per-iteration barrier count;
  they do not execute the original loads, stores, atomics, or global-lock path,
- data-dependent, divergent, helper-contained, non-canonical dynamic-loop, or
  non-`barrier0` synchronization is rejected conservatively,
- NVVM warp primitives use warp-aligned adaptation: physical block candidates
  must be multiples of 32, the logical multidimensional block shape is
  preserved, and slicing is restricted to a whole number of logical warps.
  Inline assembly that may hide warp behavior is still rejected.

## Mutator Strategy

The fuzzer mutates VCONFIG separately from arg-pack payload bytes. The current
`RapidInputMutator` selects one of five equally weighted plans: payload structure,
payload havoc, VCONFIG only, structure plus VCONFIG, or havoc plus VCONFIG. A
combined plan mutates the two channels independently and then normalizes the full
envelope. When VCONFIG mutation is disabled, only the two payload plans remain
eligible, so the VCONFIG header is preserved.

Initial strategies:

- near-original: prefer configs close to the original/static launch shape when known,
- small/extreme: try tiny logical blocks to expose boundary and serial behavior,
- random legal: sample legal dimensions under the physical cap,
- warp-aware: canonicalize the product `block_x * block_y * block_z` to a
  multiple of 32 when VConfig mutation is enabled, so multidimensional logical
  shapes do not create partial-warp slicing.

VCONFIG should become corpus-worthy only when combined with feedback novelty or diagnostics. Numeric novelty alone is not enough.

### Fuzzer Representation

The implemented fuzzer representation keeps LibAFL's `BytesInput`, but the
bytes are the full task envelope. Envelope-aware normalize/mutate code applies
arg-pack mutations only to the payload slice and VConfig mutations only to the
header slice.

The representation preserves these boundaries:

- raw `arg-pack-v1` bytes continue to encode original kernel arguments only;
- arg-pack mutators operate on the payload after `RapidTaskEnvelopeHeader`;
- VCONFIG validation and mutation remain a separate metadata path driven by
  manifest `launch_policy` bounds;
- backend H2D copy transfers `RapidTaskEnvelopeHeader + payload`, not a nested
  envelope.

## Multi-Block Future Work

Supporting logical `gridDim > 1` requires a RAPID2 runtime redesign:

- physical grid must contain multiple blocks,
- all physical blocks must agree on the current slot and iteration,
- done signaling must wait for all participating physical blocks,
- per-task coverage merge must combine all physical blocks,
- inactive logical blocks must not corrupt persistent runtime state.

This should not be folded into the first VCONFIG implementation.

## Acceptance Criteria

- Single-block logical `blockDim` can vary per task without relaunching the persistent kernel.
- The same arg-pack payload can be replayed under multiple VCONFIGs.
- Invalid VCONFIG values are clamped or rejected before device execution.
- Kernels with `vconfig_enabled=true` observe logical `gridDim`/`blockDim`
  through `RapidKernelContext::vconfig`.
- Kernels with `vconfig_enabled=false` keep the base
  `fuzzer_invoke_v1(decoded, context)` path; origin can still vary physical
  CUDA launch by relaunching per input, while persistent RAPID/RAPID2 kernels
  observe their selected physical CUDA dimensions.
- Barrier and warp-primitive kernels either get the supported sync/warp
  adaptation above or are classified conservatively instead of silently
  rewritten unsafely.
