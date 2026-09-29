# RAPID Device-Side Sanitizer Design

This document defines the sanitizer direction for RAPID's persistent CUDA execution model.

## Position

The primary sanitizer path must run inside RAPID's device execution path.

External replay through `compute-sanitizer` is not a sufficient primary design because RAPID2 keeps the persistent kernel running and executes many logical test cases inside that kernel. A process-level external sanitizer can still be useful as a later calibration tool, but it does not solve per-task fast feedback, attribution, or persistent runtime integration.

Host ASan/UBSan is also not part of this sanitizer plan. Those tools are useful for debugging the C++ harness or Rust FFI boundary, but they do not detect target kernel memory, synchronization, or logical-boundary errors. RAPID's sanitizer work should focus on device-side target behavior.

## Why LLVM IR Instrumentation

The sanitizer should be inserted at the LLVM IR/NVVM stage, not by editing user source and not as the first PTX-text rewrite.

Reasons:

- RAPID already captures and rewrites `kernel.bc` / `kernel.device.bc`.
- IR instrumentation does not modify source files; it transforms generated build artifacts.
- IR still has structured operations such as loads, stores, GEPs, calls, arguments, and debug metadata.
- Phase 2 has manifest information needed to connect pointer arguments, payload lengths, and constraints to sanitizer checks.
- PTX text is lower-level and less stable across compiler versions, optimization levels, register allocation, and address-space lowering.

PTX-level instrumentation can be considered later for checks that truly require final lowered instructions, but it should not be the first implementation path.

## Sanitizer Classes

The first useful set should be narrow and tied to RAPID's current manifest model.

### 1. Logical Bounds Sanitizer

Purpose:

- detect accesses beyond the logical payload length or VCONFIG logical boundary,
- catch OOB reads/writes that may not crash immediately,
- report near-boundary writes as structured diagnostics.

Inputs:

- manifest pointer roles,
- payload length decoded from arg-pack,
- VCONFIG logical dimensions,
- generated decode metadata.

Checks:

- instrument loads/stores derived from payload-buffer pointers,
- compute byte or element offset where possible,
- compare against logical length,
- record violation into per-task diagnostic buffer.

This is the highest-priority sanitizer because RAPID's arg-pack model already knows payload buffer lengths.

### 2. Dynamic Canary Sanitizer

Purpose:

- detect writes beyond logical boundaries when static allocation is larger than logical data,
- support persistent execution without reallocating device buffers per test case.

Design:

- before target invocation, initialize a canary region after the logical boundary,
- after invocation, verify the canary region,
- record mismatch as a sanitizer finding.

This is a good second step after direct IR bounds checks. It catches some writes even when the exact IR offset expression is hard to recover.

### 3. Initialization Sanitizer

Purpose:

- detect reads from bytes or logical elements that were never initialized by the current test input or setup path.

Initial scope:

- payload-buffer shadow bits at byte or coarse block granularity,
- mark arg-pack-provided payload bytes initialized,
- mark runtime-created canary regions separately,
- report reads from uninitialized logical regions.

This is lower priority than bounds because it requires shadow metadata and can be noisy.

### 4. Synchronization Sanitizer

Purpose:

- detect unsafe VCONFIG rewrites around block-wide barriers,
- detect likely divergent barrier participation in transformed kernels.

Initial scope:

- classification and conservative rejection during rewrite,
- optional runtime counters around known barrier sites in later phases.

This should not be implemented as a broad race detector first. Race detection is expensive and requires a more complete memory access history.

### 5. Race Sanitizer

Race detection should be future work.

Reason:

- it needs per-address access history across threads,
- it is expensive in the fast path,
- it is harder to make deterministic under persistent async execution,
- and logical bounds bugs are a clearer first target for RAPID.

## L1 Canary MVP

The initial validation lane places a 256-byte `RAPIDCAN` canary immediately
after each task payload. `origin` and `rapid` copy that redzone back during
result collection; RAPID2 includes it in the existing combined-input copy and
checks the task-owned pinned buffer before release. A mismatch is reported as
`CUDA_ERROR` at the `COLLECT` stage and logs the task ID plus the first expected
and actual byte. `RAPID_ENABLE_CANARY` controls this path and defaults to ON.
The GPU-gated corrupt path is covered by
`tests.feedback_e2e.test_canary_e2e.CanaryE2ETest.test_origin_and_rapid2_detect_corrupt_canary`.

This canary detects writes that touch the redzone. It does not detect
out-of-bounds reads, and a large-stride write that skips the entire 256-byte
redzone is not detected.

## Runtime Record Region

Sanitizer findings must be per task.

Add a per-task device-side runtime record region next to coverage and feedback buffers. It should be reset before each target invocation and copied back with task completion.

Suggested first structure:

```c++
enum RapidSanitizerKind : uint32_t {
  RAPID_SAN_NONE = 0,
  RAPID_SAN_OOB_READ = 1,
  RAPID_SAN_OOB_WRITE = 2,
  RAPID_SAN_CANARY_CORRUPTION = 3,
  RAPID_SAN_UNINIT_READ = 4,
  RAPID_SAN_SYNC_UNSUPPORTED = 5,
};

struct RapidSanitizerRecord {
  uint32_t kind;
  uint32_t flags;
  uint64_t task_id;
  uint64_t pc_or_site_id;
  uint64_t access_offset;
  uint64_t access_size;
  uint64_t logical_bound;
  uint32_t arg_index;
  uint32_t reserved;
};
```

The fast path should store only the first finding or a small fixed number of findings per task. The goal is stable fuzzing feedback and reproducible triage, not full tracing.

## Fuzzer Integration

Sanitizer findings should feed both objective and feedback:

- OOB, canary corruption, uninitialized read, and unsupported sync rewrites should map to `ExitKind::Crash` or a dedicated crash/objective class.
- Sanitizer kind and site ID should be recorded in crash metadata.
- Non-crashing near-boundary sanitizer signals can contribute to memory/index novelty feedback.

`TaskResult` can be extended directly or can carry a pointer to a diagnostic buffer, similar to `coverage_ptr`. A pointer-based design avoids copying large diagnostics through the poll API.

## Build Integration

Add sanitizer modes as build-time toggles:

- `none`: no sanitizer instrumentation,
- `bounds`: logical bounds sanitizer,
- `bounds-canary`: logical bounds plus dynamic canary,
- `debug`: heavier assertions and diagnostic metadata.

The default long-running mode should start with `bounds` once the overhead is known. Canary and initialization checks can be enabled selectively.

## Acceptance Criteria

- A fixture with a payload-buffer OOB read/write records a per-task sanitizer finding.
- The finding is attributed to the correct `task_id` after async completion.
- The fuzzer stores sanitizer-triggering inputs in the objective/crash corpus.
- Sanitizer instrumentation does not modify source files.
- Sanitizer-disabled builds preserve the current Phase 2/RAPID2 execution behavior.
- Unsupported barrier/VCONFIG cases fail closed instead of silently producing unsafe rewritten code.

## Non-Goals

- No host ASan/UBSan as part of this design.
- No external `compute-sanitizer` as the primary sanitizer lane.
- No first-version full race detector.
- No full per-thread memory trace in the fast path.
