# RAPID Feedback Design

This document is the single source of truth for RAPID's device-side feedback
semantics, instrumentation, transport, and LibAFL integration. It covers CFG
coverage, SIMT memory features, and thread/launch activity for `origin`,
`rapid`, and `rapid2`.

## Status and Metric Migration

RAPID implements the complete feedback transport path: automatic LLVM
instrumentation, per-task maps, payload bounds, backend completion attribution,
Rust novelty feedback, and cumulative RQ2 telemetry. The source implementation
now generates the `rapid-simt-memcov-v1` device-memory metric.

`rapid-simt-memcov-v1`, defined below, is the direct semantic replacement for
the historical coarse `rapid-memory-bitset-v1` metric. It is not a second
long-lived feedback mode:

```text
historical device helper semantics
rapid-memory-bitset-v1
        |
        | replace feature calculation in place
        v
current source helper semantics
rapid-simt-memcov-v1
```

The replacement keeps the existing bitmap storage and transport ABI but
changes the meaning and device-side generation of memory bits `0..61,439`.
Normal feedback-enabled builds emit only
`rapid-simt-memcov-v1`; there is no runtime selector for the old metric.

Historical artifacts remain tagged with `rapid-memory-bitset-v1`. Because the
two versions assign different meanings to bitmap bits, an old curve cannot be
continued, unioned, or normalized with a new SIMT MemCov curve. Formal RQ2
collection must rebuild all backends and rerun the full matrix after the
replacement.

The source replacement, deterministic direct-GPU semantic oracle, metadata
contract, RQ2 version checks, and fresh all-backend equivalence artifacts are
implemented. The sequential six-workload 60-second performance pilot remains
an acceptance gate. Old pre-replacement binaries still implement the
historical metric and must not be relabeled or reused for formal RQ2.

## Feedback Domains

RAPID exposes three independent domains:

```text
CFG coverage + SIMT memory features + thread/launch activity
```

They share a task/completion lifecycle but not a semantic denominator.

| Domain | Meaning | Storage |
|---|---|---|
| CFG coverage | Rewritten target basic blocks reached by any active logical thread | 65,536-byte edge map |
| SIMT MemCov | Payload-relative global-memory sectors reached by logical warps with a classified access pattern | Bits `0..61,439` of an 8 KiB bitset |
| Thread/launch activity | Coarse actual CUDA launch-shape features | Bits `61,440..65,535` of the same bitset |

An input is interesting when it improves any enabled feedback domain or
triggers an objective such as crash, timeout, or sanitizer finding. These
domains must be reported separately; their raw feature counts must not be
added and presented as one coverage percentage.

## Implemented Feedback Infrastructure

The following infrastructure is already implemented and is reused by SIMT
MemCov:

- one shared LLVM/NVVM instrumentation stage for `origin`, `rapid`, and
  `rapid2`;
- set-like task-local CFG coverage;
- the hidden trailing `RapidKernelContext *` Phase 2 entry ABI;
- generated payload bounds indexed by a dense static `arg_slot`;
- one 8 KiB task-local memory/thread bitset;
- exact task-owned maps returned through ordered `TaskResult` completion;
- structured backend status and dispatch-time watchdog handling;
- Rust `SimtMemCovFeedback` with persistent seen-state metadata;
- completion-time cumulative telemetry for RQ2;
- instrumented and uninstrumented backend build modes; and
- builtin and supported third-party feedback validation.

Exact validation commands and the boundary of the existing runtime evidence
are in [`feedback-validation.md`](feedback-validation.md).

## Feedback Channel 1: CFG Coverage

### Meaning

The LLVM pass instruments basic blocks in rewritten target device code. Each
task records whether a static CFG site was reached by at least one active
logical thread. The CFG map remains 65,536 bytes, with one byte per hashed
site. Every thread that reaches a site performs the same idempotent
`atomicOr(1)` update, so each byte is either zero or one. Dynamic warp or thread
execution counts are not part of the CFG metric.

The only supported CFG contract is `rapid-cfg-presence-v1`. Instrumentation
metadata predating the explicit result contract has no `cfg_metric_version`;
RQ2 loaders normalize that absence to `rapid-cfg-presence-v1` when writing or
reading trial and sample records. Any other CFG metric version is rejected.

### Stable sites and maps

Each instrumented block receives a stable site ID derived from kernel,
function, and block identity:

```text
site_id = stable_hash(kernel_id, function_id, basic_block_id) % 65536
```

The inserted probe is:

```c++
__rapid_feedback_bb(site_id);
```

The helper preserves the original per-thread set-like update. `origin`,
`rapid`, and `rapid2` all return a complete task map; no backend may expose a
map that is still being modified by another task.

### Known limitation: conditional expressions

By default, CFG instrumentation covers basic blocks in the rewritten entry
function. A source conditional expression that survives optimization as an
LLVM `select` creates no control-flow edge or additional basic block, so it
adds no CFG site and does not reveal which value arm was selected. Code left
in a non-inlined device helper is also outside the entry-function scope.

`rapid-feedback-instrument --instrument-selects` optionally emits one
single-sided pseudo-site when each entry-function `SelectInst` is evaluated.
The flag is off by default. Enabled metadata keeps these probes separate as
`instrumented_select_sites` and `select_sites` entries with
`site_kind=select_evaluated`; they still do not distinguish the true and false
arms. In that mode, `instrumented_cfg_sites` and `cfg_sites` include all
CFG-map probes, while `instrumented_basic_block_sites` and
`site_kind=basic_block` identify the real basic-block subset.

### Reporting

RQ2 may report:

```text
instrumented_cfg_sites = static probes emitted for the built kernel
cfg_sites(t)            = popcount(campaign union by completion time)
cfg_fraction(t)         = cfg_sites(t) / instrumented_cfg_sites
cfg_map_density(t)      = cfg_sites(t) / 65536
```

`cfg_fraction` is valid only when build metadata provides a trustworthy static
site count and comparable configurations use the same site-ID scheme and
kernel artifact. Map density is a collision/saturation diagnostic, not the
paper's device-coverage percentage.

Those formulas describe the only supported bit-only presence contract. Result
loading, verification, and plotting reject any non-presence CFG metric
version.

## Feedback Channel 2: SIMT MemCov

### Scope

Version `rapid-simt-memcov-v1` instruments only global-memory loads and stores
whose pointer provenance maps unambiguously to a generated payload
`arg_slot`. The payload base and byte length must be available in
`RapidKernelContext::feedback.bounds[arg_slot]`.

Included pointer forms are the direct, derived, spilled, and nested payload
forms already resolved by `rapid-feedback-instrument`. Unsupported memory
instructions remain counted as `unknown_memory_sites` and receive no memory
probe.

Version 1 excludes:

- shared, local, and constant memory;
- global symbols, heap allocations, and device allocations without payload
  provenance;
- raw pointer hashing;
- partially or fully out-of-bounds accesses;
- boundary/distance/alignment buckets;
- exact active masks, per-thread traces, timestamps, and access order;
- cross-event sharing tables; and
- inferred races or happens-before relationships.

Sanitizer feedback owns invalid-access reporting. SIMT MemCov is a fuzzing
signal that can steer mutation toward race-relevant access structures; it is
not a race detector.

### Formal feature

Let \(\tau\) be one completed task. Let \(e\) be one dynamic execution of an
instrumented memory instruction by one warp, and let \(m\) be one stable
payload-relative memory location touched by that event. The semantic feature
is:

\[
\boxed{
K(e,m)=
\left\langle
site(e),
argSlot(e),
relativeSector(m),
op(e),
logicalBlock(e),
warpInBlock(e),
pattern(e)
\right\rangle
}
\]

This tuple answers:

```text
which static access site
accessed which stable payload sector
as a read or write
from which logical VConfig warp
with which warp access pattern
```

It deliberately does not answer which individual thread ran first or whether
two accesses raced.

### Stable payload-relative location

For lane \(l\), let:

\[
base=base(argSlot(e)),\qquad len=lenBytes(argSlot(e))
\]

Let \(a(e,l)\) be the integer access address and \(z(e)\) the access width.
The lane contributes only when:

\[
z(e)>0,\qquad z(e)\le len,\qquad a(e,l)\ge base
\]

and, after computing \(offset(e,l)=a(e,l)-base\):

\[
offset(e,l)\le len-z(e)
\]

The comparison uses `offset <= len - width`, not an unchecked
`offset + width <= len`, so the device implementation does not depend on
overflow behavior.

Every 32-byte sector touched by the complete access contributes:

\[
q\in
\left[
\left\lfloor\frac{offset(e,l)}{32}\right\rfloor,
\left\lfloor\frac{offset(e,l)+z(e)-1}{32}\right\rfloor
\right]
\]

The stable location is:

\[
\boxed{
m(e,l,q)=\left\langle argSlot(e),q\right\rangle
}
\]

`arg_slot` identifies the stable payload object and `q` identifies the
object-relative 32-byte sector. ASLR, allocator reuse, backend allocation
order, and per-run device addresses therefore do not change the feature.

Read and write encodings are fixed:

| Operation | Encoding |
|---|---:|
| Read | `0` |
| Write | `1` |

Width is not a separate final-key field. It is static at one site and already
affects the touched-sector set and contiguous-pattern calculation.

### Logical VConfig warp

Let `(Gx, Gy, Gz)` and `(Bx, By, Bz)` be the logical grid and block dimensions
from the current task's VConfig. For an active rewritten target thread:

\[
b_v=blockIdx.x+G_x\left(blockIdx.y+G_y\,blockIdx.z\right)
\]

\[
t_v=threadIdx.x+B_x\left(threadIdx.y+B_y\,threadIdx.z\right)
\]

\[
\boxed{
w_v(e)=
\left\langle
b_v,
\left\lfloor\frac{t_v}{32}\right\rfloor
\right\rangle
}
\]

The feature retains the pair `(logical block linear ID, warp-in-block ID)`;
it does not flatten them into one campaign-wide warp number. The logical
dimensions come from VConfig rather than the persistent executor's fixed
physical launch dimensions.

The VConfig transform preserves each active physical block/thread linear
ordinal while decomposing that ordinal into rewritten logical x/y/z
coordinates. The helper therefore realizes `(logical block linear ID,
warp-in-block ID)` directly from the corresponding physical-linear ordinals.
It does not mix raw physical x/y/z coordinates with logical multidimensional
strides at every probe.

### Warp access pattern

Let \(R(e)\) be the CUDA active mask at the inserted helper call. Every lane in
that mask evaluates the overflow-safe in-bounds predicate. One ballot produces:

\[
A(e)=\{l\in R(e)\mid access(e,l)\text{ is complete and in bounds}\}
\]

Let \(n(e)=popcount(A(e))\), and let \(l_0\) be the lowest valid lane. An empty
valid mask emits no memory feature. Version 1 freezes exactly seven classes:

| Pattern | Encoding | Definition |
|---|---:|---|
| `SINGLE` | `0` | \(n=1\) |
| `FULL_BROADCAST` | `1` | \(n=32\) and every valid lane uses the same offset |
| `FULL_CONTIGUOUS` | `2` | \(n=32\) and offsets are lane-contiguous |
| `FULL_OTHER` | `3` | \(n=32\) and neither full pattern above holds |
| `PARTIAL_BROADCAST` | `4` | \(1<n<32\) and every valid lane uses the same offset |
| `PARTIAL_CONTIGUOUS` | `5` | \(1<n<32\), valid lanes form one interval, and offsets are lane-contiguous |
| `PARTIAL_OTHER` | `6` | \(1<n<32\) and neither partial pattern above holds |

Broadcast means:

\[
\forall l\in A(e),\quad offset(e,l)=offset(e,l_0)
\]

Contiguous means that valid lanes form one contiguous lane interval and:

\[
\forall l\in A(e),\quad
offset(e,l)=offset(e,l_0)+(l-l_0)\,z(e)
\]

The lane-interval requirement prevents the contiguous fast path from inventing
sectors for inactive holes. A sparse mask with otherwise regular offsets is
`PARTIAL_OTHER`.

The exact active mask is not part of the final key. It is used only for
full/partial classification and leader election, avoiding feature explosion
from incidental divergence masks.

### Versioned bitmap hash

The semantic tuple is mapped into the 61,440-bit memory partition with one
field-salted word and the existing `rapid_feedback_mix32`.

Define:

\[
fold_{64}(z)=lo_{32}(z)\oplus rotl_{32}(hi_{32}(z),16)
\]

With multiplication and XOR evaluated modulo \(2^{32}\):

\[
\begin{aligned}
x={}&site
\oplus C_1\,argSlot
\oplus C_2\,lo_{32}(sector)
\oplus C_3\,hi_{32}(sector)\\
&\oplus C_4\,op
\oplus C_5\,fold_{64}(logicalBlock)
\oplus C_6\,warpInBlock
\oplus C_7\,pattern
\end{aligned}
\]

\[
\boxed{
i(e,m)=rapid\_feedback\_mix32(x)\bmod 61440
}
\]

Version 1 freezes these distinct odd salts:

```text
C1 = 0x9e3779b9
C2 = 0x85ebca6b
C3 = 0xc2b2ae35
C4 = 0x27d4eb2f
C5 = 0x165667b1
C6 = 0xd3a2646d
C7 = 0xfd7046c5
```

It also freezes the 32-byte sector size, field order, operation and pattern
encodings, logical-warp formula, `rapid_feedback_mix32`, and memory partition.
Changing any of them requires a new metric version. An optimization that
produces exactly the same semantic tuple and bitmap index does not.

### Task bitmap, novelty, and RQ2 curve

Let \(\mathcal E_\tau\) be the dynamic warp-memory events of task \(\tau\), and
let \(\mathcal M(e)\) contain the unique payload-relative sectors touched by
event \(e\). The task bitmap is:

\[
B_{mem}(\tau)[j]=
\bigvee_{e\in\mathcal E_\tau}
\bigvee_{m\in\mathcal M(e)}
[j=i(e,m)]
\]

Repeated lanes, loop iterations, and replays that resolve to the same index
remain one set-like feature. Campaign novelty is:

\[
Seen_k=Seen_{k-1}\lor B_{mem}(\tau_k)
\]

\[
Novel_k=B_{mem}(\tau_k)\land\neg Seen_{k-1}
\]

RQ2 reports the completion-time union:

\[
\boxed{
SIMTMemCov(t)=
popcount\left(
\bigvee_{\tau_i:\ completion(\tau_i)\le t}
B_{mem}(\tau_i)
\right)
}
\]

For `rapid` and `rapid2`, telemetry updates only when the matching completed
`TaskResult` is consumed, never merely when an input is submitted.

Because the reachable tuple space is input- and VConfig-dependent and the map
is hashed, version 1 has no defensible total denominator. Reports use **SIMT
memory features** or **SIMT memory novelty**, not “X% memory coverage.”

Bitmap density remains a saturation diagnostic:

\[
density(t)=SIMTMemCov(t)/61440
\]

Curves are comparable only when metric version, bitmap partition,
instrumentation metadata, kernel artifact, initial corpus, seed policy, and
time budget match.

### Example

Suppose logical warp 2 at static store site `17` writes four bytes per lane to
payload slot `0`:

\[
offset_l=256+4l,\qquad l=0,\ldots,31
\]

The event is `FULL_CONTIGUOUS` and touches bytes `256..383`, which are sectors
`8..11`. It emits four semantic features:

```text
<17, 0,  8, W, <block=0, warp=2>, FULL_CONTIGUOUS>
<17, 0,  9, W, <block=0, warp=2>, FULL_CONTIGUOUS>
<17, 0, 10, W, <block=0, warp=2>, FULL_CONTIGUOUS>
<17, 0, 11, W, <block=0, warp=2>, FULL_CONTIGUOUS>
```

One leader records the sector range; the other 31 lanes do not perform final
hashes or atomics.

For a grid-stride loop such as `sum[i] = a[i] + b[i]`, spatial sectors may
saturate quickly because the kernel eventually covers the whole input. SIMT
MemCov can still distinguish the logical warp and access pattern reaching a
sector. It cannot distinguish executions with the same site, sector, R/W,
logical warp, and pattern but different temporal interleavings.

### Warp-level recording algorithm

The external helper shape remains unchanged:

```c++
extern "C" __device__ __noinline__ void __rapid_feedback_mem(
    RapidKernelContext *context,
    uint32_t site_id,
    uint16_t arg_slot,
    uint8_t access_kind,
    uint8_t width,
    const void *access_ptr);
```

Every lane reaching the instrumented instruction calls the helper. The helper
captures the call-site active mask, ballots valid lanes, classifies the event
with the contiguous check first, and selects one of three paths:

```text
broadcast
  first valid lane enumerates the access's sectors

contiguous
  first valid lane derives the exact continuous first..last sector range

other
  if every valid lane touches one sector, match lanes by sector and let one
  leader per distinct sector emit; those leaders can issue in parallel
  otherwise, advance by the bounded lane-local sector span and emit a sector
  only from the lowest-numbered valid lane whose range covers that sector
```

Only a broadcast/contiguous leader or elected distinct-sector leader performs:

```text
field-salted key construction
rapid_feedback_mix32
modulo 61440
atomicOr
```

Version 1 does not use per-lane final hashing, sampling, a block shared-memory
cache, exact-mask hashing, or per-event tracing. Sampling changes semantic
reachability; a cache adds shared-memory pressure and a flush protocol beside
the existing CFG allocation.

## Feedback Channel 3: Thread/Launch Activity

Thread/launch activity occupies bits `61,440..65,535` of the 8 KiB bitset. The
current helper records coarse features from the actual CUDA launch:

- physical thread count;
- physical warp count;
- partial-warp presence;
- physical grid x/y/z; and
- physical block x/y/z.

This channel verifies actual backend execution shape. VConfig is an
experimental factor rather than a fourth coverage domain: VConfig-enabled
targets additionally observe logical dimensions through
`RapidKernelContext::vconfig`, which can change CFG and SIMT MemCov.

Thread activity remains versioned and reported separately from SIMT memory
features even though both partitions share one transport buffer.

## LLVM Instrumentation and Metadata

### Build insertion point

All feedback-enabled backends use the same device-bitcode pipeline:

```text
wrapper.cu -> wrapper.bc
phase2/kernel.device.bc
llvm-link -> linked.bc
rapid-feedback-instrument -> instrumented.bc + feedback_metadata.json
llc -> PTX
```

Instrumentation occurs after `llvm-link` and before `llc`; user source files
remain unchanged.

The complete RQ2 CFG path is:

```text
benchmark/rq2/build.py
  -> benchmark/rq1/build.py backend command plan
  -> cuda-kernel/{origin,rapid,rapid2}/build.py
  -> cuda-kernel/backend_build.py
       -> rapid-feedback-instrument
  -> cuda-kernel/utils/feedback/feedback.cuh::__rapid_feedback_bb
  -> feedback_metadata.json
  -> benchmark/rq2/campaign.py trial/sample fields
  -> benchmark/rq2/verify_results.py and plot_coverage.py
```

### Probe scope

The pass instruments the rewritten target entry body. Runtime wrappers,
queues, feedback helpers, and sanitizer helpers are excluded. Basic blocks
receive `__rapid_feedback_bb`. Loads/stores receive `__rapid_feedback_mem` only
when pointer provenance resolves to a monitored payload slot.

The pass already carries:

```text
RapidKernelContext *context
mem_site_id
arg_slot
read/write
access width
access pointer
```

No additional probe argument or kernel-entry ABI field is required for SIMT
MemCov.

### Metadata contract

`feedback_metadata.json` records:

- feedback ABI version;
- the static CFG site table used by the presence metric;
- memory metric version;
- edge, memory, and thread-activity map sizes;
- memory sector size and pattern encodings;
- kernel ID and rewritten entry symbol;
- CFG site table;
- memory site table with `mem_site_id`, function/instruction, `arg_slot`, R/W,
  and width; and
- instrumented and unknown memory-site counts.

The replacement metadata includes:

```json
{
  "memory_metric_version": "rapid-simt-memcov-v1",
  "memory_map_bits": 61440,
  "memory_sector_bytes": 32,
  "thread_activity_map_bits": 4096
}
```

Unknown memory sites stay explicit; metadata never invents raw-address
identities for them.

## Kernel Context and Input ABI

Feedback uses the generated hidden trailing context:

```c++
struct RapidPayloadBounds {
  uintptr_t base;
  uint64_t len_bytes;
};

struct RapidFeedbackTaskContext {
  uintptr_t simt_memcov_bits_addr;
  uint32_t bounds_count;
  RapidPayloadBounds bounds[RAPID_PAYLOAD_SLOT_STORAGE_COUNT];
};

struct RapidKernelContext {
  RapidVConfig vconfig;
  RapidFeedbackTaskContext feedback;
};
```

`arg_slot` is a static probe constant. It directly selects one generated
payload bound, so no runtime object lookup or raw pointer table is required.
`simt_memcov_bits_addr` points to the current task's 8 KiB device bitmap.

The context address remains stable for the lifetime of a backend buffer.
Backends initialize the bitmap pointer once and update per-task VConfig and
active payload bounds after decode. They clear map contents per task; they do
not recopy a full fixed-capacity context for every execution.

Every submitted LibAFL `BytesInput` is one complete envelope:

```text
offset  size  field
0       24    RapidVConfig {grid_x, grid_y, grid_z, block_x, block_y, block_z}
24      8     payload_size, little-endian uint64
32      N     arg-pack-v1 payload bytes
```

The transfer length is exactly:

```text
sizeof(RapidTaskEnvelopeHeader) + payload_size = 32 + payload_size
```

The 8-byte size field is already part of the 32-byte header and must not be
counted twice. Detailed envelope rules are in
[`input-envelope-v1.md`](input-envelope-v1.md), while logical-dimension policy
is in [`vconfig.md`](vconfig.md).

## Runtime Collection and Backend Attribution

### Map layout

The constants remain:

```text
MAP_SIZE                    = 65536 bytes
SIMT_MEMCOV_STORAGE_SIZE    = 8192 bytes
SIMT_MEMCOV_DATA_BUCKETS    = 61440 bits
THREAD_ACTIVITY_BUCKET_BASE = 61440
THREAD_ACTIVITY_BUCKETS     = 4096 bits
```

The former coarse-metric names have been renamed: `libafl_mem_index_bits` is
now `libafl_simt_memcov_bits`; `mem_index_ptr` and `mem_index_size` are now
`simt_memcov_ptr` and `simt_memcov_size`; and `BitsetNoveltyFeedback` is now
`SimtMemCovFeedback`. These are naming-only ABI/source changes: the exported
symbol spelling changes, but bitmap size and `TaskResult` field layout do not.

### origin

`origin` executes one synchronous task, clears task maps, decodes the envelope,
publishes context bounds/VConfig, invokes the target, and copies complete maps
back before returning. Its exported observer maps are exact for that task.

### rapid

`rapid` uses ordered submit/poll/release completion. Completed tasks own stable
edge and bitset buffers until Rust releases their task IDs. Feedback attribution
is exact and ordered; there is no synchronous `libafl_target` entry point.

### rapid2

`rapid2` uses asynchronous persistent execution with task-owned double-buffered
maps. The collector copies both maps before publishing `TaskResult`; Rust then
evaluates the matching pending input and releases the task buffer.

The common completion order is:

```text
poll task result
  -> map backend status to ExitKind
  -> copy task-owned maps into observers/telemetry
  -> evaluate feedback and objectives for the matching input
  -> release task buffers
  -> restart on a context-poisoning terminal status
```

The ordered fuzzing window is capped at 32. RAPID2 pending/completed queues
hold 64 entries, so an accepted window must remain below queue capacity or a
single submit/poll thread can lose liveness.

`TaskResult.edge_ptr` and `TaskResult.simt_memcov_ptr` remain valid only until
`libafl_release_tasks()` for that task ID.

## LibAFL Novelty and Telemetry

CFG uses the normal LibAFL map observer path. The combined memory/thread bitset
uses `SimtMemCovFeedback`, which computes exact new bits against persistent
seen-state metadata.

For one completed task:

```text
current task bitset
  -> compare with persistent seen bitset
  -> retain exact new bucket IDs in testcase metadata
  -> OR current bits into seen state
```

The persistent state separates new bits by partition:

```text
SIMT memory bucket IDs      0..61439
thread activity bucket IDs 61440..65535
```

RQ2 `CoverageTelemetry` owns observation-only maps separate from corpus
admission. It ORs only completed task maps and maintains the cumulative bitmaps
and counts internally. Completion-event schema-1 logging serializes the zero
baseline, cumulative counts at real change points, exact count deltas, and the
terminal state; the full bitmap remains available only through the diagnostic
snapshot API. This schema is intentionally incompatible with the retired
polling/full-bitmap telemetry.

The existing generic telemetry field `memory_features` remains unchanged. It
must always be interpreted through `memory_metric_version`. Validators reject
mixed old/new metric aggregation and keep CuFuzz internal CFG/memory fields as
`N/A`, not zero.

## SIMT MemCov Implementation Checklist

The transport and novelty layers above are complete. The direct metric
replacement requires these scoped changes:

- [x] Add sector, pattern, and hash constants to
  `cuda-kernel/utils/feedback/feedback_constants.h`, with compile-time encoding
  checks in `tests/cuda_kernel/test_feedback_headers.py`.
- [x] Replace the coarse bucket calculation in
  `cuda-kernel/utils/feedback/feedback.cuh` with overflow-safe relative-sector,
  logical-warp, pattern, hash, and warp-leader paths.
- [x] Add deterministic GPU coverage for all seven patterns, cross-sector
  accesses, invalid lanes, repetition deduplication, and VConfig warp identity.
- [x] Emit `rapid-simt-memcov-v1` and its frozen constants from
  `tools/rapid-feedback-instrument/rapid_feedback_instrument.cpp`; preserve
  `unknown_memory_sites` behavior.
- [x] Update `benchmark/rq2/campaign.py`, `benchmark/rq2/verify_results.py`,
  `benchmark/rq2/plot_coverage.py`, and this protocol so new builds derive the
  metric from instrumentation metadata and mixed versions are rejected.
- [x] Rebuild Phase 2 and every feedback-enabled backend; old `.so` files are
  invalid after a device helper or metadata-contract change.
- [x] Prove fixed-input bitmaps match across fresh `origin`, `rapid`, and
  `rapid2` artifacts for the same kernel, input, and VConfig; ordered slot-reuse
  and four-slot completion tests cover the windowed transport independently of
  the task-local metric.
- [ ] Run a sequential 60-second pilot across all six RQ1/RQ2 kernels for
  LibAFL+, `Sys-s (w4)`, and `Sys`, comparing no feedback, the historical
  coarse helper, and the new SIMT helper. Use that comparison only as an
  engineering gate; formal RQ2 results must use the new metric throughout.

Focused verification commands are:

```bash
python3 -m unittest tests.cuda_kernel.test_feedback_headers
python3 -m unittest tests.feedback_instrument.test_instrument
RAPID_RUN_GPU_TESTS=1 python3 -m unittest tests.feedback_e2e.test_simt_memcov
cd cuda-fuzzer && cargo test coverage_telemetry
```

### Hot-path implementation constraints

Pointer comparison and payload-bounds validation remain 64-bit so device
addresses are handled safely. After proving that the complete access is inside
one payload object, the helper requires its relative end offset to fit in
32 bits and performs warp shuffle, contiguous-pattern classification, sector
iteration, and scatter grouping on 32-bit relative offsets. RAPID already caps
one input at 4 MiB, so this does not reduce the supported payload range.

The common broadcast and contiguous cases emit sectors from one warp leader.
The common single-sector `OTHER`/scatter path uses `__match_any_sync` once and
lets one leader per distinct sector emit, so 32 distinct sectors do not require
32 dependent warp-wide selection rounds. If any lane crosses a sector
boundary, a slow path advances by the lane-local sector span and elects the
lowest-numbered valid lane whose complete range covers each sector. With the
current `uint8_t` access width, this path has at most nine sector steps. It
still emits each distinct sector once when lane-local ranges overlap. The
implementation does not hash or atomically update once per lane, store exact
active masks, record temporal order, or maintain a sharing table. These
constraints are part of the metric: adding any of those dimensions requires a
separate performance and semantic review.

The contiguous check precedes the broadcast check because payload global
memory is dominated by coalesced lane-linear accesses in the evaluation
kernels. This removes one warp ballot from the common contiguous path without
changing any pattern encoding. Single-lane events also skip the first-offset
shuffle. The block-local physical thread ordinal uses 32-bit arithmetic because
CUDA permits at most 1024 threads in one block; pointer and payload bounds
remain 64-bit.

## Performance Gate

The completed two-round RTX 3090 (`sm_86`) RQ1 campaign measured the current
coarse helper's geometric-mean throughput loss against matching no-feedback
backends as:

| Configuration | `rapid-memory-bitset-v1` measured loss |
|---|---:|
| LibAFL+ | 38.78% |
| `Sys-s` (w1) | 34.62% |
| `Sys-s` (w4) | 12.27% |
| `Sys` | 7.49% |

These values are a historical comparison baseline, not measurements of SIMT
MemCov.

The premeasurement engineering estimates for new-helper loss have been
superseded by the measured final-v3 engineering screen and are omitted here.
They were never acceptance criteria. Acceptance requires:

```text
Sys six-kernel geometric-mean throughput loss <= 15%
Sys per-kernel throughput loss <= 25%
Sys-s (w4) geometric-mean throughput loss <= 20%
For each of LibAFL+, Sys-s (w4), and Sys:
  new-vs-coarse six-kernel geometric-mean throughput regression <= 5%
  new-vs-coarse per-kernel throughput regression <= 5%
Backend-equivalence bitmap mismatches = 0
```

The measurement tables, raw-artifact pointers, and claim boundaries live in
[`feedback-validation.md`](feedback-validation.md#fixed-input-performance-diagnostic),
including the
[final-v3 six-workload screen](feedback-validation.md#final-v3-six-workload-performance-screen).
The short diagnostics do not satisfy the formal 60-second repeated gate or its
per-kernel thresholds.

Define:

\[
Overhead_{new}=1-\frac{Exec/s_{new}}{Exec/s_{no-feedback}}
\]

\[
Regression_{old}=1-\frac{Exec/s_{new}}{Exec/s_{old-feedback}}
\]

If a gate fails, use a diagnostic build to count patterns, emitted sectors,
and global atomics per probe before changing metric semantics. Diagnostic
counters remain disabled in performance and RQ2 builds.

## Acceptance and Claim Boundary

The replacement is complete only when:

- all frozen constants and encodings appear in code and metadata;
- all seven patterns produce deterministic expected feature bits;
- invalid and unknown accesses contribute no SIMT memory feature;
- comparable backends produce equivalent task maps;
- async maps are attributed at completion and remain valid until release;
- Rust and RQ2 artifacts record `rapid-simt-memcov-v1`;
- old/new result aggregation is rejected;
- the 60-second performance gate is recorded from fresh artifacts; and
- paper text reports an absolute feature/novelty curve rather than a fabricated
  percentage or detected-race count.

Until all checks pass, it is valid to say that the source and direct semantic
oracle implement SIMT MemCov. It is not valid to claim that old RAPID binaries
implement it, that measured overhead matches the estimates, that formal RQ2 is
complete, or that the metric improves race detection by a measured factor.

Device sanitizer design and findings remain separate in
[`sanitizer.md`](sanitizer.md). A sanitizer or race detector is required to
turn race-relevant scheduling and memory features into a confirmed bug.
