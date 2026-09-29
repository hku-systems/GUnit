# RQ2 coverage-growth experiment protocol

Status: completion-driven collection, its verifier, and execution-axis plotter
are implemented. The previous 1 ms polling smoke is intentionally incompatible
with the current raw schema. The synthetic and six classic formal campaigns
each have five completed 60-second seeds per configuration.

This directory is the source of truth for the planned RQ2 coverage-growth
experiment. The protocol keeps memory novelty as a separate feedback domain
and fixes its recording semantics as `rapid-simt-memcov-v1` plus sparse,
completion-event schema-v1 delta telemetry.

`verify.py` is RQ2's source/provenance gate, corresponding to
`benchmark/rq1/verify.py`. `correctness.py` is the manual correctness probe
tool for checking VConfig adaptations.

## Results

Campaign ledgers, logs, and telemetry are generated under `results/` and are
not tracked. Figures are generated under `reports/figures/` and are likewise
not tracked. Keep a result only as an explicitly reviewed, compact data
snapshot; do not commit raw campaign directories.

## Research question

RQ2 asks whether RAPID reaches useful CUDA-kernel behavior faster than the
host-driven baselines, and whether per-input VConfig mutation exposes behavior
that fixed launch configurations do not reach.

The experiment uses **online collection**: each completed execution that adds
coverage reports the campaign's cumulative features immediately. It does not
use polling or offline corpus replay as the primary measurement method.

## Reproduce

Run these commands from the repository root. Build the shared RQ2 artifacts:

```bash
.venv/bin/python -m benchmark.rq2.build \
  --run-id rq2-formal-sm86-20260802
```

Run the six formal classic workloads as two GPU-local shards. Each complete
five-seed matrix stays on one GPU:

```bash
.venv/bin/python -m benchmark.rq2.campaign \
  --build-root benchmark/rq2/build/rq2-formal-sm86-20260802 \
  --output benchmark/rq2/results/formal-classic-shoc-9cfg-5seed-60s-20260802-gpu2 \
  --repetitions 5 --coverage-seconds 60 --seed-mode per-rep --gpu-device 2 \
  --workload shoc_reduction \
  --workload shoc_radix_sort \
  --workload shoc_scan

.venv/bin/python -m benchmark.rq2.campaign \
  --build-root benchmark/rq2/build/rq2-formal-sm86-20260802 \
  --output benchmark/rq2/results/formal-classic-big3-9cfg-5seed-60s-20260802-gpu3 \
  --repetitions 5 --coverage-seconds 60 --seed-mode per-rep --gpu-device 3 \
  --workload cutlass_gemm \
  --workload flashattention_device1xn \
  --workload pytorch_batchnorm

.venv/bin/python -m benchmark.rq2.campaign \
  --build-root benchmark/rq2/build/rq2-formal-sm86-20260802 \
  --output benchmark/rq2/results/formal-synth-complex-9cfg-5seed-60s-20260801-gpu2 \
  --repetitions 5 --coverage-seconds 60 --seed-mode per-rep --gpu-device 2 \
  --workload synth_complex
```

On a clean workspace, the synth command freshly regenerates the result
directory consumed by both formal plotters; the archived result with that ID
used an older build root.

Given an existing Phase 1/2 campaign ledger and built backend matrix, rerun
completion-driven coverage for the HEonGPU entry currently recorded in the
wave-3 root ledger:

```bash
.venv/bin/python -m scripts.third_party_fuzz.cli \
  --campaign-dir build/e2e/third-party-fuzz/20260802-rq2-wave3 \
  --project heongpu \
  --kernel heongpu_decryption_tile_kernel \
  coverage-matrix --coverage-seconds 60 --gpu-device 2
```

Collect and build a different campaign ledger before selecting its projects or
kernels for `coverage-matrix`.

Regenerate the formal, classic-workload, and derived-variant figures:

```bash
.venv/bin/python benchmark/rq2/variants/plot_formal_curves.py
.venv/bin/python benchmark/rq2/variants/plot_classic_curves.py
.venv/bin/python benchmark/rq2/variants/plot_variant_curves.py
.venv/bin/python benchmark/rq2/variants/plot_master_curves.py
```

## Workloads

The catalog reuses all six source-traceable RQ1 kernels and adds one
RQ2-native synthetic coverage workload. RQ2 derives the six upstream-backed
contracts from their RQ1 provenance:

- `cutlass_gemm`
- `flashattention_device1xn`
- `pytorch_batchnorm`
- `shoc_radix_sort`
- `shoc_reduction`
- `shoc_scan`

The seventh workload, `synth_complex`, has an explicit synthetic provenance
record instead of an upstream repository claim. Its checked base contract is
derived into the same RQ2 VConfig schema as the upstream-backed workloads.
It uses a variable byte payload guarded by scalar `n`, grid-stride and
block-width-displaced accesses, and 24 four-level magic-guard chains. The last
guard is deliberately five bits wide, making the cumulative deepest-path hit
probability 1/131,072 under uniform bytes. It is the tunable workload for
60-second coverage-growth experiments; the six existing workloads remain the
original small-kernel pilot set.

### VConfig-responsiveness hard filter

An A-grade RQ2 kernel must satisfy all three requirements:

1. **Free launch geometry.** The kernel's `launch_policy` must provide more
   than one legal logical block shape (`block_candidates` must not be a
   singleton), and the implementation must not be pinned by warp
   synchronization or a compile-time stride: no `CU1DBLOCK`-style constant
   stride, full-mask shuffle that requires a complete warp, or hard-coded
   `dim3(32, N)` warp layout.
2. **Warp-assignment-dependent addressing.** A memory-address expression must
   involve `threadIdx` or `blockIdx` (for example, `y * pitch + x` or
   `tid * stride`) so changing the block shape reassigns the same address to
   different warps or produces a different lane pattern.
3. **Complex branching.** Apply the existing branch-complexity criterion.

A kernel satisfying only requirements 1+3 or 2+3 is grade B. Any kernel that
fails requirement 1 must be downgraded, and its report must state
`"固定几何，仅可作阴性对照"`.

The initial corpus, mutator settings, target timeout, and random seed must be
identical across comparable configurations for a kernel.

For the WS4 select experiment, `benchmark.rq2.build --instrument-selects`
adds evaluated-select pseudo-sites to the three feedback-enabled backends.
These sites record evaluation, not the selected arm; the option is off by
default.

The audited RQ2 constraints now permit kernel-safe scalar and buffer-size
variation for a subset of workloads. Campaigns produced after this change
explore a different input search space and are not directly comparable with
earlier RQ2 results. Preserve and compare the constraints hash before combining
or contrasting runs.

Three workloads use regular RQ2-only sources with checked adaptation records
and exact patches against their unchanged RQ1 extractions: `shoc_scan`,
`shoc_reduction`, and `flashattention_device1xn`. FlashAttention runs inside a
256-thread physical envelope with logical block candidates 32, 64, 128, and
256. Its scalar `n` is the exact readable-float count and is repaired to
`1 <= n <= logical blockDim.x`; the kernel additionally uses `tidx < n` so
tails that are not divisible by `THREADS_PER_ROW=4` cannot read padded lanes.
`ROWS_PER_STG` and `STGS_PER_LOOP` retain the extracted one-stage row mapping;
because only `jj=0` executes, this adaptation does not claim to generalize a
future multi-stage loop.
The output remains fixed at 256 floats (1,024 bytes) because every active
logical thread writes `output[tidx]`. The default logical block remains 128,
and `(block_x, n)=(128,64)` is the compatibility anchor for the original
extraction.

`count_fits_buffer` establishes the safety lower bound
`len(input) >= 4*n`. Its grow-only repair may retain an oversized input and
does not establish equality between the buffer length and `n`. Consequently,
length-only mutation is not evidence that semantic read extent changed; `n`
is the access-extent control.

## Coverage and feedback domains

The domains are reported independently. They must not be added into one
percentage because they have different meanings and denominators.

### 1. External/software coverage

This is host-side coverage visible to a CuFuzz or CuFuzz-style runner. CuFuzz
does not observe RAPID's intra-kernel feedback channels. Therefore its
device-side CFG and memory fields are `N/A`, not zero.

The current repository's LibAFL-driven `cufuzz` backend deliberately clears its
device feedback maps and may not expose a meaningful host-side coverage map.
Before plotting an external-coverage curve, the pilot must verify that the
selected runner actually exports a cumulative software-coverage metric. If it
does not, report external coverage as `N/A`; do not infer coverage from corpus
size or execution count.

External coverage is not numerically comparable with device CFG coverage. It
is retained to state exactly what the baseline can observe and to explain why
wrapper-only coverage may saturate almost immediately on small kernels.

### 2. Device-side CFG coverage

For LibAFL+, `Sys-s`, and `Sys`, record the campaign-level union of CFG sites
reached by any active logical thread:

```text
cfg_sites(t) = popcount(union of task-local CFG maps observed by time t)
```

Report the absolute `cfg_sites(t)` curve. A normalized value may also be
reported when the build metadata provides a trustworthy count of instrumented
sites:

```text
cfg_fraction(t) = cfg_sites(t) / instrumented_cfg_sites
```

The same instrumentation/site-ID scheme must be used for LibAFL+, `Sys-s`, and
`Sys` for each kernel. Map density is a diagnostic, not the paper's CFG
coverage percentage.

### 3. Memory novelty

Memory remains a first-class feedback domain in the test matrix. For the
experiment, call the reported quantity `memory features` or `memory
novelty`, not `memory coverage percentage`:

```text
memory_features(t) = cumulative union of memory-domain features by time t
```

The implemented feature definition is frozen as `rapid-simt-memcov-v1` in
[`docs/feedback.md`](../../docs/feedback.md). One semantic feature combines a
static load/store site, payload argument slot, payload-relative 32-byte sector,
R/W, logical VConfig warp, and warp access-pattern class. The device-side
implementation directly replaces the historical coarse
`rapid-memory-bitset-v1` calculation while reusing the same task-local bitmap
and completion-time union.

The following boundary is frozen:

- memory remains an independent feedback domain;
- online fuzzing may admit an input when it adds a memory-domain feature;
- values from different memory metric versions must never be merged into one
  curve;
- each run records `memory_metric_version`, map size, instrumentation metadata
  hash, a zero baseline, cumulative counts at real change points, exact count
  deltas, and a terminal state; and
- changing the underlying feature definition requires rerunning the pilot.

Existing pilot artifacts produced by `rapid-memory-bitset-v1` remain useful
only as diagnostic evidence for that historical metric. They cannot be merged
with, continued by, or substituted for the formal RQ2 curves. After the SIMT
replacement, rebuild every backend and rerun the complete matrix with
`memory_metric_version=rapid-simt-memcov-v1`. New campaign loading requires the
explicit schema-2 metric contract; missing-version schema-1 metadata is
accepted only when auditing an existing historical result directory.

The hashed 61,440-bit map capacity is not a generic reachable-feature
denominator. Report absolute cumulative popcount and map density. A
workload-specific percentage is permitted only when a separate semantic-site
enumeration proves a reachable hashed ceiling; otherwise use a declared
cross-configuration, cross-repetition empirical reference union and name it as
such.

### 4. VConfig

VConfig is an experimental factor, not a coverage domain. Each run is assigned
one of these categorical values:

- `off`: the launch configuration is fixed and VConfig mutation is disabled;
- `on`: VConfig rewriting is supported and per-input VConfig mutation is
  enabled; or
- `unsupported`: the target cannot safely use VConfig rewriting. This is an
  applicability result, not an `off` data point.

The actual build report decides whether a kernel supports `on`; the label must
not be inferred only from the requested command-line option. VConfig can remain
useful even when memory hits are merged across threads: changing the logical
launch shape changes which threads are active, the logical built-ins observed
by the rewritten kernel, CFG decisions, and the set or shape of memory accesses
that enters the merged map.

## Pilot matrix

Run one 10-second campaign for every applicable row below on each catalog
kernel when validating the complete matrix. The synthetic workload's formal
coverage-growth cells use a 60-second budget. A single repetition remains a
protocol validation, not final paper evidence.

| Paper label | Backend/configuration | Window | VConfig categories | Online domains |
|---|---|---:|---|---|
| CuFuzz-style | `cufuzz` | 1 | `off` only | verified external/software metric only; otherwise `N/A` |
| LibAFL+ | feedback-enabled `origin` | 1 | `off`, `on` when supported | device CFG + memory |
| `Sys-s (w1)` | feedback-enabled `rapid` | 1 | `off`, `on` when supported | device CFG + memory |
| `Sys-s (w4)` | feedback-enabled `rapid` | 4 | `off`, `on` when supported | device CFG + memory |
| `Sys` | feedback-enabled `rapid2` | canonical asynchronous window | `off`, `on` when supported | device CFG + memory |

This is at most nine campaigns per kernel: one CuFuzz-style run plus two
VConfig categories for each of the four feedback-enabled configurations.
Unsupported `on` cells are recorded but not executed. With seven catalog
workloads, the full upper bound is 63 campaigns. The measured-time total
depends on whether the small-kernel 10-second protocol or the synthetic
60-second protocol is selected, excluding build, startup, cooldown, and
validation time.

No-feedback configurations belong to RQ1's throughput/feedback-overhead
experiment. They are not meaningful coverage-guided RQ2 configurations and are
not included here.

## Pilot procedure

Campaigns run one complete repetition at a time; do not interleave repetitions
from different rounds. For the initial single-repetition pilot:

1. Build all configurations for one kernel from the same captured Phase 2
   source and record artifact hashes.
2. Verify the requested and effective VConfig category from build metadata.
3. Start from the same minimized seed corpus and fixed random seed.
4. Run for 10 seconds after a documented startup/warm-up boundary.
5. At every execution completion, persist a record only when CFG, memory, or
   thread-activity coverage changes. Also persist the zero baseline and the
   final sample during graceful drain/shutdown. Execution-count changes alone
   do not create a row.
6. Preserve stdout/stderr, environment metadata, build metadata, the exact
   command, and raw sparse telemetry records.
7. Complete the whole configuration matrix once before starting any later
   repetition.

The 10-second budget does not establish saturation. Formal evidence still
requires multiple independent repetitions or seeds, and any unreached
threshold remains right-censored.

Run the full matrix sequentially on one GPU with an explicit budget:

```bash
.venv/bin/python -m benchmark.rq2.campaign \
  --build-root benchmark/rq2/build/<build-id> \
  --output benchmark/rq2/results/<result-id> \
  --repetitions 1 \
  --coverage-seconds 10 \
  --gpu-device 2
```

To use multiple idle GPUs, start independent commands with distinct result
directories and repeat `--workload` for each shard. Keep all nine
configurations of one workload on the same GPU. For example:

```bash
.venv/bin/python -m benchmark.rq2.campaign \
  --build-root benchmark/rq2/build/<build-id> \
  --output benchmark/rq2/results/<result-id>-gpu2 \
  --repetitions 1 --coverage-seconds 10 --gpu-device 2 \
  --workload shoc_reduction \
  --workload shoc_radix_sort \
  --workload shoc_scan

.venv/bin/python -m benchmark.rq2.campaign \
  --build-root benchmark/rq2/build/<build-id> \
  --output benchmark/rq2/results/<result-id>-gpu3 \
  --repetitions 1 --coverage-seconds 10 --gpu-device 3 \
  --workload cutlass_gemm \
  --workload flashattention_device1xn \
  --workload pytorch_batchnorm
```

Each shard records only its selected workloads in `environment.json` and is
resumed independently. Unknown, empty, or duplicate workload selections are
rejected, and selected workloads retain catalog order regardless of CLI order.

### Level and time-to-reference reporting

Report final CFG, SIMT memory, and thread-activity levels separately, together
with completed executions and executions per second. Plot coverage against
completed executions as the exploration curve and against wall time as the
system curve. For an analytical ceiling or a declared empirical reference
union, report completion/time to 50%, 90%, 95%, and 100%, plus the final
new-feature completion/time. Thresholds not reached within the fixed budget are
right-censored.

A run-local last-change point is not genuine saturation by itself. Do not call
it saturation without an analytical reachable ceiling or an explicitly named
empirical reference denominator. In particular, the observed
`shoc_reduction` pilot value of 242 memory buckets is below its enumerated
464-bucket reachable hashed ceiling and must not be presented as the achievable
maximum.

### Local tooling smoke (not paper evidence)

The historical local smoke directory
`results/simt-memcov-sm86-20260730-smoke-v2-1s-1ms/` covers six workloads,
nine configurations, one repetition, one second per cell, fixed seed 1337,
and strictly sequential execution on one GPU. It contains 54 completed trial
rows, zero failure rows, and 198 sparse enriched samples. The two generated
figures and `coverage_saturation.jsonl` validate the end-to-end collection and
plotting path only. The run is too short and has too few repetitions to support
paper claims or the SIMT MemCov performance gate. Its workload binaries
predate the final distinct-sector election optimization, which removes
duplicate atomics without changing bitmap bits; rebuild all six workloads
before the formal pilot. A later release-frontend rebuild also changes the
live build-tree binary. Current campaigns instead copy both fuzzer frontends
into the result directory and execute those snapshots, so later Cargo builds
do not alter their recorded provenance. This historical directory predates
those snapshots and uses the retired polling telemetry format, so it is not
accepted by the current verifier.

## Required online sample schema

Each sample should contain at least:

```text
timestamp_s
kernel
tool
backend
window
seed
vconfig_requested
vconfig_effective
executions_submitted
executions_completed
external_features          # nullable
cfg_sites                   # nullable
instrumented_cfg_sites      # nullable
memory_features             # nullable
memory_metric_version       # nullable
memory_map_bits             # nullable
instrumentation_metadata_sha256
```

Use completed executions for `rapid` and `rapid2` rate/coverage alignment.
Nullable values preserve the distinction between an unsupported/unobservable
metric and a measured zero.

## Plotting plan

Do not draw three vertically stacked coverage panels. The main per-kernel plot
uses two side-by-side panels against completed executions:

- left: device CFG sites over completed executions;
- right: memory features over completed executions.

Color identifies CuFuzz-style, LibAFL+, `Sys-s (w1)`, `Sys-s (w4)`, and `Sys`.
For feedback-enabled configurations, line style identifies VConfig: solid for
`on`, dashed for `off`. An unsupported VConfig cell has no curve and is noted
in the caption or applicability table.

CuFuzz's external/software metric must not be placed on either device-domain
axis as if it had the same denominator. If the runner exposes a meaningful
external curve, show it in a compact separate inset/figure or report it in a
table, explicitly labeled `external`. If it does not, show CuFuzz as `N/A` for
the two internal domains.

Use absolute feature counts for the pilot. Normalized CFG coverage may be a
secondary axis/figure only when the denominator is validated. Do not normalize
memory features to 100% until the memory feature universe and denominator have
a stable semantic definition.

## Interpretation rules

- Fast CFG saturation is expected for small or mostly straight-line kernels;
  it is a result, not automatically a collection bug.
- A curve that starts at 100% may reflect seed-corpus saturation, wrapper-only
  external coverage, too-coarse hashing, or an incorrect denominator. Check
  these possibilities before making a claim.
- Compare completed executions to fixed CFG thresholds only among
  configurations sharing the same device instrumentation and denominator.
- Compare memory curves only when `memory_metric_version`, map size, and
  instrumentation metadata hash match.
- Report VConfig's effect as an `on` versus `off` comparison within the same
  tool and kernel; do not add VConfig to CFG or memory counts.
- Keep throughput and coverage claims separate: a faster executor may achieve
  earlier coverage growth without changing final reachable coverage.

## SIMT MemCov promotion gate

The memory feature unit is no longer deferred. Before promoting a run to the
formal RQ2 result, require all of the following:

1. the backend instrumentation metadata records
   `memory_metric_version=rapid-simt-memcov-v1`;
2. all comparable configurations use the same map size, hash constants,
   pattern encodings, kernel artifact, initial corpus, seed policy, and budget;
3. origin, rapid, and rapid2 fixed-input task maps pass the equivalence test;
4. telemetry updates the campaign union at task completion rather than
   submission;
5. the result verifier rejects old/new mixed-version aggregation; and
6. bitmap density is reported as collision/saturation context, not as a
   memory-coverage percentage.
