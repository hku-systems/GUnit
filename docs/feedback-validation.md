# RAPID Feedback Validation

The historical evidence in this document validates the feedback transport and
the coarse `rapid-memory-bitset-v1` device-memory metric. The 2026-07-30 source
checks below additionally validate the direct `rapid-simt-memcov-v1` device
semantics, metadata contract, RQ2 version handling, and fresh all-backend
equivalence. The sequential six-workload 60-second performance pilot is still
required before formal RQ2 collection.

This document records the acceptance workflow for the three implemented
feedback channels:

- CFG site coverage from `libafl_cov_map`;
- versioned memory novelty from the low `61440` bitset buckets;
- thread/launch activity from the high `4096` bitset buckets.

The campaign command intentionally uses mutation. Do not add `--no-mutate` to
this workflow: that option is only for the bounded fixed-envelope smoke path.

## Direct SIMT MemCov Semantic Gate

Run the deterministic direct-helper GPU oracle with:

```bash
RAPID_RUN_GPU_TESTS=1 .venv/bin/python -m unittest \
  tests.feedback_e2e.test_simt_memcov
```

The oracle checks exact host-computed bitmap indices for all seven frozen warp
patterns, cross-sector accesses, lane-local invalid bounds, duplicate-event
deduplication, and VConfig active-warp identity. It uses the unchanged public
probe signature and the production `feedback.cuh` helper.

The metadata and RQ2 provenance gates are:

```bash
.venv/bin/python -m unittest \
  tests.cuda_kernel.test_feedback_headers \
  tests.feedback_instrument.test_instrument \
  tests.benchmark.test_rq2_campaign \
  tests.benchmark.test_rq2_plot_coverage
```

These checks require schema-2 feedback metadata for new campaigns, propagate
the build-derived metric contract to trial/sample records, preserve schema-1
coarse artifacts only through legacy verification, and reject mixed-version or
mixed-map plots. They do not measure end-to-end throughput.

### Recorded 2026-07-30 fresh gates

The current direct GPU oracle runs five tests and passes on GPU 2. In addition
to the seven patterns and bounds/VConfig cases, it counts final atomics for
overlapping scatter ranges and requires exactly one emission per distinct
sector. It also covers a full warp whose lanes touch 32 different sectors.
The device helper compiles for both `sm_75` and `sm_86`; the prior
`__reduce_min_sync` implementation failed the `sm_75` compile gate and has
been replaced by a single-sector match-any fast path plus a bounded
canonical-owner slow path.

A final-v3 fresh capture, Phase 1, Phase 2, and backend rebuild then ran the
complete 13-test feedback E2E module. It passed exact origin/rapid/rapid2
task-map equivalence together with allocation/free, feedback/no-feedback
observability, metadata, slot reuse, completion order, and four-slot
stream-completion checks. Separately, all six evaluation workloads were rebuilt
under `benchmark/rq1/build/simt-memcov-final-v3-sm86-20260730/`; their
six-workload, seven-backend, ten-run runtime gate completed 420/420 executions
with zero failures.

All six RQ2 workloads were rebuilt under
`benchmark/rq2/build/completion-events-sm86-20260730/`. Their origin, rapid,
and rapid2 metadata contracts all report `rapid-simt-memcov-v1`. The
subsequent one-second, one-repetition completion-event smoke under
`benchmark/rq2/results/completion-events-1s-sm86-20260730-gpu3/` completed all
54 cells with zero failures and produced 213 sparse enriched samples.
Completion-event raw telemetry uses the deliberately incompatible schema 1:
it writes the zero baseline, real cumulative-coverage changes, and the
terminal state without `memory_map_hex`. Historical polling/full-bitmap raw
telemetry is rejected by the current campaign loader and verifier.

The final-v3 direct oracle, E2E, and 420-run gate cover the current source. The
older one-second RQ2 smoke is not the planned 60-second performance pilot and
is not paper evidence. Its six
workload binaries predate the final distinct-sector election optimization;
because that optimization removes duplicate atomics without changing bitmap
bits, the smoke remains a tooling check, but formal RQ2 must rebuild all six
workloads from the final source. The final release frontend rebuild also
changes the recorded fuzzer hash, so the verifier now intentionally rejects
this old directory on provenance drift rather than relabeling it as current.

### Fixed-input performance diagnostic

A single-fixture, 10-second fixed-input diagnostic on GPU 2 compared matching
feedback and no-feedback builds of the superseded warp-wide-minimum scatter
implementation. It was intended only to catch a catastrophic hot-path
regression:

| Configuration | No feedback (exec/s) | SIMT feedback (exec/s) | Throughput loss |
|---|---:|---:|---:|
| LibAFL+ | 12004.68 | 8037.83 | 33.04% |
| `Sys-s (w4)` | 14485.66 | 13540.36 | 6.53% |
| `Sys` | 20525.35 | 19477.25 | 5.11% |

These rates are not measurements of the current bounded scatter
implementation, and they do not include the historical coarse helper needed
to calculate new-vs-coarse regression. They must be replaced after a fresh
three-way sequential pilot.

### Final-v3 six-workload performance screen

A later one-repetition screen ran the complete historical-coarse matrix before
the complete final-v3 SIMT matrix on GPU 2. Every cell used 100 warm-up
executions and a minimum five-second measurement. All 108 trials were valid.

| Configuration | Direct SIMT-vs-coarse regression | No-feedback-normalized regression |
|---|---:|---:|
| LibAFL+ | 1.34% | 1.75% |
| `Sys-s (w4)` | 0.81% | 2.34% |
| `Sys` | 0.35% | 0.99% |

The final-v3 helper checks contiguous patterns before broadcast patterns, so a
common contiguous access avoids one warp ballot. A same-workload 10-second A/B
on PyTorch BatchNorm measured 3.61%, 5.04%, and 3.96% higher throughput for
LibAFL+, `Sys-s (w4)`, and `Sys`, respectively, than the broadcast-first v2
build. That A/B motivated the ordering change; it is not a paper result.

The screen supports only the claim that no configuration shows a
six-workload geometric-mean regression above 5% in this short diagnostic.
PyTorch BatchNorm still reports direct five-second regressions of 5.20%, 7.46%,
and 7.81% for the three configurations. The formal 60-second repeated gate is
therefore still pending. Raw trials, logs, environment metadata, aggregate
CSVs, formulas, and limitations are retained in
`benchmark/rq2/results/simt-memcov-performance-screen-v3-5s-20260730/`.

The feedback-enabled rapid loader selected a 1024-thread physical block at 64
registers/thread, while its matching no-feedback build reported 25
registers/thread. RAPID2 selected the same 1024-thread block at 62 registers
with feedback and 25 without feedback. This confirms that the optimized helper
does not force this fixture down to a 512-thread launch. It does not establish
the six-workload geometric-mean or per-kernel acceptance thresholds.

Backend feedback semantics are deliberately different:

```text
origin  -> fuzzer       -> exact synchronous per-input feedback
rapid   -> fuzzer       -> ordered completion, task-owned maps, exact task IDs
rapid2  -> fuzzer_async -> exact task-ID delayed feedback
```

## Builtin Three-Backend Campaign

Build a fresh feedback fixture and run the GPU equivalence check:

```bash
mkdir -p .tmp-feedback-target-layout
TMPDIR="$PWD/.tmp-feedback-target-layout" \
RAPID_RUN_GPU_TESTS=1 \
.venv/bin/python -m unittest -v tests.feedback_e2e.test_feedback_e2e
```

The test performs capture, Phase 1, Phase 2, and all three backend builds from
`tests/feedback_e2e/fixtures/feedback_kernels.cu`. Each backend runs in an
independent worker process because loading synchronous CUDA backends and RAPID2
into one process can make `cudaSetDeviceFlags` reject the second driver
context. It checks exact SHA-256 equality for the edge and bitset maps.

For a real LibAFL broker/client campaign, first produce a builtin run directory
with `cuda-kernel/builtin_phase_pipeline.py`, then run:

```bash
.venv/bin/python tests/feedback_e2e/run_campaign.py \
  --run-dir /tmp/rapid-feedback-e2e/out/feedback-e2e \
  --backends origin,rapid,rapid2 \
  --runs 100000 \
  --observe-seconds 20 \
  --gpu-devices 0,1,2
```

For each backend, the runner starts the same command twice in one isolated
working directory. The first process is the LibAFL broker and the second is the
fuzzing client. It writes `broker.log`, `client.log`, and `report.json` below
the selected output directory.

For RAPID, LibAFL `executions` counts enqueue calls that returned. Periodic
`rapid_submitted`, `rapid_completed`, `rapid_pending`, and `rapid_failed`
snapshots describe backend work; the runner keeps the coherent snapshot with
the largest submission count. Periodic snapshots are necessary because the
LibAFL restarting client exits through `_exit` on a campaign signal and does
not run C++ destructors. Use `rapid_completed / observation_seconds` for GPU
throughput; its `TimeObserver` value is enqueue/backpressure latency, not
per-task GPU execution time.

Historical builtin broker/client evidence from the earlier synchronous-wait
RAPID baseline was generated on four RTX 3090 GPUs in
`/tmp/rapid-feedback-e2e-current/out/feedback-e2e/feedback_campaigns/report.json`:

| Backend | Client joined | Executions | CFG sites | Memory/index bits | Thread bits |
|---|---:|---:|---:|---:|---:|
| origin | yes | 101167 | 6 | 22 | 9 |
| rapid (old synchronous-wait baseline) | yes | 42813 | 6 | 22 | 9 |
| rapid2 | yes | 151011 | 6 | 22 | 9 |

The same-input GPU equivalence fixture reports `cfg_sites=6`,
`simt_memcov_bits=8`, and `logical_thread_bits=9` for each backend. Its edge and
memory/index SHA-256 values match across origin, rapid, and rapid2. RAPID2 also
submits an active task followed by two zero-length tasks, proves that the
active task maps remain unchanged until release, and observes the zero-length
tasks with no memory/index bits.

Ordered-backend timeout feedback is validated by a separate GPU-gated fixture:

```bash
RAPID_RUN_GPU_TESTS=1 \
python3 -m unittest tests.feedback_e2e.test_rapid2_timeout_feedback
```

The module builds `tests/feedback_e2e/fixtures/timeout_kernel.cu` through real
capture, Phase 1, Phase 2, and fresh backend builds. Its four cases verify that
both the RAPID2 watchdog and the ordered RAPID backend publish a task-owned
completion with `LibAflRunStatus::TIMEOUT`, backend stage `EXECUTE`, non-null
edge/SIMT MemCov maps, and execution time at least the configured 100 ms
budget. It also runs the ordered RAPID broker/client path until a timeout
objective is persisted and the CUDA backend has been loaded twice, proving
fresh-context restart, and verifies that RAPID2 rejects submission until an
explicit positive timeout has been configured.

VConfig feedback is validated by a separate GPU-gated fixture:

```bash
RAPID_RUN_GPU_TESTS=1 \
.venv/bin/python -m unittest -v tests.feedback_e2e.test_vconfig_e2e
```

`VConfigFeedbackE2ETest` builds
`tests/feedback_e2e/fixtures/vconfig_blockdim_kernel.cu` through real capture,
Phase 1, Phase 2, and all three backend builders. It injects a manifest
`launch_policy` with two legal logical block candidates, generates one
canonical seed, patches only the envelope `blockDim.x`, and asserts that
origin, rapid, and rapid2 observe different feedback for logical
`blockDim.x=32` versus `blockDim.x=64`. It separately checks that an origin
artifact with `vconfig_reserved: false` records VConfig as disabled and uses
physical relaunch dimensions.

`VConfigSyncWarpE2ETest` additionally builds the `vconfig_sync_loop_kernel.cu`
and `vconfig_warp_kernel.cu` fixtures. For the sync-loop fixture it checks the
non-warp-aligned VConfig metadata and proves that origin, rapid, and rapid2 can
replay logical block sizes 32 and 64 without an inactive-thread barrier
deadlock while still producing distinct feedback. For the multidimensional
`16x4x1` warp fixture it checks the warp-aligned metadata and verifies the
logical warp layout on all three backends.

## Supported Third-Party Campaigns

There are two third-party validation tracks:

- historical three-backend TensorRT validation, used to prove the feedback ABI
  across `origin`, `rapid`, and `rapid2`;
- current four-project RAPID2 validation for `gpuRIR`, `phantom-fhe`,
  `CudaSift`, and `lietorch`, used to exercise support-gated real-project
  kernel fuzzing.

### Historical TensorRT three-backend campaign

The recorded real-project capture is under:

```text
/tmp/rapid-third-party-phase3-20260702-101739/third-party-capture
```

Refresh Phase 2 and rebuild the chosen target before campaign validation. This
is required whenever the generated entry ABI changes.

```bash
THIRD=/tmp/rapid-third-party-phase3-20260702-101739/third-party-capture
KERNEL=_Z10clipKernelIiiLj512EEviT_S0_PKT0_PS1___136e29ff

.venv/bin/python scripts/kernel-rewrite/cli.py --run-dir "$THIRD"

for backend in origin rapid rapid2; do
  .venv/bin/python "cuda-kernel/$backend/build.py" \
    --phase2-dir "$THIRD/kernels/$KERNEL/phase2" \
    --out-dir "$THIRD/kernels/$KERNEL/phase2/backends/$backend" \
    --cuda-path /usr/local/cuda --cuda-arch sm_86
done

.venv/bin/python tests/feedback_e2e/run_campaign.py \
  --run-dir "$THIRD" --kernel-id "$KERNEL" \
  --backends origin,rapid,rapid2 \
  --runs 100000 --observe-seconds 18 --gpu-devices 0,1,2
```

The 2026-07-13 TensorRT `clipKernel<int,int>` instrumentation-baseline campaign
is recorded in `/tmp/rapid-feedback-perf-20260713/report.json`. It is retained
for the enabled/disabled comparison below; it predates the generated-layout
transport optimization. All campaigns used mutation, the same fixed seed, GPU
0, and a 30-second broker/client observation window:

| Backend | Client joined | LibAFL executions | GPU-completed metric | CFG | Memory/index | Thread |
|---|---:|---:|---:|---:|---:|---:|
| origin | yes | 151921 | 151921 | 9 | 15 | 9 |
| rapid | yes | 45899 | 44800 (`rapid_completed`) | 8 | 13 | 9 |
| rapid2 | yes | 225138 | 225138 | 9 | 15 | 9 |

The latest generated-layout and task-envelope functional evidence from
`/tmp/rapid-third-party-phase3-20260702-101739/third-party-capture/feedback_campaigns/report.json`
was generated on 2026-07-16 after refreshing Phase 2 and rebuilding all three
backends with the generated two-slot target layout and one-time context
initialization.
This campaign used the real two-process broker/client runner with mutation,
`--runs 100000`, `--observe-seconds 18`, and GPUs 0, 1, and 2:

| Backend | Client joined | LibAFL executions | GPU-completed metric | CFG | Memory/index | Thread |
|---|---:|---:|---:|---:|---:|---:|
| origin | yes | 90207 | 90207 | 9 | 15 | 9 |
| rapid | yes | 29623 | 28416 (`rapid_completed`) | 9 | 15 | 9 |
| rapid2 | yes | 136187 | 136187 | 9 | 15 | 9 |

All six broker/client processes exited with return code 0 and every client
joined. The generated header defines `RAPID_PAYLOAD_SLOT_COUNT=2`, so the
target-specific `RapidKernelContext` is 72 bytes instead of the former fixed
552 bytes. The per-task hot paths contain no full-context H2D copy.

Phase 1 discovered 13 third-party entries. Phase 2 currently builds four
TensorRT `clipKernel` instantiations. The remaining nine FlashAttention/CUTLASS
entries are recorded as `phase2_input_invalid` with
`params.mainloop:const_assignment_blocker`; they are support-boundary
exclusions, not feedback failures.

### Four-project RAPID2 support-gated campaign

The current four-project campaign layout is:

```text
build/e2e/third-party-fuzz/20260719-final
```

It contains resolved Phase 1 and Phase 2 outputs for:

```text
third_party/gpuRIR
third_party/phantom-fhe
third_party/CudaSift
third_party/lietorch
```

The support gate is intentionally strict. It runs kernels that Phase 2 builds
and that can be represented by the current value-constraint materializer and
RAPID2 physical-launch model. It skips structural gaps such as pointer-to-
pointer payloads, external CUDA texture objects, and private/protected C++ data
fields. The recorded 20260719 campaign predates Phase 2 VConfig integration, so
its `vconfig_required` skips are historical support-boundary evidence and must
be refreshed from fresh Phase 2/backend artifacts before being treated as a
current limitation.

Recorded support-boundary counts from the resolved 20260719 campaign are:

| Project | Phase 1 kernels | Phase 2 built | Runnable | Skipped |
|---|---:|---:|---:|---:|
| gpuRIR | 10 | 9 | 2 | 8 |
| phantom-fhe | 102 | 100 | 96 | 6 |
| CudaSift | 15 | 15 | 2 | 13 |
| lietorch | 15 | 0 | 0 | 15 |
| Total | 142 | 124 | 100 | 42 |

The current skip-reason distribution is:

| Reason | Count | Meaning |
|---|---:|---|
| `phase2_input_invalid` | 18 | Phase 2 rejected a structural input shape such as protected/private field materialization. |
| `vconfig_required` | 18 | Historical skip from before Phase 2 virtual-dimension rewriting was integrated. Refresh before using as a current support boundary. |
| `pointer_to_pointer_not_supported` | 3 | The kernel takes pointer-to-pointer arguments that are not representable as standalone payload buffers. |
| `protected_data_field` | 1 | A payload type contains a protected CUDA/C++ data field; generic field materialization is deferred. |
| `external_texture_object_not_supported` | 1 | The kernel depends on an external CUDA texture object/resource. |
| `target_bug_candidate_divergent_barrier` | 1 | `zero_coeff_count_kernel` has a control-dependent `__syncthreads()` and is tracked as a target-side bug candidate. |

A local provisional result under `/tmp/thirdparty.summary.json` observed
`100/100` runnable kernels passing both bounded fixed/no-mutate and broker/
client mutation fuzz, with `fuzzer_normal=true`. Because that file is outside
the repository and may be stale relative to a later Phase 2 or backend rebuild,
it is not treated as durable validation evidence.

Refresh the durable four-project evidence with:

```bash
.venv/bin/python -m scripts.third_party_fuzz.cli \
  --campaign-dir build/e2e/third-party-fuzz/20260719-final collect

.venv/bin/python -m scripts.third_party_fuzz.cli \
  --campaign-dir build/e2e/third-party-fuzz/20260719-final \
  build --cuda-path /usr/local/cuda --cuda-arch sm_86

.venv/bin/python -m scripts.third_party_fuzz.cli \
  --campaign-dir build/e2e/third-party-fuzz/20260719-final \
  fuzz --mode fixed --fixed-runs 100 --timeout-seconds 30

.venv/bin/python -m scripts.third_party_fuzz.cli \
  --campaign-dir build/e2e/third-party-fuzz/20260719-final \
  fuzz --mode mutation --mutation-runs 1000 --timeout-seconds 30

.venv/bin/python -m scripts.third_party_fuzz.cli \
  --campaign-dir build/e2e/third-party-fuzz/20260719-final report
```

The durable outputs are:

```text
build/e2e/third-party-fuzz/20260719-final/kernel_results.json
build/e2e/third-party-fuzz/20260719-final/summary.json
build/e2e/third-party-fuzz/20260719-final/SUMMARY.md
build/e2e/third-party-fuzz/20260719-final/BUG_CANDIDATES.md
```

Do not claim four-project fuzzer normality from the default campaign directory
until these files show every runnable kernel has completed fresh backend,
fixed/no-mutate, and mutation stages, or has an explicit support-boundary skip
reason.

## Three-Backend Instrumentation Baseline

The same TensorRT kernel was built twice for every backend from the same Phase
2 bitcode. The enabled build inserts automatic CFG and memory/index probes; the
disabled control passes `--feedback-instrumentation disabled`. This control
still preserves the feedback ABI, fixed map transport, and wrapper-recorded
thread activity, so it isolates probe instrumentation rather than removing the
entire feedback transport implementation.

Recorded same-GPU results from the 2026-07-13
`/tmp/rapid-feedback-perf-20260713/report.json` baseline are:

| Backend | Feedback enabled exec/s | Instrumentation disabled exec/s | Slowdown |
|---|---:|---:|---:|
| origin | 5064.03 | 5613.30 | 9.79% |
| rapid (`rapid_completed`) | 1493.33 | 1518.93 | 1.69% |
| rapid2 | 7504.60 | 13672.80 | 45.11% |

RAPID reports every 256 completed tasks, so its 30-second rate has at most 255
unreported completions, or 8.5 exec/s of downward truncation. Both RAPID runs
reported `rapid_failed=0`; the latest snapshots had a full 1024-entry pending
queue, confirming that the producer was ahead of the GPU worker.

For a repeatable RAPID2-only A/B run, build the instrumented target normally,
then build an uninstrumented control from the same `phase2/` directory:

```bash
THIRD=/tmp/rapid-third-party-phase3-20260702-101739/third-party-capture
KERNEL=_Z10clipKernelIiiLj512EEviT_S0_PKT0_PS1___136e29ff

.venv/bin/python cuda-kernel/rapid2/build.py \
  --phase2-dir "$THIRD/kernels/$KERNEL/phase2" \
  --out-dir "$THIRD/kernels/$KERNEL/phase2/backends/rapid2" \
  --cuda-path /usr/local/cuda --cuda-arch sm_86

.venv/bin/python cuda-kernel/rapid2/build.py \
  --phase2-dir "$THIRD/kernels/$KERNEL/phase2" \
  --out-dir "$THIRD/kernels/$KERNEL/phase2/backends/rapid2-uninstrumented" \
  --cuda-path /usr/local/cuda --cuda-arch sm_86 \
  --feedback-instrumentation disabled

.venv/bin/python tests/feedback_e2e/run_rapid2_benchmark.py \
  --run-dir "$THIRD" --kernel-id "$KERNEL" \
  --instrumented-build "$THIRD/kernels/$KERNEL/phase2/backends/rapid2/backend_build.json" \
  --uninstrumented-build "$THIRD/kernels/$KERNEL/phase2/backends/rapid2-uninstrumented/backend_build.json" \
  --runs 100000 --observe-seconds 18 --gpu-device 0
```

The benchmark report records both campaigns' executions/sec and
`slowdown = 1 - instrumented_rate / uninstrumented_rate`. It rejects mismatched
kernel identity, a missing broker/client join, or zero executions.

The 2026-07-13 all-backend report contains this RAPID2 subset:

| Mode | Client joined | Executions | Observe seconds | Executions/sec |
|---|---:|---:|---:|---:|
| instrumented | yes | 225138 | 30.0 | 7504.60 |
| uninstrumented | yes | 410184 | 30.0 | 13672.80 |

Measured slowdown:

```text
1 - 7504.60 / 13672.80 = 0.4511 = 45.11%
```

## Unit and Regression Checks

```bash
.venv/bin/python -m unittest \
  tests.cuda_kernel.test_feedback_headers \
  tests.cuda_kernel.test_builtin_phase_pipeline \
  tests.cuda_kernel.test_cufuzz_backend \
  tests.cuda_kernel.test_rapid_delayed_feedback \
  tests.cuda_kernel.test_rapid2_feedback_ffi \
  tests.feedback_instrument.test_instrument \
  tests.kernel_pipeline_rewrite.drivers.test_phase2_contract \
  tests.feedback_e2e.test_campaign_runner

RUSTFLAGS='-A function-casts-as-integer -A unstable-name-collisions' \
  cargo test --manifest-path cuda-fuzzer/Cargo.toml --lib
```

The real-GPU builtin E2E is intentionally separate and uses the
`RAPID_RUN_GPU_TESTS=1` command at the start of this document.
