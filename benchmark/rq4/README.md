# RQ4: cumulative ablation and performance profile

RQ4 consumes the canonical workload identities in `benchmark/workloads/` and
the 12-workload selection in `suite.json`. It does not maintain another copy
of any kernel or adapter.

## Scope

RQ1 owns the Release-build throughput comparison. RQ4 does not rerun that
matrix or generate a second throughput-ablation figure. Its paper artifact is
the eight-lane execution-time breakdown generated from profiling traces and
device-side timing counters.

The five profiled configurations are:

1. `cufuzz` (`cufuzz` artifact): per-input allocation/copy/launch/sync/free,
   with no device feedback;
2. `libafl` (`origin-no-feedback`): buffer reuse and the synchronous LibAFL
   frontend, without device feedback;
3. `libafl-plus` (`origin`): the same serial path with SIMT-aware device
   feedback enabled;
4. `gunit-sync` (paper label `GUnit-s`, `rapid`, `W=2`): persistent execution with synchronous
   feedback consumption;
5. `gunit` (`rapid2`, `W=2`): persistent execution with asynchronous feedback
   processing.

`W=2` is the canonical RQ4 setting and matches the default window size in the
RQ1 campaign code. Earlier `W=4` profiling results remain archival evidence and
must not be relabeled or reused as `W=2` measurements.

## Profiling

RQ4 uses a profiling-only campaign, separate from the Release throughput data
reported by RQ1:

- short Nsight Systems CUDA traces retain interval and thread identity for
  allocation/free, memory-copy, launch, GPU-kernel, and GPU-memory activity;
- timing-enabled `origin`, `rapid`, and `rapid2` artifacts record device-side
  input decoding, target execution, explicit feedback preparation/merge,
  completion signaling, bookkeeping, and idle/wait cycles;
- profiling-only Rust frontends use the unified profiling core to record the
  seven exclusive host-loop segments and feedback predicate/metadata handling
  without changing the RQ1 Release binaries;
- the same schema-v2 JSONL record contains device `KernelTimingStats` buckets
  and drives both the eight-lane RQ4 view and the host-loop diagnostic view.

The canonical `run_profile.sh` defaults to `mutating` mode and runs the
ordinary restarting-manager fuzz path: an unprofiled broker starts first, then
Nsight wraps the GPU client driven by `--mutate-seconds S`. At the deadline the
client drains outstanding work,
writes the unified profile, and shuts down the restarting campaign without a
respawn. Every configuration uses `RAPID_FIXED_SEED=1`, so mutation starts from
the same deterministic seed. Mutating collection uses the fuzzer's default two
retirement workers and two supply threads. The retained `fixed` mode uses the
original `--benchmark` path as an explicit legacy comparison and automatically
keeps that specialized path single-threaded. In either mode the profiling
frontend resets persistent device counters immediately before the Nsight
capture and snapshots them only after the measured work has drained.

The paper profile contains eight independently normalized lanes: `CuFuzz`,
`LibAFL`, `LibAFL+`, `GUnit-s-GPU`, `GUnit-s-CPU`, `GUnit-GPU`, `GUnit-Coll`,
and `GUnit-Disp`. The former `GUnit-s-FB` and `GUnit-FB` lanes were removed;
their concurrent durations are not added to `GUnit-s-CPU` or
`GUnit-Coll`, because doing so would combine overlapping thread time. Those two
remaining lanes are single-thread views. CPU and GPU time are never added into
one denominator. For each serial or host-thread lane, `Idle` is the measurement
window minus the measured active components. A negative residual fails
validation. The two persistent GPU lanes use device timing counters directly,
including measured idle cycles.

Nsight SQLite export is only a transient conversion format. Each profile is
immediately reduced to a compact evidence JSON that retains thread-scoped CUDA
API durations and GPU intervals. The same compact evidence supplies the audit
totals, so the profiling path does not run additional `nsys stats` reports.
The SQLite file is removed after the compact file is written successfully
unless `--keep-sqlite` is requested. The final paper input is
`aggregate/rq4_overhead_breakdown.csv`, which records every workload,
configuration, lane, repetition, category, raw value, unit, share, and
measurement domain.

The only retained result snapshot is
[`results/rq4-latest-20260811-50658ea.csv`](results/rq4-latest-20260811-50658ea.csv).
It contains the reviewed 12-workload mutating profile in the aggregate schema;
raw profile directories and intermediate reports remain reproducible outputs
and are not tracked.

`run_profile.sh` is the canonical RQ4 entry point.

## Run the profile

### Prerequisites

- CUDA, an accessible GPU, and Nsight Systems at `/usr/local/bin/nsys`.
- The repository `.venv/` and Rust/Cargo toolchain. The script builds the
  profiling-only release fuzzers with the `profiling` feature.
- Runtime-verified RQ1 build roots that together cover the 12 workloads
  exactly once and each contain `runtime_verification_summary.json`. Normally
  these are the six-workload standalone root produced by
  `benchmark.rq1.build` and the six-workload imported root produced by
  `benchmark.rq1.import_phase2`; see [RQ1](../rq1/README.md).

Pass the roots explicitly as a space-separated environment value:

```bash
RQ4_BUILD_ROOTS="benchmark/rq1/build/standalone-sm86-YYYYMMDD benchmark/rq1/build/imported-sm86-YYYYMMDD" \
RQ4_GPU=0 \
RQ4_CUDA_ARCH=sm_86 \
bash benchmark/rq4/run_profile.sh
```

This defaults to a 30-second mutating collection. Run the
legacy fixed-input profile explicitly with `RQ4_MODE=fixed`.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `RQ4_BUILD_ROOTS` | none; required | Space-separated RQ1 build roots; each becomes one `--build-root` argument |
| `RQ4_MODE` | `mutating` | `mutating` uses broker/client ordinary fuzzing; `fixed` uses the legacy benchmark path |
| `RQ4_MUTATE_SECONDS` | `30` | Measured fuzzing duration per mutating profile |
| `RQ4_GPU` | `0` | CUDA device exposed to the profiling subprocesses |
| `RQ4_CUDA_ARCH` | `sm_86` | CUDA architecture used for timing-enabled backend builds |

The script first runs the RQ4 unit tests, builds the profiling fuzzers and
timing-enabled backends, and profiles `apex_maybe_cast` as a pilot. The full
12-workload, five-configuration, single-repetition profile starts only if the
pilot has target activity for every configuration, exactly eight complete
lanes, non-negative residuals, normalized lane shares, positive
persistent-device iterations, and distinct RAPID2 collection/dispatch thread
identities. Mutating rows additionally require the requested profile duration,
positive execution progress, mutations (unless the cell already found a
solution), and a clean host and backend drain. Feedback-enabled configurations
additionally require nonzero CFG or SIMT coverage bits, proving that the
feedback channel is active. Corpus growth is advisory because a valid
30-second run may find no new interesting input after early coverage
saturation. `cufuzz` and `libafl` skip the coverage activity check because
their backends intentionally omit device feedback.

Crash-related cell states have distinct meanings:

- `solutions > 0` means the fuzzer found at least one objective input. The row
  is retained for attribution but excluded from timing lanes, even when the
  timed run drained normally.
- `capture_incomplete: true` means the profiler did not produce final mutating
  statistics, usually because of a client timeout or crash loop. Any partial
  Nsight or unified-profile evidence is retained, but the row is excluded.
- `lock_residual` is a teardown warning, not a `profiles.jsonl` status. It
  means the GPU-client lock remained busy after holder termination and the
  bounded cleanup wait; the next cell preflight retries cleanup. A residual
  broker port or live process group remains a hard boundary failure.

The output tag ends in `git rev-parse --short HEAD`. Generated artifacts are:

```text
benchmark/rq4/build/timing-w2-feedback-v2-<short-sha>/
benchmark/rq4/build/rq4-profile-target-<short-sha>/
benchmark/rq4/results/diagnostic-profile-pilot-apex-mutating-30s-seed1-r1-w2-feedback-v3-<short-sha>/
benchmark/rq4/results/profile-mutating-30s-seed1-r1-w2-feedback-v3-<short-sha>/
```

The fixed comparison retains the previous `...apex-5s...` and
`profile-10s...` result names.

The full result contains `profiles.jsonl`, one `raw/*.profiling.jsonl` per
profile process, raw Nsight reports and compact evidence, `aggregate/*.csv`,
and `overhead_breakdown.pdf`. Unified profiling records start at
`schema_version: 2`; every line is one aggregate segment in the `main_loop`,
`cpu_feedback`, or `device_kernel` domain. Each `profiles.jsonl` row records its
`mode`, the configuration's effective `window_size` (`null` when not
applicable), and mutating rows record `fixed_seed: 1` plus timed-run/drain
statistics. Aggregation rejects mixed modes, non-canonical mutating seeds, and
rows that do not match the expected configuration. Fixed trace windows are
checked against `RAPID_BENCHMARK_RESULT.elapsed_ns`; mutating trace windows are
checked against the summed exclusive main-loop time plus its nested CPU
feedback time from the unified profiling record, both with a five-percent
relative tolerance.
`aggregate/rq4_phase_diagnostic.csv` is the seven-segment host-loop view from
the same raw records. Generated `build/` and raw `results/` trees remain
ignored. Promoting a result requires an explicitly reviewed compact snapshot
and a matching ignore-rule exception; do not check in raw traces, logs, or
derived CSV/PDF report directories.
