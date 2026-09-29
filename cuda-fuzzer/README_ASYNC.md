# CUDA Fuzzer – Async vs Original

This directory ships two binaries for CUDA fuzzing:

1. `fuzzer` – Synchronous baseline using LibAFL’s `StdFuzzer` + `InProcessExecutor`
2. `fuzzer_async` – Asynchronous pipeline for RAPID2 persistent kernel

## Architecture

### Original (synchronous)
- `InProcessExecutor` calls into the harness and blocks per input.
- Coverage is read immediately after each execution.
- Simple, good for baseline and debugging.

### Async (RAPID2)
- `GpuExecutor` only submits inputs (non‑blocking).
- `AsyncBatchFuzzer` batches inputs, polls GPU for completed tasks, and evaluates coverage/results later.
- Normal mutating fuzz runs default to a bounded host input queue with two
  producers. `--supply-threads N` selects `0..=32` producers; `0` restores the
  original single-threaded mutation path. Each receive keeps one deadline across
  its `recv_timeout` loop, so a transient empty queue waits for a producer rather
  than immediately falling back to owner-thread mutation.
- Retirement defaults to a bounded two-worker materialization pool.
  `--retire-worker N` selects `0..=32` workers; `0` restores the original
  event-loop retirement path. One submit/poll thread hands acquired results to
  the workers. Each worker waits for its share of the configured batch, copies
  and sparsifies every task-local coverage map in that share, and releases the
  acquired backend slots together. The single LibAFL owner restores sequence
  order, revalidates novelty, and alone updates feedback, corpus, and events.
  `--retire-batch N` sets the maximum owner commit batch size; the pool splits
  it into `ceil(N / workers)`-task materializer batches (the default `8 / 2`
  split gives each worker a four-task batch).
- The owner admits another stage while fewer than 64 submissions await owner
  retirement. At that watermark it retires enough prepared completions to make
  progress; shutdown and failures drain the remainder. The submit/poll thread
  waits for device completion only when its GPU window is full.
- Retirement statistics distinguish `max_in_flight` (submitted-to-retired
  end-to-end backlog), `max_gpu_pending` (actual submitted GPU tasks), and
  `max_retirement_credits_held` (peak acquired admission credits).

## Build

```bash
cd cuda-fuzzer
cargo build --release
```

Outputs:
- `target/release/fuzzer`
- `target/release/fuzzer_async`

## Run

Option A: helper script
- `./run_comparison.sh /path/to/lib.so`

Option B: direct
- Baseline: `./target/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json`
- Async:    `./target/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json`

Normal fuzzing uses LibAFL's restarting manager: launch the same command twice,
first for the broker and then for the GPU client. The async command needs no
worker flags; the client defaults to two retirement and two supply workers.

Phase3 smoke helpers:
- Dump the manifest-driven canonical full task envelope without loading the `.so`:
  `./target/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --dump-seed`
- Run the normal LibAFL fuzz loop while forcing each stage iteration back to
  the manifest-driven canonical full envelope:
  `./target/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --no-mutate`
  `./target/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json --no-mutate`
- Run a bounded fixed-envelope LibAFL loop:
  `./target/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --no-mutate --runs 1000`
  `./target/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json --no-mutate --runs 1000`
- Run a bounded normal fuzz loop:
  `./target/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --runs 3`
  `./target/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json --runs 3`
- Override the host-side producer count:
  `./target/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json --supply-threads 4`
- Override the retirement materializer count:
  `./target/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json --retire-worker 4 --retire-batch 8`

The normal fuzz loop uses envelope-aware VConfig/arg-pack mutators first, then
one canonical envelope normalization/constraint repair after the complete
mutation plan. Stage and supply-pool outputs carry that canonical guarantee to
`GpuExecutor` and bypass its repair pass; unmarked imported/legacy inputs still
use the executor safety net.
Use `--no-mutate` when you want fixed canonical envelope execution while still
exercising the LibAFL scheduler/stage/feedback path. Use `--no-mutate --runs N`
for a bounded version of that same fixed-seed path. Use `--runs N` without
`--no-mutate` for bounded normal fuzzing.

## Performance controls

- `--supply-threads N` selects `0..=32` host mutation producers and defaults to
  `2` in normal mutating fuzz mode; `0` disables the pool. Specialized and
  fixed-input modes select their compatible single-threaded paths automatically.
- `--retire-worker N` selects `0..=32` coverage materializers and defaults to
  `2` in normal mutating fuzz mode; `0` uses owner-thread retirement. Two
  workers are the validated default across both light and heavy kernels.
  `--retire-batch N` defaults to `8`, controls the owner-side commit batch, and
  requires a positive worker count when specified.
- Benchmark, coverage-campaign, raw-throughput, and seed-dump commands keep
  their specialized single-threaded paths automatically. RQ1 benchmark and RQ2
  coverage commands therefore need no worker overrides, while RQ4 mutating
  profiles use the normal two-retirement/two-supply default.
- `--raw-throughput-seconds S` runs an isolated RAPID2 submit/poll measurement.
  `RAPID_RAW_SKIP_COV_READ=1` skips Rust coverage reads; additionally setting
  `RAPID2_SKIP_COVERAGE_COPY=1` skips backend coverage copies and requires the
  Rust skip. The mode and both environment switches are off by default.
- Unified profiling requires `cargo build --release --bins --features
  profiling` and `RAPID_PROFILE=1`. Set `RAPID_PROFILE_OUTPUT` to select the
  schema-v2 JSONL destination; otherwise it defaults to `rapid-profile.jsonl`.
  Normal builds contain no profiling instrumentation.
- In restarting-manager fuzz runs, `fuzzer_async` permits one client for the
  first device in `CUDA_VISIBLE_DEVICES`. The initial broker does not acquire
  the GPU lock; a later client invocation does.

## Execution Model

Original:
```
Submit -> Execute (block) -> Read coverage -> Feedback -> Next
```

Async:
```
Submit -> Return immediately
Poll completed tasks -> Update coverage map -> Feedback/objective -> Corpus/solutions
```

Async with retirement worker:
```
submit/poll: submit -> poll -> dispatch acquired TaskResult
                           |
                 N materialize workers
                 fill share -> copy CFG+SIMT -> sparsify -> release batch
                           |
                 ordered LibAFL owner
                 O(1) observer-map swap -> exact novelty recheck
                 -> feedback/objective -> corpus/events
```

The restarting manager is set up before any helper thread is created. In the
fuzzing client, only the main execution context touches LibAFL state. Workers
never write `libafl_cov_map` and never call feedback. A task remains acquired
until a worker has copied both task-owned feedback maps for the whole
materializer batch; only then does that worker call `libafl_release_tasks()`
for that batch. The owner uses an owned `StdMapObserver`
and exchanges its `OwnedMutSlice` backing with the prepared CFG `Vec<u8>`;
installation and restoration move allocation ownership in O(1), without a
64-KiB owner-side copy. Sparse words are only a prefilter: the owner checks them
against current state immediately before each ordered commit. Fixed-capacity
channels and 64 credits bound acquired tasks and prepared host results.

## Files

```
cuda-fuzzer/
├── src/
│   ├── cuda_backend/       # Strict sync/async backend loaders
│   ├── async_fuzzer.rs     # AsyncBatchFuzzer (implements Fuzzer/Evaluator)
│   ├── gpu_executor.rs     # Non‑blocking executor (submit only)
│   ├── retirement.rs       # Bounded materialization and ordered handoff
│   ├── supply_pool.rs      # Snapshot-based host mutation producers
│   ├── submission_context.rs # Canonical-input submission marker
│   ├── gpu_client_guard.rs # One async client per visible GPU
│   ├── mutators/           # Example mutators
│   └── bin/
│       ├── fuzzer.rs       # Baseline binary
│       └── fuzzer_async.rs # Async binary
└── Cargo.toml
```

## C++/CUDA interface (RAPID2)

Provided by the generated RAPID2 Phase2 backend library (`librapid2_target.so`):
- `libafl_submit_with_id(const uint8_t*, size_t) -> uint64_t` (`0` means the
  submission was rejected, including when the target timeout has not been
  configured yet)
- `libafl_poll_results(TaskResult*, size_t) -> size_t` (pointer-based; returns
  task-owned edge and memory/index maps without embedding them in the result)
- `libafl_release_tasks(const uint64_t*, size_t) -> size_t` (release acquired task IDs)
- `libafl_set_target_timeout_ms(uint64_t)` (positive runtime target-timeout
  budget owned by the fuzzer)
- `libafl_get_queue_counts(LibAflQueueCounts*)`
- `libafl_wait()` (wait pending tasks)
- `libafl_stop()` (shutdown)

Rust side maps these symbols in `cuda_backend/async.rs`. The coverage map size must match across C++/Rust (`MAP_SIZE`/`EDGES_MAP_SIZE` = 65536).

## Monitoring

- Baseline prints standard LibAFL stats.
- Async also logs pending/evaluated counts and submissions. Use `RUST_LOG` to adjust verbosity.

## Notes

- RAPID2 async completion reports CUDA/kernel errors through
  `TaskResult.status: LibAflRunStatus`. `AsyncBatchFuzzer` maps `OK` to
  `ExitKind::Ok`, `TIMEOUT` to `ExitKind::Timeout`, and any other non-OK status
  to `ExitKind::Crash`.
- RAPID2 completion statuses that can poison the CUDA context (`CUDA_ERROR`,
  `TIMEOUT`, and `INTERNAL_ERROR`) are processed once for feedback/objective
  accounting and then abort the client process. The restarting manager is
  expected to relaunch a fresh process and CUDA context.
- RAPID2 target timeout is enforced in the backend active-task watchdog,
  measured from dispatch into a double-buffer slot rather than from
  `libafl_submit_with_id()` enqueue time. This avoids counting host/backend
  queue wait as target execution time. `fuzzer_async` configures the budget via
  `libafl_set_target_timeout_ms()`. On RAPID2 timeout, the backend publishes a
  `TaskResult` with `LibAflRunStatus::TIMEOUT`; a materializer copies its maps
  and releases the task, then the owner evaluates `TimeoutFeedback` before
  aborting the client so the restarting manager can start a fresh CUDA context.
- Throughput tuning: `with_batch_size()` sets the maximum number of completions
  pulled per non-blocking poll, while `with_max_pending()` controls when the
  fuzzer forces a drain. The batch size defaults to 8 and the pending window
  defaults to 2; `with_batch_size(0)` is rejected. The ordered window is capped
  at 32. RAPID2 uses queues with capacity 64, and the configured window must
  never exceed the backend queue capacity.
