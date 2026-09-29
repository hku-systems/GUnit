Agent Guide and Persistent Context

Scope: This file applies to the entire repository and serves as persistent memory and a development guide for agents.

Agent Memory – Quick Reference

- Project Goals
  - Standardize fuzz testing for CUDA kernels using LibAFL.
  - Optimized variant uses a persistent kernel + double-buffered pipeline (RAPID2) to remove launch/transfer overhead.

- Backend Architecture and Paths
  - Traditional-style synchronous backend
    - Backend: `cuda-kernel/origin/` (built from Phase2 artifacts)
    - Fuzzer: `cuda-fuzzer/src/bin/fuzzer.rs` (StdFuzzer + InProcessExecutor)
  - Ordered (RAPID, asynchronous with ordered completion)
    - Backend: `cuda-kernel/rapid/` (ordered completion queue, built from Phase2 artifacts)
    - Fuzzer: `fuzzer` auto-selects the ordered backend when the library lacks the sync ABI
      (`libafl_target`); `--window-size` (1..=32) sets the completion window
  - Optimized (RAPID2, asynchronous)
    - Backend: `cuda-kernel/rapid2/` (persistent kernel + double buffer, built from Phase2 artifacts)
    - Fuzzer: `cuda-fuzzer/src/bin/fuzzer_async.rs`
    - Design doc: `cuda-kernel/rapid2/README.md`
  - Baseline (cufuzz)
    - Backend: `cuda-kernel/cufuzz/` (upstream-style baseline used by the RQ1 benchmark)

- Async Problem and Motivation
  - Under RAPID2, inputs can be continuously enqueued while execution/coverage completion is asynchronous.
  - The classic “submit → wait → read coverage” assumption no longer holds; the fuzzer must be redesigned for async.

- Current Async Implementation (Key Files)
  - `cuda-fuzzer/src/async_fuzzer.rs`
    - AsyncBatchFuzzer implements LibAFL’s Fuzzer/Evaluator/EventProcessor.
    - Decouples submission from evaluation; maintains a `pending_tasks` queue.
    - Periodically polls GPU completions; writes coverage to the global cov_map before running feedback/objective.
    - `force_evaluation()` calls RAPID2 `wait` to wait for outstanding work on shutdown.
  - `cuda-fuzzer/src/gpu_executor.rs`
    - GpuExecutor only submits (non-blocking) and returns `ExitKind::Ok`; observers read from the shared coverage map.
  - `cuda-fuzzer/src/cuda_backend/`
    - `async.rs` / `sync.rs` / `ordered.rs`: dynamically load the shared library and symbols:
      `libafl_submit_with_id`, `libafl_poll_results`, `libafl_release_tasks`,
      `libafl_get_queue_counts`, `libafl_wait`, `libafl_wait_for_completion`, `libafl_stop`,
      `libafl_set_target_timeout_ms`, and the global `libafl_cov_map`.
    - Provides helpers: `poll_completed_tasks()`, `get_queue_counts()`, etc.
    - `fuzzer.rs` picks `SyncCudaBackend` when the library exports `libafl_target`,
      otherwise the ordered backend (`ordered_fuzzer.rs`) with the default window.
  - Binaries
    - Synchronous: `fuzzer`
    - Asynchronous: `fuzzer_async`
    - Normal mutating `fuzzer_async` runs default to `2` retirement materializers plus `2` supply threads;
      benchmark, coverage, bounded `--no-mutate`, raw-throughput, and `--dump-seed` modes select `0+0` automatically.
      Explicit `--retire-worker 0` / `--supply-threads 0` still disable either pool.
    - Retirement admits up to `64` end-to-end tasks; default `8/2` batches release four tasks after full copies.
      Supply waits via `recv_timeout`; metrics split `max_in_flight`, `max_gpu_pending`, and `max_retirement_credits_held`.

- RAPID2 C API (Key)
  - Location: `cuda-kernel/rapid2/libafl_interface_shared.cuh` (exported by the Phase2 backend harness)
  - Functions:
    - `libafl_submit_with_id(const uint8_t*, size_t) -> uint64_t` (non-blocking submit; returns task_id)
    - `libafl_poll_results(TaskResult*, size_t) -> size_t` (non-blocking pull of completions)
    - `libafl_release_tasks(const uint64_t*, size_t) -> size_t` (release completed tasks back to the pool; required after consuming results)
    - `libafl_get_queue_counts(LibAflQueueCounts*)` (pending/completed counts in one call)
    - `libafl_wait()` (wait outstanding tasks), `libafl_wait_for_completion()`
    - `libafl_set_target_timeout_ms(uint64_t)` (backend watchdog budget)
    - `libafl_stop()` (shutdown)
  - TaskResult fields: `task_id`, `input_ptr`, `edge_ptr`, `simt_memcov_ptr`, `edge_size`,
    `simt_memcov_size`, `status` (`LibAflRunStatus`), `exec_time_ns`.
  - Crash/timeout are wired: `status` maps to `ExitKind::Ok/Timeout/Crash`; non-recoverable
    statuses abort the client so the restarting manager relaunches with a fresh CUDA context.
  - The ordered `rapid` backend exports the same submit/poll/release ABI plus
    `libafl_get_ordered_queue_counts`; it has no `libafl_target` sync entry point.

- Invariants
  - Coverage bitmap size must match: C++ `MAP_SIZE` == Rust `EDGES_MAP_SIZE` (both 65536).
  - Observers directly read the shared coverage map; always copy GPU results into this map before evaluation.
  - Window/queue liveness contract: the fuzz completion window is capped at 32
    (`MAX_ORDERED_WINDOW_SIZE`) and RAPID2 task queues hold 64 slots; the window must
    never exceed backend queue capacity, or the single submit/poll thread can deadlock.
  - Each RAPID2 `TaskData` owns a full-capacity `MAX_INPUT_SIZE` pinned buffer; pinned
    buffers are only freed at pipeline teardown, never while the persistent kernel runs.

- Build and Run (Quick)
  - Build Phase2 backends, including RAPID2: `cd cuda-kernel && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target builtin_phase_pipeline`
  - Build fuzzers: `cd cuda-fuzzer && cargo build --release`
  - Run (sync): `./target/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json`
  - Run (async): `./target/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json`
  - More details: `README.md`, `cuda-fuzzer/README_ASYNC.md`

- Testing (Quick)
  - `test_main`: run from `cuda-kernel/build/cuda-kernel/` against one generated target library, e.g. `./test_main /path/to/libphase2_origin_target.so` (or `libphase2_rapid_target.so` / `librapid2_target.so`).
  - `fuzzer` / `fuzzer_async`: both use restarting manager; to actually fuzz you must start twice:
    - first process starts the broker
    - second process starts the fuzzing client

- Status and TODOs
  - Async submission and delayed evaluation form a working end-to-end path; `fuzzer_async` runs.
  - Crash/timeout propagation is wired (`TaskResult.status` → `ExitKind`), with the backend
    watchdog and fresh-context restart validated in third-party campaigns.
  - VConfig (per-input logical launch configuration) is implemented for origin/rapid/rapid2;
    see `docs/vconfig.md`. Multi-block logical grids remain future work.
  - The RQ1 benchmark suite lives in `benchmark/rq1/` and selects the canonical
    12-workload corpus in `benchmark/workloads/` (7 backend variants / 9 campaign
    configurations); its perf workloads opt out of VConfig via
    `vconfig_reserved: false`.
  - The RQ4 eight-lane `LibAFL+` performance profile lives in `benchmark/rq4/`;
    its canonical entry point is `benchmark/rq4/run_profile.sh`.
  - Potential improvements: `AsyncMutationalStage` (batched mutations), adaptive batching, multi-GPU, richer stats/monitoring.

- Reference Docs
  - Overview & usage: `README.md`
  - Async design: `cuda-fuzzer/ASYNC_ARCHITECTURE.md`, `cuda-fuzzer/README_ASYNC.md`
  - RAPID2 details & APIs: `cuda-kernel/rapid2/README.md`
  - Implementation roadmap: `IMPLEMENTATION_PLAN.md`
  - Phase 0 interface baseline: `docs/phase0-interface-mapping.md`
  - Phase 1 artifact metadata schema: `docs/artifact-metadata.schema.json`

- Roadmap Working Agreement
  - `IMPLEMENTATION_PLAN.md` is the single roadmap source of truth.
  - Keep phase boundaries aligned with the roadmap:
    - Phase 1 outputs manifest/rewrite-plan metadata (see `docs/kernel-smoke-pipeline.md`).
    - Phase 2 executes `__global__ -> __device__` rewrite + decode/invoke integration (see `docs/kernel-rewrite-pipeline.md`).
    - Phase 3 does fuzzer-side manifest-driven input encoding.
  - Keep `origin`/`rapid`/`rapid2` decode/invoke semantics aligned (see `docs/phase0-interface-mapping.md`).

Contribution and Style Guidelines

- Principles: simple and clear; prioritize readability and maintainability; align with common Linux and LibAFL practices.
- Commits
  - Small, atomic changes; one logical change per commit.
  - Short subject (≤ 50 chars recommended); body explains motivation and effect when needed.
  - Commit bodies must use itemized bullets (use `- ...` for each key point).
  - Use Signed-off-by (as in README): `git commit -s -m "..."`.
- Pull Requests and Merge Policy
  - After creating a PR, do NOT merge proactively.
  - Merge is allowed only after an explicit second confirmation from the user/reviewer.
  - When merging PRs, use rebase merge only (`Rebase and merge`); do not use merge commit or squash merge.
- C/C++ (CUDA/RAPID2)
  - Prefer straightforward, predictable code; avoid excessive abstraction/template magic.
  - Keep cross-language structs (e.g., TaskResult) POD/standard-layout; stable fields and alignment.
  - RAII for resources; unified cleanup paths; avoid unsynchronized/un-aligned device access.
  - Minimize lock/sync scope; explicit atomic/memory ordering semantics.
  - Critical paths: reduce copies/allocations/branches; keep NVTX/debug logging behind optional flags.
- Rust (LibAFL integration)
  - Match LibAFL architecture: keep the Executor simple; put async complexity into Fuzzer/Evaluator/EventProcessor.
  - Explicit trait bounds and lifetimes; avoid unnecessary `unsafe` and global mutable state.
  - Use `log`/`env_logger`; keep cross-FFI constants consistent (e.g., `EDGES_MAP_SIZE`).
  - Prefer small, focused modules/functions; return semantic errors (`anyhow`/`Error`), don’t swallow failures.
  - Before feedback evaluation, ensure the shared coverage map is up to date.

Agent Tips

- When changing coverage logic, preserve invariants and order: poll → write shared coverage map → run feedback/objective.
- Avoid unrelated complexity; keep changes minimal and aligned with LibAFL traits.
- Use `apply_patch` for edits; do not add license headers; follow existing code style.
- Consult `README.md`, `cuda-fuzzer/README_ASYNC.md`, `cuda-kernel/rapid2/README.md` before modifying async paths.

Development Philosophy (IMPORTANT - Must Understand and Follow)

These principles are fundamental to maintaining code quality and project integrity. Every contributor and agent must internalize and practice them:

- Be ashamed of guessing APIs in the dark; be proud of reading the docs carefully.
- Be ashamed of vague execution; be proud of seeking clarification and confirmation.
- Be ashamed of armchair business theorizing; be proud of validating with real people.
- Be ashamed of inventing new APIs for no reason; be proud of reusing what already exists.
- Be ashamed of skipping validation; be proud of proactive testing.
- Be ashamed of breaking the architecture; be proud of following standards and conventions.
- Be ashamed of pretending to understand; be proud of honest "I don't know."
- Be ashamed of blind edits; be proud of careful refactoring.
