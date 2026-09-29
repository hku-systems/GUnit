# GUnit

GUnit is a persistent and asynchronous kernel testing framework for heterogeneous computing. It transforms repeated host-driven GPU kernel testing into a continuous host-device pipeline, reducing launch and synchronization overhead while enabling fine-grained exploration of kernel behavior. This release implements CUDA kernel fuzzing on top of [LibAFL](https://github.com/AFLplusplus/LibAFL).

## Key Ideas

- **PerKernel** keeps a device-side testing harness resident across test cases and repeatedly invokes the target kernel without relaunching it for every input.
- **Asynchronous Testing Engine** decouples input generation, device execution, and feedback processing so that host and device work can overlap.
- **VConfig Rewriting** virtualizes CUDA launch configurations, allowing each test case to exercise a different logical block configuration under a fixed physical launch (multi-block logical grids are future work).
- **SIMT-Aware Kernel Feedback** distinguishes kernel executions through device-side behaviors such as warp participation and memory-access patterns.

The implementation has two complementary parts:

- Baseline synchronous fuzzing using a traditional harness and LibAFL’s standard `StdFuzzer` and `InProcessExecutor`.
- An optimized system using a persistent kernel (RAPID2) with a double‑buffered pipeline to eliminate launch overhead. This requires an async‑aware fuzzer that decouples input submission from coverage evaluation.

> Naming note: `origin` / `rapid` / `rapid2` are the historical names of the
> three execution backends (synchronous, ordered-completion, and persistent
> asynchronous). They are kept as identifiers throughout the code and docs;
> the persistent asynchronous stack is the GUnit design described above.

## Why two systems?

- Traditional GPU fuzzing submits one input, waits for the kernel to finish, reads coverage, then repeats. This is simple but under‑utilizes the GPU due to launch and transfer overhead.
- RAPID2 runs a persistent kernel and overlaps host/device work with double buffering. Inputs can stream continuously, but coverage and crashes arrive asynchronously. The classic synchronous fuzzer loop is no longer correct.
- To handle this, we introduce an async fuzzer (`fuzzer_async`) that submits inputs immediately and evaluates coverage later when results are available.

## Repository Layout

- `LibAFL/` – upstream LibAFL submodule; apply `patches/libafl/` before building the fuzzer.
- `cuda-kernel/` – CUDA harnesses and libraries
  - `origin/` – Original harness (blocking, lock‑step)
  - `rapid/` – Persistent harness with ordered completion (async submit/poll/release ABI)
  - `rapid2/` – Persistent kernel with double buffering and async interfaces
  - `cufuzz/` – Upstream-style baseline backend used by the RQ1 benchmark
- `cuda-fuzzer/` – Rust fuzzer(s)
  - `src/bin/fuzzer.rs` – Baseline fuzzer; selects the sync backend when the library
    exports `libafl_target`, otherwise the ordered backend (`--window-size`, max 32)
  - `src/bin/fuzzer_async.rs` – Async fuzzer for RAPID2
  - `src/async_fuzzer.rs` – `AsyncBatchFuzzer` implementation (decouples execution/evaluation)
  - `src/gpu_executor.rs` – `GpuExecutor` that only submits to GPU (non‑blocking)
  - `src/cuda_backend/` – Shared-library loaders (`sync.rs`, `ordered.rs`, `async.rs`)
- `benchmark/rq1/` – [RQ1 throughput benchmark](benchmark/rq1/README.md):
  12-workload dual-root build, runtime gate, campaign, and paper export
- `benchmark/rq2/` – [RQ2 coverage-growth benchmark](benchmark/rq2/README.md):
  build, completion-driven campaign, validation, and plots
- `benchmark/rq4/` – [RQ4 eight-lane performance profile](benchmark/rq4/README.md):
  canonical entry point `benchmark/rq4/run_profile.sh`
- `benchmark/reproduce.sh` – unified RQ1/RQ2/RQ4 reproduction driver
  (`verify` / `smoke` / `full`; campaigns resume, `REPRO_*` env overrides)
- `tests/`
  - `vconfig/` – LLVM/Phase2 virtual-dimension rewrite tests
  - `feedback_e2e/` – GPU-gated feedback, timeout, and VConfig backend E2E tests
  - `benchmark/` – RQ1/RQ2/RQ4 build, campaign, profile, and catalog unit tests
- Top‑level scripts: `docker_build.sh`, `docker_run.sh`

## Build

Prerequisites: CUDA toolkit, a CUDA‑capable GPU, Rust (stable), `cmake` (>= 3.20), and a C++ compiler.

```bash
git clone https://github.com/hku-systems/gunit.git
cd gunit
git submodule update --init rapid-llvm LibAFL
```

### Submodules and patched dependencies

Initialize the submodules before building. `rapid-llvm/` uses stock LLVM
`llvmorg-21.1.8` for the capture compiler. The historical fork's VirtualDim
pass is superseded in the current pipeline by
`tools/rapid-vconfig-instrument/` and is not needed. The tools under `tools/`
build against LLVM 22; that toolchain is separate from the LLVM 21.1.8 capture
compiler.

`LibAFL/` is pinned to upstream commit `824f5535` (the PR #3372 merge).
The fuzzer uses its crates through local path dependencies, so apply the three
repository patches before building `cuda-fuzzer`:

```bash
git submodule update --init rapid-llvm LibAFL
(cd LibAFL && git am ../patches/libafl/*.patch)
```

See [the LibAFL patch notes](patches/libafl/README.md) for what each patch
changes. Applying them advances only the local LibAFL checkout; the parent
repository remains pinned to the upstream commit.

### Python environment

When running Python tooling in this repository, first look for the repo-local
virtual environment at `.venv/`. If it exists, use it explicitly:

```bash
.venv/bin/python scripts/kernel-smoke/cli.py ...
.venv/bin/python scripts/kernel-rewrite/cli.py ...
```

If `.venv/` does not exist and the task needs Python dependencies, create it
before installing packages:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r scripts/kernel-smoke/requirements.txt
```

For build systems that spawn Python subprocesses, pass the environment through
explicitly instead of relying on an activated shell:

```bash
VIRTUAL_ENV="$(pwd)/.venv" \
PATH="$(pwd)/.venv/bin:$PATH" \
PYTHON="$(pwd)/.venv/bin/python" \
<build command>
```

Third-party Python/CUDA builds may need additional packages in `.venv`; for
example FlashAttention requires PyTorch before its `setup.py` can construct
CUDA extension build commands.

### One-command (top-level CMake)

Build kernels + fuzzers from the repo root:

- Configure (defaults to `Release`): `cmake -S . -B build`
- Configure Debug: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`
- Build everything: `cmake --build build --target rapid_all -j`

Notes:
- CUDA SM arch is auto-detected via CUDA Runtime API during CMake configure (no `nvidia-smi`).
- If your environment cannot run CUDA runtime queries during configure (common in restricted containers), configure will fail; either fix the driver/GPU access for the build environment or override it:
  - `cmake -S . -B build -DRAPID_CUDA_ARCHITECTURES=86`
- Final artifacts are placed under the build directory:
  - CUDA libs + `test_main`: `build/cuda-kernel/`
  - Fuzzer binaries (cargo output): `build/cuda-fuzzer/<debug|release>/`
- Useful kernel build toggles:
  - Timing stats: `-DRAPID_ENABLE_TIMING=ON` (defines `ENABLE_KERNEL_TIMING=1`)
  - NVTX markers: `-DRAPID_ENABLE_NVTX_PROFILING=ON` (defines `ENABLE_NVTX_PROFILING=1`)
- Generate `compile_commands.json` (for clangd/IDE):
  - `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
  - Output: `build/compile_commands.json`

### Manual (per-component)

1) Build CUDA harness libraries
- `cd cuda-kernel`
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build Phase2 backends, including RAPID2: `cmake --build build --target builtin_phase_pipeline -j`
- Build the shared-library smoke-test tool: `cmake --build build --target test_main -j`

2) Build fuzzer binaries
- `cd cuda-fuzzer`
- `cargo build --release`
- Binaries: `target/release/fuzzer` (sync) and `target/release/fuzzer_async` (async)

## Run

- Generate backend libraries with `builtin_phase_pipeline`, then pass one of the generated `.so` files and its artifact `manifest.json` to the fuzzer.
- RAPID2 async example:
  - `./build/cuda-fuzzer/release/fuzzer_async /path/to/librapid2_target.so --manifest /path/to/manifest.json`
- Baseline sync example:
  - `./build/cuda-fuzzer/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json`
- Phase3 payload smoke helpers:
  - `./build/cuda-fuzzer/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --dump-seed`
  - `./build/cuda-fuzzer/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --no-mutate`
  - `./build/cuda-fuzzer/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --no-mutate --runs 1000`
  - `./build/cuda-fuzzer/release/fuzzer /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json --runs 3`
  - Batch-check Phase2 built kernels that already have backend artifacts:
    `.venv/bin/python scripts/phase3_smoke.py --run-dir /path/to/kernel-smoke/run --backend origin --profile release --build-fuzzer --runs 1000`
  - Batch-check manifest-driven seed generation without loading backend `.so` files:
    `.venv/bin/python scripts/phase3_smoke.py --run-dir /path/to/kernel-smoke/run --dump-seeds --profile release --build-fuzzer`
- Helper script: `./cuda-fuzzer/run_comparison.sh <path_to_lib.so>`

## Logging

The fuzzer uses Rust's `env_logger`. Control log level with `RUST_LOG`.

- Levels (most→least verbose): `trace`, `debug`, `info` (default), `warn`, `error`

Examples with cargo run:

```bash
# Show all logs
RUST_LOG=trace cargo run --release --bin fuzzer         -- /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json
RUST_LOG=trace cargo run --release --bin fuzzer_async   -- /path/to/librapid2_target.so --manifest /path/to/manifest.json

# Only info/warn/error
RUST_LOG=info  cargo run --release --bin fuzzer_async   -- /path/to/librapid2_target.so --manifest /path/to/manifest.json

# Only warnings and errors
RUST_LOG=warn  cargo run --release --bin fuzzer_async   -- /path/to/librapid2_target.so --manifest /path/to/manifest.json

# Errors only
RUST_LOG=error cargo run --release --bin fuzzer_async   -- /path/to/librapid2_target.so --manifest /path/to/manifest.json

# Filter by module
RUST_LOG=cuda_fuzzer=debug cargo run --release --bin fuzzer_async -- /path/to/librapid2_target.so --manifest /path/to/manifest.json
```

Examples with built binaries:

```bash
RUST_LOG=trace ./cuda-fuzzer/target/release/fuzzer        /path/to/libphase2_origin_target.so --manifest /path/to/manifest.json
RUST_LOG=trace ./cuda-fuzzer/target/release/fuzzer_async  /path/to/librapid2_target.so --manifest /path/to/manifest.json
```

## Key Concepts

- RAPID2 persistent kernel keeps the GPU busy and hides transfer latency via double buffering. See `cuda-kernel/rapid2/README.md`.
- Async fuzzer (`AsyncBatchFuzzer`) separates input submission from evaluation:
  - Submits inputs via `GpuExecutor` without blocking.
  - Polls completed GPU tasks and updates the global coverage map.
  - Runs LibAFL feedback/objective checks when results arrive.
- Phase3 fuzzer input encoding is manifest-driven:
  - `--manifest` is required; the fuzzer no longer falls back to a repository-root manifest.
  - `--dump-seed` prints the canonical full RAPID task envelope as hex without loading the `.so`; the bytes after `RapidTaskEnvelopeHeader` are the canonical `arg-pack-v1` payload.
  - `--no-mutate` still runs the normal LibAFL fuzz loop/stage, but each stage iteration rewrites the input to the manifest-driven canonical full envelope and submits that fixed input. Use this when validating long-running Phase2 harness execution without semantic mutation.
  - `--runs N` bounds the current LibAFL fuzzing mode to `N` loop iterations. Use `--runs N` for bounded normal fuzzing and `--no-mutate --runs N` for bounded canonical-seed smoke runs.
  - The default fuzz loop uses envelope-aware VConfig/arg-pack mutators followed by canonical normalize/constraint repair.
- VConfig is an envelope-level control channel, not part of the raw arg-pack
  payload. Phase 2 requests the virtual-dimension rewrite by default and
  records `vconfig_requested`, `vconfig_enabled`, and an optional
  `vconfig_disabled_reason` in its generated artifacts. Enabled targets read
  logical `gridDim`/`blockDim` from `RapidKernelContext::vconfig`; unsupported
  divergent/helper/data-dependent barrier and inline-assembly cases keep the
  base backend path and record a stable disabled reason. Fixed main-path
  `barrier0` sites are supported by inserting matching virtual sync points
  before inactive sliced threads return. Fixed-trip loop barriers are supported
  for the pass's simple PHI self-loop form and Clang `optnone`
  header/body/latch form by emitting an inactive sync-only loop. NVVM warp
  intrinsics force warp-aligned logical slicing plus warp-aligned physical
  block candidates.
- The CUDA library exports async APIs consumed by `cuda_backend/` loaders:
  - `libafl_submit_with_id`, `libafl_poll_results`,
    `libafl_release_tasks`, `libafl_set_target_timeout_ms`,
    `libafl_get_queue_counts`, `libafl_wait`, `libafl_stop`.
  - Coverage map size must match (`MAP_SIZE` in C++ == `EDGES_MAP_SIZE` in Rust, 65536).

## Evaluation

The evaluation harness used in the paper lives under `benchmark/`:

- `benchmark/rq1/` – throughput benchmark (12 workloads, dual build roots, paper CSV export)
- `benchmark/rq2/` – coverage-growth benchmark (device CFG, SIMT memory novelty, VConfig on/off)
- `benchmark/rq4/` – eight-lane performance profile (`benchmark/rq4/run_profile.sh`)
- `benchmark/reproduce.sh` – unified `verify` / `smoke` / `full` reproduction driver

## Documentation

- Async architecture rationale and design: `cuda-fuzzer/ASYNC_ARCHITECTURE.md`
- RAPID2 pipeline, buffering and exported APIs: `cuda-kernel/rapid2/README.md`
- How the async fuzzer works and how to run it: `cuda-fuzzer/README_ASYNC.md`
- VConfig transport and Phase 2 virtual-dimension policy: `docs/vconfig.md`

## Docker (optional)

- Build: `./docker_build.sh`
- Run: `./docker_run.sh`

## Paper

**GUnit: A Persistent and Asynchronous Kernel Testing Framework for Heterogeneous Computing**

Citation information and artifact instructions will be linked here when they are available.

## Contributing

- Install hooks: `./scripts/install-hooks.sh`
- Use signed commits: `git commit -s -m "..."`

## License

GUnit is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).
Third-party components and their licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
