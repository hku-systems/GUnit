# CuFuzz-Style Per-Input-Allocation Backend

This backend provides the repository's controlled, deliberately naive CUDA
execution baseline. It uses the existing LibAFL synchronous fuzzer and the
same Phase2 task envelope, decoder, invocation adapter, manifest, mutators,
and launch policy as `origin`. For every valid test case, its host harness
allocates device memory, copies the input and invocation context, launches the
target once, synchronizes, copies the payload back, and frees the allocations.

This is a **CuFuzz-style**, **LibAFL-naive**, or **per-launch** baseline. It is
not the official LibFuzzer-based CuFuzz implementation and must not be
reported as such.

## Build

```bash
python3 cuda-kernel/cufuzz/build.py \
  --phase2-dir /path/to/kernel/phase2
```

The default artifact is:

```text
/path/to/kernel/phase2/backends/cufuzz/libphase2_cufuzz_target.so
```

## Fuzz

Use the same synchronous LibAFL frontend as `origin`:

```bash
./cuda-fuzzer/target/release/fuzzer \
  /path/to/libphase2_cufuzz_target.so \
  --manifest /path/to/kernel/manifest.json \
  --runs 1000
```

`--runs` bounds LibAFL stage iterations, not exact CUDA target invocations.
Use the reported `executions` counter for campaign budgets, or the allocation
probe below when an exact invocation count is required.

For a fixed-input execution comparison that still traverses LibAFL's normal
executor and feedback path:

```bash
./cuda-fuzzer/target/release/fuzzer \
  /path/to/libphase2_cufuzz_target.so \
  --manifest /path/to/kernel/manifest.json \
  --no-mutate --runs 1000
```

The exported host coverage and memory/index maps intentionally stay zero. The
builder lowers the pre-instrumentation linked bitcode directly, so it neither
runs RAPID's LLVM feedback pass nor collects device feedback. Input kernels
must not contain hand-written CUDA coverage calls; LLVM-only experiment targets
carry no device coverage map in the CuFuzz-style PTX.

The backend also exports `libafl_get_cufuzz_allocation_stats` so tests can
verify the per-input allocation/free lifecycle without adding feedback to the
fuzzer.
