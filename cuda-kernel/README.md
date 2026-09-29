# CUDA Kernel Phase2 Backends

This directory contains the CUDA target example and the four backend builders
used by the official Phase1/Phase2 pipeline:

- `cufuzz/`
- `origin/`
- `rapid/`
- `rapid2/`

All final backend shared libraries are built from Phase2 artifacts. The old
direct-compile demo targets have been removed.

## Builtin kernel pipeline

For the in-repository `kernel.cu` example, run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target builtin_phase_pipeline
```

Or invoke the driver directly:

```bash
python3 cuda-kernel/builtin_phase_pipeline.py \
  --out-root /tmp/builtin-phase \
  --run-id builtin-kernel \
  --cuda-arch sm_86 \
  --cuda-path /usr/local/cuda
```

The driver performs:

1. Capture the builtin CUDA target with the wrapped compiler.
2. Run `scripts/kernel-smoke/cli.py run` to produce Phase1 artifacts.
3. Run `scripts/kernel-rewrite/cli.py` to produce Phase2 artifacts.
4. Invoke each backend builder:
   - `cuda-kernel/cufuzz/build.py`
   - `cuda-kernel/origin/build.py`
   - `cuda-kernel/rapid/build.py`
   - `cuda-kernel/rapid2/build.py`

Generated libraries are placed under:

```text
<out-root>/out/<run-id>/kernels/<kernel-id>/phase2/backends/<backend>/
```

The final shared libraries are:

- `libphase2_cufuzz_target.so`
- `libphase2_origin_target.so`
- `libphase2_rapid_target.so`
- `librapid2_target.so`

## Building a backend from an existing Phase2 directory

Each backend can consume a single kernel's `phase2/` directory directly:

```bash
python3 cuda-kernel/cufuzz/build.py --phase2-dir /path/to/kernel/phase2
python3 cuda-kernel/origin/build.py --phase2-dir /path/to/kernel/phase2
python3 cuda-kernel/rapid/build.py  --phase2-dir /path/to/kernel/phase2
python3 cuda-kernel/rapid2/build.py --phase2-dir /path/to/kernel/phase2
```

Each builder compiles its `wrapper.cu`, links it with `kernel.device.bc`, emits
PTX, embeds that PTX into the host harness, and links the final `.so`.

The builtin pipeline also produces `origin-no-feedback/` by building `origin`
with `--feedback-instrumentation disabled`. This disables both the LLVM
feedback pass and wrapper/host feedback work. It is the controlled comparison
for `cufuzz`, which differs by allocating and freeing device memory for every
input:

```bash
python3 cuda-kernel/origin/build.py \
  --phase2-dir /path/to/kernel/phase2 \
  --out-dir /path/to/kernel/phase2/backends/origin-no-feedback \
  --feedback-instrumentation disabled
```

`cufuzz` is a LibAFL-driven CuFuzz-style/per-launch baseline, not an execution
of the official LibFuzzer-based CuFuzz implementation. See
`cuda-kernel/cufuzz/README.md` for the exact naming and run commands.

Both disabled configurations select the linked bitcode before RAPID's LLVM
feedback pass, disable wrapper feedback, and leave LibAFL's host feedback maps
zero. Use kernels without repository-specific coverage calls for formal LLVM
feedback experiments.

## Backend source layout

For each backend:

- `build.py` builds the final `.so` from Phase2 artifacts.
- `wrapper.cu` defines the device-side launch entry and includes the generated
  invoke/decode header via `FUZZER_INVOKE_HEADER`.
- `harness.cpp` is the host-side shared-library entrypoint.

RAPID2 additionally keeps its queue/runtime implementation in `rapid2/*.cuh`.
