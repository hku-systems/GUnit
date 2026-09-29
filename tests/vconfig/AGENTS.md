# Agents: vconfig virtual dims

## Dynamic virtual dimensions (persistent kernel)
- We use the LLVM `VirtualDim` pass to rewrite `VDIM_KERNEL` kernels so their
  effective grid/block dimensions can be changed at runtime.
- `vdim_set` now takes **device pointers**: `vdim_set(dim3 *vgrid, dim3 *vblock)`.
  The pointers must reference device‑accessible memory (device allocation +
  host→device copy, or unified memory).
- The host stub passes these pointers into the kernel. The kernel reads the
  `dim3` values from device memory on each launch, so a persistent kernel can
  adjust its effective parallelism by updating the device values between
  iterations instead of relaunching.
- If the device pointers are null, the kernel falls back to physical
  `gridDim/blockDim` values (safety default).

## Marking kernels for the pass
- Use `VDIM_KERNEL` (e.g., `__attribute__((annotate("vdim")))`) on the kernel.
- The pass recognizes:
  - `vdim` / `vdim-args`: use runtime virtual dims.
  - `vdim=Gx,Gy,Gz,Bx,By,Bz`: use compile‑time constant virtual dims.

## Usage pattern (example)
- Allocate `dim3` on device or unified memory.
- Copy/initialize `vgrid` and `vblock`.
- Call `vdim_set(d_vgrid, d_vblock)`.
- Launch the persistent kernel; update `*d_vgrid/*d_vblock` as needed.

## Behavior notes
- `vdim_set` only stores the **device pointers**; it does not copy data.
- Virtual dims must be `<=` physical dims; threads outside virtual dims return
  immediately (guard inserted by the pass).
- Updates to `*d_vgrid/*d_vblock` take effect on the next kernel entry.
  If you need updates to be visible inside a long‑running persistent loop,
  the kernel must explicitly reload those values.

## Build & run (end‑to‑end test)
### 1) Build LLVM tools (once)
- The VConfig pass plugin lives in `tools/rapid-vconfig-instrument`.
- You need clang/opt/llc from LLVM 22; RAPID no longer requires modifying or
  rebuilding the `VirtualDimPassPlugin` target inside `rapid-llvm`.
- Recommended CMake configure (Ninja or Make both fine):
  - `-DLLVM_ENABLE_PROJECTS=clang`
  - `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"`
  - `-DLLVM_ENABLE_PLUGINS=ON`
  - `-DCMAKE_BUILD_TYPE=Release`

Example:
```
cmake -S rapid-llvm/llvm -B rapid-llvm/build -G Ninja \
  -DLLVM_ENABLE_PROJECTS=clang \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_PLUGINS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build rapid-llvm/build --target clang opt llc --parallel
```

### 2) Build the vdim end‑to‑end test
- Script: `tests/vconfig/build_vdim_end2end.sh`
- It uses `LLVM_BUILD` to locate `clang/opt/llc`.
- If `VDIM_PLUGIN` is not set, it builds the out-of-tree plugin with
  `tools/rapid-vconfig-instrument/build.py`.

Example:
```
LLVM_BUILD=$PWD/rapid-llvm/build \
  tests/vconfig/build_vdim_end2end.sh
```

### 3) Run
```
tests/vconfig/bin/vdim_end2end
```

## One‑step build (clang plugin)
- Script: `tests/vconfig/build_vdim_one_step.sh`
- It uses `VDIM_PLUGIN` to locate the pass plugin.

Example:
```
VDIM_PLUGIN=$PWD/tests/vconfig/bin/rapid-vconfig-pass-plugin.so \
  tests/vconfig/build_vdim_one_step.sh
```

## Notes / troubleshooting
- If your LLVM build has an empty default target triple, pass an explicit
  `--target=x86_64-unknown-linux-gnu` to clang or configure
  `LLVM_DEFAULT_TARGET_TRIPLE` in CMake.
