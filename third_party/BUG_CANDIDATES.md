# Third-party target bug candidates

This file records target-side bug candidates found while preparing bounded RAPID2 fuzz validation for vendored third-party CUDA projects. These entries are not counted as fuzzer failures unless later evidence shows the fuzzer generated an invalid input outside the reviewed manifest constraints.

## phantom-fhe: `zero_coeff_count_kernel`

- Symbol: `_Z23zero_coeff_count_kernelPjPKmm`
- Kernel id: `_Z23zero_coeff_count_kernelPjPKmm__4fc80c15`
- Source: `third_party/phantom-fhe/src/polymath.cu:667`
- Status: `target_bug_candidate_divergent_barrier`
- Upstream report: https://github.com/encryptorion-lab/phantom-fhe/issues/27

Evidence:

- The kernel computes `tid = blockIdx.x * blockDim.x + threadIdx.x`.
- It enters the body only when `tid < ct1_size`.
- Inside that conditional body it executes `__syncthreads()` twice.

Why this is a target bug candidate:

CUDA block-level barriers must be reached uniformly by all live threads in a block. With RAPID2 physical launches, and also with many native launches when `ct1_size < blockDim.x`, only the first `ct1_size` threads enter the conditional. The remaining threads skip both barriers. That is a divergent barrier and can deadlock independently of fuzzer mutation correctness.

Current fuzz handling:

- `third_party/fuzz/support/phantom_fhe.json` marks this kernel as `skip`.
- The skip reason is `target_bug_candidate_divergent_barrier`.
- Report generation counts it under target bug candidates instead of runtime fuzzer failures.

## CudaSift: `FindMaxCorr10`

- Symbol: `_Z13FindMaxCorr10P9SiftPointS0_ii`
- Kernel id: `_Z13FindMaxCorr10P9SiftPointS0_ii__845fd768`
- Source: `third_party/CudaSift/matching.cu:301`
- Unsafe reads: `third_party/CudaSift/matching.cu:396-397`
- Status: `target_bug_candidate_negative_match_index_oob`
- Upstream report: not filed

Evidence:

- The public `MatchSiftData` API rejects zero point counts but does not require `numPts2 >= 32`.
- `MatchSiftData` selects mode 10 and launches `FindMaxCorr10`.
- `FindMaxCorr10` initializes every candidate match index to `-1`.
- The `sift2` tile loop executes only while `bp2 < numPts2 - 32 + 1`. For `1 <= numPts2 < 32`, it executes zero times, so the match index remains `-1`.
- The final `threadIdx.y == 0` path reads `sift2[index].xpos` and `sift2[index].ypos` without checking that `index` is nonnegative.

Why this is a target bug candidate:

The public API accepts positive `numPts2` values below 32, and those values deterministically leave the selected match index at `-1` before the kernel reads `sift2[-1]`. A clean minimal configuration is `numPts1=32, numPts2=1`; keeping `numPts1` equal to the kernel's 32-point output width avoids conflating this defect with a separate partial-output-block condition.

Runtime confirmation status:

- The direct-kernel reproducer is
  `third_party/fuzz/repro/findmaxcorr10_oob_repro.cu`.
- On an RTX 3090 (`sm_86`), it produced `sift1[0].match = -1` and read
  `match_xpos` / `match_ypos` through `sift2[-1]`.
- `compute-sanitizer --tool memcheck` reported no errors, so this run confirms
  the negative-index dereference but not a sanitizer-detected allocation
  violation. A public-API reproducer has not been run.

Build from the repository root and run under Compute Sanitizer:

```bash
/usr/local/cuda/bin/nvcc -std=c++14 -arch=sm_86 \
  -I third_party/CudaSift -o /tmp/fmc10_repro \
  third_party/fuzz/repro/findmaxcorr10_oob_repro.cu \
  third_party/CudaSift/matching.cu third_party/CudaSift/cudaImage.cu
compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/fmc10_repro
```

Current fuzz handling:

- `third_party/fuzz/support/cudasift.json` currently marks `FindMaxCorr10` as `run`.
- `third_party/fuzz/kernel_constraints/cudasift.json` pins both `numPts1` and `numPts2` to 32, excluding the deterministic `numPts2 < 32` trigger described above.

Fuzz-pipeline replay evaluation:

- `third_party/fuzz/kernel_constraints/cudasift_replay_findmaxcorr10.json`
  permits `numPts2` from 1 to 32 while pinning `numPts1` to 32 and allocating
  at least one 576-byte `SiftPoint` for `sift2`.
- On 2026-08-01 (`sm_86`, GPU 0), the manifest-generated seed decoded to
  `numPts1=32, numPts2=1`. A fixed-seed `fuzzer_async --no-mutate --runs 10`
  replay exited successfully without a crash.
- This demonstrates that the standard input pipeline can generate and execute
  the trigger. The direct reproducer above supplies the negative-index oracle;
  the fuzzer replay itself did not report a runtime failure.
