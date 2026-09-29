# Third-Party Notices

GUnit is distributed under the Apache License, Version 2.0 (see `LICENSE`).
It builds on, links against, or redistributes parts of the following
third-party projects. Each project retains its own license; nothing here
changes those terms.

## Submodules (fetched by `git submodule update --init`)

| Component | Path | Upstream | License |
|---|---|---|---|
| LibAFL | `LibAFL/` | https://github.com/AFLplusplus/LibAFL | Apache-2.0 OR MIT (`LICENSE-APACHE`, `LICENSE-MIT`); pinned at `824f5535` plus the patches in `patches/libafl/` |
| LLVM | `rapid-llvm/` | https://github.com/llvm/llvm-project | Apache-2.0 WITH LLVM-exception (`LICENSE.TXT`); pinned at `llvmorg-21.1.8` |
| gpuRIR | `third_party/gpuRIR` | https://github.com/DavidDiazGuerra/gpuRIR | AGPLv3 |
| CudaSift | `third_party/CudaSift` | https://github.com/Celebrandil/CudaSift | MIT |
| phantom-fhe | `third_party/phantom-fhe` | https://github.com/encryptorion-lab/phantom-fhe | GPLv3 / see upstream |
| lietorch | `third_party/lietorch` | https://github.com/princeton-vl/lietorch | BSD-3-Clause / see upstream |
| TensorRT | `third_party/TensorRT` | https://github.com/NVIDIA/TensorRT | Apache-2.0 |
| CUTLASS | `third_party/cutlass` | https://github.com/NVIDIA/cutlass | BSD-3-Clause |
| FlashAttention | `third_party/flash-attention` | https://github.com/Dao-AILab/flash-attention | BSD-3-Clause |
| llama.cpp | `third_party/llama.cpp` | https://github.com/ggml-org/llama.cpp | MIT |
| CUDA Samples | `third_party/cuda-samples` | https://github.com/NVIDIA/cuda-samples | BSD-3-Clause / see upstream |
| GPUJPEG | `third_party/GPUJPEG` | https://github.com/CESNET/GPUJPEG | BSD-2-Clause |
| Kaldi | `third_party/kaldi` | https://github.com/kaldi-asr/kaldi | Apache-2.0 |
| HEonGPU | `third_party/HEonGPU` | https://github.com/Alisah-Ozcan/HEonGPU | Apache-2.0 |
| NVIDIA Apex | `third_party/apex` | https://github.com/NVIDIA/apex | BSD-3-Clause |
| DeepSpeed | `third_party/DeepSpeed` | https://github.com/microsoft/DeepSpeed | Apache-2.0 |
| Caffe | `third_party/caffe` | https://github.com/BVLC/caffe | BSD-2-Clause |
| Darknet | `third_party/darknet` | https://github.com/pjreddie/darknet | public domain / see upstream |

License names above are summaries for convenience; the authoritative text is
in each upstream repository.

## Extracted kernel sources

`benchmark/workloads/*/kernel.cu` (and selected kernels under
`benchmark/rq2/variants/`) are verbatim or adapted extracts from upstream
projects, used as fuzzing/evaluation targets. Each such directory carries a
`LICENSE.upstream` (and `provenance.json` / `upstream.source` /
`extraction.patch` where applicable) recording its origin and license:

- SHOC benchmark suite (`shoc_radix_sort`, `shoc_reduction`, `shoc_scan`)
- CUTLASS reference kernels (`cutlass_gemm`)
- FlashAttention-derived kernel (`flashattention_device1xn`)
- PyTorch-derived kernel (`pytorch_batchnorm`)

Extracted third-party adapters under `benchmark/rq2/variants/` record their
capture provenance in the directory README; the `shoc_radix_sort_fix` variant
derives from the SHOC extract and carries the same `LICENSE.upstream`.

## Rust crates

`cuda-fuzzer/` depends on crates.io packages resolved by Cargo at build time
(LibAFL itself is used through local path dependencies into the `LibAFL/`
submodule). Their licenses are recorded in the Cargo metadata and are not
redistributed in this repository.
