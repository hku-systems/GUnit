# RQ2 geometry-freed variants (A/B class)

Derived variants and direct VConfig entries for already-integrated kernels.
Rewrites keep the originals untouched as negative controls and ship a
bitwise/atol correctness probe; direct entries reuse unchanged, already
geometry-safe kernel bodies. Gains must come from the original kernel's
addressing, never from adapter-invented scheduling.

| Variant | Class | Change | Result |
|---|---|---|---|
| `kaldi_power_spectrum_blockdim/` | B (pseudo-pinned) | stride `CU1DBLOCK` → `blockDim.x` (1 line) + block candidates 32/64/128/256 | 104 → 198 (+90%); all backends consistent; probe bitwise-passed |
| `apex_index_mul_vgeo/` | B (pseudo-pinned) | launch `dim3(32,8)` → block.x ∈ {4,8,16,32}, y=8 fixed | 801 → 1786 (+123%); all backends consistent; probe bitwise-passed |
| `cutlass_syrk_dispatch/` | A (template-pinned) | dispatcher over `SyrkTile<8,8>/<16,16>/<32,8>` selected by a `tile_x` scalar paired to the vconfig shape via `scalar_eq_logical_block_dim` normalization | 692 → 942 (+36%); all backends consistent; probe bitwise-passed |
| `cutlass_gemm_dispatch/` | A (template-pinned) | dispatcher over `GemmTile<8,8>/<16,16>/<32,8>` with `tile_x` paired to the VConfig shape | 645 → 978 (+52%) |
| `inverseCNDKernel_vgeo` | B (already geometry-safe) | direct VConfig entry reusing the unchanged block-strided vendor kernel for logical block.x 32/64/128 | 72 → 128 (+78%) |
| `batched_extract_window_vgeo` | B (already geometry-safe) | direct VConfig entry reusing the unchanged block-strided Kaldi kernel for logical block.x 32/64/128/256 | 44 → 44 (on = off); the original registry entry was already open-policy, so this duplicate entry exposes no new geometry |
| `gpujpeg_dct_dispatch/` | A (template-pinned) | dispatcher over 2/4/8-warp DCT bodies with `warp_count` paired to block.y | 351 → 351 (on = off): per-block memory accesses are invariant; CFG 7 → 10 comes from dispatcher switch/guard sites |
| `cutlass_trmm_dispatch/` | A (template-pinned) | dispatcher over `TrmmTile<8,8>/<16,16>/<32,8>` with `tile_x` paired to the VConfig shape | 365 → 574 (+57%) |

## Captured third-party adapters

| Adapter | Tracked source | Capture provenance |
|---|---|---|
| `apex_adam_cuda_kernel` | [`apex_optimizer_adapters.cu`](apex_adam_maybe_cast/apex_optimizer_adapters.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave3/apex/capture_sources/apex_optimizer_adapters.cu` |
| `apex_maybe_cast_kernel` | [`apex_optimizer_adapters.cu`](apex_adam_maybe_cast/apex_optimizer_adapters.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave3/apex/capture_sources/apex_optimizer_adapters.cu` |
| `deepspeed_lamb_cuda_kernel_part3` | [`deepspeed_lamb_part3_adapter.cu`](deepspeed_lamb_part3/deepspeed_lamb_part3_adapter.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave3/deepspeed/capture_sources/deepspeed_lamb_part3_adapter.cu` |
| `flash_bwd_convert_dq_pod_kernel` | [`flash_bwd_convert_dq_pod.cu`](flash_attention_bwd_convert_dq_pod/flash_bwd_convert_dq_pod.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave3/flash_attention/capture_sources/flash_bwd_convert_dq_pod.cu` |
| `flash_bwd_dot_do_o_pod_kernel` | [`flash_bwd_dot_do_o_pod.cu`](flash_attention_bwd_dot_do_o_pod/flash_bwd_dot_do_o_pod.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave3/flash_attention/capture_sources/flash_bwd_dot_do_o_pod.cu` |
| `heongpu_decryption_tile_kernel` | [`heongpu_decryption_tile_adapter.cu`](heongpu_decryption_tile/heongpu_decryption_tile_adapter.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave3/heongpu/capture_sources/heongpu_decryption_tile_adapter.cu` |
| `cholesky_solve6x6_forward_raw_kernel` | [`cholesky_solve6x6_forward_raw.cu`](lietorch_cholesky_solve6x6_forward_raw/cholesky_solve6x6_forward_raw.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave2/lietorch/capture_sources/cholesky_solve6x6_forward_raw.cu` |
| `topk_moe_cuda_128_no_bias_adapter` | [`topk_moe_cuda_128_no_bias_adapter.cu`](llama_cpp_topk_moe_cuda_128_no_bias/topk_moe_cuda_128_no_bias_adapter.cu) | `build/e2e/third-party-fuzz/20260802-rq2-wave3/llama_cpp/capture_sources/topk_moe_cuda_128_no_bias_adapter.cu` |

## Recorded blocker: DeepSpeed `scan_sort` (NO-GO, L)

The "parameterize the 512" plan is rejected. The premise that the kernel is
CUB/bitonic-based is wrong:
[`third_party/DeepSpeed/csrc/random_ltd/token_sort.cu`](../../../third_party/DeepSpeed/csrc/random_ltd/token_sort.cu)
implements a
cooperative-groups, hand-written two-level warp prefix scan. The constant 512
is wired into `iter_idx = i*512 + tid` (lines 52/64/142/161), the launch
template dispatch (lines 179-190), and — the hard part — the scan itself: the
algorithm scans exactly `4*blockDim.x` histogram slots and warp 0 unconditionally
reads 16 warp subtotals (lines 100-115), so any block smaller than 512 both
misses scan range and reads subtotals that nonexistent warps never wrote.
Runtime-parameterizing it means redesigning items-per-thread, active-warp
count, `VALS_PER_THREAD`, and the shared-memory bounds, then re-validating
`cg::thread_block` semantics — an L-sized rewrite of the algorithm, not a
mechanical parameterization. A narrow-domain toy (capping `original_tokens` to
`4*min_block`) would change the input domain and break comparability with the
original kernel, so it is not used either. The kernel stays a fixed-geometry
negative control.
