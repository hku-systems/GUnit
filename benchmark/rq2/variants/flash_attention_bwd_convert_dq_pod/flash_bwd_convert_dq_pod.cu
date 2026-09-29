#include <cuda_fp16.h>

// POD extraction of flash_bwd_convert_dq_kernel for one batch/head tile.
// Split planes are compact and contiguous; each plane contains exactly
// seqlen_q * head_dim fp32 accumulator elements.
extern "C" __global__ void flash_bwd_convert_dq_pod_kernel(
    const float* dq_accum, __half* dq, const unsigned int seqlen_q,
    const unsigned int head_dim, const unsigned int nsplits,
    const float scale) {
  const unsigned long long elements =
      static_cast<unsigned long long>(seqlen_q) * head_dim;
  for (unsigned long long index = threadIdx.x; index < elements;
       index += blockDim.x) {
    float value = 0.0f;
    for (unsigned int split = 0; split < nsplits; ++split) {
      value += dq_accum[static_cast<unsigned long long>(split) * elements +
                        index];
    }
    dq[index] = __float2half_rn(value * scale);
  }
}
