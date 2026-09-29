#include <cuda_fp16.h>

// POD extraction of flash_bwd_dot_do_o_kernel's non-deterministic path for
// one batch/head tile. The upstream path computes sum(dO * O) * p_dropout
// for each query row and clears the corresponding fp32 dQ accumulator tile.
extern "C" __global__ void flash_bwd_dot_do_o_pod_kernel(
    const __half* dout, const __half* out, float* softmax_d, float* dq_accum,
    const unsigned int seqlen_q, const unsigned int head_dim,
    const float scale) {
  for (unsigned int row = threadIdx.x; row < seqlen_q; row += blockDim.x) {
    const unsigned long long row_offset =
        static_cast<unsigned long long>(row) * head_dim;
    float dot = 0.0f;
    for (unsigned int col = 0; col < head_dim; ++col) {
      const unsigned long long index = row_offset + col;
      dot += __half2float(dout[index]) * __half2float(out[index]);
    }
    softmax_d[row] = dot * scale;
    for (unsigned int col = 0; col < head_dim; ++col) {
      dq_accum[row_offset + col] = 0.0f;
    }
  }
}
