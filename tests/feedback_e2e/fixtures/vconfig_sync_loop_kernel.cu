#include <stdint.h>

extern "C" __global__ void vconfig_sync_loop_kernel(uint32_t *out,
                                                      uint32_t seed) {
  __shared__ uint32_t scratch[64];

  scratch[threadIdx.x] = seed + threadIdx.x;

  for (uint32_t i = 0; i < 4; ++i) {
    __syncthreads();
    if (threadIdx.x == 0) {
      if (blockDim.x == 32) {
        scratch[0] += 3u + i;
      } else if (blockDim.x == 64) {
        scratch[0] += 7u + i;
      } else {
        scratch[0] += 1u + i;
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    if (blockDim.x == 32) {
      out[0] = scratch[0] + 32u;
    } else if (blockDim.x == 64) {
      out[0] = scratch[0] + 64u;
    } else {
      out[0] = scratch[0] + 1u;
    }
  }
}
