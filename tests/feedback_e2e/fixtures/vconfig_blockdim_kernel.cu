#include <stdint.h>

// Keep each launch-shape branch in its own basic block so edge coverage can
// distinguish them even when backends are built with -O3 (which would
// otherwise flatten the if/else chain into branchless selects).
__device__ __noinline__ static uint32_t blockdim_branch_32(uint32_t seed) {
  return seed + 32u;
}

__device__ __noinline__ static uint32_t blockdim_branch_64(uint32_t seed) {
  return seed + 64u;
}

__device__ __noinline__ static uint32_t blockdim_branch_other(uint32_t seed) {
  return seed + 1u;
}

extern "C" __global__ void vconfig_blockdim_kernel(uint32_t *out,
                                                     uint32_t seed) {
  if (threadIdx.x != 0) {
    return;
  }

  if (blockDim.x == 32) {
    out[0] = blockdim_branch_32(seed);
  } else if (blockDim.x == 64) {
    out[0] = blockdim_branch_64(seed);
  } else {
    out[0] = blockdim_branch_other(seed);
  }
}
