#include <stdint.h>

extern "C" __global__ void vconfig_warp_kernel(uint32_t *out, uint32_t seed) {
  uint32_t lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31u;
  uint32_t value = seed + lane;
  uint32_t shifted = __shfl_down_sync(0xffffffffu, value, 1);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (blockDim.x != 16 || blockDim.y != 4 || shifted != seed + 1u) {
      asm("trap;");
    }
    out[0] = shifted;
  }
}
