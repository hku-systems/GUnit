#include <stdint.h>

extern "C" __global__ void rapid_oob_crash_kernel(uint8_t *buffer) {
  (void)buffer;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    volatile uint8_t *invalid = reinterpret_cast<volatile uint8_t *>(0x1);
    invalid[0] = 0x5aU;
  }
}
