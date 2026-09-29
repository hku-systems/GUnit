#include <stdint.h>

extern "C" __global__ void rapid_timeout_kernel() {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned long long start = clock64();
    while (clock64() - start < 1000000000000ULL) {
      asm volatile("" ::: "memory");
    }
  }
}
