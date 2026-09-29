#include <stddef.h>
#include <stdint.h>

extern "C" __global__ void canary_corrupt_kernel(size_t payload_size,
                                                    int nbytes,
                                                    uint8_t *payload) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < nbytes; ++i) {
    payload[payload_size + static_cast<size_t>(i)] = 0xff;
  }
}
