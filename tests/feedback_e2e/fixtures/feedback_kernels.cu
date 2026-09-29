#include <stddef.h>
#include <stdint.h>

extern "C" __global__ void feedback_kernel(uint8_t *input, uint8_t *output,
                                             size_t size) {
  const size_t index =
      threadIdx.x + blockIdx.x * static_cast<size_t>(blockDim.x);
  if (index >= size)
    return;

  const uint8_t value = input[index];
  if ((value & 1U) != 0)
    output[index] = value ^ 0x5aU;
  else
    output[index] = value + 1U;
}
