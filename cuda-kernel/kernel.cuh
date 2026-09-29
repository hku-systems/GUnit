// kernel.cuh
#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__
#include <cuda_runtime.h>

#include <cstdint>

extern "C" __device__ void vulnerable_kernel(volatile uint8_t *input,
                                              volatile uint8_t *output,
                                              size_t size);

#endif // __KERNEL_CUH__
