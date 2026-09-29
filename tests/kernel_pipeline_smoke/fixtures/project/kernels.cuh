#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cuda_runtime.h>

__global__ void vec_add(int* out, const int* a, const int* b, int n);

#endif // PROJECT_KERNELS_CUH
