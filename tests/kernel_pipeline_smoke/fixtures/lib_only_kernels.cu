#include <cuda_runtime.h>

#include "common_helpers.cuh"

__device__ inline unsigned int clamp_u32_local(unsigned int x, unsigned int lo, unsigned int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

extern "C" __global__ void lib_only_axpy(
    float* out,
    const float* x,
    const float* y,
    float alpha,
    int n
) {
    int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
          + static_cast<int>(threadIdx.x);
    if (i < n) {
        out[i] = alpha * x[i] + y[i];
    }
}

extern "C" __global__ void lib_only_clamp_u32(
    unsigned int* data,
    unsigned int lo,
    unsigned int hi,
    int n
) {
    int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
          + static_cast<int>(threadIdx.x);
    if (i < n) {
        unsigned int v = data[i];
        data[i] = clamp_u32_local(v, lo, hi);
    }
}
