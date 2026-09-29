#include "utils.cuh"
#include "kernels.cuh"

__global__ void vec_add(int* out, const int* a, const int* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = safe_add(a[i], b[i]);
  }
}

__global__ void vec_lerp(float* out, const float* a, const float* b, float t, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = lerp(a[i], b[i], t);
  }
}
