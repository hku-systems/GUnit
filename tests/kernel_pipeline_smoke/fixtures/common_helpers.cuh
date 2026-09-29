#ifndef COMMON_HELPERS_CUH
#define COMMON_HELPERS_CUH

__device__ inline int clamp_int(int x, int lo, int hi) {
  return (x < lo) ? lo : (x > hi) ? hi : x;
}

__device__ inline float scale_float(float x, float factor) {
  return x * factor;
}

__device__ inline int bitwise_blend_impl(int a, int b, int mask) {
  return (a & mask) | (b & ~mask);
}

__device__ inline int reduce_pair_impl(int a, int b) {
  return (a + b) / 2;
}

__device__ inline float fused_mul_add_impl(float a, float b, float c) {
  return a * b + c;
}

#endif // COMMON_HELPERS_CUH
