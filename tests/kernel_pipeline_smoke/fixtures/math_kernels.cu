#include "common_helpers.cuh"
#include "fixture_api.cuh"
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void add_kernel(int* out, int a, int b) {
  out[0] = a + b;
}

extern "C" __global__ void mul_kernel(int* out, int a, int b) {
  out[0] = a * b;
}

__global__ void clamped_add(int* out, int a, int b, int lo, int hi) {
  out[0] = clamp_int(a + b, lo, hi);
}

__global__ void fused_mul_add(float* out, float a, float b, float c) {
  out[0] = fused_mul_add_impl(a, b, c);
}

/* ---- Host-callable test wrappers ---- */

static bool check_cuda(cudaError_t err, const char* ctx) {
  if (err == cudaSuccess) return true;
  printf("  CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
  return false;
}

extern "C" int test_add_kernel(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  add_kernel<<<1,1>>>(d_out, 3, 4);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 7) { printf("  FAIL add_kernel: expected 7 got %d\n", h_out); return 1; }
  printf("  PASS add_kernel\n");
  return 0;
}

extern "C" int test_mul_kernel(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  mul_kernel<<<1,1>>>(d_out, 5, 6);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 30) { printf("  FAIL mul_kernel: expected 30 got %d\n", h_out); return 1; }
  printf("  PASS mul_kernel\n");
  return 0;
}

extern "C" int test_clamped_add(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  clamped_add<<<1,1>>>(d_out, 30, 40, 0, 50);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 50) { printf("  FAIL clamped_add: expected 50 got %d\n", h_out); return 1; }
  printf("  PASS clamped_add\n");
  return 0;
}

extern "C" int test_fused_mul_add(void) {
  float *d_out;
  float h_out = 0.0f;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(float)), "malloc")) return 1;
  /* 3.0 * 4.0 + 5.0 = 17.0 */
  fused_mul_add<<<1,1>>>(d_out, 3.0f, 4.0f, 5.0f);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  float expected = 17.0f;
  if (h_out < expected - 0.01f || h_out > expected + 0.01f) {
    printf("  FAIL fused_mul_add: expected %.2f got %.2f\n", expected, h_out);
    return 1;
  }
  printf("  PASS fused_mul_add\n");
  return 0;
}
