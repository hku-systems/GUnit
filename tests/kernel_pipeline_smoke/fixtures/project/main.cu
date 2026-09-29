#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "kernels.cu"

static bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  printf("CUDA error at %s: %s\n", what, cudaGetErrorString(status));
  return false;
}

int main() {
  const int N = 16;
  int h_a[N], h_b[N], h_out[N];
  for (int i = 0; i < N; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  int *d_a, *d_b, *d_out;
  if (!check_cuda(cudaMalloc(&d_a, N * sizeof(int)), "cudaMalloc(d_a)")) return 1;
  if (!check_cuda(cudaMalloc(&d_b, N * sizeof(int)), "cudaMalloc(d_b)")) return 1;
  if (!check_cuda(cudaMalloc(&d_out, N * sizeof(int)), "cudaMalloc(d_out)")) return 1;

  if (!check_cuda(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_a)")) return 1;
  if (!check_cuda(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy(d_b)")) return 1;

  vec_add<<<1, N>>>(d_out, d_a, d_b, N);
  if (!check_cuda(cudaGetLastError(), "kernel launch")) return 1;
  if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return 1;

  if (!check_cuda(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(h_out)")) return 1;

  int ok = 1;
  for (int i = 0; i < N; i++) {
    if (h_out[i] != h_a[i] + h_b[i]) {
      printf("FAIL at %d: expected %d got %d\n", i, h_a[i] + h_b[i], h_out[i]);
      ok = 0;
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  if (ok) printf("PASS\n");
  return ok ? 0 : 1;
}
