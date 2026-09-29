#include "common_helpers.cuh"
#include "fixture_api.cuh"
#include <cstdio>
#include <cuda_runtime.h>

#ifdef HAVE_CUBLAS
#include <cublas_v2.h>
#endif

/*
 * Project-owned __global__ kernels used alongside cuBLAS host API.
 * The discover pipeline must find these but NOT any cuBLAS internal
 * or library symbols (e.g. cublasSaxpy, cublasCreate_v2, etc.).
 */

__global__ void vec_init(float* out, float value, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    out[tid] = value;
  }
}

__global__ void vec_scale_inplace(float* data, float factor, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    data[tid] = scale_float(data[tid], factor);
  }
}

/* ---- Host-callable test wrappers ---- */

static bool check_cuda(cudaError_t err, const char* ctx) {
  if (err == cudaSuccess) return true;
  printf("  CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
  return false;
}

extern "C" int test_cublas_host_api_smoke(void) {
  const int N = 8;
  float *d_x;
  float h_x[N];

  /* Part (a): launch project kernel and validate */
  if (!check_cuda(cudaMalloc(&d_x, N * sizeof(float)), "malloc")) return 1;

  vec_init<<<1, N>>>(d_x, 3.0f, N);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_x); return 1; }

  vec_scale_inplace<<<1, N>>>(d_x, 2.0f, N);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_x); return 1; }

  cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    float expected = 6.0f; /* 3.0 * 2.0 */
    if (h_x[i] < expected - 0.01f || h_x[i] > expected + 0.01f) {
      printf("  FAIL cublas_host_api_smoke: vec_scale h_x[%d]=%.2f expected %.2f\n",
             i, h_x[i], expected);
      cudaFree(d_x);
      return 1;
    }
  }
  printf("  PASS cublas_host_api_smoke (project kernels)\n");

  /* Part (b): exercise cuBLAS host API when available */
#ifdef HAVE_CUBLAS
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("  SKIP cublas_host_api_smoke (cublasCreate failed: %d)\n", (int)stat);
    cudaFree(d_x);
    return 0;
  }

  /* Re-init d_x to 1.0 for saxpy: y = alpha*x + y */
  vec_init<<<1, N>>>(d_x, 1.0f, N);
  cudaDeviceSynchronize();

  float *d_y;
  cudaMalloc(&d_y, N * sizeof(float));
  vec_init<<<1, N>>>(d_y, 10.0f, N);
  cudaDeviceSynchronize();

  float alpha = 5.0f;
  stat = cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("  FAIL cublas_host_api_smoke (cublasSaxpy failed: %d)\n", (int)stat);
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);
    return 1;
  }
  cudaDeviceSynchronize();

  float h_y[N];
  cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    float expected = 15.0f; /* 5.0 * 1.0 + 10.0 */
    if (h_y[i] < expected - 0.01f || h_y[i] > expected + 0.01f) {
      printf("  FAIL cublas_host_api_smoke: saxpy h_y[%d]=%.2f expected %.2f\n",
             i, h_y[i], expected);
      cublasDestroy(handle);
      cudaFree(d_x);
      cudaFree(d_y);
      return 1;
    }
  }

  printf("  PASS cublas_host_api_smoke (cuBLAS saxpy)\n");
  cublasDestroy(handle);
  cudaFree(d_y);
#else
  printf("  SKIP cublas_host_api_smoke (cuBLAS not available at build time)\n");
#endif

  cudaFree(d_x);
  return 0;
}
