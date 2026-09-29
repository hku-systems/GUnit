#include "common_helpers.cuh"
#include "fixture_api.cuh"
#include "scoped_header_types.cuh"
#include <cstdio>
#include <cuda_runtime.h>

/*
 * Namespaced multi-level __global__ kernels (non-extern-C).
 * These exercise the discover pipeline's ability to find kernels
 * inside nested C++ namespaces.
 */

namespace ns1 {
namespace ns2 {

struct ns_pair {
  int x;
  float y;
};

__global__ void ns_add(int* out, int a, int b) {
  out[0] = a + b;
}

__global__ void ns_scale(float* out, float value, float factor) {
  out[0] = scale_float(value, factor);
}

__global__ void ns_record_ptr_copy(ns_pair* out, const ns_pair* in) {
  out[0] = in[0];
}

struct scoped_owner {
  struct nested_pair {
    int x;
    float y;
  };
};

__global__ void class_nested_record_ptr_copy(scoped_owner::nested_pair* out,
                                            const scoped_owner::nested_pair* in) {
  out[0] = in[0];
}

}  // namespace ns2
}  // namespace ns1

__global__ void header_nested_params_kernel(header_scope::outer_box::params params, float *out) {
  out[0] = params.data ? params.data[0] + static_cast<float>(params.count) : static_cast<float>(params.count);
}

/*
 * CUDA library / runtime symbol stress: use various CUDA runtime calls
 * and math intrinsics so that AST/IR contains many external symbols.
 * The discover pipeline must exclude all of these from the kernels list.
 */
__global__ void cuda_api_stress(int* out, int n) {
  /* Use a variety of CUDA device-side built-ins / intrinsics */
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    float f = __int2float_rn(tid);
    float s = __sinf(f);
    float c = __cosf(f);
    out[tid] = __float2int_rn(s + c);
  }
}

/* ---- Host-callable test wrappers ---- */

static bool check_cuda(cudaError_t err, const char* ctx) {
  if (err == cudaSuccess) return true;
  printf("  CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
  return false;
}

extern "C" int test_ns_add(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  ns1::ns2::ns_add<<<1,1>>>(d_out, 13, 29);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 42) { printf("  FAIL ns_add: expected 42 got %d\n", h_out); return 1; }
  printf("  PASS ns_add\n");
  return 0;
}

extern "C" int test_ns_scale(void) {
  float *d_out;
  float h_out = 0.0f;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(float)), "malloc")) return 1;
  ns1::ns2::ns_scale<<<1,1>>>(d_out, 4.0f, 2.5f);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  float expected = 10.0f;
  if (h_out < expected - 0.01f || h_out > expected + 0.01f) {
    printf("  FAIL ns_scale: expected %.2f got %.2f\n", expected, h_out);
    return 1;
  }
  printf("  PASS ns_scale\n");
  return 0;
}

extern "C" int test_cuda_api_stress(void) {
  const int N = 16;
  int *d_out;
  int h_out[N];
  if (!check_cuda(cudaMalloc(&d_out, N * sizeof(int)), "malloc")) return 1;
  if (!check_cuda(cudaMemset(d_out, 0, N * sizeof(int)), "memset")) { cudaFree(d_out); return 1; }
  cuda_api_stress<<<1, N>>>(d_out, N);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  /* Just verify the kernel ran without errors; values depend on intrinsics */
  printf("  PASS cuda_api_stress\n");
  return 0;
}
