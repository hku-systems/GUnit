#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void index_mul_2d_float_orig(float* out, const float* in1, const float* in2,
                                        const int64_t* idx1, int64_t size, int64_t fea_dim);
__global__ void index_mul_2d_float_vgeo(float* out, const float* in1, const float* in2,
                                        const int64_t* idx1, int64_t size, int64_t fea_dim);

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

struct Case {
  int64_t size;
  int64_t fea_dim;
  std::vector<int64_t> idx1;
};

void run_case(const Case& test_case) {
  const int64_t rows = test_case.size;
  const size_t in1_bytes = static_cast<size_t>(rows * test_case.fea_dim) * sizeof(float);
  const size_t out_bytes = static_cast<size_t>(test_case.size * test_case.fea_dim) * sizeof(float);
  const size_t idx_bytes = static_cast<size_t>(test_case.size) * sizeof(int64_t);

  std::vector<float> in1(static_cast<size_t>(rows * test_case.fea_dim));
  std::vector<float> in2(static_cast<size_t>(test_case.size * test_case.fea_dim));
  std::vector<float> orig(in2.size());
  std::vector<float> vgeo(in2.size());
  for (size_t i = 0; i < in1.size(); ++i) {
    in1[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.25f;
  }
  for (size_t i = 0; i < in2.size(); ++i) {
    in2[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.5f;
  }

  float* device_in1 = nullptr;
  float* device_in2 = nullptr;
  float* device_orig = nullptr;
  float* device_vgeo = nullptr;
  int64_t* device_idx1 = nullptr;
  check_cuda(cudaMalloc(&device_in1, in1_bytes), "cudaMalloc(in1)");
  check_cuda(cudaMalloc(&device_in2, out_bytes), "cudaMalloc(in2)");
  check_cuda(cudaMalloc(&device_orig, out_bytes), "cudaMalloc(orig)");
  check_cuda(cudaMalloc(&device_vgeo, out_bytes), "cudaMalloc(vgeo)");
  check_cuda(cudaMalloc(&device_idx1, idx_bytes), "cudaMalloc(idx1)");

  // cudaMalloc supplies alignment stricter than the vendor float4 path's 16-byte precondition.
  check_cuda(cudaMemcpy(device_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(in1)");
  check_cuda(cudaMemcpy(device_in2, in2.data(), out_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(in2)");
  check_cuda(cudaMemcpy(device_idx1, test_case.idx1.data(), idx_bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(idx1)");

  index_mul_2d_float_orig<<<1, dim3(32, 8)>>>(device_orig, device_in1, device_in2,
                                              device_idx1, test_case.size,
                                              test_case.fea_dim);
  check_cuda(cudaGetLastError(), "launch(orig)");
  check_cuda(cudaDeviceSynchronize(), "synchronize(orig)");
  check_cuda(cudaMemcpy(orig.data(), device_orig, out_bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy(orig)");

  for (unsigned int block_x : {4u, 8u, 16u, 32u}) {
    check_cuda(cudaMemset(device_vgeo, 0xa5, out_bytes), "cudaMemset(vgeo)");
    index_mul_2d_float_vgeo<<<1, dim3(block_x, 8)>>>(
        device_vgeo, device_in1, device_in2, device_idx1, test_case.size,
        test_case.fea_dim);
    check_cuda(cudaGetLastError(), "launch(vgeo)");
    check_cuda(cudaDeviceSynchronize(), "synchronize(vgeo)");
    check_cuda(cudaMemcpy(vgeo.data(), device_vgeo, out_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(vgeo)");
    if (std::memcmp(orig.data(), vgeo.data(), out_bytes) != 0) {
      std::fprintf(stderr, "mismatch: size=%lld fea_dim=%lld block.x=%u\n",
                   static_cast<long long>(test_case.size),
                   static_cast<long long>(test_case.fea_dim), block_x);
      std::exit(1);
    }
  }

  check_cuda(cudaFree(device_idx1), "cudaFree(idx1)");
  check_cuda(cudaFree(device_vgeo), "cudaFree(vgeo)");
  check_cuda(cudaFree(device_orig), "cudaFree(orig)");
  check_cuda(cudaFree(device_in2), "cudaFree(in2)");
  check_cuda(cudaFree(device_in1), "cudaFree(in1)");
}

}  // namespace

int main() {
  for (const Case& test_case : {
           Case{1, 1, {0}},
           Case{5, 7, {3, 1, 3, 0, 1}},
           Case{7, 33, {6, 2, 2, 4, 0, 6, 1}},
           Case{8, 63, {7, 0, 3, 3, 5, 1, 7, 2}},
       }) {
    run_case(test_case);
  }
  std::puts("apex index_mul generic forward: bitwise equal");
  return 0;
}
