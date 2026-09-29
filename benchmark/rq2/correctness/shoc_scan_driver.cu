#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void rq1_shoc_scan(float* g_block_sums, int n);

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool run_case(int block, int n) {
  constexpr int kCapacity = 256;
  constexpr float kSentinel = -8192.0f;
  std::vector<float> input(kCapacity, kSentinel);
  std::vector<float> expected(kCapacity, kSentinel);
  float prefix = 0.0f;
  for (int i = 0; i < n; ++i) {
    input[i] = static_cast<float>((i % 11) - 5) * 0.125f;
    expected[i] = prefix;
    prefix += input[i];
  }

  float* device = nullptr;
  if (!check_cuda(cudaMalloc(&device, input.size() * sizeof(float)), "cudaMalloc")) {
    return false;
  }
  bool ok = check_cuda(
      cudaMemcpy(device, input.data(), input.size() * sizeof(float),
                 cudaMemcpyHostToDevice),
      "cudaMemcpy H2D");
  if (ok) {
    rq1_shoc_scan<<<1, block, 2048>>>(device, n);
    ok = check_cuda(cudaGetLastError(), "scan launch") &&
         check_cuda(cudaDeviceSynchronize(), "scan synchronize");
  }
  std::vector<float> actual(kCapacity);
  if (ok) {
    ok = check_cuda(
        cudaMemcpy(actual.data(), device, actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  }
  check_cuda(cudaFree(device), "cudaFree");
  if (!ok) {
    return false;
  }

  for (int i = 0; i < kCapacity; ++i) {
    const float tolerance = 1.0e-5f * std::fmax(1.0f, std::fabs(expected[i]));
    if (std::fabs(actual[i] - expected[i]) > tolerance) {
      std::fprintf(stderr,
                   "scan mismatch block=%d n=%d index=%d expected=%g actual=%g\n",
                   block, n, i, expected[i], actual[i]);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "expected at least one block candidate\n");
    return 2;
  }
  int cases = 0;
  for (int arg = 1; arg < argc; ++arg) {
    const int block = std::atoi(argv[arg]);
    if (block <= 0 || block > 256) {
      std::fprintf(stderr, "invalid block candidate %s\n", argv[arg]);
      return 2;
    }
    const int sizes[] = {1, block / 2, block};
    for (int n : sizes) {
      if (!run_case(block, n)) {
        return 1;
      }
      ++cases;
    }
  }
  std::printf("shoc_scan correctness passed: %d cases\n", cases);
  return 0;
}
