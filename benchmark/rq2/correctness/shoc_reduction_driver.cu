#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void rq1_shoc_reduction(const float* g_idata,
                                                float* g_odata,
                                                unsigned int n);

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool run_case(int block) {
  constexpr unsigned int kCount = 512;
  std::vector<float> input(kCount);
  double expected = 0.0;
  for (unsigned int i = 0; i < kCount; ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    expected += static_cast<double>(input[i]);
  }

  float* device_input = nullptr;
  float* device_output = nullptr;
  bool ok = check_cuda(cudaMalloc(&device_input, input.size() * sizeof(float)),
                       "cudaMalloc input") &&
            check_cuda(cudaMalloc(&device_output, sizeof(float)),
                       "cudaMalloc output");
  if (ok) {
    ok = check_cuda(
        cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  }
  if (ok) {
    rq1_shoc_reduction<<<1, block, 1024>>>(device_input, device_output, kCount);
    ok = check_cuda(cudaGetLastError(), "reduction launch") &&
         check_cuda(cudaDeviceSynchronize(), "reduction synchronize");
  }
  float actual = 0.0f;
  if (ok) {
    ok = check_cuda(
        cudaMemcpy(&actual, device_output, sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");
  }
  if (device_output != nullptr) {
    check_cuda(cudaFree(device_output), "cudaFree output");
  }
  if (device_input != nullptr) {
    check_cuda(cudaFree(device_input), "cudaFree input");
  }
  if (!ok) {
    return false;
  }

  const double tolerance = 1.0e-5 * std::fmax(1.0, std::fabs(expected));
  if (std::fabs(static_cast<double>(actual) - expected) > tolerance) {
    std::fprintf(stderr,
                 "reduction mismatch block=%d expected=%g actual=%g\n", block,
                 expected, static_cast<double>(actual));
    return false;
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
    if (!run_case(block)) {
      return 1;
    }
    ++cases;
  }
  std::printf("shoc_reduction correctness passed: %d cases\n", cases);
  return 0;
}
