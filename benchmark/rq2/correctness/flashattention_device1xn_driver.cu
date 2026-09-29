#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

extern "C" __global__ void rq1_flashattention_device1xn(
    const float* input, float* output, unsigned int n);

namespace {

constexpr int kCapacity = 256;
constexpr float kSentinel = -8192.0f;

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

float input_value(int index) {
  switch (index % 5) {
    case 0:
      return -static_cast<float>(index + 1) * 0.125f;
    case 1:
      return 0.0f;
    default:
      return static_cast<float>(index + 1) * 0.0625f;
  }
}

bool run_case(int block, unsigned int n) {
  std::vector<float> input(kCapacity);
  std::vector<float> initial_output(kCapacity, kSentinel);
  for (int index = 0; index < kCapacity; ++index) {
    input[index] = input_value(index);
  }

  float* device_input = nullptr;
  float* device_output = nullptr;
  bool ok = check_cuda(
                cudaMalloc(&device_input, input.size() * sizeof(float)),
                "cudaMalloc input") &&
            check_cuda(
                cudaMalloc(&device_output, initial_output.size() * sizeof(float)),
                "cudaMalloc output");
  if (ok) {
    ok = check_cuda(
             cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy input H2D") &&
         check_cuda(
             cudaMemcpy(device_output, initial_output.data(),
                        initial_output.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy output H2D");
  }
  if (ok) {
    rq1_flashattention_device1xn<<<1, block>>>(device_input, device_output, n);
    ok = check_cuda(cudaGetLastError(), "flashattention launch") &&
         check_cuda(cudaDeviceSynchronize(), "flashattention synchronize");
  }

  std::vector<float> actual(kCapacity);
  if (ok) {
    ok = check_cuda(
        cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy output D2H");
  }
  if (device_output != nullptr) {
    ok = check_cuda(cudaFree(device_output), "cudaFree output") && ok;
  }
  if (device_input != nullptr) {
    ok = check_cuda(cudaFree(device_input), "cudaFree input") && ok;
  }
  if (!ok) {
    return false;
  }

  for (int tidx = 0; tidx < block; ++tidx) {
    const float expected = tidx < static_cast<int>(n)
                               ? std::max(input[tidx], 0.0f) / 16.0f
                               : 0.0f;
    const float tolerance = 1.0e-6f * std::fmax(1.0f, std::fabs(expected));
    if (std::fabs(actual[tidx] - expected) > tolerance) {
      std::fprintf(
          stderr,
          "flashattention mismatch block=%d n=%u index=%d expected=%g actual=%g\n",
          block, n, tidx, expected, actual[tidx]);
      return false;
    }
  }
  for (int tidx = block; tidx < kCapacity; ++tidx) {
    if (actual[tidx] != kSentinel) {
      std::fprintf(
          stderr,
          "flashattention sentinel corrupted block=%d n=%u index=%d actual=%g\n",
          block, n, tidx, actual[tidx]);
      return false;
    }
  }
  return true;
}

std::set<unsigned int> case_sizes(int block) {
  return {
      1u,
      3u,
      4u,
      5u,
      static_cast<unsigned int>(block / 2 - 1),
      static_cast<unsigned int>(block / 2),
      static_cast<unsigned int>(block / 2 + 1),
      static_cast<unsigned int>(block - 1),
      static_cast<unsigned int>(block),
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "expected at least one block candidate\n");
    return 2;
  }

  int cases = 0;
  bool compatibility_anchor_seen = false;
  for (int arg = 1; arg < argc; ++arg) {
    const int block = std::atoi(argv[arg]);
    if (block <= 0 || block > kCapacity) {
      std::fprintf(stderr, "invalid block candidate %s\n", argv[arg]);
      return 2;
    }
    for (unsigned int n : case_sizes(block)) {
      if (!run_case(block, n)) {
        return 1;
      }
      compatibility_anchor_seen |= block == 128 && n == 64;
      ++cases;
    }
  }

  if (!compatibility_anchor_seen) {
    if (!run_case(128, 64)) {
      return 1;
    }
    ++cases;
  }
  std::printf("flashattention_device1xn correctness passed: %d cases\n", cases);
  return 0;
}
