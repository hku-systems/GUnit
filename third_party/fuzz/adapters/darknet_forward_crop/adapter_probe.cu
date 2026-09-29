#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "adapter.cu"

namespace {

struct Case {
  uint32_t batch;
  uint32_t channels;
  uint32_t height;
  uint32_t width;
  uint32_t crop_height;
  uint32_t crop_width;
  int32_t train;
  int32_t flip;
  float angle;
  float r4;
  float r5;
  float r6;
  float r7;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

uint32_t output_count(const Case& test_case) {
  return test_case.batch * test_case.channels * test_case.crop_height *
         test_case.crop_width;
}

void launch_orig(const Case& test_case, float* input, float* output) {
  const uint32_t count = output_count(test_case);
  const uint32_t blocks = (count + 511u) / 512u;
  darknet_forward_crop_orig512<<<blocks, 512>>>(
      input, output, test_case.batch, test_case.channels, test_case.height,
      test_case.width, test_case.crop_height, test_case.crop_width,
      test_case.train, test_case.flip, test_case.angle, test_case.r4,
      test_case.r5, test_case.r6, test_case.r7);
  check_cuda(cudaGetLastError(), "launch darknet_forward_crop_orig512");
  check_cuda(cudaDeviceSynchronize(),
             "synchronize darknet_forward_crop_orig512");
}

void launch_vgeo(const Case& test_case, uint32_t threads, float* input,
                  float* output) {
  darknet_forward_crop_vgeo<<<1, threads>>>(
      input, output, test_case.batch, test_case.channels, test_case.height,
      test_case.width, test_case.crop_height, test_case.crop_width,
      test_case.train, test_case.flip, test_case.angle, test_case.r4,
      test_case.r5, test_case.r6, test_case.r7);
  check_cuda(cudaGetLastError(), "launch darknet_forward_crop_vgeo");
  check_cuda(cudaDeviceSynchronize(), "synchronize darknet_forward_crop_vgeo");
}

void run_case(const Case& test_case) {
  const size_t input_count = static_cast<size_t>(test_case.batch) *
                             test_case.channels * test_case.height *
                             test_case.width;
  const size_t out_count = output_count(test_case);
  const size_t input_bytes = input_count * sizeof(float);
  const size_t output_bytes = out_count * sizeof(float);
  std::vector<float> input(input_count);
  std::vector<float> orig(out_count);
  std::vector<float> vgeo(out_count);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int32_t>(i % 37) - 18) / 16.0f;
  }

  float* device_input = nullptr;
  float* device_orig = nullptr;
  float* device_vgeo = nullptr;
  check_cuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc(input)");
  check_cuda(cudaMalloc(&device_orig, output_bytes), "cudaMalloc(orig)");
  check_cuda(cudaMalloc(&device_vgeo, output_bytes), "cudaMalloc(vgeo)");
  check_cuda(cudaMemcpy(device_input, input.data(), input_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(input)");

  launch_orig(test_case, device_input, device_orig);
  check_cuda(cudaMemcpy(orig.data(), device_orig, output_bytes,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(orig)");

  for (uint32_t threads : {32u, 64u, 128u, 256u, 512u}) {
    check_cuda(cudaMemset(device_vgeo, 0xa5, output_bytes),
               "cudaMemset(vgeo)");
    launch_vgeo(test_case, threads, device_input, device_vgeo);
    check_cuda(cudaMemcpy(vgeo.data(), device_vgeo, output_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(vgeo)");
    if (std::memcmp(orig.data(), vgeo.data(), output_bytes) != 0) {
      std::fprintf(
          stderr,
          "Darknet crop mismatch: count=%zu block=%u train=%d flip=%d "
          "angle=%g offsets=(%g,%g,%g,%g)\n",
          out_count, threads, test_case.train, test_case.flip, test_case.angle,
          test_case.r4, test_case.r5, test_case.r6, test_case.r7);
      std::exit(1);
    }
  }

  check_cuda(cudaFree(device_vgeo), "cudaFree(vgeo)");
  check_cuda(cudaFree(device_orig), "cudaFree(orig)");
  check_cuda(cudaFree(device_input), "cudaFree(input)");
}

std::vector<Case> fixed_cases() {
  return {
      {1, 1, 1, 1, 1, 1, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      {1, 3, 7, 9, 7, 9, 1, 0, 0.0f, 0.0f, 0.0f, 0.49f, 1.0f},
      {2, 3, 9, 11, 5, 7, 1, 1, 0.785398163f, 1.0f, 1.0f, 0.51f, 0.0f},
      {1, 3, 8, 10, 4, 6, 1, 1, 3.141592654f, 1.0f, 1.0f, 0.5f, 0.0f},
      {2, 4, 16, 16, 8, 8, 0, 1, -1.570796327f, 1.0f, 0.0f, 1.0f, 1.0f},
      {4, 4, 32, 32, 32, 32, 1, 1, 3.141592654f, 0.0f, 1.0f, 1.0f, 1.0f},
  };
}

}  // namespace

int main() {
  for (const Case& test_case : fixed_cases()) {
    run_case(test_case);
  }
  std::puts("Darknet forward crop vgeo/orig512 outputs are bitwise equal");
  return 0;
}
