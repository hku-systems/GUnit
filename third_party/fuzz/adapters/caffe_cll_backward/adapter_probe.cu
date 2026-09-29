#include <cuda_runtime.h>

#include <cmath>
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
  float margin;
  int32_t legacy_version;
  float alpha;
  std::vector<float> y;
  std::vector<float> diff;
  std::vector<float> dist_sq;
};

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

float reference_value(const Case& test_case, uint32_t i) {
  const uint32_t n = i / test_case.channels;
  if (static_cast<int32_t>(test_case.y[n]) != 0) {
    return test_case.alpha * test_case.diff[i];
  }

  float mdist;
  float beta;
  if (test_case.legacy_version != 0) {
    mdist = test_case.margin - test_case.dist_sq[n];
    beta = -test_case.alpha;
  } else {
    const float dist = std::sqrt(test_case.dist_sq[n]);
    mdist = test_case.margin - dist;
    beta = -test_case.alpha * mdist / (dist + 1.0e-4f) * test_case.diff[i];
  }
  return mdist > 0.0f ? beta : 0.0f;
}

void check_value(
    float actual, float expected, uint32_t i, int32_t legacy_version,
    uint32_t threads) {
  const float tolerance = 1.0e-6f * std::fmax(1.0f, std::fabs(expected));
  if (std::fabs(actual - expected) > tolerance) {
    std::fprintf(
        stderr,
        "CLLBackward mismatch: i=%u legacy=%d threads=%u actual=%g "
        "expected=%g\n",
        i, legacy_version, threads, actual, expected);
    std::exit(1);
  }
}

void run_host_case(const Case& test_case) {
  const uint32_t count = test_case.batch * test_case.channels;
  for (uint32_t i = 0; i < count; ++i) {
    const float actual = caffe_cll_detail::backward_value(
        i, test_case.channels, test_case.margin, test_case.legacy_version,
        test_case.alpha, test_case.y.data(), test_case.diff.data(),
        test_case.dist_sq.data());
    check_value(actual, reference_value(test_case, i), i,
                test_case.legacy_version, 0);
  }
}

void run_gpu_case(const Case& test_case, uint32_t threads) {
  const uint32_t count = test_case.batch * test_case.channels;
  const size_t batch_bytes = test_case.batch * sizeof(float);
  const size_t count_bytes = count * sizeof(float);
  float* device_y = nullptr;
  float* device_diff = nullptr;
  float* device_dist_sq = nullptr;
  float* device_bottom_diff = nullptr;

  check_cuda(cudaMalloc(&device_y, batch_bytes), "cudaMalloc(y)");
  check_cuda(cudaMalloc(&device_diff, count_bytes), "cudaMalloc(diff)");
  check_cuda(cudaMalloc(&device_dist_sq, batch_bytes), "cudaMalloc(dist_sq)");
  check_cuda(
      cudaMalloc(&device_bottom_diff, count_bytes), "cudaMalloc(bottom_diff)");
  check_cuda(cudaMemcpy(device_y, test_case.y.data(), batch_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(y)");
  check_cuda(cudaMemcpy(device_diff, test_case.diff.data(), count_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(diff)");
  check_cuda(cudaMemcpy(device_dist_sq, test_case.dist_sq.data(), batch_bytes,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(dist_sq)");

  caffe_cll_backward_f32<<<1, threads>>>(
      test_case.batch, test_case.channels, test_case.margin,
      test_case.legacy_version, test_case.alpha, device_y, device_diff,
      device_dist_sq, device_bottom_diff);
  check_cuda(cudaGetLastError(), "launch caffe_cll_backward_f32");
  check_cuda(cudaDeviceSynchronize(), "synchronize caffe_cll_backward_f32");

  std::vector<float> actual(count);
  check_cuda(cudaMemcpy(actual.data(), device_bottom_diff, count_bytes,
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(bottom_diff)");
  for (uint32_t i = 0; i < count; ++i) {
    check_value(actual[i], reference_value(test_case, i), i,
                test_case.legacy_version, threads);
  }

  check_cuda(cudaFree(device_y), "cudaFree(y)");
  check_cuda(cudaFree(device_diff), "cudaFree(diff)");
  check_cuda(cudaFree(device_dist_sq), "cudaFree(dist_sq)");
  check_cuda(cudaFree(device_bottom_diff), "cudaFree(bottom_diff)");
}

void run_invalid_shape_gpu_case(uint32_t threads) {
  constexpr float kSentinel = 123.0f;
  float* device_value = nullptr;
  check_cuda(cudaMalloc(&device_value, sizeof(float)), "cudaMalloc(sentinel)");
  check_cuda(cudaMemcpy(device_value, &kSentinel, sizeof(float),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(sentinel to device)");
  caffe_cll_backward_f32<<<1, threads>>>(
      32, 256, 1.0f, 0, 1.0f, device_value, device_value, device_value,
      device_value);
  check_cuda(cudaGetLastError(), "launch invalid caffe_cll_backward_f32");
  check_cuda(cudaDeviceSynchronize(),
             "synchronize invalid caffe_cll_backward_f32");
  float actual = 0.0f;
  check_cuda(cudaMemcpy(&actual, device_value, sizeof(float),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(sentinel from device)");
  if (actual != kSentinel) {
    std::fputs("CLLBackward invalid shape was not rejected\n", stderr);
    std::exit(1);
  }
  check_cuda(cudaFree(device_value), "cudaFree(sentinel)");
}

std::vector<Case> fixed_cases() {
  const std::vector<float> y = {0.0f, 1.0f, 0.0f, 0.0f};
  const std::vector<float> diff = {
      -1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 0.5f,
      3.0f, -3.0f, 0.25f, -0.5f, 1.5f, -1.5f};
  const std::vector<float> dist_sq = {
      0.0f, 0.25f, 0.99999988f, 1.00000012f};
  std::vector<Case> cases = {
      {4, 3, 1.0f, 0, 1.0f, y, diff, dist_sq},
      {4, 3, 1.0f, 1, -1.0f, y, diff, dist_sq},
      {1, 1, 0.0f, 0, 0.0f, {0.0f}, {1.0f}, {1.0e-30f}},
  };
  Case grid_stride = {
      8, 256, 1.0f, 0, 1.0f, std::vector<float>(8),
      std::vector<float>(2048), std::vector<float>(8)};
  for (uint32_t n = 0; n < grid_stride.batch; ++n) {
    grid_stride.y[n] = static_cast<float>(n % 2);
    grid_stride.dist_sq[n] = dist_sq[n % dist_sq.size()];
  }
  for (size_t i = 0; i < grid_stride.diff.size(); ++i) {
    grid_stride.diff[i] = static_cast<float>(static_cast<int32_t>(i % 17) - 8) /
                          8.0f;
  }
  cases.push_back(grid_stride);
  return cases;
}

}  // namespace

int main(int argc, char** argv) {
  const bool host_only = argc == 2 && std::strcmp(argv[1], "--host-only") == 0;
  if (argc > 1 && !host_only) {
    std::fprintf(stderr, "usage: %s [--host-only]\n", argv[0]);
    return 2;
  }

  if (!caffe_cll_detail::valid_shape(32, 64) ||
      !caffe_cll_detail::valid_shape(8, 256) ||
      caffe_cll_detail::valid_shape(32, 256)) {
    std::fputs("CLLBackward shape guard mismatch\n", stderr);
    return 1;
  }

  const std::vector<Case> cases = fixed_cases();
  for (const Case& test_case : cases) {
    run_host_case(test_case);
  }
  if (host_only) {
    std::puts("Caffe CLLBackward<float> adapter host probe passed");
    return 0;
  }

  for (uint32_t threads : {32u, 64u, 128u, 256u, 512u}) {
    for (const Case& test_case : cases) {
      run_gpu_case(test_case, threads);
    }
    run_invalid_shape_gpu_case(threads);
  }
  std::puts("Caffe CLLBackward<float> adapter GPU probe passed");
  return 0;
}
