#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

extern "C" __global__ void rq1_shoc_radix_sort(
    unsigned int, unsigned int, uint4*, uint4*, uint4*, uint4*);
extern "C" __global__ void rq1_shoc_radix_sort_scanfix(
    unsigned int, unsigned int, uint4*, uint4*, uint4*, uint4*);

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kNbits = 4;
constexpr unsigned int kStartbit = 0;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

unsigned int& component(uint4& value, unsigned int lane) {
  switch (lane) {
    case 0:
      return value.x;
    case 1:
      return value.y;
    case 2:
      return value.z;
    default:
      return value.w;
  }
}

unsigned int component(const uint4& value, unsigned int lane) {
  switch (lane) {
    case 0:
      return value.x;
    case 1:
      return value.y;
    case 2:
      return value.z;
    default:
      return value.w;
  }
}

std::pair<std::vector<uint4>, std::vector<uint4>> reference_sort(
    const std::vector<uint4>& keys,
    const std::vector<uint4>& values,
    unsigned int block) {
  std::vector<uint4> expected_keys = keys;
  std::vector<uint4> expected_values = values;
  const unsigned int scalars_per_block = 4 * block;
  const unsigned int mask = (1u << kNbits) - 1;

  for (unsigned int base = 0; base < 4 * kThreads;
       base += scalars_per_block) {
    std::vector<std::pair<unsigned int, unsigned int>> items;
    items.reserve(scalars_per_block);
    for (unsigned int offset = 0; offset < scalars_per_block; ++offset) {
      const unsigned int scalar = base + offset;
      items.emplace_back(component(keys[scalar / 4], scalar % 4),
                         component(values[scalar / 4], scalar % 4));
    }
    std::stable_sort(items.begin(), items.end(), [mask](const auto& lhs,
                                                        const auto& rhs) {
      return ((lhs.first >> kStartbit) & mask) <
             ((rhs.first >> kStartbit) & mask);
    });
    for (unsigned int offset = 0; offset < scalars_per_block; ++offset) {
      const unsigned int scalar = base + offset;
      component(expected_keys[scalar / 4], scalar % 4) = items[offset].first;
      component(expected_values[scalar / 4], scalar % 4) =
          items[offset].second;
    }
  }
  return {expected_keys, expected_values};
}

void compare_output(const std::vector<uint4>& reference,
                    const std::vector<uint4>& candidate,
                    const char* name,
                    unsigned int block) {
  if (std::memcmp(reference.data(), candidate.data(),
                  reference.size() * sizeof(uint4)) == 0) {
    return;
  }
  std::fprintf(stderr, "%s mismatch at block.x=%u\n", name, block);
  std::exit(1);
}

void initialize_inputs(std::vector<uint4>& keys, std::vector<uint4>& values) {
  for (unsigned int scalar = 0; scalar < 4 * kThreads; ++scalar) {
    const unsigned int item = scalar / 4;
    const unsigned int lane = scalar % 4;
    // Exercise every low-nibble bucket in a deliberately non-sorted order.
    const unsigned int low_nibble = (scalar * 13u + scalar / 7u + 5u) & 0xfu;
    component(keys[item], lane) =
        ((scalar * 2654435761u) & 0xfffffff0u) | low_nibble;
    component(values[item], lane) = 0x9e370000u + scalar;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool reproduce_original =
      argc == 2 &&
      std::string_view(argv[1]) == "--reproduce-original-block32";
  if (argc > 2 || (argc == 2 && !reproduce_original)) {
    std::fprintf(stderr, "usage: %s [--reproduce-original-block32]\n", argv[0]);
    return 2;
  }

  std::vector<uint4> keys(kThreads);
  std::vector<uint4> values(kThreads);
  initialize_inputs(keys, values);

  const size_t bytes = keys.size() * sizeof(uint4);
  uint4* device_keys_in = nullptr;
  uint4* device_values_in = nullptr;
  uint4* device_keys_reference = nullptr;
  uint4* device_values_reference = nullptr;
  uint4* device_keys_candidate = nullptr;
  uint4* device_values_candidate = nullptr;
  check_cuda(cudaMalloc(&device_keys_in, bytes), "cudaMalloc keys input");
  check_cuda(cudaMalloc(&device_values_in, bytes), "cudaMalloc values input");
  check_cuda(cudaMalloc(&device_keys_reference, bytes), "cudaMalloc reference keys");
  check_cuda(cudaMalloc(&device_values_reference, bytes), "cudaMalloc reference values");
  check_cuda(cudaMalloc(&device_keys_candidate, bytes), "cudaMalloc candidate keys");
  check_cuda(cudaMalloc(&device_values_candidate, bytes), "cudaMalloc candidate values");
  check_cuda(cudaMemcpy(device_keys_in, keys.data(), bytes, cudaMemcpyHostToDevice),
             "copy keys input");
  check_cuda(cudaMemcpy(device_values_in, values.data(), bytes,
                        cudaMemcpyHostToDevice),
             "copy values input");

  if (reproduce_original) {
    rq1_shoc_radix_sort<<<kThreads / 32, 32>>>(
        kNbits, kStartbit, device_keys_reference, device_values_reference,
        device_keys_in, device_values_in);
    check_cuda(cudaGetLastError(), "launch original block-32 reproducer");
    const cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaErrorIllegalAddress) {
      std::puts("reproduced original block-32 shared-memory illegal address");
      return 0;
    }
    if (status == cudaSuccess) {
      std::fputs("original block-32 launch unexpectedly completed\n", stderr);
      return 1;
    }
    std::fprintf(stderr, "original block-32 launch returned %s, expected %s\n",
                 cudaGetErrorString(status),
                 cudaGetErrorString(cudaErrorIllegalAddress));
    return 1;
  }

  rq1_shoc_radix_sort<<<2, 128>>>(kNbits, kStartbit, device_keys_reference,
                                  device_values_reference, device_keys_in,
                                  device_values_in);
  check_cuda(cudaGetLastError(), "launch block-128 reference");
  check_cuda(cudaDeviceSynchronize(), "synchronize block-128 reference");

  std::vector<uint4> observed_keys(kThreads);
  std::vector<uint4> observed_values(kThreads);
  std::vector<uint4> candidate_keys(kThreads);
  std::vector<uint4> candidate_values(kThreads);
  check_cuda(cudaMemcpy(observed_keys.data(), device_keys_reference, bytes,
                        cudaMemcpyDeviceToHost),
             "copy reference keys");
  check_cuda(cudaMemcpy(observed_values.data(), device_values_reference, bytes,
                        cudaMemcpyDeviceToHost),
             "copy reference values");

  const auto [reference_keys_128, reference_values_128] =
      reference_sort(keys, values, 128);
  compare_output(reference_keys_128, observed_keys, "original keys", 128);
  compare_output(reference_values_128, observed_values, "original values", 128);

  for (unsigned int block : {32u, 64u, 128u, 256u}) {
    check_cuda(cudaMemset(device_keys_candidate, 0xa5, bytes),
               "initialize candidate keys");
    check_cuda(cudaMemset(device_values_candidate, 0x5a, bytes),
               "initialize candidate values");
    rq1_shoc_radix_sort_scanfix<<<kThreads / block, block>>>(
        kNbits, kStartbit, device_keys_candidate, device_values_candidate,
        device_keys_in, device_values_in);
    check_cuda(cudaGetLastError(), "launch fix candidate");
    check_cuda(cudaDeviceSynchronize(), "synchronize fix candidate");
    check_cuda(cudaMemcpy(candidate_keys.data(), device_keys_candidate, bytes,
                          cudaMemcpyDeviceToHost),
               "copy candidate keys");
    check_cuda(cudaMemcpy(candidate_values.data(), device_values_candidate,
                          bytes, cudaMemcpyDeviceToHost),
               "copy candidate values");
    const auto [reference_keys, reference_values] =
        reference_sort(keys, values, block);
    compare_output(reference_keys, candidate_keys, "fixed keys", block);
    compare_output(reference_values, candidate_values, "fixed values", block);
  }

  check_cuda(cudaFree(device_values_candidate), "cudaFree candidate values");
  check_cuda(cudaFree(device_keys_candidate), "cudaFree candidate keys");
  check_cuda(cudaFree(device_values_reference), "cudaFree reference values");
  check_cuda(cudaFree(device_keys_reference), "cudaFree reference keys");
  check_cuda(cudaFree(device_values_in), "cudaFree values input");
  check_cuda(cudaFree(device_keys_in), "cudaFree keys input");
  std::puts("shoc radix-sort fix probe: all outputs match CPU references");
  return 0;
}
