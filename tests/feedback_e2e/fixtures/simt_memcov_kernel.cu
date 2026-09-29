#define RAPID_PAYLOAD_SLOT_COUNT 1u
#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT 1u
#define RAPID_DEFINE_DEVICE_COVERAGE_MAP 1

__device__ unsigned int rapid_test_atomic_calls = 0u;

__device__ inline unsigned int rapid_test_atomic_or(unsigned int *address,
                                                     unsigned int value) {
  atomicAdd(&rapid_test_atomic_calls, 1u);
  return atomicOr(address, value);
}

#define atomicOr rapid_test_atomic_or
#include "feedback/feedback.cuh"
#undef atomicOr

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" uint8_t libafl_cov_map[MAP_SIZE] = {0};
extern "C" uint8_t libafl_simt_memcov_bits[SIMT_MEMCOV_STORAGE_SIZE] = {0};

namespace {

constexpr uint32_t kSiteId = 0x12345678u;
constexpr size_t kPayloadBytes = 4096u;

enum Case : int {
  kSingle = 0,
  kFullBroadcast = 1,
  kFullContiguous = 2,
  kFullOther = 3,
  kPartialBroadcast = 4,
  kPartialContiguous = 5,
  kPartialOther = 6,
  kCrossSector = 7,
  kInvalidLanes = 8,
  kDeduplicate = 9,
  kVConfig32 = 10,
  kVConfig64 = 11,
  kOverlappingOther = 12,
  kDistinctSectorOther = 13,
};

void check(cudaError_t error, const char *operation) {
  if (error == cudaSuccess)
    return;
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  std::exit(1);
}

__global__ void exercise_case(RapidKernelContext *context, uint8_t *payload,
                              int selected_case) {
  const uint32_t physical_thread =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const uint32_t lane = physical_thread & 31u;
  const uint32_t logical_threads = context->vconfig.block_x *
                                   context->vconfig.block_y *
                                   context->vconfig.block_z;
  if (physical_thread >= logical_threads)
    return;

  bool participate = true;
  uint64_t offset = 0u;
  uint8_t width = 4u;
  switch (selected_case) {
  case kSingle:
    participate = lane == 0u;
    break;
  case kFullBroadcast:
    break;
  case kFullContiguous:
    offset = static_cast<uint64_t>(physical_thread) * 4u;
    break;
  case kFullOther:
    offset = static_cast<uint64_t>(lane) * 8u;
    break;
  case kPartialBroadcast:
    participate = lane < 16u;
    break;
  case kPartialContiguous:
    participate = lane < 16u;
    offset = static_cast<uint64_t>(lane) * 4u;
    break;
  case kPartialOther:
    participate = (lane & 1u) == 0u;
    offset = static_cast<uint64_t>(lane) * 4u;
    break;
  case kCrossSector:
    participate = lane == 0u;
    offset = 28u;
    width = 8u;
    break;
  case kInvalidLanes:
    offset = lane < 16u ? static_cast<uint64_t>(lane) * 4u
                        : static_cast<uint64_t>(kPayloadBytes);
    break;
  case kDeduplicate:
    participate = lane == 0u;
    break;
  case kVConfig32:
  case kVConfig64:
    offset = static_cast<uint64_t>(physical_thread) * 4u;
    break;
  case kOverlappingOther:
    participate = lane == 0u || lane == 2u;
    offset = lane == 0u ? 28u : 32u;
    width = 8u;
    break;
  case kDistinctSectorOther:
    offset = static_cast<uint64_t>(lane) * 32u;
    break;
  default:
    return;
  }

  if (!participate)
    return;
  const void *pointer = payload + offset;
  __rapid_feedback_mem(context, kSiteId, 0u, RAPID_FEEDBACK_READ, width,
                       pointer);
  if (selected_case == kDeduplicate) {
    __rapid_feedback_mem(context, kSiteId, 0u, RAPID_FEEDBACK_READ, width,
                         pointer);
  }
}

int parse_case(const char *name) {
  if (std::strcmp(name, "single") == 0)
    return kSingle;
  if (std::strcmp(name, "full_broadcast") == 0)
    return kFullBroadcast;
  if (std::strcmp(name, "full_contiguous") == 0)
    return kFullContiguous;
  if (std::strcmp(name, "full_other") == 0)
    return kFullOther;
  if (std::strcmp(name, "partial_broadcast") == 0)
    return kPartialBroadcast;
  if (std::strcmp(name, "partial_contiguous") == 0)
    return kPartialContiguous;
  if (std::strcmp(name, "partial_other") == 0)
    return kPartialOther;
  if (std::strcmp(name, "cross_sector") == 0)
    return kCrossSector;
  if (std::strcmp(name, "invalid_lanes") == 0)
    return kInvalidLanes;
  if (std::strcmp(name, "deduplicate") == 0)
    return kDeduplicate;
  if (std::strcmp(name, "vconfig32") == 0)
    return kVConfig32;
  if (std::strcmp(name, "vconfig64") == 0)
    return kVConfig64;
  if (std::strcmp(name, "overlapping_other") == 0)
    return kOverlappingOther;
  if (std::strcmp(name, "distinct_sector_other") == 0)
    return kDistinctSectorOther;
  return -1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s CASE\n", argv[0]);
    return 2;
  }
  const int selected_case = parse_case(argv[1]);
  if (selected_case < 0) {
    std::fprintf(stderr, "unknown case: %s\n", argv[1]);
    return 2;
  }

  uint8_t *payload = nullptr;
  uint8_t *bitmap = nullptr;
  RapidKernelContext *device_context = nullptr;
  check(cudaMalloc(&payload, kPayloadBytes), "cudaMalloc(payload)");
  check(cudaMalloc(&bitmap, SIMT_MEMCOV_STORAGE_SIZE), "cudaMalloc(bitmap)");
  check(cudaMalloc(&device_context, sizeof(RapidKernelContext)),
        "cudaMalloc(context)");
  check(cudaMemset(bitmap, 0, SIMT_MEMCOV_STORAGE_SIZE), "cudaMemset(bitmap)");

  const uint32_t logical_threads = selected_case == kVConfig32 ? 32u :
                                   selected_case == kVConfig64 ? 64u : 32u;
  RapidKernelContext context{};
  context.vconfig = {1u, 1u, 1u, logical_threads, 1u, 1u};
  context.feedback.simt_memcov_bits_addr = reinterpret_cast<uintptr_t>(bitmap);
  context.feedback.bounds_count = 1u;
  context.feedback.bounds[0] = {reinterpret_cast<uintptr_t>(payload),
                                kPayloadBytes};
  check(cudaMemcpy(device_context, &context, sizeof(context),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy(context)");

  const uint32_t physical_threads = logical_threads == 64u ? 64u : 32u;
  exercise_case<<<1u, physical_threads>>>(device_context, payload,
                                          selected_case);
  check(cudaGetLastError(), "exercise_case launch");
  check(cudaDeviceSynchronize(), "exercise_case synchronize");

  uint8_t host_bitmap[SIMT_MEMCOV_STORAGE_SIZE]{};
  check(cudaMemcpy(host_bitmap, bitmap, sizeof(host_bitmap),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(bitmap)");
  bool first = true;
  for (uint32_t bit = 0u; bit < SIMT_MEMCOV_BUCKETS; ++bit) {
    if ((host_bitmap[bit >> 3u] & (1u << (bit & 7u))) == 0u)
      continue;
    std::printf(first ? "%u" : ",%u", bit);
    first = false;
  }
  std::printf("\n");

  unsigned int atomic_calls = 0u;
  check(cudaMemcpyFromSymbol(&atomic_calls, rapid_test_atomic_calls,
                             sizeof(atomic_calls)),
        "cudaMemcpyFromSymbol(rapid_test_atomic_calls)");
  std::fprintf(stderr, "atomic_calls=%u\n", atomic_calls);

  check(cudaFree(device_context), "cudaFree(context)");
  check(cudaFree(bitmap), "cudaFree(bitmap)");
  check(cudaFree(payload), "cudaFree(payload)");
  return 0;
}
