#ifndef __UTILS_COVERAGE_HELPER_CUH__
#define __UTILS_COVERAGE_HELPER_CUH__
#ifndef __UTILS_COVERAGE_CUH__
#error "coverage_helper.cuh should only be included from coverage.cuh"
#endif
#include "cuda_utils.cuh"

// Reset the edge map
extern "C" __attribute__((visibility("default"))) void
init_edges(cudaStream_t copy_stream = 0) {
  // Copy host libafl_cov_map to device to init coverage map
  CUDA_CHECK(cudaMemcpyToSymbolAsync(device_cov_map, libafl_cov_map, MAP_SIZE,
                                     0, cudaMemcpyHostToDevice, copy_stream));
}

extern "C" __attribute__((visibility("default"))) void
get_edges(cudaStream_t copy_stream = 0) {
  // Copy device device_cov_map to host libafl_cov_map
  CUDA_CHECK(cudaMemcpyFromSymbolAsync(libafl_cov_map, device_cov_map, MAP_SIZE,
                                       0, cudaMemcpyDeviceToHost, copy_stream));
}

__device__ inline void __libafl_edge(uint16_t _index, uint8_t *_cov_map) {
  if (_cov_map == nullptr) {
    return;
  }
  uint16_t offset = _index % 4;
  uint16_t index_mask = _index - offset;
  atomicOr(reinterpret_cast<uint32_t *>(&_cov_map[index_mask]),
           (uint32_t)1 << (offset * 8));
}

__device__ inline uint8_t *rapid_current_coverage_map() {
  return rapid_bound_cov_map == nullptr ? device_cov_map : rapid_bound_cov_map;
}

__device__ inline void rapid_bind_coverage_map(uint8_t *coverage_map) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
      blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    rapid_bound_cov_map = coverage_map;
  }
  __syncthreads();
}

static_assert(MAP_SIZE % sizeof(uint32_t) == 0,
              "MAP_SIZE must be divisible by the clear word size");

__device__ inline void rapid_clear_coverage_words() {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  uint32_t *coverage_words =
      reinterpret_cast<uint32_t *>(rapid_current_coverage_map());
  constexpr uint32_t word_count = MAP_SIZE / sizeof(uint32_t);
  for (uint32_t index = tid; index < word_count; index += stride) {
    coverage_words[index] = 0u;
  }
}

// Compatibility no-op. Coverage writes already target the bound global map.
__device__ inline void merge_shared_coverage(uint8_t* device_cov_map,
                                             uint8_t* shared_cov_map) {
  (void)device_cov_map;
  (void)shared_cov_map;
  __syncthreads();
}

__device__ inline void merge_shared_coverage() {
  __syncthreads();
}

#endif /* __UTILS_COVERAGE_HELPER_CUH__ */
