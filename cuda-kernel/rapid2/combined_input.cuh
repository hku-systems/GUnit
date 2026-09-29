#ifndef __COMBINED_INPUT_CUH__
#define __COMBINED_INPUT_CUH__

#include "config.h"
#include "coverage/coverage_constants.h"
#include "feedback/feedback_constants.h"
#include "input/task_envelope.cuh"
#include "sanitizer/canary.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

template <typename T> struct PinnedHostDeleter {
  void operator()(T *ptr) const noexcept {
    if (ptr) {
      (void)cudaFreeHost(ptr);
    }
  }
};

template <typename T>
using PinnedHostPtr = std::unique_ptr<T, PinnedHostDeleter<T>>;

template <typename T>
PinnedHostPtr<T> allocatePinnedHostArray(size_t count) {
  T *ptr = nullptr;
  const cudaError_t error =
      cudaHostAlloc(reinterpret_cast<void **>(&ptr), sizeof(T) * count,
                    cudaHostAllocDefault);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
  std::memset(ptr, 0, sizeof(T) * count);
  return PinnedHostPtr<T>(ptr);
}

class HostFeedbackBuffer {
public:
  static constexpr size_t kSize = MAP_SIZE + SIMT_MEMCOV_STORAGE_SIZE;

  HostFeedbackBuffer() = default;

  static HostFeedbackBuffer allocate() {
    return HostFeedbackBuffer(allocatePinnedHostArray<uint8_t>(kSize));
  }

  uint8_t *data() { return storage_.get(); }
  const uint8_t *data() const { return storage_.get(); }
  uint8_t *edgeData() { return data(); }
  const uint8_t *edgeData() const { return data(); }
  uint8_t *memIndexData() { return data() + MAP_SIZE; }
  const uint8_t *memIndexData() const { return data() + MAP_SIZE; }
  constexpr size_t size() const { return kSize; }
  explicit operator bool() const { return storage_ != nullptr; }

private:
  explicit HostFeedbackBuffer(PinnedHostPtr<uint8_t> storage)
      : storage_(std::move(storage)) {}

  PinnedHostPtr<uint8_t> storage_;
};

// Device-side combined input keeps a fixed-size buffer for the persistent
// kernel.
struct CombinedInput {
  RapidTaskEnvelopeHeader header;
  uint8_t data[MAX_INPUT_SIZE + rapid::canary::kStorageSize];

  __host__ __device__ size_t getTransferSize() const {
    return rapid::canary::guarded_size(
        rapid_task_envelope_size(static_cast<size_t>(header.payload_size)));
  }
};

// Host-side combined input uses a full-capacity allocation; each transfer is
// limited to the current envelope plus the optional canary redzone.
struct HostCombinedInput;
using HostCombinedInputPtr = PinnedHostPtr<HostCombinedInput>;

struct HostCombinedInput {
  RapidTaskEnvelopeHeader header;
  uint8_t data[1];

  size_t getTransferSize() const {
    return rapid::canary::guarded_size(
        rapid_task_envelope_size(static_cast<size_t>(header.payload_size)));
  }

  static HostCombinedInputPtr allocate(size_t input_size) {
    const size_t alloc_size = std::max(
        offsetof(HostCombinedInput, data) +
            rapid::canary::guarded_size(input_size),
        sizeof(HostCombinedInput));
    HostCombinedInput *ptr = nullptr;
    const cudaError_t error =
        cudaHostAlloc(reinterpret_cast<void **>(&ptr), alloc_size,
                      cudaHostAllocDefault);
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    return HostCombinedInputPtr(ptr);
  }
};

#endif // __COMBINED_INPUT_CUH__
