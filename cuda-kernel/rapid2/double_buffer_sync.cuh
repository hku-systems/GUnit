#ifndef __DOUBLE_BUFFER_SYNC_CUH__
#define __DOUBLE_BUFFER_SYNC_CUH__

#include "config.h"
#include "cuda_utils.cuh"
#include "nvtx_profiler.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Device-side synchronization pointers structure
struct DeviceSyncPtrs {
  volatile uint64_t *buffer_ready[kRapid2MaxSlotCount]{};
  volatile uint64_t *buffer_done[kRapid2MaxSlotCount]{};
  volatile bool *should_exit;
};

// Double buffer synchronization class for CPU-GPU coordination
class DoubleBufferSync {
public:
  // Constructor - RAII initialization
  DoubleBufferSync() {
    nvtx_profiler_ = new NVTXProfiler<RAPID2_SLOT_COUNT>();

    // Each slot owns its ready staging value so an asynchronous H2D copy cannot
    // observe a later generation written for another slot.
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      CUDA_CHECK(cudaHostAlloc((void **)&h_pinned_ready[slot],
                               sizeof(uint64_t), cudaHostAllocDefault));
      CUDA_CHECK(cudaMalloc(&d_buffer_ready[slot], sizeof(uint64_t)));

      // Done generations use mapped memory so the CPU can poll without a
      // cudaMemcpy on every check.
      CUDA_CHECK(cudaHostAlloc((void **)&h_buffer_done[slot], sizeof(uint64_t),
                               cudaHostAllocMapped));
      CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_buffer_done[slot],
                                          (void *)h_buffer_done[slot], 0));
    }

    CUDA_CHECK(cudaMalloc(&d_should_exit, sizeof(bool)));

    // Initialize flags to zero
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      *h_pinned_ready[slot] = 0;
      CUDA_CHECK(cudaMemset(d_buffer_ready[slot], 0, sizeof(uint64_t)));
      *h_buffer_done[slot] = 0;
    }
    CUDA_CHECK(cudaMemset(d_should_exit, 0, sizeof(bool)));

    initialized = true;
  }

  // Destructor - RAII cleanup
  ~DoubleBufferSync() {
    if (initialized) {
      for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
        if (h_pinned_ready[slot])
          cudaFreeHost(h_pinned_ready[slot]);
        if (d_buffer_ready[slot])
          cudaFree(d_buffer_ready[slot]);
        if (h_buffer_done[slot])
          cudaFreeHost((void *)h_buffer_done[slot]);
      }
      if (d_should_exit)
        cudaFree(d_should_exit);

      // Cleanup NVTX profiler
      if (nvtx_profiler_) {
        delete nvtx_profiler_;
        nvtx_profiler_ = nullptr;
      }
    }
  }

  // Disable copy and move
  DoubleBufferSync(const DoubleBufferSync &) = delete;
  DoubleBufferSync &operator=(const DoubleBufferSync &) = delete;
  DoubleBufferSync(DoubleBufferSync &&) = delete;
  DoubleBufferSync &operator=(DoubleBufferSync &&) = delete;

  // CPU-side interface with task ID for better NVTX visualization
  void signalBufferReady(int buffer_idx, uint64_t generation,
                         cudaStream_t stream = 0) {
    if (buffer_idx < 0 || static_cast<size_t>(buffer_idx) >= kRapid2SlotCount)
      return;

    *h_pinned_ready[buffer_idx] = generation;
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_ready[buffer_idx],
                               h_pinned_ready[buffer_idx], sizeof(uint64_t),
                               cudaMemcpyHostToDevice, stream));

    // Start NVTX kernel processing range with task ID
    NVTX_BEGIN_KERNEL(nvtx_profiler_, buffer_idx, generation);
  }

  CUresult enqueueBufferReady(int buffer_idx, uint64_t generation,
                              cudaStream_t stream) {
    if (buffer_idx < 0 ||
        static_cast<size_t>(buffer_idx) >= kRapid2SlotCount) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult result = cuStreamWriteValue64(
        reinterpret_cast<CUstream>(stream),
        reinterpret_cast<CUdeviceptr>(d_buffer_ready[buffer_idx]), generation,
        CU_STREAM_WRITE_VALUE_DEFAULT);
    if (result == CUDA_SUCCESS) {
      NVTX_BEGIN_KERNEL(nvtx_profiler_, buffer_idx, generation);
    }
    return result;
  }

  bool checkBufferDone(int buffer_idx, uint64_t generation) const {
    if (buffer_idx < 0 || static_cast<size_t>(buffer_idx) >= kRapid2SlotCount)
      return false;
    // Direct read from mapped memory - no cudaMemcpy needed
    // CPU can directly access the value written by GPU
    return *h_buffer_done[buffer_idx] == generation;
  }

  CUresult enqueueCompletionWait(int buffer_idx, uint64_t generation,
                                 cudaStream_t stream) const {
    if (buffer_idx < 0 ||
        static_cast<size_t>(buffer_idx) >= kRapid2SlotCount) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return cuStreamWaitValue64(
        reinterpret_cast<CUstream>(stream),
        reinterpret_cast<CUdeviceptr>(d_buffer_done[buffer_idx]), generation,
        CU_STREAM_WAIT_VALUE_EQ);
  }

  void signalExit(cudaStream_t stream = 0) {
    bool exit_val = true;
    // Set exit flag
    CUDA_CHECK(cudaMemcpyAsync(d_should_exit, &exit_val, sizeof(bool),
                               cudaMemcpyHostToDevice, stream));
  }

  // Get device pointers for kernel use
  DeviceSyncPtrs getDevicePtrs() const {
    DeviceSyncPtrs ptrs{};
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      ptrs.buffer_ready[slot] = d_buffer_ready[slot];
      ptrs.buffer_done[slot] = d_buffer_done[slot];
    }
    ptrs.should_exit = d_should_exit;
    return ptrs;
  }

  bool isInitialized() const { return initialized; }

  // Record buffer completion event for profiling
  void recordBufferDone(int buffer_idx, uint64_t task_id) {
    if (buffer_idx < 0 || static_cast<size_t>(buffer_idx) >= kRapid2SlotCount)
      return;

    // End NVTX kernel processing range with task_id validation
    NVTX_END_KERNEL(nvtx_profiler_, buffer_idx, task_id);
  }

  // Get NVTX profiler for external use
  NVTXProfiler<RAPID2_SLOT_COUNT> *getNVTXProfiler() {
    return nvtx_profiler_;
  }

private:
  // Synchronization flags for each buffer
  uint64_t *h_pinned_ready[RAPID2_SLOT_COUNT]{};
  uint64_t *d_buffer_ready[RAPID2_SLOT_COUNT]{};

  // Done flags use mapped memory for zero-copy CPU-GPU synchronization
  volatile uint64_t *h_buffer_done[RAPID2_SLOT_COUNT]{};
  uint64_t *d_buffer_done[RAPID2_SLOT_COUNT]{};

  bool *d_should_exit = nullptr; // Exit flag

  NVTXProfiler<RAPID2_SLOT_COUNT> *nvtx_profiler_ = nullptr;

  // RAII resource management
  bool initialized = false;
};

#endif // __DOUBLE_BUFFER_SYNC_CUH__
