#ifndef __RAPID2_KERNEL_BACKEND_API_CUH__
#define __RAPID2_KERNEL_BACKEND_API_CUH__

#include "combined_input.cuh"
#include "double_buffer_sync.cuh"
#include "feedback/feedback_context.cuh"
#include "utils/timing_stats.cuh"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <array>

namespace rapid2_kernel_backend {

struct PersistentKernelLaunchParams {
  dim3 grid_dim{};
  dim3 block_dim{};
  size_t shared_mem_size = 0;
  cudaStream_t kernel_stream = nullptr;
  CombinedInput *d_combined_input[kRapid2MaxSlotCount]{};
  uint8_t *d_coverage[kRapid2MaxSlotCount]{};
  RapidKernelContext *d_context[kRapid2MaxSlotCount]{};
  DeviceSyncPtrs sync_ptrs{};
  int kernel_id = 0;
};

struct PersistentKernelLaunchConfig {
  dim3 grid_dim{1u, 1u, 1u};
  dim3 block_dim{1024u, 1u, 1u};
  size_t shared_mem_size = 0;
  unsigned int physical_block_threads = 1024u;
};

bool initialize_global_coverage(const uint8_t *host_cov_map);
template <size_t GridN, size_t BlockN>
bool select_persistent_launch_config(
    PersistentKernelLaunchConfig *out,
    const std::array<unsigned int, GridN> &grid,
    const std::array<unsigned int, BlockN> &block_candidates,
    size_t target_dynamic_shared_bytes);
bool launch_persistent_kernel(const PersistentKernelLaunchParams &params);
bool copy_timing_stats_from_device(KernelTimingStats *out,
                                   cudaStream_t stream);
bool snapshot_timing_stats_from_device(KernelTimingStats *out,
                                       cudaStream_t stream);
bool reset_timing_stats_on_device(KernelTimingStats *scratch,
                                  cudaStream_t stream);
void shutdown_backend();

} // namespace rapid2_kernel_backend

#endif // __RAPID2_KERNEL_BACKEND_API_CUH__
