#ifndef __RAPID2_KERNEL_BACKEND_CUH__
#define __RAPID2_KERNEL_BACKEND_CUH__

#include "kernel_backend_api.cuh"
#include "module_runtime.cuh"

namespace rapid2_kernel_backend {

inline bool initialize_global_coverage(const uint8_t *host_cov_map) {
  return rapid2_module_runtime::copy_global_coverage_to_device(host_cov_map);
}

template <size_t GridN, size_t BlockN>
inline bool select_persistent_launch_config(
    PersistentKernelLaunchConfig *out,
    const std::array<unsigned int, GridN> &grid,
    const std::array<unsigned int, BlockN> &block_candidates,
    size_t target_dynamic_shared_bytes) {
  if (out == nullptr) {
    return false;
  }
  dim3 selected_grid{};
  dim3 selected_block{};
  size_t selected_shared_mem_size = 0;
  unsigned int selected_block_threads = 0;
  if (!rapid2_module_runtime::select_persistent_launch_config(
          grid.data(), grid.size(), block_candidates.data(),
          block_candidates.size(), target_dynamic_shared_bytes, &selected_grid,
          &selected_block, &selected_shared_mem_size,
          &selected_block_threads)) {
    return false;
  }
  out->grid_dim = selected_grid;
  out->block_dim = selected_block;
  out->shared_mem_size = selected_shared_mem_size;
  out->physical_block_threads = selected_block_threads;
  return true;
}

inline bool launch_persistent_kernel(const PersistentKernelLaunchParams &params) {
  if (!rapid2_module_runtime::ensure_context_current()) {
    return false;
  }

  CUDA_CHECK(cudaSetDevice(0));

  CUstream kernel_stream_driver = reinterpret_cast<CUstream>(params.kernel_stream);
  CUdeviceptr combined0_arg = reinterpret_cast<CUdeviceptr>(params.d_combined_input[0]);
  CUdeviceptr combined1_arg = reinterpret_cast<CUdeviceptr>(params.d_combined_input[1]);
  CUdeviceptr combined2_arg = reinterpret_cast<CUdeviceptr>(params.d_combined_input[2]);
  CUdeviceptr combined3_arg = reinterpret_cast<CUdeviceptr>(params.d_combined_input[3]);
  CUdeviceptr coverage0_arg = reinterpret_cast<CUdeviceptr>(params.d_coverage[0]);
  CUdeviceptr coverage1_arg = reinterpret_cast<CUdeviceptr>(params.d_coverage[1]);
  CUdeviceptr coverage2_arg = reinterpret_cast<CUdeviceptr>(params.d_coverage[2]);
  CUdeviceptr coverage3_arg = reinterpret_cast<CUdeviceptr>(params.d_coverage[3]);
  CUdeviceptr d_context0 = reinterpret_cast<CUdeviceptr>(params.d_context[0]);
  CUdeviceptr d_context1 = reinterpret_cast<CUdeviceptr>(params.d_context[1]);
  CUdeviceptr d_context2 = reinterpret_cast<CUdeviceptr>(params.d_context[2]);
  CUdeviceptr d_context3 = reinterpret_cast<CUdeviceptr>(params.d_context[3]);
  CUdeviceptr ready0_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_ready[0]));
  CUdeviceptr ready1_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_ready[1]));
  CUdeviceptr ready2_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_ready[2]));
  CUdeviceptr ready3_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_ready[3]));
  CUdeviceptr done0_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_done[0]));
  CUdeviceptr done1_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_done[1]));
  CUdeviceptr done2_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_done[2]));
  CUdeviceptr done3_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.buffer_done[3]));
  CUdeviceptr should_exit_arg = static_cast<CUdeviceptr>(
      reinterpret_cast<uintptr_t>(params.sync_ptrs.should_exit));
  int kernel_id = params.kernel_id;
  void *kernel_args[] = {
      &combined0_arg,
      &combined1_arg,
      &combined2_arg,
      &combined3_arg,
      &coverage0_arg,
      &coverage1_arg,
      &coverage2_arg,
      &coverage3_arg,
      &d_context0,
      &d_context1,
      &d_context2,
      &d_context3,
      &ready0_arg,
      &ready1_arg,
      &ready2_arg,
      &ready3_arg,
      &done0_arg,
      &done1_arg,
      &done2_arg,
      &done3_arg,
      &should_exit_arg,
      &kernel_id};

  return rapid2_module_runtime::check_driver(
      cuLaunchKernel(rapid2_module_runtime::persistent_kernel(),
                     params.grid_dim.x, params.grid_dim.y, params.grid_dim.z,
                     params.block_dim.x, params.block_dim.y, params.block_dim.z,
                     params.shared_mem_size, kernel_stream_driver, kernel_args,
                     nullptr),
      "cuLaunchKernel(rapid2_persistent_kernel)");
}

inline bool copy_timing_stats_from_device(KernelTimingStats *out,
                                          cudaStream_t stream) {
#if ENABLE_KERNEL_TIMING
  return rapid2_module_runtime::copy_timing_stats_from_device(
      out, reinterpret_cast<CUstream>(stream));
#else
  (void)out;
  (void)stream;
  return false;
#endif
}

inline bool snapshot_timing_stats_from_device(KernelTimingStats *out,
                                              cudaStream_t stream) {
#if ENABLE_KERNEL_TIMING
  return rapid2_module_runtime::snapshot_timing_stats_from_device(
      out, reinterpret_cast<CUstream>(stream));
#else
  (void)out;
  (void)stream;
  return false;
#endif
}

inline bool reset_timing_stats_on_device(KernelTimingStats *scratch,
                                         cudaStream_t stream) {
#if ENABLE_KERNEL_TIMING
  return rapid2_module_runtime::reset_timing_stats_on_device(
      scratch, reinterpret_cast<CUstream>(stream));
#else
  (void)scratch;
  (void)stream;
  return true;
#endif
}

inline void shutdown_backend() { rapid2_module_runtime::unload_module(); }

} // namespace rapid2_kernel_backend

#endif // __RAPID2_KERNEL_BACKEND_CUH__
