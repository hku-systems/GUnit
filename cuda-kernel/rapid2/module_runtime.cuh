#ifndef __RAPID2_MODULE_RUNTIME_CUH__
#define __RAPID2_MODULE_RUNTIME_CUH__

#include "config.h"
#include "coverage/coverage_constants.h"
#include "utils/timing_stats.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace rapid2_embedded_ptx {
extern const unsigned char kEmbeddedPtx[];
extern const size_t kEmbeddedPtxSize;
inline constexpr const char *kKernelName = "rapid2_persistent_kernel";
#if RAPID_ENABLE_INNER_FEEDBACK
inline constexpr const char *kCoverageGlobalName = "device_cov_map";
#endif
#if ENABLE_KERNEL_TIMING
inline constexpr const char *kTimingGlobalName = "g_rapid2_timing_stats";
#endif
} // namespace rapid2_embedded_ptx

namespace rapid2_module_runtime {

struct ModuleState {
  CUcontext ctx = nullptr;
  CUmodule module = nullptr;
  CUfunction persistent_kernel = nullptr;
  CUdeviceptr d_cov_map = 0;
  size_t d_cov_map_size = 0;
#if ENABLE_KERNEL_TIMING
  CUdeviceptr d_timing_stats = 0;
  size_t d_timing_stats_size = 0;
#endif
  bool loaded = false;
};

struct LaunchResourceInfo {
  int regs_per_thread = 0;
  int static_shared_bytes = 0;
  int function_max_threads_per_block = 0;
  int device_max_threads_per_block = 0;
  int device_max_registers_per_block = 0;
  int device_max_shared_bytes_per_block = 0;
};

static ModuleState g_module_state;
static std::mutex g_module_mutex;

inline bool check_driver(CUresult status, const char *what) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &desc);
  std::fprintf(stderr, "[rapid2-module] %s failed: %s (%s)\n", what,
               name ? name : "CUDA_ERROR_UNKNOWN",
               desc ? desc : "no description");
  return false;
}

inline bool ensure_module_loaded() {
  std::lock_guard<std::mutex> lock(g_module_mutex);
  if (g_module_state.loaded) {
    return true;
  }

  if (!check_driver(cuInit(0), "cuInit")) {
    return false;
  }

  // Ensure the CUDA runtime has established the primary context before loading
  // the external PTX module into it.
  const cudaError_t runtime_status = cudaFree(nullptr);
  if (runtime_status != cudaSuccess && runtime_status != cudaErrorCudartUnloading) {
    std::fprintf(stderr, "[rapid2-module] cudaFree(0) failed: %s\n",
                 cudaGetErrorString(runtime_status));
    return false;
  }

  CUcontext ctx = nullptr;
  if (!check_driver(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent")) {
    return false;
  }
  if (ctx == nullptr) {
    std::fprintf(stderr,
                 "[rapid2-module] no active CUDA context for module load\n");
    return false;
  }
  g_module_state.ctx = ctx;

  if (rapid2_embedded_ptx::kEmbeddedPtxSize == 0) {
    std::fprintf(stderr, "[rapid2-module] embedded PTX image is empty\n");
    return false;
  }

  if (!check_driver(cuModuleLoadDataEx(&g_module_state.module,
                                       rapid2_embedded_ptx::kEmbeddedPtx, 0,
                                       nullptr, nullptr),
                    "cuModuleLoadDataEx")) {
    return false;
  }
  if (!check_driver(cuModuleGetFunction(
                        &g_module_state.persistent_kernel,
                        g_module_state.module,
                        rapid2_embedded_ptx::kKernelName),
                    "cuModuleGetFunction(rapid2_persistent_kernel)")) {
    cuModuleUnload(g_module_state.module);
    g_module_state.module = nullptr;
    return false;
  }
#if RAPID_ENABLE_INNER_FEEDBACK
  if (!check_driver(cuModuleGetGlobal(&g_module_state.d_cov_map,
                                      &g_module_state.d_cov_map_size,
                                      g_module_state.module,
                                      rapid2_embedded_ptx::kCoverageGlobalName),
                    "cuModuleGetGlobal(device_cov_map)")) {
    cuModuleUnload(g_module_state.module);
    g_module_state.module = nullptr;
    g_module_state.persistent_kernel = nullptr;
    return false;
  }
  if (g_module_state.d_cov_map_size < MAP_SIZE) {
    std::fprintf(stderr,
                 "[rapid2-module] device_cov_map too small: got=%zu want=%u\n",
                 g_module_state.d_cov_map_size, MAP_SIZE);
    cuModuleUnload(g_module_state.module);
    g_module_state.module = nullptr;
    g_module_state.persistent_kernel = nullptr;
    g_module_state.d_cov_map = 0;
    g_module_state.d_cov_map_size = 0;
    return false;
  }
#endif

#if ENABLE_KERNEL_TIMING
  (void)cuModuleGetGlobal(&g_module_state.d_timing_stats,
                          &g_module_state.d_timing_stats_size,
                          g_module_state.module,
                          rapid2_embedded_ptx::kTimingGlobalName);
#endif

  g_module_state.loaded = true;
  return true;
}

inline void unload_module() {
  std::lock_guard<std::mutex> lock(g_module_mutex);
  if (g_module_state.module != nullptr) {
    (void)cuModuleUnload(g_module_state.module);
  }
  g_module_state = ModuleState{};
}

inline CUfunction persistent_kernel() { return g_module_state.persistent_kernel; }

inline bool ensure_context_current() {
  if (!ensure_module_loaded()) {
    return false;
  }
  return check_driver(cuCtxSetCurrent(g_module_state.ctx), "cuCtxSetCurrent");
}

inline bool copy_global_coverage_to_device(const uint8_t *src) {
#if RAPID_ENABLE_INNER_FEEDBACK
  if (!ensure_context_current()) {
    return false;
  }
  return check_driver(cuMemcpyHtoD(g_module_state.d_cov_map, src, MAP_SIZE),
                      "cuMemcpyHtoD(device_cov_map)");
#else
  (void)src;
  return true;
#endif
}

inline bool query_launch_resources(LaunchResourceInfo *out) {
  if (out == nullptr) {
    return false;
  }
  if (!ensure_context_current()) {
    return false;
  }
  CUdevice device = 0;
  if (!check_driver(cuCtxGetDevice(&device), "cuCtxGetDevice")) {
    return false;
  }
  bool ok = true;
  ok &= check_driver(
      cuFuncGetAttribute(&out->regs_per_thread,
                         CU_FUNC_ATTRIBUTE_NUM_REGS,
                         g_module_state.persistent_kernel),
      "cuFuncGetAttribute(NUM_REGS)");
  ok &= check_driver(
      cuFuncGetAttribute(&out->static_shared_bytes,
                         CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                         g_module_state.persistent_kernel),
      "cuFuncGetAttribute(SHARED_SIZE_BYTES)");
  ok &= check_driver(
      cuFuncGetAttribute(&out->function_max_threads_per_block,
                         CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                         g_module_state.persistent_kernel),
      "cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)");
  ok &= check_driver(
      cuDeviceGetAttribute(&out->device_max_threads_per_block,
                           CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                           device),
      "cuDeviceGetAttribute(MAX_THREADS_PER_BLOCK)");
  ok &= check_driver(
      cuDeviceGetAttribute(&out->device_max_registers_per_block,
                           CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                           device),
      "cuDeviceGetAttribute(MAX_REGISTERS_PER_BLOCK)");
  ok &= check_driver(
      cuDeviceGetAttribute(&out->device_max_shared_bytes_per_block,
                           CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                           device),
      "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK)");
  return ok;
}

inline bool select_persistent_launch_config(
    const unsigned int *grid, size_t grid_count,
    const unsigned int *block_candidates, size_t block_candidate_count,
    size_t target_dynamic_shared_bytes, dim3 *out_grid, dim3 *out_block,
    size_t *out_shared_mem_size, unsigned int *out_physical_block_threads) {
  if (grid == nullptr || grid_count < 3 || block_candidates == nullptr ||
      block_candidate_count == 0 || out_grid == nullptr ||
      out_block == nullptr || out_shared_mem_size == nullptr ||
      out_physical_block_threads == nullptr) {
    return false;
  }
  LaunchResourceInfo resources{};
  if (!query_launch_resources(&resources)) {
    return false;
  }

  const unsigned int function_thread_limit =
      resources.function_max_threads_per_block > 0
          ? static_cast<unsigned int>(resources.function_max_threads_per_block)
          : 1024u;
  const unsigned int device_thread_limit =
      resources.device_max_threads_per_block > 0
          ? static_cast<unsigned int>(resources.device_max_threads_per_block)
          : 1024u;
  const unsigned int thread_limit =
      function_thread_limit < device_thread_limit ? function_thread_limit
                                                  : device_thread_limit;
  const size_t static_shared =
      resources.static_shared_bytes > 0
          ? static_cast<size_t>(resources.static_shared_bytes)
          : 0u;
  const size_t shared_required = static_shared + target_dynamic_shared_bytes;
  const size_t shared_limit =
      resources.device_max_shared_bytes_per_block > 0
          ? static_cast<size_t>(resources.device_max_shared_bytes_per_block)
          : static_cast<size_t>(48 * 1024);

  if (shared_required > shared_limit) {
    std::fprintf(stderr,
                 "[rapid2-module] dynamic shared requirement too large: "
                 "static=%zu target=%zu limit=%zu\n",
                 static_shared, target_dynamic_shared_bytes, shared_limit);
    return false;
  }

  unsigned int selected = 0u;
  for (size_t idx = 0; idx < block_candidate_count; ++idx) {
    const unsigned int candidate = block_candidates[idx];
    if (candidate == 0u || candidate > thread_limit) {
      continue;
    }
    if (resources.regs_per_thread > 0 &&
        resources.device_max_registers_per_block > 0) {
      const size_t regs_required =
          static_cast<size_t>(resources.regs_per_thread) * candidate;
      if (regs_required >
          static_cast<size_t>(resources.device_max_registers_per_block)) {
        continue;
      }
    }
    selected = candidate;
    break;
  }

  if (selected == 0u) {
    std::fprintf(stderr,
                 "[rapid2-module] no launch block candidate fits: "
                 "regs/thread=%d max_regs/block=%d func_max_threads=%d "
                 "device_max_threads=%d static_shared=%d target_shared=%zu\n",
                 resources.regs_per_thread,
                 resources.device_max_registers_per_block,
                 resources.function_max_threads_per_block,
                 resources.device_max_threads_per_block,
                 resources.static_shared_bytes, target_dynamic_shared_bytes);
    return false;
  }

  if (target_dynamic_shared_bytes > 0) {
    if (!check_driver(
            cuFuncSetAttribute(
                g_module_state.persistent_kernel,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                static_cast<int>(target_dynamic_shared_bytes)),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED)")) {
      return false;
    }
  }

  *out_grid = dim3(grid[0], grid[1], grid[2]);
  *out_block = dim3(selected, 1u, 1u);
  *out_shared_mem_size = target_dynamic_shared_bytes;
  *out_physical_block_threads = selected;

  std::fprintf(stderr,
               "[rapid2-module] selected block=%u grid=(%u,%u,%u) "
               "regs/thread=%d static_shared=%d target_shared=%zu "
               "coverage_memory=global\n",
               selected, grid[0], grid[1], grid[2], resources.regs_per_thread,
               resources.static_shared_bytes, target_dynamic_shared_bytes);
  return true;
}

#if ENABLE_KERNEL_TIMING
inline bool copy_timing_stats_from_device(KernelTimingStats *out,
                                          CUstream stream) {
  if (out == nullptr || stream == nullptr) {
    return false;
  }
  if (!ensure_context_current()) {
    return false;
  }
  if (g_module_state.d_timing_stats == 0 ||
      g_module_state.d_timing_stats_size < sizeof(KernelTimingStats)) {
    return false;
  }
  if (!check_driver(cuMemcpyDtoHAsync(out, g_module_state.d_timing_stats,
                                     sizeof(KernelTimingStats), stream),
                    "cuMemcpyDtoHAsync(g_rapid2_timing_stats)")) {
    return false;
  }
  return check_driver(cuStreamSynchronize(stream),
                      "cuStreamSynchronize(rapid2 timing snapshot)");
}

inline bool request_timing_stats_on_device(uint64_t request,
                                           KernelTimingStats *scratch,
                                           CUstream stream) {
  if (scratch == nullptr || stream == nullptr || !ensure_context_current()) {
    return false;
  }
  if (g_module_state.d_timing_stats == 0 ||
      g_module_state.d_timing_stats_size < sizeof(KernelTimingStats)) {
    return false;
  }
  const CUdeviceptr iterations =
      g_module_state.d_timing_stats + offsetof(KernelTimingStats, iterations);
  if (!check_driver(cuMemcpyHtoDAsync(iterations, &request, sizeof(request),
                                     stream),
                    "cuMemcpyHtoDAsync(g_rapid2_timing_stats request)") ||
      !check_driver(cuStreamSynchronize(stream),
                    "cuStreamSynchronize(rapid2 timing request)")) {
    return false;
  }
  for (size_t attempt = 0; attempt < 1000; ++attempt) {
    if (!copy_timing_stats_from_device(scratch, stream)) {
      return false;
    }
    if (scratch->iterations != request) {
      return true;
    }
  }
  return false;
}

inline bool snapshot_timing_stats_from_device(KernelTimingStats *out,
                                              CUstream stream) {
  if (!request_timing_stats_on_device(KERNEL_TIMING_SNAPSHOT_REQUEST, out,
                                      stream)) {
    return false;
  }
  // The acknowledgement publishes iterations after a system fence. Read the
  // now-stable structure once more so the snapshot cannot mix old and new
  // fields from the acknowledgement copy.
  return copy_timing_stats_from_device(out, stream);
}

inline bool reset_timing_stats_on_device(KernelTimingStats *scratch,
                                         CUstream stream) {
  return request_timing_stats_on_device(KERNEL_TIMING_RESET_REQUEST, scratch,
                                        stream);
}
#endif

} // namespace rapid2_module_runtime

#endif // __RAPID2_MODULE_RUNTIME_CUH__
