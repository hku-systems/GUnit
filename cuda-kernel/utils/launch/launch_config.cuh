#ifndef __RAPID_UTILS_LAUNCH_CONFIG_CUH__
#define __RAPID_UTILS_LAUNCH_CONFIG_CUH__

#include "feedback/feedback_context.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace rapid_launch {

struct LaunchConfig {
  dim3 grid_dim{1u, 1u, 1u};
  dim3 block_dim{1u, 1u, 1u};
  dim3 logical_grid_dim{1u, 1u, 1u};
  dim3 logical_block_dim{1u, 1u, 1u};
  size_t shared_mem_size = 0u;
  unsigned int physical_block_threads = 1u;
  bool has_logical_vconfig_bounds = false;
};

struct LaunchResourceInfo {
  int regs_per_thread = 0;
  int static_shared_bytes = 0;
  int function_max_threads_per_block = 0;
  int device_max_threads_per_block = 0;
  int device_max_registers_per_block = 0;
  int device_max_shared_bytes_per_block = 0;
};

inline const char *log_prefix_or_default(const char *log_prefix) {
  return log_prefix ? log_prefix : "rapid-launch";
}

inline bool check_driver(CUresult status, const char *log_prefix,
                         const char *what) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &desc);
  std::fprintf(stderr, "[%s] %s failed: %s (%s)\n",
               log_prefix_or_default(log_prefix), what,
               name ? name : "CUDA_ERROR_UNKNOWN",
               desc ? desc : "no description");
  return false;
}

inline bool query_launch_resources(CUfunction function, CUdevice device,
                                   const char *log_prefix,
                                   LaunchResourceInfo *out) {
  if (function == nullptr || out == nullptr) {
    return false;
  }
  bool ok = true;
  ok &= check_driver(cuFuncGetAttribute(&out->regs_per_thread,
                                        CU_FUNC_ATTRIBUTE_NUM_REGS, function),
                     log_prefix, "cuFuncGetAttribute(NUM_REGS)");
  ok &= check_driver(
      cuFuncGetAttribute(&out->static_shared_bytes,
                         CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
      log_prefix, "cuFuncGetAttribute(SHARED_SIZE_BYTES)");
  ok &= check_driver(
      cuFuncGetAttribute(&out->function_max_threads_per_block,
                         CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
      log_prefix, "cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)");
  ok &= check_driver(
      cuDeviceGetAttribute(&out->device_max_threads_per_block,
                           CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device),
      log_prefix, "cuDeviceGetAttribute(MAX_THREADS_PER_BLOCK)");
  ok &= check_driver(
      cuDeviceGetAttribute(&out->device_max_registers_per_block,
                           CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device),
      log_prefix, "cuDeviceGetAttribute(MAX_REGISTERS_PER_BLOCK)");
  ok &= check_driver(
      cuDeviceGetAttribute(&out->device_max_shared_bytes_per_block,
                           CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                           device),
      log_prefix, "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK)");
  return ok;
}

inline bool select_cuda_launch_config(
    CUfunction function, CUdevice device, const unsigned int *grid,
    size_t grid_count, const unsigned int *block_candidates,
    size_t block_candidate_count, size_t target_dynamic_shared_bytes,
    const char *log_prefix, LaunchConfig *out) {
  if (function == nullptr || grid == nullptr || grid_count < 3 ||
      block_candidates == nullptr || block_candidate_count == 0 ||
      out == nullptr) {
    return false;
  }

  LaunchResourceInfo resources{};
  if (!query_launch_resources(function, device, log_prefix, &resources)) {
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
      std::min(function_thread_limit, device_thread_limit);
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
                 "[%s] dynamic shared requirement too large: static=%zu "
                 "target=%zu limit=%zu coverage_memory=global\n",
                 log_prefix_or_default(log_prefix), static_shared,
                 target_dynamic_shared_bytes, shared_limit);
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
                 "[%s] no launch block candidate fits: regs/thread=%d "
                 "max_regs/block=%d func_max_threads=%d "
                 "device_max_threads=%d static_shared=%d target_shared=%zu "
                 "coverage_memory=global\n",
                 log_prefix_or_default(log_prefix), resources.regs_per_thread,
                 resources.device_max_registers_per_block,
                 resources.function_max_threads_per_block,
                 resources.device_max_threads_per_block,
                 resources.static_shared_bytes, target_dynamic_shared_bytes);
    return false;
  }

  if (target_dynamic_shared_bytes > 0) {
    if (!check_driver(cuFuncSetAttribute(
                          function,
                          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                          static_cast<int>(target_dynamic_shared_bytes)),
                      log_prefix,
                      "cuFuncSetAttribute(MAX_DYNAMIC_SHARED)")) {
      return false;
    }
  }

  out->grid_dim = dim3(grid[0], grid[1], grid[2]);
  out->block_dim = dim3(selected, 1u, 1u);
  out->shared_mem_size = target_dynamic_shared_bytes;
  out->physical_block_threads = selected;

  std::fprintf(stderr,
               "[%s] selected block=%u grid=(%u,%u,%u) regs/thread=%d "
               "static_shared=%d target_shared=%zu coverage_memory=global\n",
               log_prefix_or_default(log_prefix), selected, grid[0], grid[1],
               grid[2], resources.regs_per_thread,
               resources.static_shared_bytes, target_dynamic_shared_bytes);
  return true;
}

template <size_t GridN, size_t BlockN>
inline bool select_cuda_launch_config(
    CUfunction function, CUdevice device,
    const std::array<unsigned int, GridN> &grid,
    const std::array<unsigned int, BlockN> &block_candidates,
    size_t target_dynamic_shared_bytes, const char *log_prefix,
    LaunchConfig *out) {
  return select_cuda_launch_config(
      function, device, grid.data(), grid.size(), block_candidates.data(),
      block_candidates.size(), target_dynamic_shared_bytes, log_prefix, out);
}

inline uint32_t clamp_launch_dim(uint32_t value, unsigned int max_value) {
  const uint32_t max_dim = max_value == 0u ? 1u : max_value;
  if (value == 0u) {
    return 1u;
  }
  return std::min<uint32_t>(value, max_dim);
}

inline uint64_t dim_product(const dim3 &dims) {
  return static_cast<uint64_t>(dims.x) * static_cast<uint64_t>(dims.y) *
         static_cast<uint64_t>(dims.z);
}

inline bool dims_are_positive(const unsigned int *dims, size_t count) {
  return dims != nullptr && count >= 3 && dims[0] > 0u && dims[1] > 0u &&
         dims[2] > 0u;
}

inline bool apply_logical_vconfig_bounds(LaunchConfig *launch_config,
                                         const unsigned int *logical_grid,
                                         size_t logical_grid_count,
                                         const unsigned int *logical_block,
                                         size_t logical_block_count,
                                         const char *log_prefix) {
  if (launch_config == nullptr ||
      !dims_are_positive(logical_grid, logical_grid_count) ||
      !dims_are_positive(logical_block, logical_block_count)) {
    return false;
  }

  const dim3 grid_bounds(logical_grid[0], logical_grid[1], logical_grid[2]);
  const dim3 block_bounds(logical_block[0], logical_block[1],
                          logical_block[2]);
#ifndef NDEBUG
  if (dim_product(grid_bounds) > dim_product(launch_config->grid_dim)) {
    std::fprintf(stderr,
                 "[%s] logical grid exceeds physical grid envelope: "
                 "logical=(%u,%u,%u) physical=(%u,%u,%u)\n",
                 log_prefix_or_default(log_prefix), grid_bounds.x,
                 grid_bounds.y, grid_bounds.z, launch_config->grid_dim.x,
                 launch_config->grid_dim.y, launch_config->grid_dim.z);
    return false;
  }
  if (dim_product(block_bounds) >
      static_cast<uint64_t>(launch_config->physical_block_threads)) {
    std::fprintf(stderr,
                 "[%s] logical block threads exceed physical block envelope: "
                 "logical=(%u,%u,%u) physical_threads=%u\n",
                 log_prefix_or_default(log_prefix), block_bounds.x,
                 block_bounds.y, block_bounds.z,
                 launch_config->physical_block_threads);
    return false;
  }
#endif

  launch_config->logical_grid_dim = grid_bounds;
  launch_config->logical_block_dim = block_bounds;
  launch_config->has_logical_vconfig_bounds = true;
  return true;
}

template <size_t GridN, size_t BlockN>
inline bool apply_logical_vconfig_bounds(
    LaunchConfig *launch_config,
    const std::array<unsigned int, GridN> &logical_grid,
    const std::array<unsigned int, BlockN> &logical_block,
    const char *log_prefix) {
  return apply_logical_vconfig_bounds(launch_config, logical_grid.data(),
                                      logical_grid.size(),
                                      logical_block.data(),
                                      logical_block.size(), log_prefix);
}

inline RapidVConfig clamp_vconfig_to_launch_bounds(
    const RapidVConfig &vconfig, const LaunchConfig &launch_config) {
  const dim3 grid_bounds = launch_config.has_logical_vconfig_bounds
                               ? launch_config.logical_grid_dim
                               : launch_config.grid_dim;
  const dim3 block_bounds = launch_config.has_logical_vconfig_bounds
                                ? launch_config.logical_block_dim
                                : launch_config.block_dim;
  RapidVConfig out = vconfig;
  out.grid_x = clamp_launch_dim(out.grid_x, grid_bounds.x);
  out.grid_y = clamp_launch_dim(out.grid_y, grid_bounds.y);
  out.grid_z = clamp_launch_dim(out.grid_z, grid_bounds.z);
  out.block_x = clamp_launch_dim(out.block_x, block_bounds.x);
  out.block_y = clamp_launch_dim(out.block_y, block_bounds.y);
  out.block_z = clamp_launch_dim(out.block_z, block_bounds.z);
  return out;
}

} // namespace rapid_launch

#endif // __RAPID_UTILS_LAUNCH_CONFIG_CUH__
