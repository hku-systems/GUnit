#ifndef __UTILS_FEEDBACK_CUH__
#define __UTILS_FEEDBACK_CUH__

#include "feedback_context.cuh"
#include "coverage/coverage.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

enum RapidFeedbackAccessKind : uint8_t {
  RAPID_FEEDBACK_READ = 0u,
  RAPID_FEEDBACK_WRITE = 1u,
};

__device__ inline uint32_t rapid_feedback_mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__device__ inline uint32_t *
rapid_feedback_words(RapidKernelContext *context) {
  if (context == nullptr ||
      context->feedback.simt_memcov_bits_addr == static_cast<uintptr_t>(0)) {
    return nullptr;
  }
  return reinterpret_cast<uint32_t *>(
      context->feedback.simt_memcov_bits_addr);
}

__device__ inline void rapid_feedback_set_bucket(RapidKernelContext *context,
                                                  uint32_t bucket) {
  uint32_t *words = rapid_feedback_words(context);
  if (words == nullptr || bucket >= SIMT_MEMCOV_BUCKETS) {
    return;
  }
  atomicOr(&words[bucket >> 5u], 1u << (bucket & 31u));
}

__device__ inline void
rapid_feedback_clear_mem_bits_no_sync(RapidKernelContext *context) {
  uint32_t *words = rapid_feedback_words(context);
  if (words == nullptr) {
    return;
  }
  constexpr uint32_t word_count = SIMT_MEMCOV_BUCKETS / 32u;
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t index = tid; index < word_count; index += stride) {
    words[index] = 0u;
  }
}

__device__ inline void
rapid_feedback_clear_mem_bits(RapidKernelContext *context) {
  rapid_feedback_clear_mem_bits_no_sync(context);
  __syncthreads();
}

// Reset both task-owned feedback maps and publish the cleared state with one
// block barrier before any instrumented target code runs.
__device__ inline void
rapid_feedback_clear_task_maps(RapidKernelContext *context) {
  rapid_clear_coverage_words();
  rapid_feedback_clear_mem_bits_no_sync(context);
  __syncthreads();
}

extern "C" __device__ __noinline__ void
__rapid_feedback_bb(uint32_t site_id) {
  __libafl_edge(static_cast<uint16_t>(site_id % MAP_SIZE),
                rapid_current_coverage_map());
}

__device__ inline uint32_t rapid_feedback_rotl32(uint32_t value,
                                                uint32_t amount) {
  return (value << amount) | (value >> (32u - amount));
}

__device__ inline uint32_t rapid_feedback_fold64(uint64_t value) {
  return static_cast<uint32_t>(value) ^
         rapid_feedback_rotl32(static_cast<uint32_t>(value >> 32u), 16u);
}

__device__ inline uint64_t rapid_feedback_block_linear() {
  return static_cast<uint64_t>(blockIdx.x) +
         static_cast<uint64_t>(gridDim.x) *
             (static_cast<uint64_t>(blockIdx.y) +
              static_cast<uint64_t>(gridDim.y) * blockIdx.z);
}

__device__ inline uint32_t rapid_feedback_thread_linear() {
  return threadIdx.x +
         blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ inline uint32_t rapid_simt_mem_bucket(
    uint32_t site_id, uint16_t arg_slot, uint64_t sector,
    uint8_t access_kind, uint64_t logical_block, uint32_t warp_in_block,
    uint32_t pattern) {
  uint32_t key = site_id;
  key ^= static_cast<uint32_t>(arg_slot) * RAPID_SIMT_MEM_ARG_SLOT_SALT;
  key ^= static_cast<uint32_t>(sector) * RAPID_SIMT_MEM_SECTOR_LOW_SALT;
  key ^= static_cast<uint32_t>(sector >> 32u) *
         RAPID_SIMT_MEM_SECTOR_HIGH_SALT;
  key ^= static_cast<uint32_t>(access_kind) *
         RAPID_SIMT_MEM_ACCESS_KIND_SALT;
  key ^= rapid_feedback_fold64(logical_block) *
         RAPID_SIMT_MEM_LOGICAL_BLOCK_SALT;
  key ^= warp_in_block * RAPID_SIMT_MEM_WARP_IN_BLOCK_SALT;
  key ^= pattern * RAPID_SIMT_MEM_PATTERN_SALT;
  return rapid_feedback_mix32(key) % SIMT_MEMCOV_DATA_BUCKETS;
}

__device__ inline void rapid_simt_mem_emit_sector(
    RapidKernelContext *context, uint32_t site_id, uint16_t arg_slot,
    uint64_t sector, uint8_t access_kind, uint64_t logical_block,
    uint32_t warp_in_block, uint32_t pattern) {
  rapid_feedback_set_bucket(
      context, rapid_simt_mem_bucket(site_id, arg_slot, sector, access_kind,
                                     logical_block, warp_in_block, pattern));
}

__device__ __noinline__ void rapid_simt_mem_emit_scatter_ranges(
    RapidKernelContext *context, uint32_t site_id, uint16_t arg_slot,
    uint8_t access_kind, uint64_t logical_block, uint32_t warp_in_block,
    uint32_t pattern, uint32_t valid_mask, uint32_t lane,
    uint32_t first_sector, uint32_t last_sector) {
  uint32_t sector_step = 0u;
  for (;;) {
    const uint32_t current_sector = first_sector + sector_step;
    const bool pending = current_sector <= last_sector;
    if (__ballot_sync(valid_mask, pending) == 0u) {
      break;
    }

    bool canonical_owner = pending;
    uint32_t source_lanes = valid_mask;
    while (source_lanes != 0u) {
      const uint32_t source_lane =
          static_cast<uint32_t>(__ffs(source_lanes) - 1);
      const uint32_t source_first_sector = __shfl_sync(
          valid_mask, first_sector, static_cast<int>(source_lane));
      const uint32_t source_last_sector = __shfl_sync(
          valid_mask, last_sector, static_cast<int>(source_lane));
      if (source_lane < lane && current_sector >= source_first_sector &&
          current_sector <= source_last_sector) {
        canonical_owner = false;
      }
      source_lanes &= source_lanes - 1u;
    }

    if (canonical_owner) {
      rapid_simt_mem_emit_sector(context, site_id, arg_slot, current_sector,
                                 access_kind, logical_block, warp_in_block,
                                 pattern);
    }
    ++sector_step;
  }
}

extern "C" __device__ __noinline__ void __rapid_feedback_mem(
    RapidKernelContext *context, uint32_t site_id, uint16_t arg_slot,
    uint8_t access_kind, uint8_t width, const void *access_ptr) {
  const uint32_t call_mask = __activemask();
  const bool context_valid =
      context != nullptr && arg_slot < RAPID_PAYLOAD_SLOT_COUNT &&
      arg_slot < context->feedback.bounds_count;

  uintptr_t base = 0u;
  uint64_t len = 0u;
  if (context_valid) {
    const RapidPayloadBounds &bounds = context->feedback.bounds[arg_slot];
    base = bounds.base;
    len = bounds.len_bytes;
  }

  const uintptr_t address = reinterpret_cast<uintptr_t>(access_ptr);
  const uint64_t access_width = static_cast<uint64_t>(width);
  uint64_t checked_offset = 0u;
  bool lane_valid = context_valid && access_ptr != nullptr && width != 0u &&
                    access_width <= len && address >= base;
  if (lane_valid) {
    checked_offset = static_cast<uint64_t>(address - base);
    lane_valid = checked_offset <= len - access_width &&
                 checked_offset <=
                     static_cast<uint64_t>(UINT32_MAX) - (access_width - 1u);
  }

  const uint32_t valid_mask = __ballot_sync(call_mask, lane_valid);
  if (valid_mask == 0u || !lane_valid) {
    return;
  }

  const uint32_t offset = static_cast<uint32_t>(checked_offset);
  const uint32_t access_width32 = static_cast<uint32_t>(width);
  const uint32_t thread_linear = rapid_feedback_thread_linear();
  const uint32_t lane = thread_linear & 31u;
  const uint32_t first_lane = static_cast<uint32_t>(__ffs(valid_mask) - 1);
  const uint32_t valid_lanes = static_cast<uint32_t>(__popc(valid_mask));

  uint32_t pattern = RAPID_SIMT_MEM_PATTERN_SINGLE;
  if (valid_lanes > 1u) {
    const uint32_t first_offset =
        __shfl_sync(valid_mask, offset, static_cast<int>(first_lane));
    const uint32_t interval_mask =
        valid_lanes == 32u
            ? 0xffffffffu
            : ((1u << valid_lanes) - 1u) << first_lane;
    const uint32_t expected_offset =
        first_offset + (lane - first_lane) * access_width32;
    const bool contiguous =
        interval_mask == valid_mask &&
        __ballot_sync(valid_mask, offset == expected_offset) == valid_mask;
    if (contiguous) {
      pattern = valid_lanes == 32u
                    ? RAPID_SIMT_MEM_PATTERN_FULL_CONTIGUOUS
                    : RAPID_SIMT_MEM_PATTERN_PARTIAL_CONTIGUOUS;
    } else {
      const bool broadcast =
          __ballot_sync(valid_mask, offset == first_offset) == valid_mask;
      if (broadcast) {
        pattern = valid_lanes == 32u ? RAPID_SIMT_MEM_PATTERN_FULL_BROADCAST
                                     : RAPID_SIMT_MEM_PATTERN_PARTIAL_BROADCAST;
      } else {
        pattern = valid_lanes == 32u ? RAPID_SIMT_MEM_PATTERN_FULL_OTHER
                                     : RAPID_SIMT_MEM_PATTERN_PARTIAL_OTHER;
      }
    }
  }

  const uint64_t logical_block = rapid_feedback_block_linear();
  const uint32_t warp_in_block = static_cast<uint32_t>(thread_linear >> 5u);
  const uint32_t first_sector = offset / RAPID_SIMT_MEM_SECTOR_BYTES;
  const uint32_t last_sector =
      (offset + access_width32 - 1u) / RAPID_SIMT_MEM_SECTOR_BYTES;

  const bool broadcast_pattern =
      pattern == RAPID_SIMT_MEM_PATTERN_FULL_BROADCAST ||
      pattern == RAPID_SIMT_MEM_PATTERN_PARTIAL_BROADCAST ||
      pattern == RAPID_SIMT_MEM_PATTERN_SINGLE;
  if (broadcast_pattern) {
    if (lane == first_lane) {
      for (uint32_t sector = first_sector; sector <= last_sector; ++sector) {
        rapid_simt_mem_emit_sector(context, site_id, arg_slot, sector,
                                   access_kind, logical_block, warp_in_block,
                                   pattern);
      }
    }
    return;
  }

  const bool contiguous_pattern =
      pattern == RAPID_SIMT_MEM_PATTERN_FULL_CONTIGUOUS ||
      pattern == RAPID_SIMT_MEM_PATTERN_PARTIAL_CONTIGUOUS;
  if (contiguous_pattern) {
    const uint32_t last_lane = 31u - static_cast<uint32_t>(__clz(valid_mask));
    const uint32_t final_offset =
        __shfl_sync(valid_mask, offset, static_cast<int>(last_lane));
    if (lane == first_lane) {
      const uint32_t final_sector =
          (final_offset + access_width32 - 1u) /
          RAPID_SIMT_MEM_SECTOR_BYTES;
      for (uint32_t sector = first_sector; sector <= final_sector; ++sector) {
        rapid_simt_mem_emit_sector(context, site_id, arg_slot, sector,
                                   access_kind, logical_block, warp_in_block,
                                   pattern);
      }
    }
    return;
  }

  const bool lane_single_sector = first_sector == last_sector;
  const uint32_t single_sector_mask =
      __ballot_sync(valid_mask, lane_single_sector);
  if (single_sector_mask == valid_mask) {
    const uint32_t same_sector_mask =
        __match_any_sync(valid_mask, first_sector);
    const uint32_t sector_leader =
        static_cast<uint32_t>(__ffs(same_sector_mask) - 1);
    if (lane == sector_leader) {
      rapid_simt_mem_emit_sector(context, site_id, arg_slot, first_sector,
                                 access_kind, logical_block, warp_in_block,
                                 pattern);
    }
    return;
  }

  rapid_simt_mem_emit_scatter_ranges(
      context, site_id, arg_slot, access_kind, logical_block, warp_in_block,
      pattern, valid_mask, lane, first_sector, last_sector);
}

__device__ inline void
rapid_feedback_record_thread_value(RapidKernelContext *context, uint32_t kind,
                                   uint32_t value) {
  const uint32_t key = rapid_feedback_mix32(kind * 0x9e3779b9u ^ value);
  rapid_feedback_set_bucket(
      context,
      THREAD_ACTIVITY_BUCKET_BASE + key % THREAD_ACTIVITY_BUCKETS);
}

__device__ inline void
rapid_feedback_record_thread_activity(RapidKernelContext *context) {
  if (context == nullptr ||
      (threadIdx.x + blockIdx.x * blockDim.x) != 0u) {
    return;
  }

  const uint64_t physical_threads =
      static_cast<uint64_t>(gridDim.x) * gridDim.y * gridDim.z * blockDim.x *
      blockDim.y * blockDim.z;
  const uint32_t capped_threads =
      physical_threads > UINT32_MAX ? UINT32_MAX
                                    : static_cast<uint32_t>(physical_threads);
  const uint32_t physical_warps = (capped_threads + 31u) / 32u;

  rapid_feedback_record_thread_value(context, 0u, capped_threads);
  rapid_feedback_record_thread_value(context, 1u, physical_warps);
  rapid_feedback_record_thread_value(context, 2u, capped_threads % 32u != 0u);
  rapid_feedback_record_thread_value(context, 3u, gridDim.x);
  rapid_feedback_record_thread_value(context, 4u, gridDim.y);
  rapid_feedback_record_thread_value(context, 5u, gridDim.z);
  rapid_feedback_record_thread_value(context, 6u, blockDim.x);
  rapid_feedback_record_thread_value(context, 7u, blockDim.y);
  rapid_feedback_record_thread_value(context, 8u, blockDim.z);
}

#endif /* __UTILS_FEEDBACK_CUH__ */
