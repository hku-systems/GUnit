#ifndef __UTILS_FEEDBACK_CONSTANTS_H__
#define __UTILS_FEEDBACK_CONSTANTS_H__

#include <stdint.h>

#define SIMT_MEMCOV_BUCKETS 65536u
#define SIMT_MEMCOV_STORAGE_SIZE (SIMT_MEMCOV_BUCKETS / 8u)
#define SIMT_MEMCOV_DATA_BUCKETS 61440u
#define THREAD_ACTIVITY_BUCKET_BASE 61440u
#define THREAD_ACTIVITY_BUCKETS 4096u
#define RAPID_MAX_PAYLOAD_SLOTS 32u
#define RAPID_SIMT_MEM_SECTOR_BYTES 32u

enum RapidSimtMemPattern : uint8_t {
  RAPID_SIMT_MEM_PATTERN_SINGLE = 0u,
  RAPID_SIMT_MEM_PATTERN_FULL_BROADCAST = 1u,
  RAPID_SIMT_MEM_PATTERN_FULL_CONTIGUOUS = 2u,
  RAPID_SIMT_MEM_PATTERN_FULL_OTHER = 3u,
  RAPID_SIMT_MEM_PATTERN_PARTIAL_BROADCAST = 4u,
  RAPID_SIMT_MEM_PATTERN_PARTIAL_CONTIGUOUS = 5u,
  RAPID_SIMT_MEM_PATTERN_PARTIAL_OTHER = 6u,
};

#define RAPID_SIMT_MEM_ARG_SLOT_SALT 0x9e3779b9u
#define RAPID_SIMT_MEM_SECTOR_LOW_SALT 0x85ebca6bu
#define RAPID_SIMT_MEM_SECTOR_HIGH_SALT 0xc2b2ae35u
#define RAPID_SIMT_MEM_ACCESS_KIND_SALT 0x27d4eb2fu
#define RAPID_SIMT_MEM_LOGICAL_BLOCK_SALT 0x165667b1u
#define RAPID_SIMT_MEM_WARP_IN_BLOCK_SALT 0xd3a2646du
#define RAPID_SIMT_MEM_PATTERN_SALT 0xfd7046c5u

static_assert(SIMT_MEMCOV_BUCKETS % 32u == 0u,
              "feedback bitset must contain complete 32-bit words");
static_assert(SIMT_MEMCOV_DATA_BUCKETS + THREAD_ACTIVITY_BUCKETS ==
                  SIMT_MEMCOV_BUCKETS,
              "feedback bucket partitions must cover the complete bitset");
static_assert(RAPID_SIMT_MEM_SECTOR_BYTES == 32u,
              "SIMT MemCov sector size changed");
static_assert(RAPID_SIMT_MEM_PATTERN_SINGLE == 0u &&
                  RAPID_SIMT_MEM_PATTERN_FULL_BROADCAST == 1u &&
                  RAPID_SIMT_MEM_PATTERN_FULL_CONTIGUOUS == 2u &&
                  RAPID_SIMT_MEM_PATTERN_FULL_OTHER == 3u &&
                  RAPID_SIMT_MEM_PATTERN_PARTIAL_BROADCAST == 4u &&
                  RAPID_SIMT_MEM_PATTERN_PARTIAL_CONTIGUOUS == 5u &&
                  RAPID_SIMT_MEM_PATTERN_PARTIAL_OTHER == 6u,
              "SIMT MemCov pattern encoding changed");
static_assert((RAPID_SIMT_MEM_ARG_SLOT_SALT & 1u) != 0u &&
                  (RAPID_SIMT_MEM_SECTOR_LOW_SALT & 1u) != 0u &&
                  (RAPID_SIMT_MEM_SECTOR_HIGH_SALT & 1u) != 0u &&
                  (RAPID_SIMT_MEM_ACCESS_KIND_SALT & 1u) != 0u &&
                  (RAPID_SIMT_MEM_LOGICAL_BLOCK_SALT & 1u) != 0u &&
                  (RAPID_SIMT_MEM_WARP_IN_BLOCK_SALT & 1u) != 0u &&
                  (RAPID_SIMT_MEM_PATTERN_SALT & 1u) != 0u,
              "SIMT MemCov hash salts must remain odd");

extern "C" {
__attribute__((visibility("default"), aligned(4096))) extern uint8_t
    libafl_simt_memcov_bits[SIMT_MEMCOV_STORAGE_SIZE];
}

#endif /* __UTILS_FEEDBACK_CONSTANTS_H__ */
