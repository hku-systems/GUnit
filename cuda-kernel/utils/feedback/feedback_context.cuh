#ifndef __UTILS_FEEDBACK_CONTEXT_CUH__
#define __UTILS_FEEDBACK_CONTEXT_CUH__

#include "feedback_constants.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifndef RAPID_PAYLOAD_SLOT_COUNT
#error "rapid_target_layout.v1.h must be included before feedback_context.cuh"
#endif

#ifndef RAPID_PAYLOAD_SLOT_STORAGE_COUNT
#error "rapid target layout must define RAPID_PAYLOAD_SLOT_STORAGE_COUNT"
#endif

struct RapidVConfig {
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
};

struct RapidPayloadBounds {
  uintptr_t base;
  uint64_t len_bytes;
};

struct RapidFeedbackTaskContext {
  uintptr_t simt_memcov_bits_addr;
  uint32_t bounds_count;
  RapidPayloadBounds bounds[RAPID_PAYLOAD_SLOT_STORAGE_COUNT];
};

struct RapidKernelContext {
  RapidVConfig vconfig;
  RapidFeedbackTaskContext feedback;
};

static_assert(sizeof(uintptr_t) == 8u,
              "RAPID feedback ABI requires 64-bit pointers");
static_assert(sizeof(RapidVConfig) == 24u,
              "RapidVConfig ABI size changed");
static_assert(sizeof(RapidPayloadBounds) == 16u,
              "RapidPayloadBounds ABI size changed");
static_assert(offsetof(RapidFeedbackTaskContext, bounds) == 16u,
              "RapidFeedbackTaskContext bounds offset changed");
static_assert(sizeof(RapidFeedbackTaskContext) ==
                  16u + 16u * RAPID_PAYLOAD_SLOT_STORAGE_COUNT,
              "RapidFeedbackTaskContext ABI size changed");
static_assert(offsetof(RapidKernelContext, feedback) == 24u,
              "RapidKernelContext feedback offset changed");
static_assert(sizeof(RapidKernelContext) ==
                  40u + 16u * RAPID_PAYLOAD_SLOT_STORAGE_COUNT,
              "RapidKernelContext ABI size changed");
static_assert(std::is_standard_layout_v<RapidKernelContext>,
              "RapidKernelContext must remain standard layout");
static_assert(std::is_trivially_copyable_v<RapidKernelContext>,
              "RapidKernelContext must remain trivially copyable");

#endif /* __UTILS_FEEDBACK_CONTEXT_CUH__ */
