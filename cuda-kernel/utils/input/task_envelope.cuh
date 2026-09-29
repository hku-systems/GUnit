#ifndef __UTILS_INPUT_TASK_ENVELOPE_CUH__
#define __UTILS_INPUT_TASK_ENVELOPE_CUH__

#include "feedback/feedback_context.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__CUDA__)
#define RAPID_HOST_DEVICE __host__ __device__
#else
#define RAPID_HOST_DEVICE
#endif

struct RapidTaskEnvelopeHeader {
  RapidVConfig vconfig;
  uint64_t payload_size;
};

RAPID_HOST_DEVICE inline uint8_t *
rapid_task_payload(RapidTaskEnvelopeHeader *envelope) {
  return reinterpret_cast<uint8_t *>(envelope + 1);
}

RAPID_HOST_DEVICE inline const uint8_t *
rapid_task_payload(const RapidTaskEnvelopeHeader *envelope) {
  return reinterpret_cast<const uint8_t *>(envelope + 1);
}

RAPID_HOST_DEVICE inline size_t rapid_task_envelope_size(size_t payload_size) {
  return sizeof(RapidTaskEnvelopeHeader) + payload_size;
}

static_assert(sizeof(RapidTaskEnvelopeHeader) == 32u,
              "RapidTaskEnvelopeHeader ABI size changed");
static_assert(offsetof(RapidTaskEnvelopeHeader, payload_size) == 24u,
              "RapidTaskEnvelopeHeader payload offset changed");
static_assert(std::is_standard_layout_v<RapidTaskEnvelopeHeader>,
              "RapidTaskEnvelopeHeader must remain standard layout");
static_assert(std::is_trivially_copyable_v<RapidTaskEnvelopeHeader>,
              "RapidTaskEnvelopeHeader must remain trivially copyable");

#undef RAPID_HOST_DEVICE

#endif /* __UTILS_INPUT_TASK_ENVELOPE_CUH__ */
