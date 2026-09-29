#ifndef __RAPID_TASK_DATA_H__
#define __RAPID_TASK_DATA_H__

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "coverage/coverage_constants.h"
#include "feedback/feedback_constants.h"
#include "sanitizer/canary.h"
#include "status/run_status.h"

struct RapidTaskData {
  using Clock = std::chrono::steady_clock;

  uint64_t task_id;
  uintptr_t input_ptr;
  std::unique_ptr<uint8_t[]> envelope;
  size_t envelope_size;
  size_t payload_size;
  std::unique_ptr<uint8_t[]> output;
  std::array<uint8_t, MAP_SIZE> coverage{};
  std::array<uint8_t, SIMT_MEMCOV_STORAGE_SIZE> simt_memcov{};
  LibAflRunStatus status = libafl_run_status_ok();
  Clock::time_point enqueue_time = Clock::now();
  Clock::time_point dispatch_time{};
  Clock::time_point complete_time{};
  bool has_dispatch_time = false;

  RapidTaskData(uint64_t id, uintptr_t submitted_input_ptr,
                const uint8_t *submitted_envelope, size_t submitted_size,
                size_t submitted_payload_size)
      : task_id(id),
        input_ptr(submitted_input_ptr),
        envelope(new uint8_t[std::max<size_t>(
            rapid::canary::guarded_size(submitted_size), 1)]),
        envelope_size(submitted_size),
        payload_size(submitted_payload_size),
        output(new uint8_t[std::max<size_t>(submitted_payload_size, 1)]) {
    if (submitted_size > 0) {
      std::memcpy(envelope.get(), submitted_envelope, submitted_size);
    }
    rapid::canary::initialize(envelope.get() + submitted_size);
    std::memset(output.get(), 0, std::max<size_t>(submitted_payload_size, 1));
  }
};

#endif  // __RAPID_TASK_DATA_H__
