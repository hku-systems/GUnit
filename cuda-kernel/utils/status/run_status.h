#ifndef __UTILS_STATUS_RUN_STATUS_H__
#define __UTILS_STATUS_RUN_STATUS_H__

#include <cstddef>
#include <cstdint>

enum LibAflRunStatusCode : uint16_t {
  LIBAFL_RUN_STATUS_OK = 0,
  LIBAFL_RUN_STATUS_CUDA_ERROR = 1,
  LIBAFL_RUN_STATUS_TIMEOUT = 2,
  LIBAFL_RUN_STATUS_INVALID_INPUT = 3,
  LIBAFL_RUN_STATUS_INTERNAL_ERROR = 4,
};

enum LibAflRunBackendStage : uint16_t {
  LIBAFL_BACKEND_STAGE_UNKNOWN = 0,
  LIBAFL_BACKEND_STAGE_INIT = 1,
  LIBAFL_BACKEND_STAGE_SUBMIT = 2,
  LIBAFL_BACKEND_STAGE_LAUNCH = 3,
  LIBAFL_BACKEND_STAGE_EXECUTE = 4,
  LIBAFL_BACKEND_STAGE_COLLECT = 5,
};

struct LibAflRunStatus {
  uint16_t code;
  uint16_t stage;
  uint32_t detail;
};

static_assert(sizeof(LibAflRunStatus) == 8,
              "LibAflRunStatus ABI size changed");
static_assert(offsetof(LibAflRunStatus, code) == 0,
              "LibAflRunStatus code offset changed");
static_assert(offsetof(LibAflRunStatus, stage) == 2,
              "LibAflRunStatus stage offset changed");
static_assert(offsetof(LibAflRunStatus, detail) == 4,
              "LibAflRunStatus detail offset changed");

inline constexpr LibAflRunStatus libafl_run_status_ok() {
  return LibAflRunStatus{LIBAFL_RUN_STATUS_OK,
                         LIBAFL_BACKEND_STAGE_UNKNOWN, 0};
}

inline constexpr LibAflRunStatus
libafl_run_status_cuda_error(LibAflRunBackendStage stage, uint32_t detail) {
  return LibAflRunStatus{LIBAFL_RUN_STATUS_CUDA_ERROR, stage, detail};
}

inline constexpr LibAflRunStatus
libafl_run_status_timeout(LibAflRunBackendStage stage, uint32_t detail) {
  return LibAflRunStatus{LIBAFL_RUN_STATUS_TIMEOUT, stage, detail};
}

inline constexpr LibAflRunStatus
libafl_run_status_invalid_input(LibAflRunBackendStage stage, uint32_t detail) {
  return LibAflRunStatus{LIBAFL_RUN_STATUS_INVALID_INPUT, stage, detail};
}

inline constexpr LibAflRunStatus
libafl_run_status_internal_error(LibAflRunBackendStage stage,
                                 uint32_t detail) {
  return LibAflRunStatus{LIBAFL_RUN_STATUS_INTERNAL_ERROR, stage, detail};
}

#endif /* __UTILS_STATUS_RUN_STATUS_H__ */
