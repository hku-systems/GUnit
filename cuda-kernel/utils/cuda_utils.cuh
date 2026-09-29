#ifndef __UTILS_CUDA_UTILS_CUH__
#define __UTILS_CUDA_UTILS_CUH__

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "status/run_status.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error:\n");                                        \
      fprintf(stderr, "    File:       %s\n", __FILE__);                       \
      fprintf(stderr, "    Line:       %d\n", __LINE__);                       \
      fprintf(stderr, "    Error code: %d\n", error_code);                     \
      fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code)); \
      fflush(stderr);                                                          \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// Kernel launch check macro
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

// Async version for rapid2 - aborts if task is null, otherwise sets error info
// This allows graceful error handling when a task context is available
#define CUDA_CHECK_ASYNC_STAGE(call, task, rapid_stage)                        \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error:\n");                                        \
      fprintf(stderr, "    File:       %s\n", __FILE__);                       \
      fprintf(stderr, "    Line:       %d\n", __LINE__);                       \
      fprintf(stderr, "    Error code: %d\n", error_code);                     \
      fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code)); \
      fflush(stderr);                                                          \
      if (task) {                                                              \
        task->has_error = true;                                                \
        task->error_code = static_cast<uint32_t>(error_code);                  \
        task->status = libafl_run_status_cuda_error(                           \
            rapid_stage, static_cast<uint32_t>(error_code));                   \
      } else {                                                                 \
        abort();  /* Fallback to abort if no task context */                   \
      }                                                                        \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_ASYNC(call, task)                                           \
  CUDA_CHECK_ASYNC_STAGE(call, task, LIBAFL_BACKEND_STAGE_UNKNOWN)

#endif /* __UTILS_CUDA_UTILS_CUH__ */
