#ifndef __PIPELINED_KERNEL_IMPL_CUH__
#define __PIPELINED_KERNEL_IMPL_CUH__

#include "config.h"
#include "combined_input.cuh"
#if RAPID_ENABLE_INNER_FEEDBACK
#include "coverage/coverage.cuh"
#include "feedback/feedback.cuh"
#endif
#include "utils/timing_stats.cuh"

#include <cuda_runtime.h>

// Sleep for a short time in nanoseconds
__device__ void device_sleep(int nanoseconds) {
#if __CUDA_ARCH__ >= 700
  __nanosleep(nanoseconds);
#else
  // Fallback for older architectures - simple busy wait
  clock_t start = clock();
  clock_t cycles = nanoseconds; // Approximation
  while (clock() - start < cycles)
    ;
#endif
}

#if ENABLE_KERNEL_TIMING
// Device-side timing aggregation for RAPID2 persistent kernel.
__device__ volatile KernelTimingStats g_rapid2_timing_stats;

__device__ inline void publish_rapid2_timing_stats(
    const KernelTimingStats &stats) {
  g_rapid2_timing_stats.idle_cycles = stats.idle_cycles;
  g_rapid2_timing_stats.init_cov_cycles = stats.init_cov_cycles;
  g_rapid2_timing_stats.decode_cycles = stats.decode_cycles;
  g_rapid2_timing_stats.feedback_prepare_cycles =
      stats.feedback_prepare_cycles;
  g_rapid2_timing_stats.kernel_cycles = stats.kernel_cycles;
  g_rapid2_timing_stats.merge_cov_cycles = stats.merge_cov_cycles;
  g_rapid2_timing_stats.signal_cycles = stats.signal_cycles;
  g_rapid2_timing_stats.sync_overhead_cycles = stats.sync_overhead_cycles;
  __threadfence_system();
  g_rapid2_timing_stats.iterations = stats.iterations;
}

__device__ inline bool service_rapid2_timing_request(
    KernelTimingStats &stats) {
  const uint64_t request = g_rapid2_timing_stats.iterations;
  if (request == KERNEL_TIMING_RESET_REQUEST) {
    stats = KernelTimingStats{};
  } else if (request != KERNEL_TIMING_SNAPSHOT_REQUEST) {
    return false;
  }
  publish_rapid2_timing_stats(stats);
  return true;
}
#endif

template <typename T>
__device__ inline T *rapid2_select_slot(int slot, T *slot0, T *slot1,
                                        T *slot2, T *slot3) {
#if RAPID2_SLOT_COUNT == 2
  (void)slot2;
  (void)slot3;
  return slot == 0 ? slot0 : slot1;
#else
  switch (slot) {
  case 0:
    return slot0;
  case 1:
    return slot1;
  case 2:
    return slot2;
  default:
    return slot3;
  }
#endif
}

__device__ inline void double_buffered_persistent_kernel_impl(
    CombinedInput *d_combined0, CombinedInput *d_combined1,
    CombinedInput *d_combined2, CombinedInput *d_combined3,
    uint8_t *d_coverage0, uint8_t *d_coverage1, uint8_t *d_coverage2,
    uint8_t *d_coverage3, RapidKernelContext *d_context0,
    RapidKernelContext *d_context1, RapidKernelContext *d_context2,
    RapidKernelContext *d_context3, volatile uint64_t *ready0,
    volatile uint64_t *ready1, volatile uint64_t *ready2,
    volatile uint64_t *ready3, volatile uint64_t *done0,
    volatile uint64_t *done1, volatile uint64_t *done2,
    volatile uint64_t *done3,
    volatile bool *should_exit, int kernel_id) {
  (void)kernel_id;

  __shared__ uint64_t iteration_count;
  __shared__ bool exit_requested;
#if ENABLE_KERNEL_TIMING
  __shared__ KernelTimingStats timing_stats;
  __shared__ bool timing_request_serviced;
#endif

  int global_tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (global_tid == 0) {
    iteration_count = 0;
    exit_requested = false;
#if ENABLE_KERNEL_TIMING
    timing_stats = KernelTimingStats{};
    timing_request_serviced = false;
#endif
  }
  __syncthreads();

  while (true) {
    if (global_tid == 0) {
      exit_requested = *should_exit;
    }
    __syncthreads();
    if (exit_requested) {
      break;
    }
#if ENABLE_KERNEL_TIMING
    uint64_t iter_start = clock64();
#endif

    const int current_buffer =
        static_cast<int>(iteration_count & kRapid2SlotMask);
    // This generation must equal the host task_id. That contract requires
    // dense IDs from 1, FIFO dispatch, one pipeline, and one submitter thread.
    const uint64_t current_generation = iteration_count + 1ULL;

    volatile uint64_t *current_ready = rapid2_select_slot(
        current_buffer, ready0, ready1, ready2, ready3);
    volatile uint64_t *current_done = rapid2_select_slot(
        current_buffer, done0, done1, done2, done3);
    CombinedInput *current_combined = rapid2_select_slot(
        current_buffer, d_combined0, d_combined1, d_combined2, d_combined3);
    uint8_t *current_coverage = rapid2_select_slot(
        current_buffer, d_coverage0, d_coverage1, d_coverage2, d_coverage3);
    RapidKernelContext *current_context = rapid2_select_slot(
        current_buffer, d_context0, d_context1, d_context2, d_context3);

#if ENABLE_KERNEL_TIMING
    uint64_t wait_start = clock64();
#endif
    while (threadIdx.x == 0 && *current_ready != current_generation &&
           !*should_exit
#if ENABLE_KERNEL_TIMING
           && g_rapid2_timing_stats.iterations !=
                  KERNEL_TIMING_RESET_REQUEST &&
           g_rapid2_timing_stats.iterations !=
               KERNEL_TIMING_SNAPSHOT_REQUEST
#endif
    ) {
      // device_sleep(100);
    }
    if (global_tid == 0) {
      exit_requested = *should_exit;
#if ENABLE_KERNEL_TIMING
      timing_request_serviced =
          !exit_requested && *current_ready != current_generation &&
          service_rapid2_timing_request(timing_stats);
#endif
    }
    __syncthreads();
    if (exit_requested) {
      break;
    }
#if ENABLE_KERNEL_TIMING
    if (timing_request_serviced) {
      continue;
    }
    uint64_t wait_end = clock64();
#endif

#if ENABLE_KERNEL_TIMING
    uint64_t init_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
    rapid_bind_coverage_map(current_coverage);
    rapid_feedback_clear_task_maps(current_context);
#endif
#if ENABLE_KERNEL_TIMING
    uint64_t init_end = clock64();
#endif

#if ENABLE_KERNEL_TIMING
    uint64_t decode_start = clock64();
#endif
    DecodedKernelArgs decoded =
        fuzzer_decode_v1(current_combined->data,
                         static_cast<size_t>(
                             current_combined->header.payload_size));
#if ENABLE_KERNEL_TIMING
    __syncthreads();
    uint64_t decode_end = clock64();
    uint64_t feedback_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
    fuzzer_feedback_prepare_v1(decoded, current_combined->header.vconfig,
                               current_context);
    __syncthreads();
    rapid_feedback_record_thread_activity(current_context);
#endif
#if ENABLE_KERNEL_TIMING
    __syncthreads();
    uint64_t feedback_end = clock64();
    uint64_t kernel_start = clock64();
#endif
    fuzzer_invoke_v1(decoded, current_context);

    // Thread 0 publishes the mapped done flag. This barrier is required for
    // target-output completion independently of feedback collection.
    __syncthreads();
#if ENABLE_KERNEL_TIMING
    uint64_t kernel_end = clock64();
#endif

#if ENABLE_KERNEL_TIMING
    uint64_t merge_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
    merge_shared_coverage();
#endif
#if ENABLE_KERNEL_TIMING
    uint64_t merge_end = clock64();
#endif

#if ENABLE_KERNEL_TIMING
    uint64_t signal_start = clock64();
#endif
    if (global_tid == 0) {
      __threadfence_system();
      *current_done = current_generation;
      iteration_count = current_generation;
    }
    __syncthreads();
#if ENABLE_KERNEL_TIMING
    uint64_t signal_end = clock64();
#endif

#if ENABLE_KERNEL_TIMING
    if (global_tid == 0) {
      KernelTimingStats &stats = timing_stats;
      stats.idle_cycles += wait_end - wait_start;
      stats.init_cov_cycles += init_end - init_start;
      stats.decode_cycles += decode_end - decode_start;
      stats.feedback_prepare_cycles += feedback_end - feedback_start;
      stats.kernel_cycles += kernel_end - kernel_start;
      stats.merge_cov_cycles += merge_end - merge_start;
      stats.signal_cycles += signal_end - signal_start;

      const uint64_t total_cycles = signal_end - iter_start;
      const uint64_t accounted_cycles =
          (wait_end - wait_start) + (init_end - init_start) +
          (decode_end - decode_start) +
          (feedback_end - feedback_start) + (kernel_end - kernel_start) +
          (merge_end - merge_start) +
          (signal_end - signal_start);
      if (total_cycles > accounted_cycles) {
        stats.sync_overhead_cycles += total_cycles - accounted_cycles;
      }

      stats.iterations++;
    }
#endif
  }

#if ENABLE_KERNEL_TIMING
  if (global_tid == 0) {
    publish_rapid2_timing_stats(timing_stats);
  }
#endif
}

#endif // __PIPELINED_KERNEL_IMPL_CUH__
