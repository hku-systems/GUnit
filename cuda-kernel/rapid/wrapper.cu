#include "rapid_target_layout.v1.h"

#ifndef RAPID_ENABLE_INNER_FEEDBACK
#define RAPID_ENABLE_INNER_FEEDBACK 1
#endif

#if RAPID_ENABLE_INNER_FEEDBACK
#include "coverage/coverage.cuh"
#include "feedback/feedback.cuh"
#endif
#include "input/task_envelope.cuh"

#include <cstddef>
#include <cstdint>

#include "signal.cuh"
#include "utils/timing_stats.cuh"

#ifndef FUZZER_INVOKE_HEADER
#error "FUZZER_INVOKE_HEADER must point to a generated invoke header"
#endif

#include FUZZER_INVOKE_HEADER

#if ENABLE_KERNEL_TIMING
__device__ volatile KernelTimingStats g_rapid_timing_stats;

__device__ inline void publish_rapid_timing_stats(
    const KernelTimingStats &stats) {
  g_rapid_timing_stats.idle_cycles = stats.idle_cycles;
  g_rapid_timing_stats.init_cov_cycles = stats.init_cov_cycles;
  g_rapid_timing_stats.decode_cycles = stats.decode_cycles;
  g_rapid_timing_stats.feedback_prepare_cycles =
      stats.feedback_prepare_cycles;
  g_rapid_timing_stats.kernel_cycles = stats.kernel_cycles;
  g_rapid_timing_stats.merge_cov_cycles = stats.merge_cov_cycles;
  g_rapid_timing_stats.signal_cycles = stats.signal_cycles;
  g_rapid_timing_stats.sync_overhead_cycles = stats.sync_overhead_cycles;
  __threadfence_system();
  g_rapid_timing_stats.iterations = stats.iterations;
}
#endif

extern "C" __global__ void rapid_persistent_kernel(
    RapidTaskEnvelopeHeader *envelope,
    volatile int *data_ready_sem, volatile int *data_processed_sem,
    volatile bool *run, RapidKernelContext *context) {
  Semaphore semaphore = {data_ready_sem, data_processed_sem, run};

#if ENABLE_KERNEL_TIMING
  int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ KernelTimingStats timing_stats;
  __shared__ bool timing_reset_requested;
  if (global_tid == 0) {
    timing_stats = KernelTimingStats{};
    timing_reset_requested = false;
  }
  __syncthreads();
#endif

#if RAPID_ENABLE_INNER_FEEDBACK
  rapid_bind_coverage_map(device_cov_map);
#endif

  while (*run) {
#if ENABLE_KERNEL_TIMING
    uint64_t iter_start = clock64();
    uint64_t wait_start = clock64();
#endif
    gpu_wait(semaphore);
    // stop_persistent_kernel wakes gpu_wait so the block can exit. Do not
    // decode and execute the previous envelope after that lifecycle wakeup.
    if (!*run) {
      break;
    }
#if ENABLE_KERNEL_TIMING
    uint64_t wait_end = clock64();
    if (global_tid == 0) {
      timing_reset_requested =
          g_rapid_timing_stats.iterations == KERNEL_TIMING_RESET_REQUEST;
      if (timing_reset_requested) {
        timing_stats = KernelTimingStats{};
        publish_rapid_timing_stats(timing_stats);
      }
    }
    __syncthreads();
    if (timing_reset_requested) {
      gpu_signal(semaphore);
      continue;
    }
    uint64_t init_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
    rapid_feedback_clear_task_maps(context);
#endif
#if ENABLE_KERNEL_TIMING
    uint64_t init_end = clock64();
    uint64_t decode_start = clock64();
#endif
    volatile uint8_t *input = rapid_task_payload(envelope);
    const size_t size = static_cast<size_t>(envelope->payload_size);
    DecodedKernelArgs decoded = fuzzer_decode_v1(input, size);
#if ENABLE_KERNEL_TIMING
    __syncthreads();
    uint64_t decode_end = clock64();
    uint64_t feedback_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
    fuzzer_feedback_prepare_v1(decoded, envelope->vconfig, context);
    __syncthreads();
    rapid_feedback_record_thread_activity(context);
#endif
#if ENABLE_KERNEL_TIMING
    __syncthreads();
    uint64_t feedback_end = clock64();
    uint64_t kernel_start = clock64();
#endif
    fuzzer_invoke_v1(decoded, context);
    // The completion semaphore is published by thread 0. Keep this barrier
    // even without feedback so all target threads finish writing the payload.
    __syncthreads();
#if ENABLE_KERNEL_TIMING
    uint64_t kernel_end = clock64();
    uint64_t merge_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
    merge_shared_coverage();
#endif
#if ENABLE_KERNEL_TIMING
    uint64_t merge_end = clock64();
    uint64_t signal_start = clock64();
    if (global_tid == 0) {
      int *data_ready = const_cast<int *>(semaphore.data_ready_sem);
      __threadfence_system();
      atomicExch(data_ready, 0);
    }
    __syncthreads();
    uint64_t signal_end = clock64();
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
      publish_rapid_timing_stats(stats);
      // The completion flag is the ordered backend's publication point. Keep
      // it last so a completed task always has a matching timing snapshot.
      int *data_processed = const_cast<int *>(semaphore.data_processed_sem);
      __threadfence_system();
      atomicExch(data_processed, 1);
    }
#else
    gpu_signal(semaphore);
#endif
  }

#if ENABLE_KERNEL_TIMING
  if (global_tid == 0) {
    publish_rapid_timing_stats(timing_stats);
  }
#endif
}
