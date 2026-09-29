#include "rapid_target_layout.v1.h"

#ifndef RAPID_ENABLE_INNER_FEEDBACK
#define RAPID_ENABLE_INNER_FEEDBACK 1
#endif

#if RAPID_ENABLE_INNER_FEEDBACK
#include "coverage/coverage.cuh"
#include "feedback/feedback.cuh"
#endif
#include "input/task_envelope.cuh"
#include "utils/timing_stats.cuh"

#include <cstddef>
#include <cstdint>

#ifndef FUZZER_INVOKE_HEADER
#error "FUZZER_INVOKE_HEADER must point to a generated invoke header"
#endif

#include FUZZER_INVOKE_HEADER

#if ENABLE_KERNEL_TIMING
__device__ volatile KernelTimingStats g_origin_timing_stats;
#endif

extern "C" __global__ void origin_wrapper(RapidTaskEnvelopeHeader *envelope,
                                           RapidKernelContext *context) {
#if ENABLE_KERNEL_TIMING
  const int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_tid == 0 &&
      g_origin_timing_stats.iterations == KERNEL_TIMING_RESET_REQUEST) {
    g_origin_timing_stats.idle_cycles = 0;
    g_origin_timing_stats.init_cov_cycles = 0;
    g_origin_timing_stats.decode_cycles = 0;
    g_origin_timing_stats.feedback_prepare_cycles = 0;
    g_origin_timing_stats.kernel_cycles = 0;
    g_origin_timing_stats.merge_cov_cycles = 0;
    g_origin_timing_stats.signal_cycles = 0;
    g_origin_timing_stats.sync_overhead_cycles = 0;
    g_origin_timing_stats.iterations = 0;
  }
  __syncthreads();
  const uint64_t iteration_start = clock64();
  const uint64_t init_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
  rapid_bind_coverage_map(device_cov_map);
  rapid_feedback_clear_task_maps(context);
#endif

#if ENABLE_KERNEL_TIMING
  __syncthreads();
  const uint64_t init_end = clock64();
  const uint64_t decode_start = clock64();
#endif

  volatile uint8_t *data = rapid_task_payload(envelope);
  const size_t size = static_cast<size_t>(envelope->payload_size);
  DecodedKernelArgs decoded = fuzzer_decode_v1(data, size);
#if ENABLE_KERNEL_TIMING
  __syncthreads();
  const uint64_t decode_end = clock64();
  const uint64_t feedback_start = clock64();
#endif
#if RAPID_ENABLE_INNER_FEEDBACK
  fuzzer_feedback_prepare_v1(decoded, envelope->vconfig, context);
  __syncthreads();
  rapid_feedback_record_thread_activity(context);
#endif
#if ENABLE_KERNEL_TIMING
  __syncthreads();
  const uint64_t feedback_end = clock64();
  const uint64_t kernel_start = clock64();
#endif
  fuzzer_invoke_v1(decoded, context);

#if ENABLE_KERNEL_TIMING
  __syncthreads();
  const uint64_t kernel_end = clock64();
  const uint64_t merge_start = clock64();
#endif

#if RAPID_ENABLE_INNER_FEEDBACK
  __syncthreads();
  merge_shared_coverage();
#endif
#if ENABLE_KERNEL_TIMING
  __syncthreads();
  const uint64_t merge_end = clock64();
  if (global_tid == 0) {
    g_origin_timing_stats.init_cov_cycles += init_end - init_start;
    g_origin_timing_stats.decode_cycles += decode_end - decode_start;
    g_origin_timing_stats.feedback_prepare_cycles +=
        feedback_end - feedback_start;
    g_origin_timing_stats.kernel_cycles += kernel_end - kernel_start;
    g_origin_timing_stats.merge_cov_cycles += merge_end - merge_start;
    const uint64_t accounted = (init_end - init_start) +
                               (decode_end - decode_start) +
                               (feedback_end - feedback_start) +
                               (kernel_end - kernel_start) +
                               (merge_end - merge_start);
    const uint64_t total = merge_end - iteration_start;
    if (total > accounted) {
      g_origin_timing_stats.sync_overhead_cycles += total - accounted;
    }
    g_origin_timing_stats.iterations++;
  }
#endif
}
