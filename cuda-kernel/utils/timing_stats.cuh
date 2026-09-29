#ifndef __TIMING_STATS_CUH__
#define __TIMING_STATS_CUH__

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "cuda_utils.cuh"

#ifndef ENABLE_KERNEL_TIMING
#define ENABLE_KERNEL_TIMING 0
#endif

// Timing-only buckets shared by serial and persistent backend wrappers.
struct KernelTimingStats {
  uint64_t idle_cycles;           // Waiting for work
  uint64_t init_cov_cycles;       // bind/clear task feedback maps
  uint64_t decode_cycles;         // input envelope/argument decoding
  uint64_t feedback_prepare_cycles;  // feedback prepare/thread activity
  uint64_t kernel_cycles;         // target invocation only
  uint64_t merge_cov_cycles;      // merge_shared_coverage
  uint64_t signal_cycles;         // ready/done or gpu_signal
  uint64_t sync_overhead_cycles;  // leftover bookkeeping per iteration
  uint64_t iterations;
};

inline constexpr uint64_t KERNEL_TIMING_RESET_REQUEST = ~uint64_t{0};
inline constexpr uint64_t KERNEL_TIMING_SNAPSHOT_REQUEST =
    KERNEL_TIMING_RESET_REQUEST - 1;

// Print timing stats in both cycles and per-iteration microseconds.
// Uses the current device's SM clock rate for conversion.
inline void print_timing_stats(const char *label,
                               const KernelTimingStats &stats,
                               int device_id = -1) {
#if ENABLE_KERNEL_TIMING
  fprintf(stderr, "[%s] timing stats (cycles):\n", label);
  fprintf(stderr, "  iterations: %llu\n",
          static_cast<unsigned long long>(stats.iterations));

  if (stats.iterations == 0) {
    return;
  }

  int resolved_device = device_id;
  if (resolved_device < 0) {
    CUDA_CHECK(cudaGetDevice(&resolved_device));
  }

  cudaDeviceProp prop{};
  double clock_hz = 0.0;
  if (cudaGetDeviceProperties(&prop, resolved_device) == cudaSuccess &&
      prop.clockRate > 0) {
    clock_hz = static_cast<double>(prop.clockRate) * 1000.0; // kHz -> Hz
  }

  auto to_us_per_iter = [&](uint64_t cycles) -> double {
    if (clock_hz <= 0.0) {
      return 0.0;
    }
    double total_us = (static_cast<double>(cycles) / clock_hz) * 1e6;
    return total_us / static_cast<double>(stats.iterations);
  };

  auto avg_cycles = [&](uint64_t cycles) -> double {
    return static_cast<double>(cycles) / static_cast<double>(stats.iterations);
  };

  fprintf(stderr,
          "  idle       : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.idle_cycles),
          avg_cycles(stats.idle_cycles), to_us_per_iter(stats.idle_cycles));
  fprintf(stderr,
          "  init_cov   : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.init_cov_cycles),
          avg_cycles(stats.init_cov_cycles),
          to_us_per_iter(stats.init_cov_cycles));
  fprintf(stderr,
          "  decode     : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.decode_cycles),
          avg_cycles(stats.decode_cycles), to_us_per_iter(stats.decode_cycles));
  fprintf(stderr,
          "  feedback   : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.feedback_prepare_cycles),
          avg_cycles(stats.feedback_prepare_cycles),
          to_us_per_iter(stats.feedback_prepare_cycles));
  fprintf(stderr,
          "  kernel     : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.kernel_cycles),
          avg_cycles(stats.kernel_cycles), to_us_per_iter(stats.kernel_cycles));
  fprintf(stderr,
          "  merge_cov  : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.merge_cov_cycles),
          avg_cycles(stats.merge_cov_cycles),
          to_us_per_iter(stats.merge_cov_cycles));
  fprintf(stderr,
          "  signal     : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.signal_cycles),
          avg_cycles(stats.signal_cycles),
          to_us_per_iter(stats.signal_cycles));
  fprintf(stderr,
          "  sync_extra : %llu (avg %.1f cycles | %.3f us per iteration)\n",
          static_cast<unsigned long long>(stats.sync_overhead_cycles),
          avg_cycles(stats.sync_overhead_cycles),
          to_us_per_iter(stats.sync_overhead_cycles));
#else
  (void)label;
  (void)stats;
  (void)device_id;
#endif
}

#endif // __TIMING_STATS_CUH__
