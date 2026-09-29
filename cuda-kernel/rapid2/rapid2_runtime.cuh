#ifndef __RAPID2_RUNTIME_CUH__
#define __RAPID2_RUNTIME_CUH__

#include "config.h"
#include "coverage/coverage_constants.h"
#include "kernel_backend_api.cuh"
#include "multi_pipeline_manager.cuh"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <memory>

// Global multi-pipeline manager instance.
//
// NOTE: these are defined in a header and rely on the backend harness
// having a single host translation unit (harness.cpp). If that changes, move
// these to a .cpp/.cu file and keep only extern declarations here.
static std::unique_ptr<MultiPipelineManager> g_pipeline_manager;
static std::atomic<bool> g_initialized{false};
static std::atomic<bool> g_init_error_reported{false};
static std::atomic<uint64_t> g_target_timeout_ms{0};

static inline void rapid2_set_target_timeout_ms(uint64_t timeout_ms) {
  g_target_timeout_ms.store(timeout_ms, std::memory_order_release);
  if (g_pipeline_manager) {
    g_pipeline_manager->setTargetTimeoutMs(timeout_ms);
  }
}

static inline uint64_t rapid2_get_target_timeout_ms() {
  return g_target_timeout_ms.load(std::memory_order_acquire);
}

static inline bool rapid2_ensure_initialized(int num_kernels,
                                             size_t max_input_size) {
  if (num_kernels != 1) {
    std::fprintf(stderr,
                 "[RAPID2] Initialization failed: exactly one pipeline is "
                 "required (got %d)\n",
                 num_kernels);
    return false;
  }

  if (g_initialized.load(std::memory_order_acquire)) {
    return static_cast<bool>(g_pipeline_manager);
  }

  // Only one thread should perform init.
  const bool already = g_initialized.exchange(true, std::memory_order_acq_rel);
  if (already) {
    return static_cast<bool>(g_pipeline_manager);
  }

  int device_count = 0;
  const cudaError_t device_count_err = cudaGetDeviceCount(&device_count);
  if (device_count_err != cudaSuccess || device_count <= 0) {
    const bool already_reported =
        g_init_error_reported.exchange(true, std::memory_order_acq_rel);
    if (!already_reported) {
      std::fprintf(stderr, "[RAPID2] cudaGetDeviceCount failed: %s\n",
                   cudaGetErrorString(device_count_err));
    }
    g_initialized.store(false, std::memory_order_release);
    return false;
  }

  // Enable host-mapped pinned memory for zero-copy CPU<->GPU synchronization.
  //
  // This must be called before the CUDA runtime creates a context.
  // If the context is already active (for example, initialized by user code),
  // we can't change flags anymore; continue and rely on the existing config.
  const cudaError_t map_host_err = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (map_host_err == cudaErrorSetOnActiveProcess) {
    (void)cudaGetLastError();
    std::fprintf(stderr,
                 "[RAPID2] Warning: cudaSetDeviceFlags(cudaDeviceMapHost) called after CUDA context initialization; host-mapped sync may be unavailable\n");
  } else {
    if (map_host_err != cudaSuccess) {
      const bool already_reported =
          g_init_error_reported.exchange(true, std::memory_order_acq_rel);
      if (!already_reported) {
        std::fprintf(stderr, "[RAPID2] cudaSetDeviceFlags failed: %s\n",
                     cudaGetErrorString(map_host_err));
      }
      g_initialized.store(false, std::memory_order_release);
      return false;
    }
  }

  try {
    g_pipeline_manager =
        std::make_unique<MultiPipelineManager>(num_kernels, QUEUE_CAPACITY);
    g_pipeline_manager->setTargetTimeoutMs(rapid2_get_target_timeout_ms());
    g_pipeline_manager->initialize(max_input_size);
    g_pipeline_manager->start();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "[RAPID2] Initialization failed: %s\n", error.what());
    g_pipeline_manager.reset();
    g_initialized.store(false, std::memory_order_release);
    return false;
  } catch (...) {
    std::fprintf(stderr, "[RAPID2] Initialization failed: unknown exception\n");
    g_pipeline_manager.reset();
    g_initialized.store(false, std::memory_order_release);
    return false;
  }

  std::printf("[RAPID2] Pipeline system initialized with %d kernels\n",
              num_kernels);
  g_init_error_reported.store(false, std::memory_order_release);
  return true;
}

static inline void rapid2_shutdown_if_running() {
  g_target_timeout_ms.store(0, std::memory_order_release);
  if (!g_pipeline_manager) {
    return;
  }

  auto stats = g_pipeline_manager->getStatistics();
  std::printf("[RAPID2] Stopping pipeline system\n");
  std::printf("[RAPID2] Total submitted: %lu\n", stats.total_submitted);
  std::printf("[RAPID2] Total completed: %lu\n", stats.total_completed);

  g_pipeline_manager->stop();
  g_pipeline_manager.reset();
  g_initialized.store(false, std::memory_order_release);
  g_init_error_reported.store(false, std::memory_order_release);
  rapid2_kernel_backend::shutdown_backend();
}

#endif // __RAPID2_RUNTIME_CUH__
