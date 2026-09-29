#ifndef __SINGLE_KERNEL_PIPELINE_CUH__
#define __SINGLE_KERNEL_PIPELINE_CUH__

#include "config.h"
#include "combined_input.cuh"
#include "cuda_utils.cuh"
#include "double_buffer_sync.cuh"
#include "feedback/feedback_context.cuh"
#include "kernel_backend_api.cuh"
#include "utils/timing_stats.cuh"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <stdexcept>

#ifndef RAPID_LAUNCH_CONFIG_HEADER
#define RAPID_LAUNCH_CONFIG_HEADER "launch/default_launch_config.h"
#endif

#include RAPID_LAUNCH_CONFIG_HEADER
#include "launch/launch_config.cuh"

struct SubmittedEnvelope {
  RapidTaskEnvelopeHeader header{};
  const uint8_t *payload = nullptr;
  size_t payload_size = 0;
};

inline bool parseSubmittedEnvelope(const uint8_t *input, size_t size,
                                   SubmittedEnvelope *out) {
  if (input == nullptr || out == nullptr ||
      size < sizeof(RapidTaskEnvelopeHeader)) {
    return false;
  }
  RapidTaskEnvelopeHeader header{};
  std::memcpy(&header, input, sizeof(RapidTaskEnvelopeHeader));
  const size_t payload_size = static_cast<size_t>(header.payload_size);
  if (payload_size > MAX_INPUT_SIZE ||
      rapid_task_envelope_size(payload_size) != size) {
    return false;
  }
  out->header = header;
  out->payload = input + sizeof(RapidTaskEnvelopeHeader);
  out->payload_size = payload_size;
  return true;
}

inline bool submitted_envelope_payload_size(const uint8_t *input, size_t size,
                                            size_t *out_payload_size) {
  SubmittedEnvelope envelope{};
  if (!parseSubmittedEnvelope(input, size, &envelope)) {
    return false;
  }
  if (out_payload_size) {
    *out_payload_size = envelope.payload_size;
  }
  return true;
}

inline bool rapid2_skip_coverage_copy_from_env() {
  const char *value = std::getenv("RAPID2_SKIP_COVERAGE_COPY");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

// Task data structure
struct TaskData {
  // Task identifier
  uint64_t task_id;

  // Pointer value of the input buffer passed by the caller at submission time.
  // This is used as an opaque identifier for tracking corpus input reuse across
  // the submit->complete lifecycle.
  uintptr_t input_ptr = 0;

  // Combined input data (size + data in one structure)
  HostCombinedInputPtr h_combined;
  size_t h_combined_capacity = 0;
  size_t payload_size = 0;

  // Task-owned edge and memory/index feedback in one pinned allocation.
  HostFeedbackBuffer h_feedback;

  // Timestamps
  std::chrono::time_point<std::chrono::steady_clock> enqueue_time;
  std::chrono::time_point<std::chrono::steady_clock> submit_time;
  std::chrono::time_point<std::chrono::steady_clock> dispatch_time;
  std::chrono::time_point<std::chrono::steady_clock> complete_time;
  bool has_dispatch_time = false;

  // Error tracking
  bool has_error = false;
  uint32_t error_code = 0;
  LibAflRunStatus status = libafl_run_status_ok();

  // Constructor
  TaskData(uint64_t id, const uint8_t *input, size_t size)
      : task_id(id), has_error(false), error_code(0),
        status(libafl_run_status_ok()) {
    input_ptr = reinterpret_cast<uintptr_t>(input);
    SubmittedEnvelope envelope{};
    if (!parseSubmittedEnvelope(input, size, &envelope)) {
      throw std::invalid_argument("invalid RAPID task envelope");
    }
    // Always allocate the full envelope capacity. Reallocating a pinned host
    // buffer on the submit path would free the old allocation with
    // cudaFreeHost(), which implicitly synchronizes the whole device and can
    // therefore deadlock against the never-ending persistent kernel.
    h_combined = HostCombinedInput::allocate(MAX_INPUT_SIZE);
    h_combined_capacity = MAX_INPUT_SIZE;
    std::memcpy(&h_combined->header, &envelope.header,
                sizeof(RapidTaskEnvelopeHeader));
    std::memcpy(h_combined->data, envelope.payload, envelope.payload_size);
    payload_size = envelope.payload_size;
    rapid::canary::initialize(h_combined->data + envelope.payload_size);
    h_feedback = HostFeedbackBuffer::allocate();
    enqueue_time = std::chrono::steady_clock::now();
    submit_time = enqueue_time;
  }

  void reset(uint64_t id, const uint8_t *input, size_t size) {
    task_id = id;
    input_ptr = reinterpret_cast<uintptr_t>(input);
    SubmittedEnvelope envelope{};
    if (!parseSubmittedEnvelope(input, size, &envelope)) {
      throw std::invalid_argument("invalid RAPID task envelope");
    }

    if (!h_combined || envelope.payload_size > h_combined_capacity) {
      // See the constructor: pinned buffers are allocated at full capacity so
      // this path never frees a pinned allocation while the persistent kernel
      // is running (cudaFreeHost() implies a device-wide synchronize).
      h_combined = HostCombinedInput::allocate(MAX_INPUT_SIZE);
      h_combined_capacity = MAX_INPUT_SIZE;
    }
    std::memcpy(&h_combined->header, &envelope.header,
                sizeof(RapidTaskEnvelopeHeader));
    std::memcpy(h_combined->data, envelope.payload, envelope.payload_size);
    payload_size = envelope.payload_size;
    rapid::canary::initialize(h_combined->data + envelope.payload_size);

    if (!h_feedback) {
      h_feedback = HostFeedbackBuffer::allocate();
    }
    enqueue_time = std::chrono::steady_clock::now();
    submit_time = enqueue_time;
    has_dispatch_time = false;
    has_error = false;
    error_code = 0;
    status = libafl_run_status_ok();
  }

  void markDispatched() {
    dispatch_time = std::chrono::steady_clock::now();
    has_dispatch_time = true;
  }

  void markTimeout(uint32_t elapsed_ms) {
    has_error = true;
    error_code = elapsed_ms;
    status =
        libafl_run_status_timeout(LIBAFL_BACKEND_STAGE_EXECUTE, elapsed_ms);
  }

  void markCudaError(LibAflRunBackendStage stage, uint32_t detail) {
    has_error = true;
    error_code = detail;
    status = libafl_run_status_cuda_error(stage, detail);
  }

  void clampVConfig(const rapid_launch::LaunchConfig &launch_config) {
    h_combined->header.vconfig = rapid_launch::clamp_vconfig_to_launch_bounds(
        h_combined->header.vconfig, launch_config);
  }

  // Move semantics
  TaskData(TaskData &&) = default;
  TaskData &operator=(TaskData &&) = default;

  // Disable copy
  TaskData(const TaskData &) = delete;
  TaskData &operator=(const TaskData &) = delete;
};

struct ActiveTaskTimeoutReport {
  uint64_t task_id = 0;
  int kernel_id = -1;
  int buffer_idx = -1;
  uint64_t age_ms = 0;
  uint64_t timeout_ms = 0;
};

enum class SlotState : uint8_t {
  Empty,
  Dispatching,
  Active,
  Collecting,
  Poisoned,
};

// Single kernel pipeline manager
class SingleKernelPipeline {
public:
  // Constructor - initialize double buffer sync mechanism
  SingleKernelPipeline(int kernel_id = 0)
      : kernel_id_(kernel_id),
        skip_coverage_copy_(rapid2_skip_coverage_copy_from_env()),
        initialized_(false) {
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      slot_states_[slot] = SlotState::Empty;
    }
    // Will be initialized in initialize() method
  }

  // Destructor - cleanup resources and sync mechanism
  ~SingleKernelPipeline() {
    if (initialized_) {
      shutdown();
      cleanup();
    }
  }

  // Initialize the pipeline
  void initialize() {
    if (initialized_)
      return;

    if (skip_coverage_copy_) {
      std::fprintf(stderr,
                   "[RAPID2] RAPID2_SKIP_COVERAGE_COPY=1: coverage D2H "
                   "copies disabled\n");
    }

    // Allocate physical slots on device.
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      CUDA_CHECK(cudaMalloc(&d_combined_input_[slot], sizeof(CombinedInput)));
#if RAPID_ENABLE_INNER_FEEDBACK
      CUDA_CHECK(cudaMalloc(&d_feedback_[slot], HostFeedbackBuffer::kSize));
      d_coverage_[slot] = d_feedback_[slot];
      d_simt_memcov_bits_[slot] = d_feedback_[slot] + MAP_SIZE;
#endif
      CUDA_CHECK(cudaMalloc(&d_context_[slot], sizeof(RapidKernelContext)));
#if RAPID_ENABLE_INNER_FEEDBACK
      CUDA_CHECK(cudaMemset(d_feedback_[slot], 0, HostFeedbackBuffer::kSize));
#endif
      initializeContext(static_cast<int>(slot));
    }
    // Create CUDA streams
    CUDA_CHECK(cudaStreamCreate(&kernel_stream_));
    CUDA_CHECK(cudaStreamCreate(&dispatch_stream_));
    CUDA_CHECK(cudaStreamCreate(&collect_stream_));
#if ENABLE_KERNEL_TIMING
    h_timing_stats_ = allocatePinnedHostArray<KernelTimingStats>(1);
#endif
#if RAPID2_USE_STREAM_COMPLETION
    CUdevice current_device = 0;
    int supports_stream_wait64 = 0;
    if (cuCtxGetDevice(&current_device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&supports_stream_wait64,
                             CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                             current_device) != CUDA_SUCCESS ||
        supports_stream_wait64 == 0) {
      std::fprintf(stderr,
                   "[RAPID2] 64-bit stream memory operations are required "
                   "for stream completion mode\n");
      std::abort();
    }
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      CUDA_CHECK(cudaEventCreateWithFlags(&dispatch_ready_events_[slot],
                                          cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&completion_events_[slot],
                                          cudaEventDisableTiming));
    }
#endif

    // Configure kernel. Pick the largest candidate that satisfies the loaded
    // persistent wrapper's per-block register/thread/shared-memory limits.
    // Per-input VConfig is supplied by the fuzzer task envelope and clamped to
    // the actual persistent launch before H2D copy.
    rapid2_kernel_backend::PersistentKernelLaunchConfig launch_config{};
    const size_t target_dynamic_shared_bytes =
        rapid_launch_config::kTargetDynamicSharedBytes;
    if (!rapid2_kernel_backend::select_persistent_launch_config(
            &launch_config, rapid_launch_config::kGrid,
            rapid_launch_config::kBlockCandidates,
            target_dynamic_shared_bytes)) {
      fprintf(stderr, "[RAPID2] Failed to select persistent launch config\n");
      abort();
    }
    grid_dim_ = launch_config.grid_dim;
    block_dim_ = launch_config.block_dim;
    shared_mem_size_ = launch_config.shared_mem_size;
    launch_config_.grid_dim = launch_config.grid_dim;
    launch_config_.block_dim = launch_config.block_dim;
    launch_config_.shared_mem_size = launch_config.shared_mem_size;
    launch_config_.physical_block_threads =
        launch_config.physical_block_threads;
    if (rapid_launch_config::kHasLogicalVConfigBounds &&
        !rapid_launch::apply_logical_vconfig_bounds(
            &launch_config_, rapid_launch_config::kLogicalGrid,
            rapid_launch_config::kLogicalBlock, "RAPID2")) {
      fprintf(stderr, "[RAPID2] Failed to apply logical VConfig bounds\n");
      abort();
    }

    initialized_ = true;
  }

  // Launch the persistent kernel
  void launch() {
    if (!initialized_ || kernel_launched_.load(std::memory_order_acquire))
      return;

    // Get sync pointers
    DeviceSyncPtrs sync_ptrs = sync_.getDevicePtrs();

    rapid2_kernel_backend::PersistentKernelLaunchParams launch_params{};
    launch_params.grid_dim = grid_dim_;
    launch_params.block_dim = block_dim_;
    launch_params.shared_mem_size = shared_mem_size_;
    launch_params.kernel_stream = kernel_stream_;
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      launch_params.d_combined_input[slot] = d_combined_input_[slot];
      launch_params.d_coverage[slot] = d_coverage_[slot];
      launch_params.d_context[slot] = d_context_[slot];
    }
    launch_params.sync_ptrs = sync_ptrs;
    launch_params.kernel_id = kernel_id_;

    if (!rapid2_kernel_backend::launch_persistent_kernel(launch_params)) {
      fprintf(stderr, "[RAPID2] Failed to launch persistent kernel\n");
      abort();
    }

    accepting_tasks_ = true;
    kernel_launched_.store(true, std::memory_order_release);
  }

  // Wait for the next double-buffer slot and submit under one lock acquisition.
  bool waitAndSubmitTask(std::unique_ptr<TaskData> task) {
    if (!initialized_ || !task) {
      return false;
    }

    std::unique_lock<std::mutex> lock(task_mutex_);
    buffer_available_cv_.wait(lock, [this] {
      return !accepting_tasks_ ||
             (kernel_launched_.load(std::memory_order_acquire) &&
              slot_states_[next_submit_buffer_] == SlotState::Empty);
    });
    if (!accepting_tasks_ ||
        !kernel_launched_.load(std::memory_order_acquire)) {
      if (kernel_launched_.load(std::memory_order_acquire)) {
        // cudaFreeHost may wait for the persistent kernel. Keep this one
        // dispatcher-owned task alive until process teardown, matching the
        // terminal-result lifetime policy.
        (void)task.release();
      }
      return false;
    }
    const int buffer_idx = next_submit_buffer_;
    assert(slot_states_[buffer_idx] == SlotState::Empty);
    // done[slot] retains the previous generation. Equality here means the
    // host/device task-id generation sequence has already diverged.
    assert(!sync_.checkBufferDone(buffer_idx, task->task_id));

    buffer_tasks_[buffer_idx] = std::move(task);
    TaskData *const task_ptr = buffer_tasks_[buffer_idx].get();
    slot_states_[buffer_idx] = SlotState::Dispatching;
    lock.unlock();

    // Start NVTX range for the entire buffer lifecycle
    NVTX_BEGIN_LIFECYCLE(sync_.getNVTXProfiler(), buffer_idx,
                         task_ptr->task_id);

    // Direct copy from task's combined input - no extra memcpy needed
    task_ptr->clampVConfig(launch_config_);
    CUDA_CHECK_ASYNC_STAGE(
        cudaMemcpyAsync(d_combined_input_[buffer_idx],
                        task_ptr->h_combined.get(),
                        task_ptr->h_combined->getTransferSize(),
                        cudaMemcpyHostToDevice, dispatch_stream_),
        task_ptr, LIBAFL_BACKEND_STAGE_SUBMIT);

    if (!task_ptr->has_error) {
      task_ptr->markDispatched();
#if RAPID2_USE_STREAM_COMPLETION
      const CUresult ready_result = sync_.enqueueBufferReady(
          buffer_idx, task_ptr->task_id, dispatch_stream_);
      if (ready_result != CUDA_SUCCESS) {
        const char *error_text = nullptr;
        (void)cuGetErrorString(ready_result, &error_text);
        std::fprintf(stderr,
                     "[RAPID2] cuStreamWriteValue64 failed: %s (%d)\n",
                     error_text ? error_text : "unknown driver error",
                     static_cast<int>(ready_result));
        task_ptr->markCudaError(LIBAFL_BACKEND_STAGE_SUBMIT,
                                static_cast<uint32_t>(ready_result));
      }
#else
      sync_.signalBufferReady(buffer_idx, task_ptr->task_id,
                              dispatch_stream_);
#endif
#if RAPID2_USE_STREAM_COMPLETION
      if (!task_ptr->has_error) {
        CUDA_CHECK_ASYNC_STAGE(
            cudaEventRecord(dispatch_ready_events_[buffer_idx],
                            dispatch_stream_),
            task_ptr, LIBAFL_BACKEND_STAGE_SUBMIT);
      }
      if (!task_ptr->has_error) {
        CUDA_CHECK_ASYNC_STAGE(
            cudaStreamWaitEvent(collect_stream_,
                                dispatch_ready_events_[buffer_idx], 0),
            task_ptr, LIBAFL_BACKEND_STAGE_COLLECT);
      }
      if (!task_ptr->has_error) {
        const CUresult wait_result = sync_.enqueueCompletionWait(
            buffer_idx, task_ptr->task_id, collect_stream_);
        if (wait_result != CUDA_SUCCESS) {
          const char *error_text = nullptr;
          (void)cuGetErrorString(wait_result, &error_text);
          std::fprintf(stderr,
                       "[RAPID2] cuStreamWaitValue64 failed: %s (%d)\n",
                       error_text ? error_text : "unknown driver error",
                       static_cast<int>(wait_result));
          task_ptr->markCudaError(LIBAFL_BACKEND_STAGE_COLLECT,
                                  static_cast<uint32_t>(wait_result));
        }
      }
      if (!task_ptr->has_error) {
        enqueueResultCopies(task_ptr, buffer_idx);
      }
      if (!task_ptr->has_error) {
        CUDA_CHECK_ASYNC_STAGE(
            cudaEventRecord(completion_events_[buffer_idx], collect_stream_),
            task_ptr, LIBAFL_BACKEND_STAGE_COLLECT);
      }
      if (!task_ptr->has_error) {
        completion_enqueued_[buffer_idx].store(true,
                                                std::memory_order_release);
      }
#endif
    }

    lock.lock();
    assert(slot_states_[buffer_idx] == SlotState::Dispatching);
    assert(buffer_tasks_[buffer_idx].get() == task_ptr);
    slot_states_[buffer_idx] = SlotState::Active;
    const unsigned slot_bit = 1U << static_cast<unsigned>(buffer_idx);
    active_generations_[buffer_idx].store(task_ptr->task_id,
                                          std::memory_order_release);
    if (task_ptr->has_error) {
      error_slots_.fetch_or(slot_bit, std::memory_order_release);
    }
    active_slots_.fetch_or(slot_bit, std::memory_order_release);

    // Update statistics
    tasks_submitted_++;

    // Switch to next buffer for next submission
    next_submit_buffer_ =
        (next_submit_buffer_ + 1) & static_cast<int>(kRapid2SlotMask);
    lock.unlock();
#if RAPID2_USE_STREAM_COMPLETION
    active_slot_cv_.notify_one();
#endif

    return true;
  }

  // Check for completed tasks - returns ownership of completed task
  std::unique_ptr<TaskData> checkCompletedTask() {
    if (!initialized_ || !kernel_launched_.load(std::memory_order_acquire))
      return nullptr;

    const int buffer_idx = next_collect_buffer_;
#if RAPID2_USE_STREAM_COMPLETION
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      active_slot_cv_.wait(lock, [this, buffer_idx] {
        return !accepting_tasks_ ||
               slot_states_[buffer_idx] == SlotState::Active;
      });
      if (slot_states_[buffer_idx] != SlotState::Active) {
        return nullptr;
      }
    }
#endif
    const unsigned slot_bit = 1U << static_cast<unsigned>(buffer_idx);
    if ((active_slots_.load(std::memory_order_acquire) & slot_bit) == 0) {
      return nullptr;
    }

    bool force_complete =
        (error_slots_.load(std::memory_order_acquire) & slot_bit) != 0;
    const uint64_t active_generation =
        active_generations_[buffer_idx].load(std::memory_order_acquire);
    if (active_generation == 0) {
      return nullptr;
    }
    cudaError_t stream_err = cudaSuccess;
    LibAflRunBackendStage stream_error_stage =
        LIBAFL_BACKEND_STAGE_EXECUTE;
#if RAPID2_USE_STREAM_COMPLETION
    if (!force_complete) {
      if (!completion_enqueued_[buffer_idx].load(std::memory_order_acquire)) {
        return nullptr;
      }
      stream_err = cudaEventSynchronize(completion_events_[buffer_idx]);
      const bool device_done =
          sync_.checkBufferDone(buffer_idx, active_generation);
      if (stream_err == cudaSuccess && !device_done) {
        stream_err = cudaErrorUnknown;
      }
      if (device_done) {
        stream_error_stage = LIBAFL_BACKEND_STAGE_COLLECT;
      }
      force_complete = stream_err != cudaSuccess;
    }
#else
    const bool device_done =
        sync_.checkBufferDone(buffer_idx, active_generation);
    if (!device_done && !force_complete) {
      ++stream_query_poll_count_;
      if ((stream_query_poll_count_ & 0x0fffU) != 0) {
        return nullptr;
      }
      stream_err = cudaStreamQuery(kernel_stream_);
      if (stream_err == cudaSuccess || stream_err == cudaErrorNotReady) {
        return nullptr;
      }
      force_complete = true;
    }
#endif

    TaskData *task = nullptr;
    {
      std::lock_guard<std::mutex> lock(task_mutex_);
      if (slot_states_[buffer_idx] != SlotState::Active) {
        return nullptr;
      }
      task = buffer_tasks_[buffer_idx].get();
      if (!task) {
        active_slots_.fetch_and(~slot_bit, std::memory_order_release);
        error_slots_.fetch_and(~slot_bit, std::memory_order_release);
        active_generations_[buffer_idx].store(0, std::memory_order_release);
#if RAPID2_USE_STREAM_COMPLETION
        completion_enqueued_[buffer_idx].store(false,
                                                std::memory_order_release);
#endif
        slot_states_[buffer_idx] = SlotState::Empty;
        next_collect_buffer_ =
            (next_collect_buffer_ + 1) & static_cast<int>(kRapid2SlotMask);
        buffer_available_cv_.notify_one();
        return nullptr;
      }
      slot_states_[buffer_idx] = SlotState::Collecting;
      active_slots_.fetch_and(~slot_bit, std::memory_order_release);
      error_slots_.fetch_and(~slot_bit, std::memory_order_release);
      active_generations_[buffer_idx].store(0, std::memory_order_release);
#if RAPID2_USE_STREAM_COMPLETION
      completion_enqueued_[buffer_idx].store(false,
                                              std::memory_order_release);
#endif
    }

    if (stream_err != cudaSuccess) {
      CUDA_CHECK_ASYNC_STAGE(stream_err, task, stream_error_stage);
    }

    // Force complete if the kernel reports an error before it can signal
    // completion.
    force_complete = force_complete || task->has_error;
    if (force_complete) {
      kernel_launched_.store(false, std::memory_order_release);
    }

    // Record buffer completion event for Nsight Systems visualization
    // This marks the end of processing for this buffer with task_id validation
    sync_.recordBufferDone(buffer_idx, task->task_id);

#if !RAPID2_USE_STREAM_COMPLETION
    // Polling mode schedules copies only after the host observes device done.
    enqueueResultCopies(task, buffer_idx);

    // Wait for copy to complete
    CUDA_CHECK_ASYNC_STAGE(cudaStreamSynchronize(collect_stream_), task,
                           LIBAFL_BACKEND_STAGE_COLLECT);
#endif

    if (!task->has_error) {
      rapid::canary::Mismatch mismatch{};
      const uint8_t *const canary =
          task->h_combined->data + task->payload_size;
      if (!rapid::canary::validate(canary, &mismatch)) {
        rapid::canary::report("RAPID2", task->task_id, mismatch);
        task->markCudaError(LIBAFL_BACKEND_STAGE_COLLECT,
                            rapid::canary::kStatusDetail);
      }
    }

    // Never publish feedback bytes left over from a previous pooled task when
    // dispatch, execution, or collection failed.
    if (task->has_error) {
      std::memset(task->h_feedback.data(), 0, task->h_feedback.size());
    }

    // End NVTX range for the entire buffer lifecycle
    NVTX_END_LIFECYCLE(sync_.getNVTXProfiler(), buffer_idx);

    std::unique_ptr<TaskData> completed_task;
    {
      std::lock_guard<std::mutex> lock(task_mutex_);
      assert(slot_states_[buffer_idx] == SlotState::Collecting);
      assert(buffer_tasks_[buffer_idx].get() == task);
      if (task->status.code == LIBAFL_RUN_STATUS_CUDA_ERROR ||
          task->status.code == LIBAFL_RUN_STATUS_TIMEOUT ||
          task->status.code == LIBAFL_RUN_STATUS_INTERNAL_ERROR) {
        accepting_tasks_ = false;
      }
      completed_task = std::move(buffer_tasks_[buffer_idx]);
      slot_states_[buffer_idx] = SlotState::Empty;
      next_collect_buffer_ =
          (next_collect_buffer_ + 1) & static_cast<int>(kRapid2SlotMask);
    }
    buffer_available_cv_.notify_one();

    // Update timestamp
    completed_task->complete_time = std::chrono::steady_clock::now();

    // Update statistics
    tasks_completed_++;

    return completed_task;
  }

  // Shutdown the pipeline
  void shutdown() {
    if (!initialized_ ||
        !kernel_launched_.load(std::memory_order_acquire))
      return;

    stopAcceptingTasks();

    // Signal kernel to exit
    sync_.signalExit(dispatch_stream_);

    // Wait for kernel to finish. Shutdown is a lifecycle operation; do not
    // publish cleanup errors as per-input target results.
    const cudaError_t shutdown_err = cudaStreamSynchronize(kernel_stream_);
    if (shutdown_err != cudaSuccess) {
      fprintf(stderr,
              "[RAPID2] CUDA error while shutting down persistent kernel: "
              "%s (%d)\n",
              cudaGetErrorString(shutdown_err),
              static_cast<int>(shutdown_err));
    }
    dumpTimingStats();

    kernel_launched_.store(false, std::memory_order_release);
  }

  void stopAcceptingTasks() {
    {
      std::lock_guard<std::mutex> lock(task_mutex_);
      accepting_tasks_ = false;
    }
    buffer_available_cv_.notify_all();
#if RAPID2_USE_STREAM_COMPLETION
    active_slot_cv_.notify_all();
#endif
  }

  std::unique_ptr<TaskData>
  takeTimedOutTask(std::chrono::steady_clock::time_point now,
                   std::chrono::milliseconds timeout,
                   ActiveTaskTimeoutReport *report) {
    if (!initialized_ || timeout.count() <= 0) {
      return nullptr;
    }

    std::lock_guard<std::mutex> lock(task_mutex_);
    for (int buffer_idx = 0;
         buffer_idx < static_cast<int>(kRapid2SlotCount); ++buffer_idx) {
      if (slot_states_[buffer_idx] != SlotState::Active) {
        continue;
      }
      TaskData *task = buffer_tasks_[buffer_idx].get();
      if (!task || !task->has_dispatch_time || task->has_error) {
        continue;
      }
      const auto age = now - task->dispatch_time;
      if (age < timeout) {
        continue;
      }
      const auto age_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(age).count();
      const uint32_t detail =
          static_cast<uint32_t>(age_ms > UINT32_MAX ? UINT32_MAX : age_ms);
      task->markTimeout(detail);
      task->complete_time = now;
      std::memset(task->h_feedback.data(), 0, task->h_feedback.size());
      if (report) {
        report->task_id = task->task_id;
        report->kernel_id = kernel_id_;
        report->buffer_idx = buffer_idx;
        report->age_ms = static_cast<uint64_t>(age_ms);
        report->timeout_ms = static_cast<uint64_t>(timeout.count());
      }

      // The persistent kernel may still be executing the timed-out target.
      // Publish a host-side timeout result for fuzzer feedback, but do not
      // clear the device done flag or make this buffer reusable in this
      // process. Rust aborts the client after consuming the fatal result.
      kernel_launched_.store(false, std::memory_order_release);
      const unsigned slot_bit = 1U << static_cast<unsigned>(buffer_idx);
      active_slots_.fetch_and(~slot_bit, std::memory_order_release);
      error_slots_.fetch_and(~slot_bit, std::memory_order_release);
      active_generations_[buffer_idx].store(0, std::memory_order_release);
#if RAPID2_USE_STREAM_COMPLETION
      completion_enqueued_[buffer_idx].store(false,
                                              std::memory_order_release);
#endif
      slot_states_[buffer_idx] = SlotState::Poisoned;
      return std::move(buffer_tasks_[buffer_idx]);
    }
    return nullptr;
  }

  // Check if a buffer is available for submission
  bool isBufferAvailable(int buffer_idx) const {
    if (buffer_idx < 0 || static_cast<size_t>(buffer_idx) >= kRapid2SlotCount)
      return false;
    std::lock_guard<std::mutex> lock(task_mutex_);
    return slot_states_[buffer_idx] == SlotState::Empty;
  }

  // Get processed count
  uint64_t getProcessedCount() const { return tasks_completed_.load(); }

  // Get submitted count
  uint64_t getSubmittedCount() const { return tasks_submitted_.load(); }

  // Copy and print timing stats from the device.
  void dumpTimingStats() {
#if ENABLE_KERNEL_TIMING
    if (!h_timing_stats_) {
      return;
    }
    if (!rapid2_kernel_backend::copy_timing_stats_from_device(
            h_timing_stats_.get(), collect_stream_)) {
      return;
    }
    const KernelTimingStats stats = *h_timing_stats_;
    char label[128];
    snprintf(label, sizeof(label),
             "rapid2/double_buffered_persistent_kernel[%d]", kernel_id_);
    print_timing_stats(label, stats);
#endif
  }

  bool resetTimingStats() {
#if ENABLE_KERNEL_TIMING
    if (!h_timing_stats_) {
      return false;
    }
    if (!rapid2_kernel_backend::reset_timing_stats_on_device(
            h_timing_stats_.get(), dispatch_stream_)) {
      return false;
    }
#endif
    return true;
  }

  bool copyTimingStats(KernelTimingStats *out) {
#if ENABLE_KERNEL_TIMING
    if (out == nullptr || !h_timing_stats_) {
      return false;
    }
    if (!rapid2_kernel_backend::snapshot_timing_stats_from_device(
            h_timing_stats_.get(), collect_stream_)) {
      return false;
    }
    *out = *h_timing_stats_;
    return true;
#else
    (void)out;
    return false;
#endif
  }

  bool coverageCopyEnabled() const { return !skip_coverage_copy_; }

private:
  void enqueueResultCopies(TaskData *task, int buffer_idx) {
    if (skip_coverage_copy_) {
      CUDA_CHECK_ASYNC_STAGE(
          cudaMemcpyAsync(task->h_combined.get(),
                          d_combined_input_[buffer_idx],
                          sizeof(RapidTaskEnvelopeHeader),
                          cudaMemcpyDeviceToHost, collect_stream_),
          task, LIBAFL_BACKEND_STAGE_COLLECT);
      if constexpr (rapid::canary::kStorageSize > 0) {
        if (!task->has_error) {
          CUDA_CHECK_ASYNC_STAGE(
              cudaMemcpyAsync(
                  task->h_combined->data + task->payload_size,
                  d_combined_input_[buffer_idx]->data + task->payload_size,
                  rapid::canary::kStorageSize, cudaMemcpyDeviceToHost,
                  collect_stream_),
              task, LIBAFL_BACKEND_STAGE_COLLECT);
        }
      }
    } else {
      CUDA_CHECK_ASYNC_STAGE(
          cudaMemcpyAsync(task->h_combined.get(),
                          d_combined_input_[buffer_idx],
                          task->h_combined->getTransferSize(),
                          cudaMemcpyDeviceToHost, collect_stream_),
          task, LIBAFL_BACKEND_STAGE_COLLECT);
    }
#if RAPID_ENABLE_INNER_FEEDBACK
    if (!task->has_error && !skip_coverage_copy_) {
      CUDA_CHECK_ASYNC_STAGE(
          cudaMemcpyAsync(task->h_feedback.data(), d_feedback_[buffer_idx],
                          HostFeedbackBuffer::kSize, cudaMemcpyDeviceToHost,
                          collect_stream_),
          task, LIBAFL_BACKEND_STAGE_COLLECT);
    }
#endif
  }

  void initializeContext(int buffer_idx) {
    RapidKernelContext context{};
#if RAPID_ENABLE_INNER_FEEDBACK
    context.feedback.simt_memcov_bits_addr =
        reinterpret_cast<uintptr_t>(d_simt_memcov_bits_[buffer_idx]);
#endif
    CUDA_CHECK(cudaMemcpy(d_context_[buffer_idx], &context,
                          sizeof(RapidKernelContext),
                          cudaMemcpyHostToDevice));
  }

  // Cleanup resources
  void cleanup() {
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      if (d_combined_input_[slot])
        cudaFree(d_combined_input_[slot]);
      if (d_feedback_[slot])
        cudaFree(d_feedback_[slot]);
      if (d_context_[slot])
        cudaFree(d_context_[slot]);
    }
    if (kernel_stream_)
      cudaStreamDestroy(kernel_stream_);
    if (dispatch_stream_)
      cudaStreamDestroy(dispatch_stream_);
    if (collect_stream_)
      cudaStreamDestroy(collect_stream_);
#if RAPID2_USE_STREAM_COMPLETION
    for (size_t slot = 0; slot < kRapid2SlotCount; ++slot) {
      if (dispatch_ready_events_[slot])
        cudaEventDestroy(dispatch_ready_events_[slot]);
      if (completion_events_[slot])
        cudaEventDestroy(completion_events_[slot]);
    }
#endif
  }

private:
  // Kernel identifier
  int kernel_id_;

  CombinedInput *d_combined_input_[RAPID2_SLOT_COUNT]{};
  uint8_t *d_feedback_[RAPID2_SLOT_COUNT]{};
  uint8_t *d_coverage_[RAPID2_SLOT_COUNT]{};
  RapidKernelContext *d_context_[RAPID2_SLOT_COUNT]{};
  uint8_t *d_simt_memcov_bits_[RAPID2_SLOT_COUNT]{};

  // Synchronization mechanism
  DoubleBufferSync sync_;

  // CUDA streams
  cudaStream_t kernel_stream_ = nullptr;
  cudaStream_t dispatch_stream_ = nullptr; // Stream for dispatch (H2D)
  cudaStream_t collect_stream_ = nullptr;  // Stream for collect (D2H)
#if ENABLE_KERNEL_TIMING
  PinnedHostPtr<KernelTimingStats> h_timing_stats_;
#endif
#if RAPID2_USE_STREAM_COMPLETION
  cudaEvent_t dispatch_ready_events_[RAPID2_SLOT_COUNT]{};
  cudaEvent_t completion_events_[RAPID2_SLOT_COUNT]{};
#endif

  // Pipeline state
  int next_submit_buffer_ = 0;
  int next_collect_buffer_ = 0;
  SlotState slot_states_[RAPID2_SLOT_COUNT]{};
  std::atomic<unsigned> active_slots_{0};
  std::atomic<unsigned> error_slots_{0};
  // Host task_id is also the device generation. This requires dense IDs from
  // 1, FIFO dispatch, one pipeline, and one submitter thread.
  std::atomic<uint64_t> active_generations_[RAPID2_SLOT_COUNT]{};
#if RAPID2_USE_STREAM_COMPLETION
  std::atomic<bool> completion_enqueued_[RAPID2_SLOT_COUNT]{};
#endif
  uint32_t stream_query_poll_count_ = 0;
  std::unique_ptr<TaskData> buffer_tasks_[RAPID2_SLOT_COUNT];
  mutable std::mutex task_mutex_;
  std::condition_variable buffer_available_cv_;
#if RAPID2_USE_STREAM_COMPLETION
  std::condition_variable active_slot_cv_;
#endif

  // Statistics
  std::atomic<uint64_t> tasks_submitted_{0};
  std::atomic<uint64_t> tasks_completed_{0};

  // Configuration
  const bool skip_coverage_copy_;
  dim3 grid_dim_{1};
  dim3 block_dim_{1024};
  size_t shared_mem_size_;
  rapid_launch::LaunchConfig launch_config_{};

  // State flags
  bool initialized_ = false;
  std::atomic<bool> kernel_launched_{false};
  bool accepting_tasks_ = false;
};

#endif // __SINGLE_KERNEL_PIPELINE_CUH__
