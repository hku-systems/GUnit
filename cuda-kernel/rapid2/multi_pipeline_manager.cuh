#ifndef __MULTI_PIPELINE_MANAGER_CUH__
#define __MULTI_PIPELINE_MANAGER_CUH__

#include "config.h"
#include "coverage/coverage_constants.h"
#include "kernel_backend_api.cuh"
#include "single_kernel_pipeline.cuh"
#include "task_queue.cuh"
#include <atomic>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <pthread.h> // For pthread_setname_np
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

// Multi-pipeline manager for multiple kernel instances
class MultiPipelineManager {
public:
  using PendingQueueT = Rapid2TaskQueue<TaskData>;
  using CompletedQueueT = Rapid2CompletedQueue<TaskData>;

  explicit MultiPipelineManager(int num_kernels = 1,
                                size_t queue_capacity = QUEUE_CAPACITY)
      : num_kernels_(num_kernels), queue_capacity_(queue_capacity) {
    if (num_kernels_ != 1) {
      throw std::invalid_argument("Rapid2 requires exactly one pipeline");
    }
    pending_queue_ = std::make_unique<PendingQueueT>(queue_capacity_);
    completed_queue_ = std::make_unique<CompletedQueueT>(queue_capacity_);
  }

  void setTargetTimeoutMs(uint64_t timeout_ms) {
    target_timeout_ms_.store(timeout_ms, std::memory_order_release);
  }

  uint64_t targetTimeoutMs() const {
    return target_timeout_ms_.load(std::memory_order_acquire);
  }

  // Destructor
  ~MultiPipelineManager() {
    if (running_ ||
        fatal_completion_published_.load(std::memory_order_acquire)) {
      stop();
    }
  }

  // Initialize pipelines
  void initialize(size_t max_input_size = MAX_INPUT_SIZE) {
    if (initialized_)
      return;

    // The device-side CombinedInput uses a fixed-size buffer of
    // MAX_INPUT_SIZE, so clamp any larger user-supplied value.
    max_input_size_ = std::min(max_input_size, static_cast<size_t>(MAX_INPUT_SIZE));

    // Create pipeline instances
    for (int i = 0; i < num_kernels_; i++) {
      auto pipeline = std::make_unique<SingleKernelPipeline>(i);
      pipeline->initialize();
      pipelines_.push_back(std::move(pipeline));
    }

    // Initialize global coverage map
    memset(libafl_cov_map, 0, MAP_SIZE);
#if RAPID_ENABLE_INNER_FEEDBACK
    if (!rapid2_kernel_backend::initialize_global_coverage(libafl_cov_map)) {
      fprintf(stderr, "[RAPID2] Failed to initialize device coverage map\n");
      abort();
    }
#endif

    initialized_ = true;
  }

  // Start the manager
  void start() {
    if (!initialized_ || running_)
      return;

    running_ = true;
    fatal_completion_published_.store(false, std::memory_order_release);

    // Launch all persistent kernels
    for (auto &pipeline : pipelines_) {
      pipeline->launch();
    }

    watchdog_running_.store(true, std::memory_order_release);
    watchdog_thread_ = std::thread(&MultiPipelineManager::watchdogLoop, this);

    // Start dispatcher and collector threads for each pipeline
    for (int i = 0; i < num_kernels_; i++) {
      dispatcher_threads_.emplace_back(&MultiPipelineManager::dispatcherLoop,
                                       this, i);
      collector_threads_.emplace_back(&MultiPipelineManager::collectorLoop,
                                      this, i);
    }
  }

  // Stop the manager
  void stop() {
    if (!running_ &&
        !fatal_completion_published_.load(std::memory_order_acquire))
      return;

    {
      std::lock_guard<std::mutex> lock(completion_mutex_);
      running_ = false;
    }
    completion_cv_.notify_all();
    watchdog_running_.store(false, std::memory_order_release);

    // Shutdown queues
    pending_queue_->shutdown();
    completed_queue_->shutdown();
    for (auto &pipeline : pipelines_) {
      pipeline->stopAcceptingTasks();
    }

    // Join all threads
    for (auto &thread : dispatcher_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (auto &thread : collector_threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    if (watchdog_thread_.joinable()) {
      watchdog_thread_.join();
    }

    // Shutdown all pipelines
    for (auto &pipeline : pipelines_) {
      pipeline->shutdown();
    }
  }

  // Submit input to the queue
  uint64_t submitInput(const uint8_t *input, size_t size) {
    size_t payload_size = 0;
    if (!running_ ||
        fatal_completion_published_.load(std::memory_order_acquire) ||
        target_timeout_ms_.load(std::memory_order_acquire) == 0 ||
        !submitted_envelope_payload_size(input, size, &payload_size) ||
        payload_size > max_input_size_) {
      return 0;
    }

    uint64_t task_id = next_task_id_++;

    try {
      std::unique_ptr<TaskData> task;
      {
        std::lock_guard<std::mutex> lock(reusable_mutex_);
        if (!reusable_tasks_.empty()) {
          task = std::move(reusable_tasks_.back());
          reusable_tasks_.pop_back();
        }
      }

      if (task) {
        task->reset(task_id, input, size);
      } else {
        task = std::make_unique<TaskData>(task_id, input, size);
      }

      if (pending_queue_->push(std::move(task))) {
        total_submitted_++;
        return task_id;
      }
    } catch (...) {
      // Dense task IDs require one submission thread; roll back a failed
      // allocation before the exception is reported at the C ABI boundary.
      next_task_id_--;
      throw;
    }
    // This rollback is safe only because submitInput has one calling thread.
    next_task_id_--; // Revert task ID if push failed
    return 0;        // Failed to submit
  }

  // Wait for all tasks to complete
  void waitAll() {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this] {
      return total_completed_.load(std::memory_order_acquire) >=
                 total_submitted_.load(std::memory_order_acquire) ||
             fatal_completion_published_.load(std::memory_order_acquire);
    });
  }

  bool resetTimingStats() {
    waitAll();
    for (auto &pipeline : pipelines_) {
      if (!pipeline->resetTimingStats()) {
        return false;
      }
    }
    profile_timing_baseline_ = total_completed_.load(std::memory_order_acquire);
    return true;
  }

  bool copyTimingStats(KernelTimingStats *out) {
    waitAll();
    if (out == nullptr || pipelines_.empty()) {
      return false;
    }
    const uint64_t expected =
        total_completed_.load(std::memory_order_acquire) -
        profile_timing_baseline_;
    if (!pipelines_.front()->copyTimingStats(out)) {
      return false;
    }
    return out->iterations >= expected;
  }

  void waitForCompletion() {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this] {
      return !completed_queue_->empty() ||
             fatal_completion_published_.load(std::memory_order_acquire) ||
             !running_.load(std::memory_order_acquire);
    });
  }

  // Try to pop a completed task (non-blocking)
  std::unique_ptr<TaskData> tryPopCompleted() {
    return completed_queue_->try_pop();
  }

  // Acquire a completed task for external consumption (e.g. via FFI).
  //
  // This transfers ownership out of the completed queue into an internal
  // outstanding map so we can safely hand out pointers to task-owned buffers.
  // The caller must later release it via releaseCompleted(task_id).
  TaskData *tryAcquireCompleted() {
    auto task = completed_queue_->try_pop();
    if (!task) {
      return nullptr;
    }

    TaskData *raw = task.get();
    {
      std::lock_guard<std::mutex> lock(outstanding_mutex_);
      outstanding_tasks_.emplace(raw->task_id, std::move(task));
    }
    return raw;
  }

  // Release a previously acquired completed task.
  bool releaseCompleted(uint64_t task_id) {
    std::unique_ptr<TaskData> task;
    {
      std::lock_guard<std::mutex> lock(outstanding_mutex_);
      auto it = outstanding_tasks_.find(task_id);
      if (it == outstanding_tasks_.end()) {
        return false;
      }
      task = std::move(it->second);
      outstanding_tasks_.erase(it);
    }

    if (task->status.code == LIBAFL_RUN_STATUS_CUDA_ERROR ||
        task->status.code == LIBAFL_RUN_STATUS_TIMEOUT ||
        task->status.code == LIBAFL_RUN_STATUS_INTERNAL_ERROR) {
      // The persistent kernel may still be executing in a poisoned CUDA
      // context. Destroying TaskData here would call cudaFreeHost for its
      // pinned result buffers, which can block forever waiting for that
      // kernel. The consumer aborts immediately after releasing a terminal
      // result, so leave these buffers for process teardown instead.
      (void)task.release();
      return true;
    }

    // Do not free task-owned buffers. Keep them for future reuse.
    {
      std::lock_guard<std::mutex> lock(reusable_mutex_);
      reusable_tasks_.push_back(std::move(task));
    }
    return true;
  }

  // Statistics
  struct Statistics {
    uint64_t total_submitted;
    uint64_t total_completed;
    uint64_t pending_queue_size;
    uint64_t completed_queue_size;
    int active_kernels;
  };

  Statistics getStatistics() const {
    Statistics stats;
    stats.total_submitted = total_submitted_.load();
    stats.total_completed = total_completed_.load();
    stats.pending_queue_size = pending_queue_->size();
    stats.completed_queue_size = completed_queue_->size();
    stats.active_kernels = num_kernels_;
    return stats;
  }

  bool isRunning() const { return running_; }

private:
  bool publishTimedOutTask(ActiveTaskTimeoutReport *report) {
    if (fatal_completion_published_.load(std::memory_order_acquire)) {
      return false;
    }

    const uint64_t timeout_ms =
        target_timeout_ms_.load(std::memory_order_acquire);
    if (timeout_ms == 0) {
      return false;
    }

    const auto timeout = std::chrono::milliseconds(timeout_ms);
    const auto now = std::chrono::steady_clock::now();
    for (auto &pipeline : pipelines_) {
      if (!pipeline) {
        continue;
      }

      std::unique_ptr<TaskData> timed_out_task =
          pipeline->takeTimedOutTask(now, timeout, report);
      if (!timed_out_task) {
        continue;
      }

      if (!completed_queue_->push(std::move(timed_out_task))) {
        return false;
      }
      {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        total_completed_.fetch_add(1, std::memory_order_release);
        // Publish stop/fatal only after the timeout result is durable. A
        // completion waiter must never wake before that result is pollable.
        fatal_completion_published_.store(true, std::memory_order_release);
        running_ = false;
      }
      pending_queue_->shutdown();
      for (auto &active_pipeline : pipelines_) {
        active_pipeline->stopAcceptingTasks();
      }
      completion_cv_.notify_all();
      return true;
    }
    return false;
  }

  void watchdogLoop() {
    pthread_setname_np(pthread_self(), "RAPID2-Watchdog");
    while (watchdog_running_.load(std::memory_order_acquire)) {
      ActiveTaskTimeoutReport report{};
      if (publishTimedOutTask(&report)) {
        std::fprintf(stderr,
                     "RAPID2 BACKEND WATCHDOG TIMEOUT: task_id=%lu "
                     "kernel_id=%d buffer=%d age_ms=%lu timeout_ms=%lu. "
                     "Published timeout completion for fuzzer feedback; the "
                     "client will abort after processing it so the restarting "
                     "manager can create a fresh CUDA context.\n",
                     report.task_id, report.kernel_id, report.buffer_idx,
                     report.age_ms, report.timeout_ms);
        std::fflush(stderr);
        watchdog_running_.store(false, std::memory_order_release);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Dispatcher loop - assigns tasks to a specific pipeline
  void dispatcherLoop(int pipeline_id) {
    // Set thread name for debugging
    char thread_name[16];
    snprintf(thread_name, sizeof(thread_name), "RAPID2-Disp-%d", pipeline_id);
    pthread_setname_np(pthread_self(), thread_name);

    auto &pipeline = pipelines_[pipeline_id];

    while (running_) {
      // Get a task from pending queue
      auto task = pending_queue_->pop();
      if (!task)
        break; // Shutdown

      if (!pipeline->waitAndSubmitTask(std::move(task))) {
        break;
      }
    }
  }

  // Collector loop - collects completed tasks from a specific pipeline
  void collectorLoop(int pipeline_id) {
    // Set thread name for debugging
    char thread_name[16];
    snprintf(thread_name, sizeof(thread_name), "RAPID2-Coll-%d", pipeline_id);
    pthread_setname_np(pthread_self(), thread_name);

    auto &pipeline = pipelines_[pipeline_id];
    unsigned empty_polls = 0;

    while (running_) {
      // Check this specific pipeline for completed tasks
      std::unique_ptr<TaskData> completed_task = pipeline->checkCompletedTask();
      if (completed_task) {
        const uint16_t status_code = completed_task->status.code;
        const bool fatal_completion =
            status_code == LIBAFL_RUN_STATUS_CUDA_ERROR ||
            status_code == LIBAFL_RUN_STATUS_TIMEOUT ||
            status_code == LIBAFL_RUN_STATUS_INTERNAL_ERROR;
        // Merge coverage immediately so it doesn't depend on draining the
        // bounded completed queue.
#if RAPID2_MERGE_COVERAGE_TO_GLOBAL
        if (pipeline->coverageCopyEnabled()) {
          mergeCoverageFromTask(completed_task.get());
        }
#endif
        // Move completed task to completed queue - ownership transferred
        if (!completed_queue_->push(std::move(completed_task))) {
          if (!running_.load(std::memory_order_acquire)) {
            return;
          }
          std::fprintf(stderr,
                       "RAPID2 INTERNAL ERROR: failed to publish a completed "
                       "task while the manager is running.\n");
          std::fflush(stderr);
          std::abort();
        }
        {
          std::lock_guard<std::mutex> lock(completion_mutex_);
          total_completed_.fetch_add(1, std::memory_order_release);
          if (fatal_completion) {
            fatal_completion_published_.store(true,
                                              std::memory_order_release);
            running_ = false;
          }
        }
        if (fatal_completion) {
          pending_queue_->shutdown();
          for (auto &active_pipeline : pipelines_) {
            active_pipeline->stopAcceptingTasks();
          }
          watchdog_running_.store(false, std::memory_order_release);
        }
        completion_cv_.notify_all();
        if (fatal_completion) {
          return;
        }
        empty_polls = 0;
      } else {
#if defined(__i386__) || defined(__x86_64__)
        _mm_pause();
#endif
        if (++empty_polls >= 256) {
          std::this_thread::yield();
          empty_polls = 0;
        }
      }
    }
  }

  // Merge coverage from a completed task
  void mergeCoverageFromTask(TaskData *task) {
    if (!task || !task->h_feedback) {
      return;
    }

    uint8_t *const dst = libafl_cov_map;
    const uint8_t *const src = task->h_feedback.edgeData();

#if defined(__AVX2__) && !defined(__CUDA_ARCH__)
    // Vectorized unsigned byte-wise max with "store-if-changed" to reduce
    // cache write traffic once coverage saturates.
    size_t i = 0;
    for (; i + 32 <= MAP_SIZE; i += 32) {
      const __m256i vsrc =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
      const __m256i vdst =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dst + i));
      const __m256i vmax = _mm256_max_epu8(vdst, vsrc);

      const __m256i veq = _mm256_cmpeq_epi8(vmax, vdst);
      if (_mm256_movemask_epi8(veq) != -1) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), vmax);
      }
    }
    for (; i < MAP_SIZE; ++i) {
      const uint8_t s = src[i];
      const uint8_t d = dst[i];
      dst[i] = (s > d) ? s : d;
    }
#else
    // Scalar fallback. Written branchlessly for better auto-vectorization.
    for (size_t i = 0; i < MAP_SIZE; ++i) {
      const uint8_t s = src[i];
      const uint8_t d = dst[i];
      dst[i] = (s > d) ? s : d;
    }
#endif
  }

private:
  // Configuration
  int num_kernels_;
  size_t queue_capacity_;
  size_t max_input_size_ = MAX_INPUT_SIZE;
  std::atomic<uint64_t> target_timeout_ms_{0};


  // Pipeline instances
  std::vector<std::unique_ptr<SingleKernelPipeline>> pipelines_;

  // Task queues - use abstract interface for runtime polymorphism
  std::unique_ptr<ITaskQueue<TaskData>> pending_queue_;
  std::unique_ptr<ITaskQueue<TaskData>> completed_queue_;

  // Worker threads - one dispatcher and collector per pipeline
  std::thread watchdog_thread_;
  std::vector<std::thread> dispatcher_threads_;
  std::vector<std::thread> collector_threads_;

  // Control flags
  std::atomic<bool> running_{false};
  std::atomic<bool> initialized_{false};
  std::atomic<bool> watchdog_running_{false};
  std::atomic<bool> fatal_completion_published_{false};

  // Statistics
  std::atomic<uint64_t> next_task_id_{1};
  std::atomic<uint64_t> total_submitted_{0};
  std::atomic<uint64_t> total_completed_{0};
  uint64_t profile_timing_baseline_{0};
  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;

  // Tasks that have been dequeued from completed_queue_ but not yet released by
  // the external consumer. This is needed to keep task-owned buffers alive when
  // the FFI returns pointers.
  mutable std::mutex outstanding_mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<TaskData>> outstanding_tasks_;

  // Pool of reusable TaskData objects. Objects are returned here on
  // releaseCompleted() and reused on subsequent submitInput() calls to avoid
  // repeated allocations. Note: input pointers supplied by the fuzzer may not
  // be stable across iterations, so this pool is intentionally not keyed by
  // input_ptr.
  mutable std::mutex reusable_mutex_;
  std::vector<std::unique_ptr<TaskData>> reusable_tasks_;
};

#endif // __MULTI_PIPELINE_MANAGER_CUH__
