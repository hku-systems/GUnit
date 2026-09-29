#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#if ENABLE_KERNEL_TIMING
#include <pthread.h>
#endif

#include "rapid_target_layout.v1.h"
#include "coverage/coverage_constants.h"
#include "feedback/feedback_context.cuh"
#include "input/task_envelope.cuh"
#include "ordered_completion_queue.h"
#include "rapid_task_data.h"
#include "sanitizer/canary.h"
#include "status/run_status.h"
#include "worker_idle_wait.h"
#include "utils/timing_stats.cuh"
#ifndef RAPID_LAUNCH_CONFIG_HEADER
#define RAPID_LAUNCH_CONFIG_HEADER "launch/default_launch_config.h"
#endif
#include RAPID_LAUNCH_CONFIG_HEADER
#include "launch/launch_config.cuh"

#ifndef RAPID_ENABLE_INNER_FEEDBACK
#define RAPID_ENABLE_INNER_FEEDBACK 1
#endif

extern "C" uint8_t libafl_cov_map[MAP_SIZE];

namespace rapid_embedded_ptx {
extern const unsigned char kEmbeddedPtx[];
extern const size_t kEmbeddedPtxSize;
inline constexpr const char *kKernelName = "rapid_persistent_kernel";
}  // namespace rapid_embedded_ptx

constexpr size_t QUEUE_CAPACITY = 1024;
constexpr size_t MAX_INPUT_SIZE = 4096 * 1024;
constexpr uint64_t kStatsReportInterval = 256;
constexpr auto kWorkerIdleSpinBudget = std::chrono::microseconds(100);

namespace {

using rapid_embedded_ptx::kEmbeddedPtx;
using rapid_embedded_ptx::kEmbeddedPtxSize;
using rapid_embedded_ptx::kKernelName;
using rapid_launch::apply_logical_vconfig_bounds;
using rapid_launch::clamp_vconfig_to_launch_bounds;
using rapid_launch::select_cuda_launch_config;

inline bool check_driver(CUresult status, const char *what) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &desc);
  std::fprintf(stderr, "[rapid] %s failed: %s (%s)\n", what,
               name ? name : "CUDA_ERROR_UNKNOWN",
               desc ? desc : "no description");
  return false;
}

struct SubmittedEnvelopeView {
  RapidTaskEnvelopeHeader header{};
  const uint8_t *payload = nullptr;
  size_t payload_size = 0;
};

inline bool parse_submitted_envelope(const uint8_t *input, size_t size,
                                     SubmittedEnvelopeView *out) {
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

struct TaskResult {
  uint64_t task_id;
  uintptr_t input_ptr;
  const uint8_t *edge_ptr;
  const uint8_t *simt_memcov_ptr;
  uint32_t edge_size;
  uint32_t simt_memcov_size;
  LibAflRunStatus status;
  uint64_t exec_time_ns;
};

static_assert(sizeof(TaskResult) == 56, "TaskResult ABI size changed");
static_assert(offsetof(TaskResult, edge_ptr) == 16,
              "TaskResult edge_ptr offset changed");
static_assert(offsetof(TaskResult, simt_memcov_ptr) == 24,
              "TaskResult simt_memcov_ptr offset changed");
static_assert(offsetof(TaskResult, status) == 40,
              "TaskResult status offset changed");
static_assert(offsetof(TaskResult, exec_time_ns) == 48,
              "TaskResult exec_time_ns offset changed");

struct OrderedQueueCounts {
  size_t pending;
  size_t completed;
  size_t outstanding;
};

static std::queue<std::unique_ptr<RapidTaskData>> input_queue;
static OrderedCompletionQueue<RapidTaskData> completed_queue(QUEUE_CAPACITY);
static std::mutex queue_mutex;
static std::condition_variable queue_cv;
static std::mutex capacity_mutex;
static std::condition_variable capacity_cv;
static std::condition_variable completion_cv;
static std::atomic<bool> initialized(false);
static std::atomic<bool> stop_flag(false);
static std::atomic<bool> worker_exited(false);
static std::atomic<uint64_t> pending_inputs(0);
static std::atomic<uint64_t> accepted_unreleased(0);
static std::atomic<uint64_t> submitted_inputs(0);
static std::atomic<uint64_t> completed_inputs(0);
static std::atomic<uint64_t> profile_timing_baseline(0);
static std::atomic<uint64_t> failed_inputs(0);
static std::atomic<uint64_t> next_task_id(1);
static std::atomic<uint64_t> target_timeout_ms(0);
static std::atomic<bool> reset_timing_requested(false);
static std::atomic<bool> reset_timing_completed(false);
static std::atomic<bool> reset_timing_succeeded(false);
#if ENABLE_KERNEL_TIMING
static KernelTimingStats final_timing_stats{};
static std::atomic<bool> final_timing_stats_valid(false);
#endif
static std::thread worker_thread;
static std::mutex lifecycle_mutex;
static std::mutex submit_mutex;

inline void print_rapid_feedback_stats();

inline void request_worker_stop() {
  {
    std::scoped_lock lock(queue_mutex, capacity_mutex);
    stop_flag.store(true, std::memory_order_release);
  }
  queue_cv.notify_all();
  capacity_cv.notify_all();
}

inline void publish_worker_exit() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    worker_exited.store(true, std::memory_order_release);
  }
  completion_cv.notify_all();
}

inline bool publish_completion(std::unique_ptr<RapidTaskData> task,
                               LibAflRunStatus status) {
  if (!task) {
    return false;
  }
  task->status = status;
  task->complete_time = RapidTaskData::Clock::now();
  const bool ok = status.code == LIBAFL_RUN_STATUS_OK;
  if (!completed_queue.push(std::move(task))) {
    return false;
  }
  const uint64_t completed =
      completed_inputs.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (!ok) {
    failed_inputs.fetch_add(1, std::memory_order_acq_rel);
  }
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    pending_inputs.fetch_sub(1, std::memory_order_acq_rel);
  }
  completion_cv.notify_all();
  if (completed % kStatsReportInterval == 0) {
    print_rapid_feedback_stats();
  }
  return true;
}

inline void print_rapid_feedback_stats() {
  std::fprintf(
      stderr,
      "[rapid] rapid_submitted=%llu rapid_completed=%llu rapid_pending=%llu "
      "rapid_failed=%llu rapid_completed_queued=%zu "
      "rapid_outstanding=%zu rapid_unreleased=%llu\n",
      static_cast<unsigned long long>(
          submitted_inputs.load(std::memory_order_acquire)),
      static_cast<unsigned long long>(
          completed_inputs.load(std::memory_order_acquire)),
      static_cast<unsigned long long>(
          pending_inputs.load(std::memory_order_acquire)),
      static_cast<unsigned long long>(
          failed_inputs.load(std::memory_order_acquire)),
      completed_queue.completed_size(), completed_queue.outstanding_size(),
      static_cast<unsigned long long>(
          accepted_unreleased.load(std::memory_order_acquire)));
}

inline std::vector<std::unique_ptr<RapidTaskData>> take_queued_inputs() {
  std::vector<std::unique_ptr<RapidTaskData>> tasks;
  std::lock_guard<std::mutex> lock(queue_mutex);
  tasks.reserve(input_queue.size());
  while (!input_queue.empty()) {
    tasks.push_back(std::move(input_queue.front()));
    input_queue.pop();
  }
  return tasks;
}

inline void fail_queued_inputs(LibAflRunStatus status) {
  auto tasks = take_queued_inputs();
  for (auto &task : tasks) {
    if (!publish_completion(std::move(task), status)) {
      break;
    }
  }
}

static struct {
  CUdevice device = 0;
  CUcontext ctx = nullptr;
  CUmodule module = nullptr;
  CUfunction persistent_kernel = nullptr;
  rapid_launch::LaunchConfig launch_config{};
  CUstream kernel_stream = nullptr;
  CUstream copy_stream = nullptr;
  CUdeviceptr d_envelope = 0;
  CUdeviceptr d_data_ready_sem = 0;
  CUdeviceptr d_data_processed_sem = 0;
  CUdeviceptr d_run = 0;
  CUdeviceptr d_context = 0;
  CUdeviceptr d_simt_memcov_bits = 0;
  CUdeviceptr d_cov_map = 0;
  size_t d_cov_map_size = 0;
#if ENABLE_KERNEL_TIMING
  CUdeviceptr d_timing_stats = 0;
  size_t d_timing_stats_size = 0;
  KernelTimingStats *h_timing_stats = nullptr;
#endif
  bool resources_allocated = false;
} gpu_resources;

inline bool driver_signal(CUdeviceptr d_sem, CUstream stream) {
  int h_sem = 1;
  if (!check_driver(cuMemcpyHtoDAsync(d_sem, &h_sem, sizeof(int), stream),
                    "cuMemcpyHtoDAsync(signal)")) {
    return false;
  }
  return check_driver(cuStreamSynchronize(stream),
                      "cuStreamSynchronize(signal)");
}

enum class DriverWaitOutcome { kProcessed, kTimedOut, kCudaError };

inline DriverWaitOutcome driver_wait_processed(
    uint64_t task_id, RapidTaskData::Clock::time_point dispatch_time,
    uint64_t timeout_ms, CUdeviceptr d_sem, CUstream copy_stream,
    CUstream kernel_stream, uint32_t *timeout_detail_ms) {
  int h_sem = 0;
  while (true) {
    if (!check_driver(cuMemcpyDtoHAsync(&h_sem, d_sem, sizeof(int), copy_stream),
                      "cuMemcpyDtoHAsync(wait)")) {
      return DriverWaitOutcome::kCudaError;
    }
    if (!check_driver(cuStreamSynchronize(copy_stream),
                      "cuStreamSynchronize(wait)")) {
      return DriverWaitOutcome::kCudaError;
    }
    if (h_sem != 0) {
      break;
    }
    const CUresult query = cuStreamQuery(kernel_stream);
    if (query != CUDA_SUCCESS && query != CUDA_ERROR_NOT_READY) {
      check_driver(query, "cuStreamQuery(kernel_stream)");
      return DriverWaitOutcome::kCudaError;
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                RapidTaskData::Clock::now() - dispatch_time)
                                .count();
    if (timeout_ms != 0 && elapsed_ms >= 0 &&
        static_cast<uint64_t>(elapsed_ms) >= timeout_ms) {
      const uint32_t elapsed_detail =
          elapsed_ms > std::numeric_limits<uint32_t>::max()
              ? std::numeric_limits<uint32_t>::max()
              : static_cast<uint32_t>(elapsed_ms);
      if (timeout_detail_ms != nullptr) {
        *timeout_detail_ms = elapsed_detail;
      }
      int h_ready = 0;
      const bool ready_snapshot =
          check_driver(cuMemcpyDtoHAsync(&h_ready,
                                         gpu_resources.d_data_ready_sem,
                                         sizeof(int), copy_stream),
                       "cuMemcpyDtoHAsync(timeout ready snapshot)") &&
          check_driver(cuStreamSynchronize(copy_stream),
                       "cuStreamSynchronize(timeout snapshot)");
      std::fprintf(
          stderr,
          "RAPID BACKEND WATCHDOG TIMEOUT: task_id=%llu stage=EXECUTE "
          "elapsed_ms=%u timeout_ms=%llu pending=%llu completed=%zu "
          "outstanding=%zu data_ready=%d data_processed=%d "
          "kernel_stream_query=%d snapshot_ok=%d. Published timeout "
          "completion for FIFO feedback retirement; the client will restart "
          "with a fresh CUDA context.\n",
          static_cast<unsigned long long>(task_id), elapsed_detail,
          static_cast<unsigned long long>(timeout_ms),
          static_cast<unsigned long long>(
              pending_inputs.load(std::memory_order_acquire)),
          completed_queue.completed_size(), completed_queue.outstanding_size(),
          h_ready, h_sem, static_cast<int>(query), ready_snapshot ? 1 : 0);
      std::fflush(stderr);
      return DriverWaitOutcome::kTimedOut;
    }
    std::this_thread::yield();
  }

  h_sem = 0;
  if (!check_driver(cuMemcpyHtoDAsync(d_sem, &h_sem, sizeof(int), copy_stream),
                    "cuMemcpyHtoDAsync(reset processed)")) {
    return DriverWaitOutcome::kCudaError;
  }
  if (!check_driver(cuStreamSynchronize(copy_stream),
                    "cuStreamSynchronize(reset processed)")) {
    return DriverWaitOutcome::kCudaError;
  }
  return DriverWaitOutcome::kProcessed;
}

inline bool driver_stop_persistent_kernel() {
  bool h_run = false;
  if (!check_driver(cuMemcpyHtoDAsync(gpu_resources.d_run, &h_run, sizeof(bool),
                                      gpu_resources.copy_stream),
                    "cuMemcpyHtoDAsync(stop run)")) {
    return false;
  }
  if (!check_driver(cuStreamSynchronize(gpu_resources.copy_stream),
                    "cuStreamSynchronize(stop run)")) {
    return false;
  }
  return driver_signal(gpu_resources.d_data_ready_sem,
                       gpu_resources.copy_stream);
}

inline bool initialize_gpu_resources() {
  if (gpu_resources.resources_allocated) {
    return true;
  }
  if (!check_driver(cuInit(0), "cuInit")) {
    return false;
  }
  if (!check_driver(cuDeviceGet(&gpu_resources.device, 0), "cuDeviceGet")) {
    return false;
  }
  if (!check_driver(cuCtxCreate(&gpu_resources.ctx, 0, gpu_resources.device),
                    "cuCtxCreate")) {
    return false;
  }
  if (!check_driver(cuCtxSetCurrent(gpu_resources.ctx), "cuCtxSetCurrent")) {
    return false;
  }
  if (kEmbeddedPtxSize == 0) {
    std::fprintf(stderr, "[rapid] embedded PTX image is empty\n");
    return false;
  }
  if (!check_driver(cuModuleLoadDataEx(&gpu_resources.module, kEmbeddedPtx, 0,
                                       nullptr, nullptr),
                    "cuModuleLoadDataEx")) {
    return false;
  }
  if (!check_driver(cuModuleGetFunction(&gpu_resources.persistent_kernel,
                                        gpu_resources.module,
                                        kKernelName),
                    "cuModuleGetFunction")) {
    return false;
  }
#if ENABLE_KERNEL_TIMING
  if (!check_driver(cuModuleGetGlobal(&gpu_resources.d_timing_stats,
                                      &gpu_resources.d_timing_stats_size,
                                      gpu_resources.module,
                                      "g_rapid_timing_stats"),
                    "cuModuleGetGlobal(g_rapid_timing_stats)")) {
    return false;
  }
  if (gpu_resources.d_timing_stats_size < sizeof(KernelTimingStats)) {
    std::fprintf(stderr,
                 "[rapid] timing stats too small: got=%zu want=%zu\n",
                 gpu_resources.d_timing_stats_size,
                 sizeof(KernelTimingStats));
    return false;
  }
#endif
  if (!select_cuda_launch_config(
          gpu_resources.persistent_kernel, gpu_resources.device,
          rapid_launch_config::kGrid, rapid_launch_config::kBlockCandidates,
          rapid_launch_config::kTargetDynamicSharedBytes, "rapid",
          &gpu_resources.launch_config)) {
    return false;
  }
  if (rapid_launch_config::kHasLogicalVConfigBounds &&
      !apply_logical_vconfig_bounds(&gpu_resources.launch_config,
                                    rapid_launch_config::kLogicalGrid,
                                    rapid_launch_config::kLogicalBlock,
                                    "rapid")) {
    return false;
  }
#if RAPID_ENABLE_INNER_FEEDBACK
  if (!check_driver(cuModuleGetGlobal(&gpu_resources.d_cov_map,
                                      &gpu_resources.d_cov_map_size,
                                      gpu_resources.module,
                                      "device_cov_map"),
                    "cuModuleGetGlobal(device_cov_map)")) {
    return false;
  }
  if (gpu_resources.d_cov_map_size < MAP_SIZE) {
    std::fprintf(stderr,
                 "[rapid] device_cov_map too small: got=%zu want=%u\n",
                 gpu_resources.d_cov_map_size, MAP_SIZE);
    return false;
  }
#endif
  if (!check_driver(
          cuMemAlloc(&gpu_resources.d_envelope,
                     rapid::canary::guarded_size(
                         rapid_task_envelope_size(MAX_INPUT_SIZE))),
          "cuMemAlloc(d_envelope)")) {
    return false;
  }
  if (!check_driver(cuMemAlloc(&gpu_resources.d_data_ready_sem, sizeof(int)),
                    "cuMemAlloc(d_data_ready_sem)")) {
    return false;
  }
  if (!check_driver(cuMemAlloc(&gpu_resources.d_data_processed_sem, sizeof(int)),
                    "cuMemAlloc(d_data_processed_sem)")) {
    return false;
  }
  if (!check_driver(cuMemAlloc(&gpu_resources.d_run, sizeof(bool)),
                    "cuMemAlloc(d_run)")) {
    return false;
  }
  if (!check_driver(cuMemAlloc(&gpu_resources.d_context,
                               sizeof(RapidKernelContext)),
                    "cuMemAlloc(d_context)")) {
    return false;
  }
#if RAPID_ENABLE_INNER_FEEDBACK
  if (!check_driver(cuMemAlloc(&gpu_resources.d_simt_memcov_bits,
                               SIMT_MEMCOV_STORAGE_SIZE),
                    "cuMemAlloc(d_simt_memcov_bits)")) {
    return false;
  }
#endif
  RapidKernelContext context{};
#if RAPID_ENABLE_INNER_FEEDBACK
  context.feedback.simt_memcov_bits_addr =
      static_cast<uintptr_t>(gpu_resources.d_simt_memcov_bits);
#endif
  if (!check_driver(cuMemcpyHtoD(gpu_resources.d_context, &context,
                                 sizeof(context)),
                    "cuMemcpyHtoD(init context)")) {
    return false;
  }
  const int zero = 0;
  const bool run = true;
  if (!check_driver(cuMemcpyHtoD(gpu_resources.d_data_ready_sem, &zero,
                                 sizeof(int)),
                    "cuMemcpyHtoD(init ready)")) {
    return false;
  }
  if (!check_driver(cuMemcpyHtoD(gpu_resources.d_data_processed_sem, &zero,
                                 sizeof(int)),
                    "cuMemcpyHtoD(init processed)")) {
    return false;
  }
  if (!check_driver(cuMemcpyHtoD(gpu_resources.d_run, &run, sizeof(bool)),
                    "cuMemcpyHtoD(init run)")) {
    return false;
  }
  if (!check_driver(cuStreamCreate(&gpu_resources.kernel_stream,
                                   CU_STREAM_DEFAULT),
                    "cuStreamCreate(kernel)")) {
    return false;
  }
  if (!check_driver(cuStreamCreate(&gpu_resources.copy_stream,
                                   CU_STREAM_DEFAULT),
                    "cuStreamCreate(copy)")) {
    return false;
  }
#if ENABLE_KERNEL_TIMING
  if (!check_driver(cuMemHostAlloc(
                        reinterpret_cast<void **>(&gpu_resources.h_timing_stats),
                        sizeof(KernelTimingStats), CU_MEMHOSTALLOC_PORTABLE),
                    "cuMemHostAlloc(h_timing_stats)")) {
    return false;
  }
#endif

  CUdeviceptr envelope_arg = gpu_resources.d_envelope;
  CUdeviceptr ready_arg = gpu_resources.d_data_ready_sem;
  CUdeviceptr processed_arg = gpu_resources.d_data_processed_sem;
  CUdeviceptr run_arg = gpu_resources.d_run;
  CUdeviceptr context_arg = gpu_resources.d_context;
  void *kernel_args[] = {&envelope_arg, &ready_arg, &processed_arg,
                         &run_arg, &context_arg};
  if (!check_driver(cuLaunchKernel(gpu_resources.persistent_kernel,
                                   gpu_resources.launch_config.grid_dim.x,
                                   gpu_resources.launch_config.grid_dim.y,
                                   gpu_resources.launch_config.grid_dim.z,
                                   gpu_resources.launch_config.block_dim.x,
                                   gpu_resources.launch_config.block_dim.y,
                                   gpu_resources.launch_config.block_dim.z,
                                   gpu_resources.launch_config.shared_mem_size,
                                   gpu_resources.kernel_stream, kernel_args,
                                   nullptr),
                    "cuLaunchKernel(rapid persistent)")) {
    return false;
  }

  gpu_resources.resources_allocated = true;
  return true;
}

inline bool copy_kernel_timing_stats(KernelTimingStats *out);

inline void dump_kernel_timing_stats() {
#if ENABLE_KERNEL_TIMING
  KernelTimingStats stats{};
  if (!copy_kernel_timing_stats(&stats)) {
    return;
  }
  final_timing_stats = stats;
  final_timing_stats_valid.store(true, std::memory_order_release);
  print_timing_stats("rapid/persistent_kernel", stats);
#endif
}

inline bool copy_kernel_timing_stats(KernelTimingStats *out) {
#if ENABLE_KERNEL_TIMING
  if (out == nullptr || gpu_resources.d_timing_stats == 0 ||
      gpu_resources.h_timing_stats == nullptr ||
      gpu_resources.copy_stream == nullptr) {
    return false;
  }
  CUcontext previous_context = nullptr;
  if (!check_driver(cuCtxGetCurrent(&previous_context), "cuCtxGetCurrent") ||
      !check_driver(cuCtxSetCurrent(gpu_resources.ctx),
                    "cuCtxSetCurrent(timing snapshot)")) {
    return false;
  }
  const bool copied =
      check_driver(cuMemcpyDtoHAsync(gpu_resources.h_timing_stats,
                                    gpu_resources.d_timing_stats, sizeof(*out),
                                    gpu_resources.copy_stream),
                   "cuMemcpyDtoHAsync(g_rapid_timing_stats snapshot)") &&
      check_driver(cuStreamSynchronize(gpu_resources.copy_stream),
                   "cuStreamSynchronize(timing snapshot)");
  if (copied) {
    *out = *gpu_resources.h_timing_stats;
  }
  const bool restored =
      check_driver(cuCtxSetCurrent(previous_context),
                   "cuCtxSetCurrent(restore after timing snapshot)");
  return copied && restored;
#else
  (void)out;
  return false;
#endif
}

inline bool reset_kernel_timing_stats() {
#if ENABLE_KERNEL_TIMING
  if (gpu_resources.d_timing_stats == 0) {
    return false;
  }
  const uint64_t request = KERNEL_TIMING_RESET_REQUEST;
  const CUdeviceptr iterations =
      gpu_resources.d_timing_stats + offsetof(KernelTimingStats, iterations);
  if (!check_driver(cuMemcpyHtoDAsync(iterations, &request, sizeof(request),
                                     gpu_resources.copy_stream),
                    "cuMemcpyHtoDAsync(g_rapid_timing_stats reset)")) {
    return false;
  }
  if (!check_driver(cuStreamSynchronize(gpu_resources.copy_stream),
                    "cuStreamSynchronize(reset timing request)") ||
      !driver_signal(gpu_resources.d_data_ready_sem,
                     gpu_resources.copy_stream)) {
    return false;
  }
  return driver_wait_processed(
             0, RapidTaskData::Clock::now(), 0,
             gpu_resources.d_data_processed_sem, gpu_resources.copy_stream,
             gpu_resources.kernel_stream, nullptr) ==
         DriverWaitOutcome::kProcessed;
#else
  return true;
#endif
}

inline void cleanup_gpu_resources() {
  if (gpu_resources.resources_allocated) {
    driver_stop_persistent_kernel();
    check_driver(cuStreamSynchronize(gpu_resources.kernel_stream),
                 "cuStreamSynchronize(kernel teardown)");
    dump_kernel_timing_stats();
  }

  if (gpu_resources.d_envelope != 0) {
    cuMemFree(gpu_resources.d_envelope);
    gpu_resources.d_envelope = 0;
  }
  if (gpu_resources.d_data_ready_sem != 0) {
    cuMemFree(gpu_resources.d_data_ready_sem);
    gpu_resources.d_data_ready_sem = 0;
  }
  if (gpu_resources.d_data_processed_sem != 0) {
    cuMemFree(gpu_resources.d_data_processed_sem);
    gpu_resources.d_data_processed_sem = 0;
  }
  if (gpu_resources.d_run != 0) {
    cuMemFree(gpu_resources.d_run);
    gpu_resources.d_run = 0;
  }
  if (gpu_resources.d_context != 0) {
    cuMemFree(gpu_resources.d_context);
    gpu_resources.d_context = 0;
  }
  if (gpu_resources.d_simt_memcov_bits != 0) {
    cuMemFree(gpu_resources.d_simt_memcov_bits);
    gpu_resources.d_simt_memcov_bits = 0;
  }
#if ENABLE_KERNEL_TIMING
  if (gpu_resources.h_timing_stats != nullptr) {
    cuMemFreeHost(gpu_resources.h_timing_stats);
    gpu_resources.h_timing_stats = nullptr;
  }
#endif
  if (gpu_resources.kernel_stream != nullptr) {
    cuStreamDestroy(gpu_resources.kernel_stream);
    gpu_resources.kernel_stream = nullptr;
  }
  if (gpu_resources.copy_stream != nullptr) {
    cuStreamDestroy(gpu_resources.copy_stream);
    gpu_resources.copy_stream = nullptr;
  }
  if (gpu_resources.module != nullptr) {
    cuModuleUnload(gpu_resources.module);
    gpu_resources.module = nullptr;
  }
  if (gpu_resources.ctx != nullptr) {
    cuCtxDestroy(gpu_resources.ctx);
    gpu_resources.ctx = nullptr;
  }
  gpu_resources.persistent_kernel = nullptr;
  gpu_resources.d_cov_map = 0;
  gpu_resources.d_cov_map_size = 0;
#if ENABLE_KERNEL_TIMING
  gpu_resources.d_timing_stats = 0;
  gpu_resources.d_timing_stats_size = 0;
#endif
  gpu_resources.resources_allocated = false;
}

static void gpu_producer_worker() {
#if ENABLE_KERNEL_TIMING
  pthread_setname_np(pthread_self(), "RAPID-Sync");
#endif
  if (!initialize_gpu_resources()) {
    cleanup_gpu_resources();
    request_worker_stop();
    fail_queued_inputs(
        libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_INIT, 1));
    publish_worker_exit();
    return;
  }

  LibAflRunStatus terminal_worker_status = libafl_run_status_ok();
  while (true) {
    std::unique_ptr<RapidTaskData> task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (input_queue.empty() &&
          !stop_flag.load(std::memory_order_acquire)) {
        lock.unlock();
        rapid::spin_until_work(pending_inputs, stop_flag,
                               reset_timing_requested,
                               kWorkerIdleSpinBudget);
        lock.lock();
      }
      queue_cv.wait(lock,
                    [] {
                      return !input_queue.empty() || stop_flag.load() ||
                             reset_timing_requested.load();
                    });
      if (reset_timing_requested.exchange(false,
                                          std::memory_order_acq_rel) &&
          input_queue.empty()) {
        lock.unlock();
        reset_timing_succeeded.store(reset_kernel_timing_stats(),
                                     std::memory_order_release);
        reset_timing_completed.store(true, std::memory_order_release);
        completion_cv.notify_all();
        continue;
      }
      if (stop_flag.load() && input_queue.empty()) {
        break;
      }
      if (input_queue.empty()) {
        continue;
      }
      task = std::move(input_queue.front());
      input_queue.pop();
    }

    task->dispatch_time = RapidTaskData::Clock::now();
    task->has_dispatch_time = true;
    LibAflRunStatus status = libafl_run_status_ok();
    bool fatal = false;

    if (task->payload_size > MAX_INPUT_SIZE) {
      std::fprintf(stderr,
                   "rapid input size %zu exceeds maximum %zu\n",
                   task->payload_size, MAX_INPUT_SIZE);
      status =
          libafl_run_status_invalid_input(LIBAFL_BACKEND_STAGE_EXECUTE, 1);
    }

    if (status.code == LIBAFL_RUN_STATUS_OK) {
      auto *header =
          reinterpret_cast<RapidTaskEnvelopeHeader *>(task->envelope.get());
      header->vconfig =
          clamp_vconfig_to_launch_bounds(header->vconfig,
                                         gpu_resources.launch_config);
    }

    if (status.code == LIBAFL_RUN_STATUS_OK &&
        !check_driver(cuMemcpyHtoDAsync(gpu_resources.d_envelope,
                                        task->envelope.get(),
                                        rapid::canary::guarded_size(
                                            task->envelope_size),
                                        gpu_resources.copy_stream),
                      "cuMemcpyHtoDAsync(envelope)")) {
      status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_EXECUTE, 2);
      fatal = true;
    }
    if (status.code == LIBAFL_RUN_STATUS_OK &&
        !check_driver(cuStreamSynchronize(gpu_resources.copy_stream),
                      "cuStreamSynchronize(pre-signal)")) {
      status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_EXECUTE, 5);
      fatal = true;
    }
    if (status.code == LIBAFL_RUN_STATUS_OK &&
        !driver_signal(gpu_resources.d_data_ready_sem,
                       gpu_resources.copy_stream)) {
      status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_EXECUTE, 6);
      fatal = true;
    }
    if (status.code == LIBAFL_RUN_STATUS_OK) {
      uint32_t timeout_detail = 0;
      const auto wait_outcome = driver_wait_processed(
          task->task_id, task->dispatch_time,
          target_timeout_ms.load(std::memory_order_acquire),
          gpu_resources.d_data_processed_sem, gpu_resources.copy_stream,
          gpu_resources.kernel_stream, &timeout_detail);
      if (wait_outcome == DriverWaitOutcome::kTimedOut) {
        status = libafl_run_status_timeout(LIBAFL_BACKEND_STAGE_EXECUTE,
                                           timeout_detail);
        std::fill(task->coverage.begin(), task->coverage.end(), 0);
        std::fill(task->simt_memcov.begin(), task->simt_memcov.end(), 0);
        std::memset(task->output.get(), 0,
                    std::max<size_t>(task->payload_size, 1));
        fatal = true;
      } else if (wait_outcome == DriverWaitOutcome::kCudaError) {
        status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_EXECUTE, 7);
        fatal = true;
      }
    }

    if (status.code == LIBAFL_RUN_STATUS_OK && task->payload_size > 0 &&
        !check_driver(cuMemcpyDtoHAsync(task->output.get(),
                                       gpu_resources.d_envelope +
                                           sizeof(RapidTaskEnvelopeHeader),
                                       task->payload_size,
                                       gpu_resources.copy_stream),
                      "cuMemcpyDtoHAsync(payload)")) {
      status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_COLLECT, 1);
      fatal = true;
    }
    if constexpr (rapid::canary::kEnabled) {
      if (status.code == LIBAFL_RUN_STATUS_OK &&
          !check_driver(cuMemcpyDtoHAsync(
                            task->envelope.get() + task->envelope_size,
                            gpu_resources.d_envelope + task->envelope_size,
                            rapid::canary::kSize, gpu_resources.copy_stream),
                        "cuMemcpyDtoHAsync(canary)")) {
        status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_COLLECT, 6);
        fatal = true;
      }
    }
#if RAPID_ENABLE_INNER_FEEDBACK
    if (status.code == LIBAFL_RUN_STATUS_OK &&
        !check_driver(cuMemcpyDtoHAsync(task->coverage.data(),
                                       gpu_resources.d_cov_map, MAP_SIZE,
                                       gpu_resources.copy_stream),
                      "cuMemcpyDtoHAsync(coverage)")) {
      status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_COLLECT, 2);
      fatal = true;
    }
    if (status.code == LIBAFL_RUN_STATUS_OK &&
        !check_driver(cuMemcpyDtoHAsync(task->simt_memcov.data(),
                                       gpu_resources.d_simt_memcov_bits,
                                       SIMT_MEMCOV_STORAGE_SIZE,
                                       gpu_resources.copy_stream),
                      "cuMemcpyDtoHAsync(simt_memcov_bits)")) {
      status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_COLLECT, 3);
      fatal = true;
    }
#endif
    if (status.code == LIBAFL_RUN_STATUS_OK &&
        !check_driver(cuStreamSynchronize(gpu_resources.copy_stream),
                      "cuStreamSynchronize(post-copy)")) {
      status = libafl_run_status_cuda_error(LIBAFL_BACKEND_STAGE_COLLECT, 4);
      fatal = true;
    }

    if constexpr (rapid::canary::kEnabled) {
      if (status.code == LIBAFL_RUN_STATUS_OK) {
        rapid::canary::Mismatch mismatch{};
        const uint8_t *const canary =
            task->envelope.get() + task->envelope_size;
        if (!rapid::canary::validate(canary, &mismatch)) {
          rapid::canary::report("rapid", task->task_id, mismatch);
          status = libafl_run_status_cuda_error(
              LIBAFL_BACKEND_STAGE_COLLECT, rapid::canary::kStatusDetail);
          fatal = true;
        }
      }
    }

    if (!publish_completion(std::move(task), status)) {
      terminal_worker_status = libafl_run_status_internal_error(
          LIBAFL_BACKEND_STAGE_COLLECT, 5);
      break;
    }
    if (fatal) {
      terminal_worker_status = status;
      break;
    }
  }

  request_worker_stop();
  if (terminal_worker_status.code != LIBAFL_RUN_STATUS_OK) {
    if (terminal_worker_status.code == LIBAFL_RUN_STATUS_CUDA_ERROR &&
        terminal_worker_status.stage == LIBAFL_BACKEND_STAGE_COLLECT &&
        terminal_worker_status.detail == rapid::canary::kStatusDetail) {
      fail_queued_inputs(
          libafl_run_status_internal_error(LIBAFL_BACKEND_STAGE_COLLECT, 6));
    } else {
      fail_queued_inputs(terminal_worker_status);
    }
  }

  cleanup_gpu_resources();
  publish_worker_exit();
}

static void ensure_initialized() {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex);
  if (initialized.load(std::memory_order_acquire)) {
    return;
  }

  stop_flag.store(false, std::memory_order_release);
  worker_exited.store(false, std::memory_order_release);
  pending_inputs.store(0, std::memory_order_release);
  reset_timing_requested.store(false, std::memory_order_release);
  reset_timing_completed.store(false, std::memory_order_release);
  reset_timing_succeeded.store(false, std::memory_order_release);
#if ENABLE_KERNEL_TIMING
  final_timing_stats_valid.store(false, std::memory_order_release);
#endif
  worker_thread = std::thread(gpu_producer_worker);
  initialized.store(true, std::memory_order_release);
}

static void shutdown_and_join_worker(bool drain_queue) {
  std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mutex);
  if (!initialized.load(std::memory_order_acquire)) {
    return;
  }
  if (drain_queue) {
    std::unique_lock<std::mutex> wait_lock(queue_mutex);
    completion_cv.wait(wait_lock, [] {
      return pending_inputs.load(std::memory_order_acquire) == 0 ||
             worker_exited.load(std::memory_order_acquire);
    });
  }

  request_worker_stop();

  if (worker_thread.joinable()) {
    worker_thread.join();
  }
  initialized.store(false, std::memory_order_release);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) uint64_t
libafl_submit_with_id(const uint8_t *input, size_t size) {
  std::lock_guard<std::mutex> submit_lock(submit_mutex);
  SubmittedEnvelopeView submitted{};
  if (!parse_submitted_envelope(input, size, &submitted)) {
    std::fprintf(stderr,
                 "rapid libafl_submit_with_id received invalid RAPID task envelope "
                 "size=%zu payload_max=%zu\n",
                 size, MAX_INPUT_SIZE);
    return 0;
  }
  if (target_timeout_ms.load(std::memory_order_acquire) == 0) {
    std::fprintf(stderr,
                 "rapid libafl_submit_with_id called before configuring a "
                 "nonzero target timeout\n");
    return 0;
  }
  ensure_initialized();
  {
    std::unique_lock<std::mutex> capacity_lock(capacity_mutex);
    capacity_cv.wait(capacity_lock, [] {
      return accepted_unreleased.load(std::memory_order_acquire) <
                 QUEUE_CAPACITY ||
             stop_flag.load(std::memory_order_acquire);
    });
    if (stop_flag.load(std::memory_order_acquire)) {
      return 0;
    }
  }

  const uint64_t task_id = next_task_id.fetch_add(1, std::memory_order_acq_rel);
  if (task_id == 0) {
    return 0;
  }
  std::unique_ptr<RapidTaskData> task;
  try {
    task = std::make_unique<RapidTaskData>(
        task_id, reinterpret_cast<uintptr_t>(input), input, size,
        submitted.payload_size);
  } catch (...) {
    return 0;
  }

  {
    std::scoped_lock lock(queue_mutex, capacity_mutex);
    if (stop_flag.load(std::memory_order_acquire)) {
      return 0;
    }
    input_queue.push(std::move(task));
    accepted_unreleased.fetch_add(1, std::memory_order_acq_rel);
    pending_inputs.fetch_add(1, std::memory_order_acq_rel);
    submitted_inputs.fetch_add(1, std::memory_order_acq_rel);
  }
  queue_cv.notify_one();
  return task_id;
}

extern "C" __attribute__((visibility("default"))) void
libafl_set_target_timeout_ms(uint64_t timeout_ms) {
  target_timeout_ms.store(timeout_ms, std::memory_order_release);
}

extern "C" __attribute__((visibility("default"))) size_t
libafl_poll_results(TaskResult *results, size_t max_count) {
  if (results == nullptr || max_count == 0) {
    return 0;
  }
  size_t count = 0;
  while (count < max_count) {
    RapidTaskData *task = completed_queue.try_acquire();
    if (task == nullptr) {
      break;
    }
    TaskResult &result = results[count++];
    result.task_id = task->task_id;
    result.input_ptr = task->input_ptr;
    result.edge_ptr = task->coverage.data();
    result.simt_memcov_ptr = task->simt_memcov.data();
    result.edge_size = MAP_SIZE;
    result.simt_memcov_size = SIMT_MEMCOV_STORAGE_SIZE;
    result.status = task->status;
    const auto started =
        task->has_dispatch_time ? task->dispatch_time : task->enqueue_time;
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        task->complete_time - started);
    result.exec_time_ns = elapsed.count() > 0
                              ? static_cast<uint64_t>(elapsed.count())
                              : 0;
  }
  return count;
}

extern "C" __attribute__((visibility("default"))) size_t
libafl_release_tasks(const uint64_t *task_ids, size_t count) {
  if (task_ids == nullptr || count == 0) {
    return 0;
  }
  size_t released = 0;
  for (size_t i = 0; i < count; ++i) {
    if (completed_queue.release(task_ids[i])) {
      ++released;
    }
  }
  if (released != 0) {
    {
      std::lock_guard<std::mutex> lock(capacity_mutex);
      accepted_unreleased.fetch_sub(released, std::memory_order_acq_rel);
    }
    capacity_cv.notify_all();
  }
  return released;
}

extern "C" __attribute__((visibility("default"))) void
libafl_get_ordered_queue_counts(OrderedQueueCounts *out_counts) {
  if (out_counts == nullptr) {
    return;
  }
  out_counts->pending =
      static_cast<size_t>(pending_inputs.load(std::memory_order_acquire));
  out_counts->completed = completed_queue.completed_size();
  out_counts->outstanding = completed_queue.outstanding_size();
}

extern "C" __attribute__((visibility("default"))) void libafl_wait() {
  if (!initialized.load(std::memory_order_acquire)) {
    print_rapid_feedback_stats();
    return;
  }
  std::unique_lock<std::mutex> wait_lock(queue_mutex);
  completion_cv.wait(wait_lock, [] {
    return pending_inputs.load(std::memory_order_acquire) == 0 ||
           worker_exited.load(std::memory_order_acquire);
  });
  print_rapid_feedback_stats();
}

extern "C" __attribute__((visibility("default"))) uint32_t
libafl_profile_timing_start() {
  libafl_wait();
  if (!initialized.load(std::memory_order_acquire)) {
    return 0U;
  }
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    reset_timing_completed.store(false, std::memory_order_release);
    reset_timing_succeeded.store(false, std::memory_order_release);
    reset_timing_requested.store(true, std::memory_order_release);
  }
  queue_cv.notify_one();
  std::unique_lock<std::mutex> wait_lock(queue_mutex);
  completion_cv.wait(wait_lock, [] {
    return reset_timing_completed.load(std::memory_order_acquire) ||
           worker_exited.load(std::memory_order_acquire);
  });
  if (!reset_timing_succeeded.load(std::memory_order_acquire)) {
    return 0U;
  }
  profile_timing_baseline.store(completed_inputs.load(std::memory_order_acquire),
                                std::memory_order_release);
  return 1U;
}

extern "C" __attribute__((visibility("default"))) uint32_t
libafl_profile_timing_snapshot(KernelTimingStats *out) {
  libafl_wait();
  shutdown_and_join_worker(/*drain_queue=*/true);
#if ENABLE_KERNEL_TIMING
  if (out == nullptr ||
      !final_timing_stats_valid.load(std::memory_order_acquire)) {
    return 0U;
  }
  *out = final_timing_stats;
  const uint64_t expected = completed_inputs.load(std::memory_order_acquire) -
                            profile_timing_baseline.load(std::memory_order_acquire);
  return out->iterations >= expected ? 1U : 0U;
#else
  (void)out;
  return 0U;
#endif
}

extern "C" __attribute__((visibility("default"))) void libafl_stop() {
  shutdown_and_join_worker(/*drain_queue=*/true);
  completed_queue.shutdown();
  print_rapid_feedback_stats();
}

__attribute__((destructor)) static void rapid_cleanup() {
  shutdown_and_join_worker(/*drain_queue=*/true);
  completed_queue.shutdown();
}
