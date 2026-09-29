#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>

#include "rapid_target_layout.v1.h"
#include "coverage/coverage_constants.h"
#include "feedback/feedback_context.cuh"
#include "input/task_envelope.cuh"
#include "sanitizer/canary.h"
#include "status/run_status.h"
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

namespace origin_embedded_ptx {
extern const unsigned char kEmbeddedPtx[];
extern const size_t kEmbeddedPtxSize;
inline constexpr const char *kKernelName = "origin_wrapper";
}  // namespace origin_embedded_ptx

constexpr size_t MAX_INPUT_SIZE = 4096 * 1024;

struct SubmittedEnvelopeView {
  RapidTaskEnvelopeHeader header{};
  const uint8_t *payload = nullptr;
  size_t payload_size = 0;
};

namespace {

using origin_embedded_ptx::kEmbeddedPtx;
using origin_embedded_ptx::kEmbeddedPtxSize;
using origin_embedded_ptx::kKernelName;
using rapid_launch::apply_logical_vconfig_bounds;
using rapid_launch::clamp_vconfig_to_launch_bounds;
using rapid_launch::select_cuda_launch_config;

static thread_local LibAflRunStatus g_current_run_status =
    libafl_run_status_ok();

inline bool check_driver(CUresult status, const char *what) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  g_current_run_status = libafl_run_status_cuda_error(
      LIBAFL_BACKEND_STAGE_UNKNOWN, static_cast<uint32_t>(status));
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &desc);
  std::fprintf(stderr, "[origin] %s failed: %s (%s)\n", what,
               name ? name : "CUDA_ERROR_UNKNOWN",
               desc ? desc : "no description");
  return false;
}

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

class GPUResourceManager {
 public:
  GPUResourceManager() = default;
  ~GPUResourceManager() { cleanup(); }

  bool run(const uint8_t *input, size_t size) {
    g_current_run_status = libafl_run_status_ok();
    if (!init()) {
      return false;
    }
    SubmittedEnvelopeView submitted{};
    if (!parse_submitted_envelope(input, size, &submitted)) {
      std::fprintf(stderr,
                   "[origin] invalid RAPID task envelope size=%zu "
                   "payload_max=%zu\n",
                   size, MAX_INPUT_SIZE);
      g_current_run_status =
          libafl_run_status_invalid_input(LIBAFL_BACKEND_STAGE_SUBMIT, 1);
      return false;
    }
    submitted.header.vconfig = clamp_vconfig_to_launch_bounds(
        submitted.header.vconfig, selected_launch_);
    const uint64_t task_id = next_task_id_++;

    auto *envelope = reinterpret_cast<RapidTaskEnvelopeHeader *>(
        envelope_buffer_.get());
    std::memcpy(envelope, &submitted.header, sizeof(RapidTaskEnvelopeHeader));
    if (submitted.payload_size > 0) {
      std::memcpy(rapid_task_payload(envelope), submitted.payload,
                  submitted.payload_size);
    }
    uint8_t *const canary = envelope_buffer_.get() +
                            rapid_task_envelope_size(submitted.payload_size);
    rapid::canary::initialize(canary);
    if (!check_driver(
            cuMemcpyHtoD(d_envelope_, envelope_buffer_.get(),
                         rapid::canary::guarded_size(
                             rapid_task_envelope_size(submitted.payload_size))),
            "cuMemcpyHtoD(envelope)")) {
      return false;
    }
    RapidKernelContext context{};
    context.vconfig = submitted.header.vconfig;
#if RAPID_ENABLE_INNER_FEEDBACK
    context.feedback.simt_memcov_bits_addr =
        static_cast<uintptr_t>(d_simt_memcov_bits_);
#endif
    if (!check_driver(cuMemcpyHtoD(d_context_, &context, sizeof(context)),
                      "cuMemcpyHtoD(context)")) {
      return false;
    }

    CUdeviceptr envelope_arg = d_envelope_;
    CUdeviceptr context_arg = d_context_;
    void *kernel_args[] = {&envelope_arg, &context_arg};
    const dim3 launch_grid =
        rapid_launch_config::kVConfigEnabled ? selected_launch_.grid_dim
                                             : dim3(submitted.header.vconfig.grid_x,
                                                    submitted.header.vconfig.grid_y,
                                                    submitted.header.vconfig.grid_z);
    const dim3 launch_block =
        rapid_launch_config::kVConfigEnabled ? selected_launch_.block_dim
                                             : dim3(submitted.header.vconfig.block_x,
                                                    submitted.header.vconfig.block_y,
                                                    submitted.header.vconfig.block_z);
    if (!check_driver(cuLaunchKernel(wrapper_kernel_,
                                     launch_grid.x,
                                     launch_grid.y,
                                     launch_grid.z,
                                     launch_block.x,
                                     launch_block.y,
                                     launch_block.z,
                                     selected_launch_.shared_mem_size, nullptr,
                                     kernel_args, nullptr),
                      "cuLaunchKernel")) {
      return false;
    }
    if (!check_driver(cuCtxSynchronize(), "cuCtxSynchronize")) {
      return false;
    }

    if (submitted.payload_size > 0 &&
        !check_driver(cuMemcpyDtoH(
                          output_buffer_.get(),
                          d_envelope_ + sizeof(RapidTaskEnvelopeHeader),
                          submitted.payload_size),
                      "cuMemcpyDtoH(payload)")) {
      return false;
    }
    output_size_ = submitted.payload_size;

    if constexpr (rapid::canary::kEnabled) {
      if (!check_driver(
              cuMemcpyDtoH(canary,
                           d_envelope_ + rapid_task_envelope_size(
                                             submitted.payload_size),
                           rapid::canary::kSize),
              "cuMemcpyDtoH(canary)")) {
        return false;
      }
      rapid::canary::Mismatch mismatch{};
      if (!rapid::canary::validate(canary, &mismatch)) {
        rapid::canary::report("origin", task_id, mismatch);
        g_current_run_status = libafl_run_status_cuda_error(
            LIBAFL_BACKEND_STAGE_COLLECT, rapid::canary::kStatusDetail);
        return false;
      }
    }

#if RAPID_ENABLE_INNER_FEEDBACK
    if (!check_driver(cuMemcpyDtoH(libafl_cov_map, d_cov_map_, MAP_SIZE),
                      "cuMemcpyDtoH(coverage)")) {
      return false;
    }
    if (!check_driver(cuMemcpyDtoH(libafl_simt_memcov_bits,
                                   d_simt_memcov_bits_,
                                   SIMT_MEMCOV_STORAGE_SIZE),
                      "cuMemcpyDtoH(simt_memcov_bits)")) {
      return false;
    }
#endif
    return true;
  }

  size_t output_size() const { return output_size_; }

  size_t copy_output(uint8_t *dst, size_t dst_size) const {
    if (dst == nullptr || dst_size < output_size_) {
      return 0;
    }
    if (output_size_ > 0) {
      std::memcpy(dst, output_buffer_.get(), output_size_);
    }
    return output_size_;
  }

  bool reset_profile_timing() {
#if ENABLE_KERNEL_TIMING
    if (!init() || d_timing_stats_ == 0) {
      return false;
    }
    KernelTimingStats request{};
    request.iterations = KERNEL_TIMING_RESET_REQUEST;
    return check_driver(cuMemcpyHtoD(d_timing_stats_, &request,
                                     sizeof(request)),
                        "cuMemcpyHtoD(g_origin_timing_stats reset)");
#else
    return true;
#endif
  }

  bool snapshot_profile_timing(KernelTimingStats *out) {
#if ENABLE_KERNEL_TIMING
    if (out == nullptr || !init() || d_timing_stats_ == 0) {
      return false;
    }
    return check_driver(cuMemcpyDtoH(out, d_timing_stats_, sizeof(*out)),
                        "cuMemcpyDtoH(g_origin_timing_stats snapshot)");
#else
    (void)out;
    return false;
#endif
  }

  void stop() { cleanup(); }

 private:
  bool init() {
    if (initialized_) {
      return check_driver(cuCtxSetCurrent(ctx_), "cuCtxSetCurrent");
    }

    if (!check_driver(cuInit(0), "cuInit")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuDeviceGet(&device_, 0), "cuDeviceGet")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuCtxCreate(&ctx_, 0, device_), "cuCtxCreate")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuCtxSetCurrent(ctx_), "cuCtxSetCurrent")) {
      cleanup();
      return false;
    }
    if (kEmbeddedPtxSize == 0) {
      std::fprintf(stderr, "[origin] embedded PTX image is empty\n");
      cleanup();
      return false;
    }
    if (!check_driver(cuModuleLoadDataEx(&module_, kEmbeddedPtx, 0, nullptr,
                                         nullptr),
                      "cuModuleLoadDataEx")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuModuleGetFunction(&wrapper_kernel_, module_,
                                          kKernelName),
                      "cuModuleGetFunction")) {
      cleanup();
      return false;
    }
#if ENABLE_KERNEL_TIMING
    if (!check_driver(cuModuleGetGlobal(&d_timing_stats_,
                                        &d_timing_stats_size_, module_,
                                        "g_origin_timing_stats"),
                      "cuModuleGetGlobal(g_origin_timing_stats)")) {
      cleanup();
      return false;
    }
    if (d_timing_stats_size_ < sizeof(KernelTimingStats)) {
      std::fprintf(stderr,
                   "[origin] timing stats too small: got=%zu want=%zu\n",
                   d_timing_stats_size_, sizeof(KernelTimingStats));
      cleanup();
      return false;
    }
#endif
    if (!select_cuda_launch_config(
            wrapper_kernel_, device_, rapid_launch_config::kGrid,
            rapid_launch_config::kBlockCandidates,
            rapid_launch_config::kTargetDynamicSharedBytes, "origin",
            &selected_launch_)) {
      cleanup();
      return false;
    }
    if (rapid_launch_config::kHasLogicalVConfigBounds &&
        !apply_logical_vconfig_bounds(&selected_launch_,
                                      rapid_launch_config::kLogicalGrid,
                                      rapid_launch_config::kLogicalBlock,
                                      "origin")) {
      cleanup();
      return false;
    }
#if RAPID_ENABLE_INNER_FEEDBACK
    if (!check_driver(cuModuleGetGlobal(&d_cov_map_, &d_cov_map_size_, module_,
                                        "device_cov_map"),
                      "cuModuleGetGlobal(device_cov_map)")) {
      cleanup();
      return false;
    }
    if (d_cov_map_size_ < MAP_SIZE) {
      std::fprintf(stderr,
                   "[origin] device_cov_map too small: got=%zu want=%u\n",
                   d_cov_map_size_, MAP_SIZE);
      cleanup();
      return false;
    }
#endif
    if (!check_driver(
            cuMemAlloc(&d_envelope_,
                       rapid::canary::guarded_size(
                           rapid_task_envelope_size(MAX_INPUT_SIZE))),
            "cuMemAlloc(d_envelope)")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuMemAlloc(&d_context_, sizeof(RapidKernelContext)),
                      "cuMemAlloc(d_context)")) {
      cleanup();
      return false;
    }
#if RAPID_ENABLE_INNER_FEEDBACK
    if (!check_driver(cuMemAlloc(&d_simt_memcov_bits_,
                                 SIMT_MEMCOV_STORAGE_SIZE),
                      "cuMemAlloc(d_simt_memcov_bits)")) {
      cleanup();
      return false;
    }
#endif
    envelope_buffer_ = std::make_unique<uint8_t[]>(
        rapid::canary::guarded_size(rapid_task_envelope_size(MAX_INPUT_SIZE)));
    output_buffer_ = std::make_unique<uint8_t[]>(MAX_INPUT_SIZE);
    initialized_ = true;
    return true;
  }

  void cleanup() {
#if ENABLE_KERNEL_TIMING
    if (initialized_ && d_timing_stats_ != 0) {
      KernelTimingStats stats{};
      if (check_driver(cuMemcpyDtoH(&stats, d_timing_stats_, sizeof(stats)),
                       "cuMemcpyDtoH(g_origin_timing_stats)")) {
        print_timing_stats("origin/wrapper", stats);
      }
    }
#endif
    initialized_ = false;
    output_size_ = 0;
    envelope_buffer_.reset();
    output_buffer_.reset();

    if (d_envelope_ != 0) {
      cuMemFree(d_envelope_);
      d_envelope_ = 0;
    }
    if (d_context_ != 0) {
      cuMemFree(d_context_);
      d_context_ = 0;
    }
    if (d_simt_memcov_bits_ != 0) {
      cuMemFree(d_simt_memcov_bits_);
      d_simt_memcov_bits_ = 0;
    }

    wrapper_kernel_ = nullptr;
    d_cov_map_ = 0;
    d_cov_map_size_ = 0;
    d_timing_stats_ = 0;
    d_timing_stats_size_ = 0;

    if (module_ != nullptr) {
      cuModuleUnload(module_);
      module_ = nullptr;
    }
    if (ctx_ != nullptr) {
      cuCtxDestroy(ctx_);
      ctx_ = nullptr;
    }
  }

  CUdevice device_ = 0;
  CUcontext ctx_ = nullptr;
  CUmodule module_ = nullptr;
  CUfunction wrapper_kernel_ = nullptr;
  rapid_launch::LaunchConfig selected_launch_{};
  CUdeviceptr d_envelope_ = 0;
  CUdeviceptr d_context_ = 0;
  CUdeviceptr d_simt_memcov_bits_ = 0;
  CUdeviceptr d_cov_map_ = 0;
  size_t d_cov_map_size_ = 0;
  CUdeviceptr d_timing_stats_ = 0;
  size_t d_timing_stats_size_ = 0;
  std::unique_ptr<uint8_t[]> envelope_buffer_;
  std::unique_ptr<uint8_t[]> output_buffer_;
  size_t output_size_ = 0;
  uint64_t next_task_id_ = 1;
  bool initialized_ = false;
};

static GPUResourceManager g_gpu_resources;
static std::mutex g_gpu_mutex;

static LibAflRunStatus g_last_run_status = libafl_run_status_ok();

inline LibAflRunStatus make_run_status(bool ok) {
  if (ok) {
    return libafl_run_status_ok();
  }
  if (g_current_run_status.code != LIBAFL_RUN_STATUS_OK) {
    return g_current_run_status;
  }
  return libafl_run_status_internal_error(LIBAFL_BACKEND_STAGE_UNKNOWN, 1);
}

inline void clear_host_feedback() {
  std::memset(libafl_cov_map, 0, MAP_SIZE);
  std::memset(libafl_simt_memcov_bits, 0, SIMT_MEMCOV_STORAGE_SIZE);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void libafl_target(
    uint8_t *input, size_t size) {
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  clear_host_feedback();
  g_last_run_status = make_run_status(g_gpu_resources.run(input, size));
}

extern "C" __attribute__((visibility("default"))) void
libafl_get_last_run_status(LibAflRunStatus *out_status) {
  if (out_status == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  *out_status = g_last_run_status;
}

extern "C" __attribute__((visibility("default"))) size_t
libafl_get_last_output_size() {
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  return g_gpu_resources.output_size();
}

extern "C" __attribute__((visibility("default"))) size_t
libafl_copy_last_output(uint8_t *dst, size_t dst_size) {
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  return g_gpu_resources.copy_output(dst, dst_size);
}

extern "C" __attribute__((visibility("default"))) uint32_t
libafl_profile_timing_start() {
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  return g_gpu_resources.reset_profile_timing() ? 1U : 0U;
}

extern "C" __attribute__((visibility("default"))) uint32_t
libafl_profile_timing_snapshot(KernelTimingStats *out) {
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  return g_gpu_resources.snapshot_profile_timing(out) ? 1U : 0U;
}

extern "C" __attribute__((visibility("default"))) void libafl_stop() {
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  g_gpu_resources.stop();
}
