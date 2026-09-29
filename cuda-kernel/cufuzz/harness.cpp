#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>

#include "rapid_target_layout.v1.h"
#include "coverage/coverage_constants.h"
#include "feedback/feedback_constants.h"
#include "input/task_envelope.cuh"
#include "status/run_status.h"
#ifndef RAPID_LAUNCH_CONFIG_HEADER
#define RAPID_LAUNCH_CONFIG_HEADER "launch/default_launch_config.h"
#endif
#include RAPID_LAUNCH_CONFIG_HEADER
#include "launch/launch_config.cuh"

extern "C" uint8_t libafl_cov_map[MAP_SIZE];
extern "C" uint8_t libafl_simt_memcov_bits[SIMT_MEMCOV_STORAGE_SIZE];

namespace cufuzz_embedded_ptx {
extern const unsigned char kEmbeddedPtx[];
extern const size_t kEmbeddedPtxSize;
inline constexpr const char *kKernelName = "cufuzz_wrapper";
}  // namespace cufuzz_embedded_ptx

constexpr size_t MAX_INPUT_SIZE = 4096 * 1024;

struct SubmittedEnvelopeView {
  RapidTaskEnvelopeHeader header{};
  const uint8_t *payload = nullptr;
  size_t payload_size = 0;
};

struct CuFuzzAllocationStats {
  uint64_t runs_started;
  uint64_t allocation_attempts;
  uint64_t allocations_succeeded;
  uint64_t frees;
  uint64_t runs_completed;
};

static_assert(sizeof(CuFuzzAllocationStats) == 40u,
              "CuFuzzAllocationStats ABI size changed");
static_assert(std::is_standard_layout_v<CuFuzzAllocationStats>,
              "CuFuzzAllocationStats must remain standard layout");
static_assert(std::is_trivially_copyable_v<CuFuzzAllocationStats>,
              "CuFuzzAllocationStats must remain trivially copyable");

namespace {

using cufuzz_embedded_ptx::kEmbeddedPtx;
using cufuzz_embedded_ptx::kEmbeddedPtxSize;
using cufuzz_embedded_ptx::kKernelName;
using rapid_launch::clamp_vconfig_to_launch_bounds;
using rapid_launch::select_cuda_launch_config;

static thread_local LibAflRunStatus g_current_run_status =
    libafl_run_status_ok();

bool check_driver(CUresult status, LibAflRunBackendStage stage,
                  const char *what) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  if (g_current_run_status.code == LIBAFL_RUN_STATUS_OK) {
    g_current_run_status =
        libafl_run_status_cuda_error(stage, static_cast<uint32_t>(status));
  }
  const char *name = nullptr;
  const char *desc = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &desc);
  std::fprintf(stderr, "[cufuzz] %s failed: %s (%s)\n", what,
               name ? name : "CUDA_ERROR_UNKNOWN",
               desc ? desc : "no description");
  return false;
}

bool parse_submitted_envelope(const uint8_t *input, size_t size,
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

class DeviceAllocation {
 public:
  explicit DeviceAllocation(CuFuzzAllocationStats *stats) : stats_(stats) {}
  ~DeviceAllocation() { reset(); }

  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  bool allocate(size_t size, const char *what) {
    if (ptr_ != 0 || stats_ == nullptr) {
      g_current_run_status =
          libafl_run_status_internal_error(LIBAFL_BACKEND_STAGE_SUBMIT, 1);
      return false;
    }
    ++stats_->allocation_attempts;
    if (!check_driver(cuMemAlloc(&ptr_, size), LIBAFL_BACKEND_STAGE_SUBMIT,
                      what)) {
      ptr_ = 0;
      return false;
    }
    ++stats_->allocations_succeeded;
    return true;
  }

  bool reset() {
    if (ptr_ == 0) {
      return true;
    }
    const CUdeviceptr ptr = ptr_;
    ptr_ = 0;
    if (!check_driver(cuMemFree(ptr), LIBAFL_BACKEND_STAGE_COLLECT,
                      "cuMemFree(per-input allocation)")) {
      return false;
    }
    ++stats_->frees;
    return true;
  }

  CUdeviceptr get() const { return ptr_; }

 private:
  CuFuzzAllocationStats *stats_ = nullptr;
  CUdeviceptr ptr_ = 0;
};

class GPUResourceManager {
 public:
  GPUResourceManager() = default;
  ~GPUResourceManager() { cleanup(); }

  bool run(const uint8_t *input, size_t size) {
    g_current_run_status = libafl_run_status_ok();
    output_size_ = 0;
    if (!init()) {
      return false;
    }

    SubmittedEnvelopeView submitted{};
    if (!parse_submitted_envelope(input, size, &submitted)) {
      std::fprintf(stderr,
                   "[cufuzz] invalid RAPID task envelope size=%zu "
                   "payload_max=%zu\n",
                   size, MAX_INPUT_SIZE);
      g_current_run_status =
          libafl_run_status_invalid_input(LIBAFL_BACKEND_STAGE_SUBMIT, 1);
      return false;
    }
    submitted.header.vconfig = clamp_vconfig_to_launch_bounds(
        submitted.header.vconfig, selected_launch_);

    ++allocation_stats_.runs_started;
    DeviceAllocation d_envelope(&allocation_stats_);
    DeviceAllocation d_context(&allocation_stats_);
    if (!d_envelope.allocate(rapid_task_envelope_size(submitted.payload_size),
                             "cuMemAlloc(envelope)")) {
      return false;
    }
    if (!d_context.allocate(sizeof(RapidKernelContext),
                            "cuMemAlloc(context)")) {
      return false;
    }

    auto *envelope = reinterpret_cast<RapidTaskEnvelopeHeader *>(
        envelope_buffer_.get());
    std::memcpy(envelope, &submitted.header, sizeof(RapidTaskEnvelopeHeader));
    if (submitted.payload_size > 0) {
      std::memcpy(rapid_task_payload(envelope), submitted.payload,
                  submitted.payload_size);
    }
    if (!check_driver(
            cuMemcpyHtoD(d_envelope.get(), envelope_buffer_.get(),
                         rapid_task_envelope_size(submitted.payload_size)),
            LIBAFL_BACKEND_STAGE_SUBMIT, "cuMemcpyHtoD(envelope)")) {
      return false;
    }

    RapidKernelContext context{};
    context.vconfig = submitted.header.vconfig;
    if (!check_driver(cuMemcpyHtoD(d_context.get(), &context, sizeof(context)),
                      LIBAFL_BACKEND_STAGE_SUBMIT,
                      "cuMemcpyHtoD(context)")) {
      return false;
    }

    CUdeviceptr envelope_arg = d_envelope.get();
    CUdeviceptr context_arg = d_context.get();
    void *kernel_args[] = {&envelope_arg, &context_arg};
    if (!check_driver(cuLaunchKernel(wrapper_kernel_,
                                     submitted.header.vconfig.grid_x,
                                     submitted.header.vconfig.grid_y,
                                     submitted.header.vconfig.grid_z,
                                     submitted.header.vconfig.block_x,
                                     submitted.header.vconfig.block_y,
                                     submitted.header.vconfig.block_z,
                                     selected_launch_.shared_mem_size, nullptr,
                                     kernel_args, nullptr),
                      LIBAFL_BACKEND_STAGE_LAUNCH, "cuLaunchKernel")) {
      return false;
    }
    if (!check_driver(cuCtxSynchronize(), LIBAFL_BACKEND_STAGE_EXECUTE,
                      "cuCtxSynchronize")) {
      return false;
    }

    if (submitted.payload_size > 0 &&
        !check_driver(cuMemcpyDtoH(
                          output_buffer_.get(),
                          d_envelope.get() + sizeof(RapidTaskEnvelopeHeader),
                          submitted.payload_size),
                      LIBAFL_BACKEND_STAGE_COLLECT,
                      "cuMemcpyDtoH(payload)")) {
      return false;
    }

    bool cleanup_ok = d_context.reset();
    cleanup_ok = d_envelope.reset() && cleanup_ok;
    if (!cleanup_ok) {
      return false;
    }
    output_size_ = submitted.payload_size;
    ++allocation_stats_.runs_completed;
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

  CuFuzzAllocationStats allocation_stats() const { return allocation_stats_; }

 private:
  bool init() {
    if (initialized_) {
      return check_driver(cuCtxSetCurrent(ctx_), LIBAFL_BACKEND_STAGE_INIT,
                          "cuCtxSetCurrent");
    }

    if (!check_driver(cuInit(0), LIBAFL_BACKEND_STAGE_INIT, "cuInit")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuDeviceGet(&device_, 0), LIBAFL_BACKEND_STAGE_INIT,
                      "cuDeviceGet")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuCtxCreate(&ctx_, 0, device_),
                      LIBAFL_BACKEND_STAGE_INIT, "cuCtxCreate")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuCtxSetCurrent(ctx_), LIBAFL_BACKEND_STAGE_INIT,
                      "cuCtxSetCurrent")) {
      cleanup();
      return false;
    }
    if (kEmbeddedPtxSize == 0) {
      std::fprintf(stderr, "[cufuzz] embedded PTX image is empty\n");
      g_current_run_status =
          libafl_run_status_internal_error(LIBAFL_BACKEND_STAGE_INIT, 1);
      cleanup();
      return false;
    }
    if (!check_driver(cuModuleLoadDataEx(&module_, kEmbeddedPtx, 0, nullptr,
                                         nullptr),
                      LIBAFL_BACKEND_STAGE_INIT, "cuModuleLoadDataEx")) {
      cleanup();
      return false;
    }
    if (!check_driver(cuModuleGetFunction(&wrapper_kernel_, module_,
                                          kKernelName),
                      LIBAFL_BACKEND_STAGE_INIT, "cuModuleGetFunction")) {
      cleanup();
      return false;
    }
    if (!select_cuda_launch_config(
            wrapper_kernel_, device_, rapid_launch_config::kGrid,
            rapid_launch_config::kBlockCandidates,
            rapid_launch_config::kTargetDynamicSharedBytes, "cufuzz",
            &selected_launch_)) {
      g_current_run_status =
          libafl_run_status_internal_error(LIBAFL_BACKEND_STAGE_INIT, 2);
      cleanup();
      return false;
    }

    envelope_buffer_ = std::make_unique<uint8_t[]>(
        rapid_task_envelope_size(MAX_INPUT_SIZE));
    output_buffer_ = std::make_unique<uint8_t[]>(MAX_INPUT_SIZE);
    initialized_ = true;
    return true;
  }

  void cleanup() {
    initialized_ = false;
    output_size_ = 0;
    envelope_buffer_.reset();
    output_buffer_.reset();
    wrapper_kernel_ = nullptr;

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
  std::unique_ptr<uint8_t[]> envelope_buffer_;
  std::unique_ptr<uint8_t[]> output_buffer_;
  size_t output_size_ = 0;
  CuFuzzAllocationStats allocation_stats_{};
  bool initialized_ = false;
};

GPUResourceManager g_gpu_resources;
std::mutex g_gpu_mutex;
LibAflRunStatus g_last_run_status = libafl_run_status_ok();

LibAflRunStatus make_run_status(bool ok) {
  if (ok) {
    return libafl_run_status_ok();
  }
  if (g_current_run_status.code != LIBAFL_RUN_STATUS_OK) {
    return g_current_run_status;
  }
  return libafl_run_status_internal_error(LIBAFL_BACKEND_STAGE_UNKNOWN, 1);
}

void clear_host_feedback() {
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

extern "C" __attribute__((visibility("default"))) void
libafl_get_cufuzz_allocation_stats(CuFuzzAllocationStats *out_stats) {
  if (out_stats == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_gpu_mutex);
  *out_stats = g_gpu_resources.allocation_stats();
}
