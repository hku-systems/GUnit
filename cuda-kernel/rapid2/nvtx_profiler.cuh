#ifndef __NVTX_PROFILER_CUH__
#define __NVTX_PROFILER_CUH__

#include "config.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#if ENABLE_NVTX_PROFILING
#include <nvtx3/nvToolsExt.h>

// Full implementation when NVTX is enabled
template <int N = 2> class NVTXProfiler {
public:
  // Constructor - initialize domains and ranges
  NVTXProfiler() : initialized_(false), enabled_(true) {
    // Initialize range ID vectors
    kernel_range_id_.resize(N, 0);
    lifecycle_range_id_.resize(N, 0);

    // Initialize task ID tracking for kernel processing
    kernel_task_ids_.resize(N, 0);

    // Create NVTX domains for each buffer
    nvtx_domain_kernel_.resize(N);
    nvtx_domain_lifecycle_.resize(N);

    for (int i = 0; i < N; ++i) {
      // Create domain names
      std::string kernel_domain_name = "Buffer" + std::to_string(i) + "_Kernel";
      std::string lifecycle_domain_name =
          "Buffer" + std::to_string(i) + "_Lifecycle";

      // Create NVTX domains
      nvtx_domain_kernel_[i] = nvtxDomainCreateA(kernel_domain_name.c_str());
      nvtx_domain_lifecycle_[i] =
          nvtxDomainCreateA(lifecycle_domain_name.c_str());
    }

    initialized_ = true;
  }

  // Destructor - cleanup resources
  ~NVTXProfiler() {
    if (initialized_) {
      // Destroy NVTX domains
      for (int i = 0; i < N; ++i) {
        if (nvtx_domain_kernel_[i])
          nvtxDomainDestroy(nvtx_domain_kernel_[i]);
        if (nvtx_domain_lifecycle_[i])
          nvtxDomainDestroy(nvtx_domain_lifecycle_[i]);
      }
    }
  }

  // Start kernel processing range for a buffer
  void startKernelProcessing(int buffer_idx, uint64_t task_id) {
    if (buffer_idx < 0 || buffer_idx >= N || !enabled_)
      return;

    // Store the task ID for validation during end
    kernel_task_ids_[buffer_idx] = task_id;

    // Create NVTX range in kernel domain
    char range_name[64];
    snprintf(range_name, sizeof(range_name), "Kernel Task %lu", task_id);

    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = getBufferColor(buffer_idx, true); // Kernel colors
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = range_name;

    kernel_range_id_[buffer_idx] =
        nvtxDomainRangeStartEx(nvtx_domain_kernel_[buffer_idx], &eventAttrib);
  }

  // End kernel processing range for a buffer
  void endKernelProcessing(int buffer_idx, uint64_t task_id) {
    if (buffer_idx < 0 || buffer_idx >= N || !enabled_)
      return;

    // Validate that the task_id matches the one from startKernelProcessing
    if (kernel_task_ids_[buffer_idx] != task_id) {
      fprintf(stderr, "ERROR: NVTX task_id mismatch in endKernelProcessing!\n");
      fprintf(stderr, "  Buffer %d: Expected task_id=%lu, got task_id=%lu\n",
              buffer_idx, kernel_task_ids_[buffer_idx], task_id);
      fprintf(stderr,
              "  This indicates unpaired start/end kernel processing calls\n");
      fflush(stderr);

      // Use assert to stop execution (can be caught by gdb)
      assert(kernel_task_ids_[buffer_idx] == task_id &&
             "NVTX endKernelProcessing task_id mismatch - check start/end "
             "pairing");
    }

    // End NVTX range
    if (kernel_range_id_[buffer_idx] != 0) {
      nvtxDomainRangeEnd(nvtx_domain_kernel_[buffer_idx],
                         kernel_range_id_[buffer_idx]);
      kernel_range_id_[buffer_idx] = 0;
      kernel_task_ids_[buffer_idx] = 0; // Clear the stored task_id
    }
  }

  // Start full lifecycle range for a buffer
  void startFullLifecycle(int buffer_idx, uint64_t task_id) {
    if (buffer_idx < 0 || buffer_idx >= N || !enabled_)
      return;

    // Create NVTX range in lifecycle domain
    char range_name[64];
    snprintf(range_name, sizeof(range_name), "Task %lu", task_id);

    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = getBufferColor(buffer_idx, false); // Lifecycle colors
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = range_name;

    lifecycle_range_id_[buffer_idx] = nvtxDomainRangeStartEx(
        nvtx_domain_lifecycle_[buffer_idx], &eventAttrib);
  }

  // End full lifecycle range for a buffer
  void endFullLifecycle(int buffer_idx) {
    if (buffer_idx < 0 || buffer_idx >= N || !enabled_)
      return;

    // End NVTX range
    if (lifecycle_range_id_[buffer_idx] != 0) {
      nvtxDomainRangeEnd(nvtx_domain_lifecycle_[buffer_idx],
                         lifecycle_range_id_[buffer_idx]);
      lifecycle_range_id_[buffer_idx] = 0;
    }
  }

  // Mark a specific point in time with an instant event
  void markInstant(const char *message, uint32_t color = 0xFFFF00FF) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;

    nvtxMarkEx(&eventAttrib);
  }

  // Push a range onto the default stack (for nested ranges)
  void pushRange(const char *message, uint32_t color = 0xFFFF00FF) {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;

    nvtxRangePushEx(&eventAttrib);
  }

  // Pop a range from the default stack
  void popRange() { nvtxRangePop(); }

  // Enable/disable profiling at runtime
  void setEnabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled) {
      // End any active ranges when disabling
      for (int i = 0; i < N; ++i) {
        if (kernel_range_id_[i] != 0) {
          nvtxDomainRangeEnd(nvtx_domain_kernel_[i], kernel_range_id_[i]);
          kernel_range_id_[i] = 0;
        }
        if (lifecycle_range_id_[i] != 0) {
          nvtxDomainRangeEnd(nvtx_domain_lifecycle_[i], lifecycle_range_id_[i]);
          lifecycle_range_id_[i] = 0;
        }
        // Clear stored task IDs
        kernel_task_ids_[i] = 0;
      }
    }
  }

  bool isEnabled() const { return enabled_; }

  // Get number of buffers
  int getNumBuffers() const { return N; }

private:
  // Generate distinct colors for different buffers
  uint32_t getBufferColor(int buffer_idx, bool is_kernel) {
    // Predefined color palette for up to 8 buffers
    static const uint32_t kernel_colors[] = {
        0xFF00FF00, // Green
        0xFF0000FF, // Blue
        0xFFFF00FF, // Magenta
        0xFF00FFFF, // Cyan
        0xFFFF8000, // Orange
        0xFF8000FF, // Purple
        0xFF80FF00, // Lime
        0xFFFF0080  // Pink
    };

    static const uint32_t lifecycle_colors[] = {
        0xFFFFFF00, // Yellow
        0xFF00FFFF, // Cyan
        0xFFFF00FF, // Magenta
        0xFF80FF80, // Light Green
        0xFFFFB366, // Light Orange
        0xFFB366FF, // Light Purple
        0xFF66FFB3, // Light Teal
        0xFFFF66B3  // Light Pink
    };

    // Use modulo for more than 8 buffers
    int color_idx = buffer_idx % 8;
    return is_kernel ? kernel_colors[color_idx] : lifecycle_colors[color_idx];
  }

private:
  // NVTX domains for separating different categories
  std::vector<nvtxDomainHandle_t>
      nvtx_domain_kernel_; // Kernel processing domains
  std::vector<nvtxDomainHandle_t>
      nvtx_domain_lifecycle_; // Full lifecycle domains

  // NVTX range IDs for active ranges
  std::vector<nvtxRangeId_t> kernel_range_id_;
  std::vector<nvtxRangeId_t> lifecycle_range_id_;

  // Task IDs for kernel processing validation
  std::vector<uint64_t> kernel_task_ids_;

  // State
  bool initialized_;
  bool enabled_;

  // Disable copy and move
  NVTXProfiler(const NVTXProfiler &) = delete;
  NVTXProfiler &operator=(const NVTXProfiler &) = delete;
  NVTXProfiler(NVTXProfiler &&) = delete;
  NVTXProfiler &operator=(NVTXProfiler &&) = delete;
};

#else

// Stub implementation when NVTX is disabled - all methods are empty
template <int N = 2> class NVTXProfiler {
public:
  NVTXProfiler() {}
  ~NVTXProfiler() {}
  
  void startKernelProcessing(int, uint64_t) {}
  void endKernelProcessing(int, uint64_t) {}
  void startFullLifecycle(int, uint64_t) {}
  void endFullLifecycle(int) {}
  void markInstant(const char *, uint32_t = 0xFFFF00FF) {}
  void pushRange(const char *, uint32_t = 0xFFFF00FF) {}
  void popRange() {}
  void setEnabled(bool) {}
  bool isEnabled() const { return false; }
  int getNumBuffers() const { return N; }

private:
  // Disable copy and move
  NVTXProfiler(const NVTXProfiler &) = delete;
  NVTXProfiler &operator=(const NVTXProfiler &) = delete;
  NVTXProfiler(NVTXProfiler &&) = delete;
  NVTXProfiler &operator=(NVTXProfiler &&) = delete;
};

#endif // ENABLE_NVTX_PROFILING

// Type alias for common cases
using NVTXProfiler2 = NVTXProfiler<2>; // Double buffer
using NVTXProfiler4 = NVTXProfiler<4>; // Quad buffer
using NVTXProfiler8 = NVTXProfiler<8>; // Octa buffer

// Convenient macros for NVTX profiling
#if ENABLE_NVTX_PROFILING
  #define NVTX_BEGIN_KERNEL(profiler, buffer_idx, task_id)  \
    do {                                                     \
      if (profiler) {                                        \
        (profiler)->startKernelProcessing(buffer_idx, task_id); \
      }                                                      \
    } while (0)

  #define NVTX_END_KERNEL(profiler, buffer_idx, task_id)    \
    do {                                                     \
      if (profiler) {                                        \
        (profiler)->endKernelProcessing(buffer_idx, task_id); \
      }                                                      \
    } while (0)

  #define NVTX_BEGIN_LIFECYCLE(profiler, buffer_idx, task_id) \
    do {                                                       \
      if (profiler) {                                          \
        (profiler)->startFullLifecycle(buffer_idx, task_id);  \
      }                                                        \
    } while (0)

  #define NVTX_END_LIFECYCLE(profiler, buffer_idx)            \
    do {                                                       \
      if (profiler) {                                          \
        (profiler)->endFullLifecycle(buffer_idx);             \
      }                                                        \
    } while (0)

  #define NVTX_MARK_INSTANT(profiler, message, color)         \
    do {                                                       \
      if (profiler) {                                          \
        (profiler)->markInstant(message, color);              \
      }                                                        \
    } while (0)

  #define NVTX_PUSH_RANGE(profiler, message, color)           \
    do {                                                       \
      if (profiler) {                                          \
        (profiler)->pushRange(message, color);                \
      }                                                        \
    } while (0)

  #define NVTX_POP_RANGE(profiler)                            \
    do {                                                       \
      if (profiler) {                                          \
        (profiler)->popRange();                                \
      }                                                        \
    } while (0)

#else
  // Empty macros when NVTX is disabled - properly consume parameters to avoid warnings
  #define NVTX_BEGIN_KERNEL(profiler, buffer_idx, task_id)    \
    do { (void)(profiler); (void)(buffer_idx); (void)(task_id); } while(0)
  #define NVTX_END_KERNEL(profiler, buffer_idx, task_id)      \
    do { (void)(profiler); (void)(buffer_idx); (void)(task_id); } while(0)
  #define NVTX_BEGIN_LIFECYCLE(profiler, buffer_idx, task_id) \
    do { (void)(profiler); (void)(buffer_idx); (void)(task_id); } while(0)
  #define NVTX_END_LIFECYCLE(profiler, buffer_idx)            \
    do { (void)(profiler); (void)(buffer_idx); } while(0)
  #define NVTX_MARK_INSTANT(profiler, message, color)         \
    do { (void)(profiler); (void)(message); (void)(color); } while(0)
  #define NVTX_PUSH_RANGE(profiler, message, color)           \
    do { (void)(profiler); (void)(message); (void)(color); } while(0)
  #define NVTX_POP_RANGE(profiler)                            \
    do { (void)(profiler); } while(0)
#endif

#endif // __NVTX_PROFILER_CUH__
