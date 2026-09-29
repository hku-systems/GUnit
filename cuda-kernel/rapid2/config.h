#ifndef __RAPID2_CONFIG_H__
#define __RAPID2_CONFIG_H__

#include <cstddef>

// ---- Compile-time feature switches (defaults) ----
// These are set via CMake target_compile_definitions when building the harness.
// Provide safe defaults so the headers remain usable standalone.

#ifndef ENABLE_KERNEL_TIMING
#define ENABLE_KERNEL_TIMING 0
#endif

#ifndef ENABLE_NVTX_PROFILING
#define ENABLE_NVTX_PROFILING 0
#endif

#ifndef RAPID_ENABLE_INNER_FEEDBACK
#define RAPID_ENABLE_INNER_FEEDBACK 1
#endif

// Whether to merge per-task coverage into the global shared map (libafl_cov_map)
// on the host side as tasks complete.
//
// When enabled, this supports the "observers read shared global map" workflow.
// When disabled, users are expected to consume per-task coverage via
// libafl_poll_results() and/or manage the global map externally.
#ifndef RAPID2_MERGE_COVERAGE_TO_GLOBAL
#define RAPID2_MERGE_COVERAGE_TO_GLOBAL 0
#endif

#ifndef RAPID2_SLOT_COUNT
#define RAPID2_SLOT_COUNT 2
#endif

#ifndef RAPID2_USE_STREAM_COMPLETION
#define RAPID2_USE_STREAM_COMPLETION 0
#endif

static_assert(RAPID2_SLOT_COUNT == 2 || RAPID2_SLOT_COUNT == 4,
              "RAPID2_SLOT_COUNT must be 2 or 4");
constexpr size_t kRapid2SlotCount = RAPID2_SLOT_COUNT;
constexpr size_t kRapid2SlotMask = kRapid2SlotCount - 1;
constexpr size_t kRapid2MaxSlotCount = 4;
static_assert(RAPID2_USE_STREAM_COMPLETION == 0 ||
                  RAPID2_USE_STREAM_COMPLETION == 1,
              "RAPID2_USE_STREAM_COMPLETION must be 0 or 1");

// Queue selection:
// Use alias templates instead of runtime QueueType switching.
//
// Forward-declare queue templates; full definitions are provided by
// task_queue.cuh which is included by multi_pipeline_manager.cuh.
template <typename T, bool DropOldestOnFull> class MutexTaskQueue;
template <typename T, bool DropOldestOnFull> class CasRingBufferTaskQueue;

// Pipeline configuration
constexpr size_t NUM_KERNELS = 1;
constexpr size_t MAX_INPUT_SIZE = 4096 * 1024;

// Queue configuration
// Each TaskData owns a MAX_INPUT_SIZE pinned buffer, so queue depth directly
// bounds pinned memory. Keep this capacity at least 2x the fuzzer window cap.
constexpr size_t QUEUE_CAPACITY = 64;
static_assert((QUEUE_CAPACITY & (QUEUE_CAPACITY - 1)) == 0 &&
                  QUEUE_CAPACITY > 0,
              "QUEUE_CAPACITY must be a nonzero power of two");

// Task (pending) queue: blocking on full (backpressure).
template <typename T>
using Rapid2TaskQueue = CasRingBufferTaskQueue<T, /*DropOldestOnFull=*/false>;

// Completed queue: blocking on full so every submitted task ID has exactly one
// durable terminal result for the fuzzer to acquire and release.
template <typename T>
using Rapid2CompletedQueue = CasRingBufferTaskQueue<T, /*DropOldestOnFull=*/false>;

#endif // __RAPID2_CONFIG_H__
