#ifndef __LIBAFL_INTERFACE_SHARED_CUH__
#define __LIBAFL_INTERFACE_SHARED_CUH__

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

#include "coverage/coverage_constants.h"
#include "feedback/feedback_constants.h"
#include "rapid2_runtime.cuh"
#include "status/run_status.h"

// Task result structure for LibAFL (pointer-based).
//
// The pointed-to maps remain valid until the task is released via
// libafl_release_tasks() (after which the buffers may be reused by later tasks).
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
static_assert(sizeof(LibAflRunStatus) == 8,
              "LibAflRunStatus ABI size changed");
static_assert(offsetof(TaskResult, edge_ptr) == 16,
              "TaskResult edge_ptr offset changed");
static_assert(offsetof(TaskResult, simt_memcov_ptr) == 24,
              "TaskResult simt_memcov_ptr offset changed");
static_assert(offsetof(TaskResult, status) == 40,
              "TaskResult status offset changed");
static_assert(offsetof(TaskResult, exec_time_ns) == 48,
              "TaskResult exec_time_ns offset changed");

// Queue counts (pending vs completed) for monitoring / backpressure.
struct LibAflQueueCounts {
    size_t pending;
    size_t completed;
};

static void rapid2_report_current_exception(const char *operation) noexcept {
    try {
        throw;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[RAPID2] %s failed: %s\n", operation,
                     error.what());
    } catch (...) {
        std::fprintf(stderr, "[RAPID2] %s failed: unknown exception\n",
                     operation);
    }
    std::fflush(stderr);
}

// Initialize the pipeline system
extern "C" __attribute__((visibility("default"))) void
rapid2_initialize(int num_kernels = 1,
                  size_t max_input_size = MAX_INPUT_SIZE) noexcept try {
    if (num_kernels != 1) {
        std::fprintf(stderr,
                     "[RAPID2] Initialization failed: exactly one pipeline "
                     "is required (got %d)\n",
                     num_kernels);
        std::fflush(stderr);
        return;
    }
    (void)rapid2_ensure_initialized(num_kernels, max_input_size);
} catch (...) {
    rapid2_report_current_exception("initialization");
}

// Configure the target timeout budget owned by the fuzzer runtime.
extern "C" __attribute__((visibility("default"))) void
libafl_set_target_timeout_ms(uint64_t timeout_ms) noexcept try {
    rapid2_set_target_timeout_ms(timeout_ms);
} catch (...) {
    rapid2_report_current_exception("timeout configuration");
}

// Wait for all pending tasks to complete
extern "C" __attribute__((visibility("default"))) void libafl_wait() noexcept try {
    if (g_pipeline_manager) {
        g_pipeline_manager->waitAll();
    }
} catch (...) {
    rapid2_report_current_exception("wait");
}

extern "C" __attribute__((visibility("default"))) uint32_t
libafl_profile_timing_start() noexcept try {
    if (!g_pipeline_manager) {
        return 0;
    }
    return g_pipeline_manager->resetTimingStats() ? 1U : 0U;
} catch (...) {
    rapid2_report_current_exception("profile timing reset");
    return 0;
}

extern "C" __attribute__((visibility("default"))) uint32_t
libafl_profile_timing_snapshot(KernelTimingStats *out) noexcept try {
    if (!g_pipeline_manager) {
        return 0;
    }
    return g_pipeline_manager->copyTimingStats(out) ? 1U : 0U;
} catch (...) {
    rapid2_report_current_exception("profile timing snapshot");
    return 0;
}

// Block until at least one completion can be polled. The collector publishes
// into the completed queue before notifying this condition.
extern "C" __attribute__((visibility("default"))) void
libafl_wait_for_completion() noexcept try {
    if (g_pipeline_manager) {
        g_pipeline_manager->waitForCompletion();
    }
} catch (...) {
    rapid2_report_current_exception("completion wait");
}

// Stop the pipeline system
extern "C" __attribute__((visibility("default"))) void libafl_stop() noexcept try {
    rapid2_shutdown_if_running();
} catch (...) {
    rapid2_report_current_exception("shutdown");
}

// Get pipeline statistics
extern "C" __attribute__((visibility("default"))) void
rapid2_print_stats() noexcept try {
    if (!g_pipeline_manager) {
        return;
    }
    auto stats = g_pipeline_manager->getStatistics();
    std::printf("[RAPID2] Pipeline Statistics:\n");
    std::printf("  Total submitted: %lu\n", stats.total_submitted);
    std::printf("  Total completed: %lu\n", stats.total_completed);
    std::printf("  Pending queue size: %lu\n", stats.pending_queue_size);
    std::printf("  Completed queue size: %lu\n", stats.completed_queue_size);
    std::printf("  Active kernels: %d\n", stats.active_kernels);
} catch (...) {
    rapid2_report_current_exception("statistics query");
}

// Submit input with ID (non-blocking)
extern "C" __attribute__((visibility("default"))) uint64_t
libafl_submit_with_id(const uint8_t *input, size_t size) noexcept try {
    if (rapid2_get_target_timeout_ms() == 0) {
        return 0;
    }

    // Auto-initialize on first call if not already initialized
    if (!rapid2_ensure_initialized(NUM_KERNELS, MAX_INPUT_SIZE)) {
        return 0;
    }

    if (!g_pipeline_manager || !g_pipeline_manager->isRunning()) {
        return 0;
    }

    // Submit and return immediately with task ID
    // Queue implementations are responsible for backpressure (blocking enqueue)
    // if configured as bounded queues. Do not retry here.
    return g_pipeline_manager->submitInput(input, size);
} catch (...) {
    rapid2_report_current_exception("input submission");
    return 0;
}

// Poll completed results (non-blocking) without copying feedback maps.
//
// The returned pointers reference task-owned host memory. The consumer must
// call libafl_release_tasks() after it is done consuming both maps.
extern "C" __attribute__((visibility("default"))) size_t
libafl_poll_results(TaskResult *results, size_t max_count) noexcept try {
    if (!g_pipeline_manager || !results || max_count == 0) {
        return 0;
    }

    size_t count = 0;

    // Try to get up to max_count completed tasks
    while (count < max_count) {
        TaskData *task = g_pipeline_manager->tryAcquireCompleted();
        if (!task) {
            break; // No more completed tasks available
        }

        // Fill result structure
        results[count].task_id = task->task_id;
        results[count].input_ptr = task->input_ptr;
        results[count].edge_ptr =
            task->h_feedback ? task->h_feedback.edgeData() : nullptr;
        results[count].simt_memcov_ptr =
            task->h_feedback ? task->h_feedback.memIndexData() : nullptr;
        results[count].edge_size = MAP_SIZE;
        results[count].simt_memcov_size = SIMT_MEMCOV_STORAGE_SIZE;

        if (task->status.code != LIBAFL_RUN_STATUS_OK) {
            results[count].status = task->status;
        } else if (task->has_error) {
            results[count].status = libafl_run_status_cuda_error(
                LIBAFL_BACKEND_STAGE_UNKNOWN, task->error_code);
        } else {
            results[count].status = libafl_run_status_ok();
        }

        // Calculate execution time
        std::chrono::steady_clock::duration exec_duration =
            task->complete_time - task->dispatch_time;
        if (!task->has_dispatch_time) {
            exec_duration = task->complete_time - task->enqueue_time;
        }
        results[count].exec_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(exec_duration).count();

        count++;
    }

    return count;
} catch (...) {
    rapid2_report_current_exception("result polling");
    return 0;
}

// Release previously acquired tasks.
extern "C" __attribute__((visibility("default"))) size_t
libafl_release_tasks(const uint64_t *task_ids, size_t count) noexcept try {
    if (!g_pipeline_manager || !task_ids || count == 0) {
        return 0;
    }
    size_t released = 0;
    for (size_t i = 0; i < count; ++i) {
        released += g_pipeline_manager->releaseCompleted(task_ids[i]) ? 1 : 0;
    }
    return released;
} catch (...) {
    rapid2_report_current_exception("task release");
    return 0;
}

// Get pending/completed queue counts in one call.
extern "C" __attribute__((visibility("default"))) void
libafl_get_queue_counts(LibAflQueueCounts *out_counts) noexcept try {
    if (!out_counts) {
        return;
    }
    if (!g_pipeline_manager) {
        out_counts->pending = 0;
        out_counts->completed = 0;
        return;
    }

    auto stats = g_pipeline_manager->getStatistics();
    out_counts->pending = static_cast<size_t>(stats.pending_queue_size);
    out_counts->completed = static_cast<size_t>(stats.completed_queue_size);
} catch (...) {
    if (out_counts) {
        out_counts->pending = 0;
        out_counts->completed = 0;
    }
    rapid2_report_current_exception("queue-count query");
}

// Cleanup function called when the library is unloaded
__attribute__((destructor)) static void rapid2_cleanup() noexcept try {
    rapid2_shutdown_if_running();
} catch (...) {
    rapid2_report_current_exception("library cleanup");
}

#endif // __LIBAFL_INTERFACE_SHARED_CUH__
