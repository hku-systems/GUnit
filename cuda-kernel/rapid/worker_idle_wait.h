#ifndef RAPID_WORKER_IDLE_WAIT_H_
#define RAPID_WORKER_IDLE_WAIT_H_

#include <atomic>
#include <chrono>
#include <cstdint>

namespace rapid {

enum class IdleSpinResult {
  kWorkAvailable,
  kControlRequested,
  kStopRequested,
  kTimedOut,
};

inline void idle_cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

inline IdleSpinResult spin_until_work(
    const std::atomic<uint64_t> &pending_inputs,
    const std::atomic<bool> &stop_requested,
    const std::atomic<bool> &control_requested,
    std::chrono::steady_clock::duration spin_budget) noexcept {
  const auto deadline = std::chrono::steady_clock::now() + spin_budget;
  while (true) {
    if (stop_requested.load(std::memory_order_acquire)) {
      return IdleSpinResult::kStopRequested;
    }
    if (pending_inputs.load(std::memory_order_acquire) != 0) {
      return IdleSpinResult::kWorkAvailable;
    }
    if (control_requested.load(std::memory_order_acquire)) {
      return IdleSpinResult::kControlRequested;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return IdleSpinResult::kTimedOut;
    }
    idle_cpu_relax();
  }
}

}  // namespace rapid

#endif  // RAPID_WORKER_IDLE_WAIT_H_
