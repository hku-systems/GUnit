#ifndef __TASK_QUEUE_CUH__
#define __TASK_QUEUE_CUH__

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

// Task queue interface - supports future replacement with lock-free
// implementation
template <typename T> class ITaskQueue {
public:
  virtual ~ITaskQueue() = default;

  // Core operations
  virtual bool push(std::unique_ptr<T> task) = 0;
  virtual std::unique_ptr<T> pop() = 0;
  virtual std::unique_ptr<T> try_pop() = 0; // Non-blocking pop
  virtual bool empty() const = 0;
  virtual size_t size() const = 0;

  // Lifecycle
  virtual void shutdown() = 0;
};

// Mutex-based queue.
//
// Default behavior: bounded + blocking enqueue (backpressure).
// Optional behavior: when DropOldestOnFull=true, bounded + non-blocking enqueue
// by dropping the oldest element on overflow (useful for completed queues to
// avoid blocking the collector thread).
template <typename T, bool DropOldestOnFull = false>
class MutexTaskQueue : public ITaskQueue<T> {
public:
  static constexpr bool kDropOldestOnFull = DropOldestOnFull;

  MutexTaskQueue(size_t max_size = 0) // 0 means unlimited
      : max_size_(max_size) {}

  bool push(std::unique_ptr<T> task) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (shutdown_.load(std::memory_order_relaxed)) {
      return false;
    }

    if (max_size_ > 0 && queue_.size() >= max_size_) {
      if (DropOldestOnFull) {
        queue_.pop();
      } else {
        cv_.wait(lock, [this] {
          return shutdown_.load(std::memory_order_relaxed) ||
                 queue_.size() < max_size_;
        });
        if (shutdown_.load(std::memory_order_relaxed)) {
          return false;
        }
      }
    }

    queue_.push(std::move(task));
    cv_.notify_one();
    return true;
  }

  std::unique_ptr<T> pop() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (shutdown_ && queue_.empty()) {
      return nullptr;
    }
    auto task = std::move(queue_.front());
    queue_.pop();
    // Wake a potential producer waiting for space.
    cv_.notify_one();
    return task;
  }

  std::unique_ptr<T> try_pop() override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return nullptr;
    }
    auto task = std::move(queue_.front());
    queue_.pop();
    cv_.notify_one();
    return task;
  }

  bool empty() const override {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const override {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void shutdown() override {
    std::unique_lock<std::mutex> lock(mutex_);
    shutdown_ = true;
    cv_.notify_all();
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<std::unique_ptr<T>> queue_;
  size_t max_size_;
  std::atomic<bool> shutdown_{false};
};

// Lock-free CAS ring buffer queue (MPMC).
//
// Default behavior: bounded + blocking enqueue (retries internally until it
// succeeds or shutdown).
//
// Optional behavior: when DropOldestOnFull=true, bounded + non-blocking enqueue
// by dropping one oldest element on overflow (useful for completed queues to
// avoid blocking the collector thread).
template <typename T, bool DropOldestOnFull = false>
class CasRingBufferTaskQueue : public ITaskQueue<T> {
public:
  static constexpr bool kDropOldestOnFull = DropOldestOnFull;

  explicit CasRingBufferTaskQueue(size_t capacity = 1024)
      : capacity_(normalize_capacity(capacity)),
        capacity_mask_(capacity_ - 1),
        slots_(std::make_unique<Slot[]>(capacity_)) {
    for (size_t i = 0; i < capacity_; ++i) {
      slots_[i].seq.store(i, std::memory_order_relaxed);
      slots_[i].ptr = nullptr;
    }
  }

  ~CasRingBufferTaskQueue() override {
    // Best-effort cleanup: tasks enqueued via push() transfer ownership into
    // slots_ as raw pointers. If the queue is destroyed without being fully
    // drained, delete remaining tasks by popping until empty.
    //
    // Note: like most concurrent containers, it is undefined behavior to
    // destroy the queue while other threads are concurrently using it.
    shutdown_.store(true, std::memory_order_release);
    while (try_pop());
  }

  bool push(std::unique_ptr<T> task) override {
    if (!task) {
      return true;
    }

    int backoff_iter = 0;
    while (!shutdown_.load(std::memory_order_acquire)) {
      T *raw = task.get();
      size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

      for (;;) {
        Slot &slot = slots_[pos & capacity_mask_];
        size_t seq = slot.seq.load(std::memory_order_acquire);
        intptr_t dif = (intptr_t)seq - (intptr_t)pos;

        if (dif == 0) {
          if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                                 std::memory_order_relaxed,
                                                 std::memory_order_relaxed)) {
            slot.ptr = raw;
            slot.seq.store(pos + 1, std::memory_order_release);
            task.release();
            return true;
          }
          continue;
        }

        if (dif < 0) {
          // Full.
          if (DropOldestOnFull) {
            // Make progress by dropping one completed element.
            // Only backoff if we could not actually drop an element (under
            // contention the queue may appear full, but try_pop may temporarily
            // fail to claim a slot).
            if (!try_pop()) {
              backoff(backoff_iter++);
            }
            break; // retry enqueue
          }

          // Blocking enqueue: wait for space.
          backoff(backoff_iter++);
          break; // retry enqueue
        }

        // Another producer is ahead; refresh pos and retry.
        backoff(backoff_iter++);
        pos = enqueue_pos_.load(std::memory_order_relaxed);
      }
    }

    return false;
  }

  std::unique_ptr<T> pop() override {
    int wait_cycles = 0;
    while (!shutdown_.load(std::memory_order_acquire)) {
      auto result = try_pop();
      if (result) {
        return result;
      }

      wait_cycles = std::min(wait_cycles + 1, static_cast<int>(MAX_WAIT_CYCLES));
      if (wait_cycles < SPIN_THRESHOLD) {
        for (int i = 0; i < (1 << wait_cycles); ++i) {
          std::this_thread::yield();
        }
      } else {
        std::this_thread::sleep_for(
            std::chrono::microseconds(1 << (wait_cycles - SPIN_THRESHOLD)));
      }
    }
    return nullptr;
  }

  std::unique_ptr<T> try_pop() override {
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    for (int spin = 0; spin < MAX_SPINS; ++spin) {
      Slot &slot = slots_[pos & capacity_mask_];
      size_t seq = slot.seq.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);

      if (dif == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
          T *raw = slot.ptr;
          slot.ptr = nullptr;
          slot.seq.store(pos + capacity_, std::memory_order_release);
          return std::unique_ptr<T>(raw);
        }
        continue;
      }

      if (dif < 0) {
        return nullptr; // empty
      }

      backoff(spin);
      pos = dequeue_pos_.load(std::memory_order_relaxed);
    }

    return nullptr;
  }

  bool empty() const override {
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      const Slot &slot = slots_[pos & capacity_mask_];
      const size_t seq = slot.seq.load(std::memory_order_acquire);
      const intptr_t dif =
          static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
      if (dif == 0) {
        return false;
      }
      if (dif < 0) {
        return true;
      }
      pos = dequeue_pos_.load(std::memory_order_relaxed);
    }
  }

  size_t size() const override {
    size_t t = enqueue_pos_.load(std::memory_order_relaxed);
    size_t h = dequeue_pos_.load(std::memory_order_relaxed);
    return (t >= h) ? (t - h) : 0;
  }

  void shutdown() override { shutdown_.store(true, std::memory_order_release); }

private:
  struct Slot {
    std::atomic<size_t> seq;
    T *ptr;
  };

  static size_t normalize_capacity(size_t requested) {
    size_t cap = (requested == 0) ? 1 : requested;
    if (cap & (cap - 1)) {
      size_t pow2 = 1;
      while (pow2 < cap)
        pow2 <<= 1;
      cap = pow2;
    }
    return cap;
  }

  static void backoff(int iteration) {
    if (iteration < 4) {
      std::this_thread::yield();
      return;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }

  enum : int {
    MAX_SPINS = 200,
    MAX_WAIT_CYCLES = 16,
    SPIN_THRESHOLD = 6,
  };

  size_t capacity_;
  size_t capacity_mask_;
  std::unique_ptr<Slot[]> slots_;

  alignas(64) std::atomic<size_t> enqueue_pos_{0};
  alignas(64) std::atomic<size_t> dequeue_pos_{0};
  alignas(64) std::atomic<bool> shutdown_{false};
};

#endif // __TASK_QUEUE_CUH__
