#ifndef __RAPID_ORDERED_COMPLETION_QUEUE_H__
#define __RAPID_ORDERED_COMPLETION_QUEUE_H__

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

template <typename T>
class OrderedCompletionQueue {
 public:
  explicit OrderedCompletionQueue(size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
      throw std::invalid_argument("ordered completion capacity must be nonzero");
    }
  }

  OrderedCompletionQueue(const OrderedCompletionQueue &) = delete;
  OrderedCompletionQueue &operator=(const OrderedCompletionQueue &) = delete;

  bool push(std::unique_ptr<T> task) {
    if (!task) {
      return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    space_cv_.wait(lock, [this] {
      return completed_.size() < capacity_ || shutdown_;
    });
    if (shutdown_) {
      return false;
    }
    completed_.push_back(std::move(task));
    return true;
  }

  T *try_acquire() {
    std::unique_ptr<T> task;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (completed_.empty()) {
        return nullptr;
      }
      task = std::move(completed_.front());
      completed_.pop_front();
      T *raw = task.get();
      const auto [_, inserted] =
          outstanding_.emplace(raw->task_id, std::move(task));
      if (!inserted) {
        throw std::logic_error("duplicate outstanding task ID");
      }
      space_cv_.notify_one();
      return raw;
    }
  }

  bool release(uint64_t task_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return outstanding_.erase(task_id) == 1;
  }

  size_t completed_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_.size();
  }

  size_t outstanding_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return outstanding_.size();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    space_cv_.notify_all();
  }

 private:
  const size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable space_cv_;
  std::deque<std::unique_ptr<T>> completed_;
  std::unordered_map<uint64_t, std::unique_ptr<T>> outstanding_;
  bool shutdown_ = false;
};

#endif  // __RAPID_ORDERED_COMPLETION_QUEUE_H__
