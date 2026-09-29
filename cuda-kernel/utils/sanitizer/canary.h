#ifndef __UTILS_SANITIZER_CANARY_H__
#define __UTILS_SANITIZER_CANARY_H__

#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifndef RAPID_ENABLE_CANARY
#define RAPID_ENABLE_CANARY 1
#endif

#if defined(__CUDACC__) || defined(__CUDA__)
#define RAPID_CANARY_HOST_DEVICE __host__ __device__
#else
#define RAPID_CANARY_HOST_DEVICE
#endif

namespace rapid::canary {

inline constexpr bool kEnabled = RAPID_ENABLE_CANARY != 0;
inline constexpr size_t kSize = 256;
inline constexpr size_t kStorageSize = kEnabled ? kSize : 0;
inline constexpr uint32_t kStatusDetail = 0x43414e59U;  // "CANY"
inline constexpr uint8_t kPattern[] = {0x52, 0x41, 0x50, 0x49,
                                      0x44, 0x43, 0x41, 0x4e};

struct Mismatch {
  size_t offset = 0;
  uint8_t expected = 0;
  uint8_t actual = 0;
};

RAPID_CANARY_HOST_DEVICE inline constexpr size_t guarded_size(size_t size) {
  return size + kStorageSize;
}

inline void initialize(uint8_t *guard) {
  if constexpr (kEnabled) {
    for (size_t i = 0; i < kSize; ++i) {
      guard[i] = kPattern[i % sizeof(kPattern)];
    }
  }
}

inline bool validate(const uint8_t *guard, Mismatch *mismatch = nullptr) {
  if constexpr (kEnabled) {
    for (size_t i = 0; i < kSize; ++i) {
      const uint8_t expected = kPattern[i % sizeof(kPattern)];
      if (guard[i] != expected) {
        if (mismatch != nullptr) {
          *mismatch = Mismatch{i, expected, guard[i]};
        }
        return false;
      }
    }
  }
  return true;
}

inline void report(const char *backend, uint64_t task_id,
                   const Mismatch &mismatch) {
  std::fprintf(stderr,
               "[%s] canary corrupted: task_id=%llu offset=%zu "
               "expected=0x%02x actual=0x%02x\n",
               backend, static_cast<unsigned long long>(task_id),
               mismatch.offset, static_cast<unsigned>(mismatch.expected),
               static_cast<unsigned>(mismatch.actual));
}

}  // namespace rapid::canary

#undef RAPID_CANARY_HOST_DEVICE

#endif  // __UTILS_SANITIZER_CANARY_H__
