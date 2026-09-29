#ifndef PROJECT_UTILS_CUH
#define PROJECT_UTILS_CUH

#include <limits.h>

__device__ inline int safe_add(int a, int b) {
  long long sum = (long long)a + (long long)b;
  if (sum > INT_MAX) return INT_MAX;
  if (sum < INT_MIN) return INT_MIN;
  return (int)sum;
}

__device__ inline float lerp(float a, float b, float t) {
  return a + t * (b - a);
}

#endif // PROJECT_UTILS_CUH
