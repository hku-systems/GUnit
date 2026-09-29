#include <cuda.h>

template <class T>
class SharedMem {
 public:
  __device__ inline T* getPointer() {
    extern __shared__ T s[];
    return s;
  }
};

template <>
class SharedMem<float> {
 public:
  __device__ inline float* getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <class T>
__device__ T scanLocalMem(const T val, volatile T* s_data,
                          const unsigned int logical_block_size) {
  int idx = threadIdx.x;
  s_data[idx] = 0.0f;
  idx += logical_block_size;
  s_data[idx] = val;
  __syncthreads();

#define RAPID_SCAN_STAGE(offset)                   \
  do {                                             \
    T value_at_offset = 0.0f;                     \
    if (logical_block_size > offset) {            \
      value_at_offset = s_data[idx - offset];     \
    }                                             \
    __syncthreads();                               \
    if (logical_block_size > offset) {            \
      s_data[idx] += value_at_offset;              \
    }                                             \
    __syncthreads();                               \
  } while (0)

  RAPID_SCAN_STAGE(1);
  RAPID_SCAN_STAGE(2);
  RAPID_SCAN_STAGE(4);
  RAPID_SCAN_STAGE(8);
  RAPID_SCAN_STAGE(16);
  RAPID_SCAN_STAGE(32);
  RAPID_SCAN_STAGE(64);
  RAPID_SCAN_STAGE(128);

#undef RAPID_SCAN_STAGE

  return s_data[idx - 1];
}

// RQ2 VConfig adaptation of the standalone SHOC
// scan_single_block<float, 256> extraction. The physical launch remains 256
// threads; rewritten blockDim.x supplies the selected logical block width.
extern "C" __global__ void rq1_shoc_scan(
    float* __restrict__ g_block_sums, const int n) {
  const unsigned int logical_block_size = blockDim.x;
#if __CUDA_ARCH__ <= 130
  extern volatile __shared__ float s_data[];
#else
  SharedMem<float> shared;
  volatile float* s_data = shared.getPointer();
#endif
  float val = (threadIdx.x < n) ? g_block_sums[threadIdx.x] : 0.0f;
  val = scanLocalMem<float>(val, s_data, logical_block_size);
  if (threadIdx.x < n) {
    g_block_sums[threadIdx.x] = val;
  }
}
