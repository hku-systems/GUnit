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

template <class T, int blockSize>
__device__ T scanLocalMem(const T val, volatile T* s_data) {
  int idx = threadIdx.x;
  s_data[idx] = 0.0f;
  idx += blockSize;
  s_data[idx] = val;
  __syncthreads();

  T t;
  t = s_data[idx - 1];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 2];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 4];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 8];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 16];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 32];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 64];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  t = s_data[idx - 128];
  __syncthreads();
  s_data[idx] += t;
  __syncthreads();
  return s_data[idx - 1];
}

// Standalone extraction of SHOC scan_single_block<float, 256>. This is the
// selected single-block stage, not the complete multi-kernel scan pipeline.
extern "C" __global__ void rq1_shoc_scan(
    float* __restrict__ g_block_sums, const int n) {
  constexpr int blockSize = 256;
#if __CUDA_ARCH__ <= 130
  extern volatile __shared__ float s_data[];
#else
  SharedMem<float> shared;
  volatile float* s_data = shared.getPointer();
#endif
  float val = (threadIdx.x < n) ? g_block_sums[threadIdx.x] : 0.0f;
  val = scanLocalMem<float, blockSize>(val, s_data);
  if (threadIdx.x < n) {
    g_block_sums[threadIdx.x] = val;
  }
}
