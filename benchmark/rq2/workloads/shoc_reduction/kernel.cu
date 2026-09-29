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

// RQ2 VConfig adaptation of the standalone SHOC reduce<float, 256>
// extraction. The physical launch remains 256 threads; rewritten blockDim.x
// supplies a power-of-two logical block width.
extern "C" __global__ void rq1_shoc_reduction(
    const float* __restrict__ g_idata,
    float* __restrict__ g_odata,
    const unsigned int n) {
  const unsigned int logical_block_size = blockDim.x;
  const unsigned int tid = threadIdx.x;

#if __CUDA_ARCH__ <= 130
  extern volatile __shared__ float sdata[];
#else
  SharedMem<float> shared;
  volatile float* sdata = shared.getPointer();
#endif

  float value = 0.0f;
  for (unsigned int i = tid; i < n;
       i += logical_block_size * 2 * gridDim.x) {
    if (i < n) {
      value += g_idata[i];
    }
    if (i + logical_block_size < n) {
      value += g_idata[i + logical_block_size];
    }
  }
  sdata[tid] = value;
  __syncthreads();

  if (logical_block_size >= 256 && tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if (logical_block_size >= 128 && tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();

  if (tid < warpSize) {
    if (logical_block_size >= 64) {
      sdata[tid] += sdata[tid + 32];
    }
    __syncwarp();
    if (logical_block_size >= 32 && tid < 16) {
      sdata[tid] += sdata[tid + 16];
    }
    __syncwarp();
    if (tid < 8) {
      sdata[tid] += sdata[tid + 8];
    }
    __syncwarp();
    if (tid < 4) {
      sdata[tid] += sdata[tid + 4];
    }
    __syncwarp();
    if (tid < 2) {
      sdata[tid] += sdata[tid + 2];
    }
    __syncwarp();
    if (tid < 1) {
      sdata[tid] += sdata[tid + 1];
    }
    __syncwarp();
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}
