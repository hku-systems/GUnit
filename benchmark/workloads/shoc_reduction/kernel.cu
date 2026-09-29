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

// Standalone extraction of SHOC reduce<float, 256>.
extern "C" __global__ void rq1_shoc_reduction(
    const float* __restrict__ g_idata,
    float* __restrict__ g_odata,
    const unsigned int n) {
  constexpr int blockSize = 256;
  const unsigned int tid = threadIdx.x;
  unsigned int i = (blockIdx.x * (blockDim.x * 2)) + tid;
  const unsigned int gridSize = blockDim.x * 2 * gridDim.x;

#if __CUDA_ARCH__ <= 130
  extern volatile __shared__ float sdata[];
#else
  SharedMem<float> shared;
  volatile float* sdata = shared.getPointer();
#endif

  sdata[tid] = 0.0f;
  while (i < n) {
    sdata[tid] += g_idata[i] + g_idata[i + blockSize];
    i += gridSize;
  }
  __syncthreads();

  if (tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if (tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();
  if (tid < warpSize) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
  }
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }
}
