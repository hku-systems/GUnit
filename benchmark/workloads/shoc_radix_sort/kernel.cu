#include <cuda.h>

__device__ unsigned int scanLSB(const unsigned int val, unsigned int* s_data) {
  int idx = threadIdx.x;
  s_data[idx] = 0;
  __syncthreads();
  idx += blockDim.x;

  unsigned int t;
  s_data[idx] = val;
  __syncthreads();
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
  return s_data[idx] - val;
}

__device__ uint4 scan4(uint4 idata, unsigned int* ptr) {
  uint4 val4 = idata;
  uint4 sum;
  sum.x = val4.x;
  sum.y = val4.y + sum.x;
  sum.z = val4.z + sum.y;
  unsigned int val = val4.w + sum.z;
  val = scanLSB(val, ptr);
  val4.x = val;
  val4.y = val + sum.x;
  val4.z = val + sum.y;
  val4.w = val + sum.z;
  return val4;
}

// Standalone extraction of SHOC radixSortBlocks. This is one block-local
// stage, not the full multi-kernel radix-sort pipeline.
extern "C" __global__ void rq1_shoc_radix_sort(
    const unsigned int nbits,
    const unsigned int startbit,
    uint4* keysOut,
    uint4* valuesOut,
    uint4* keysIn,
    uint4* valuesIn) {
  __shared__ unsigned int sMem[512];
  const unsigned int i = threadIdx.x + (blockIdx.x * blockDim.x);
  const unsigned int tid = threadIdx.x;
  const unsigned int localSize = blockDim.x;
  uint4 key = keysIn[i];
  uint4 value = valuesIn[i];

  for (unsigned int shift = startbit; shift < (startbit + nbits); ++shift) {
    uint4 lsb;
    lsb.x = !((key.x >> shift) & 0x1);
    lsb.y = !((key.y >> shift) & 0x1);
    lsb.z = !((key.z >> shift) & 0x1);
    lsb.w = !((key.w >> shift) & 0x1);
    uint4 address = scan4(lsb, sMem);

    __shared__ unsigned int numtrue;
    if (tid == localSize - 1) {
      numtrue = address.w + lsb.w;
    }
    __syncthreads();

    uint4 rank;
    const int idx = tid * 4;
    rank.x = lsb.x ? address.x : numtrue + idx - address.x;
    rank.y = lsb.y ? address.y : numtrue + idx + 1 - address.y;
    rank.z = lsb.z ? address.z : numtrue + idx + 2 - address.z;
    rank.w = lsb.w ? address.w : numtrue + idx + 3 - address.w;

    sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = key.x;
    sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = key.y;
    sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = key.z;
    sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = key.w;
    __syncthreads();
    key.x = sMem[tid];
    key.y = sMem[tid + localSize];
    key.z = sMem[tid + 2 * localSize];
    key.w = sMem[tid + 3 * localSize];
    __syncthreads();

    sMem[(rank.x & 3) * localSize + (rank.x >> 2)] = value.x;
    sMem[(rank.y & 3) * localSize + (rank.y >> 2)] = value.y;
    sMem[(rank.z & 3) * localSize + (rank.z >> 2)] = value.z;
    sMem[(rank.w & 3) * localSize + (rank.w >> 2)] = value.w;
    __syncthreads();
    value.x = sMem[tid];
    value.y = sMem[tid + localSize];
    value.z = sMem[tid + 2 * localSize];
    value.w = sMem[tid + 3 * localSize];
    __syncthreads();
  }
  keysOut[i] = key;
  valuesOut[i] = value;
}
