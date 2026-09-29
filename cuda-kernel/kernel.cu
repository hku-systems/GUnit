// kernel.cu
#include "kernel.cuh"

extern "C" __device__ void vulnerable_kernel(volatile uint8_t *input,
                                              volatile uint8_t *output,
                                              size_t size) {
  // Signature is intentionally kept in sync with kernelmanifest + arg-pack-v1 decode.
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid + bid * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Shared memory for collaborative processing
  __shared__ float shared_data[256];
  __shared__ int block_sum;
  __shared__ int block_max;

  // Initialize shared memory atomics
  if (tid == 0) {
    block_sum = 0;
    block_max = 0;
  }
  __syncthreads();

  // Multiple condition branches for coverage testing
  if (size > 2048) {
    // Complex mathematical operations
    float temp = __sinf(float(idx)) * __cosf(float(size));
    shared_data[tid % 256] = temp;
  } else if (size > 1024) {
    // Different mathematical path
    float temp = __expf(float(idx) / 1000.0f) + __logf(float(size) + 1.0f);
    shared_data[tid % 256] = temp;
  } else if (size > 512) {
    // Trigonometric operations
    float temp = __tanf(float(idx)) * sqrtf(float(size));
    shared_data[tid % 256] = temp;
  } else {
    shared_data[tid % 256] = float(idx);
  }

  __syncthreads();

  // Grid-stride loop for handling large data
  for (size_t i = idx; i < size; i += stride) {
    // Complex data processing with multiple paths
    uint8_t val = input[i];

    // Pattern matching with multiple branches
    if (val == 0x41) { // 'A'
      output[i] = val ^ 0xFF;
      atomicAdd(&block_sum, 1);
    } else if (val == 0x42) { // 'B'
      output[i] = (val << 2) | (val >> 6);
      atomicMax(&block_max, val);
    } else if (val > 128) {
      // Complex bit manipulation
      uint8_t shifted = (val << 3) | (val >> 5);
      uint8_t xored = shifted ^ 0xAA;
      output[i] = (xored & 0xF0) | (~xored & 0x0F);

      // Use shared memory value in computation
      float shared_val = shared_data[(i + tid) % 256];
      output[i] = uint8_t(float(output[i]) * fabs(shared_val) / 256.0f);
    } else if (val < 32) {
      // Mathematical transformation
      float angle = float(val) * 3.14159f / 32.0f;
      float result = __sinf(angle) * 127.0f + 128.0f;
      output[i] = uint8_t(result);
    } else {
      // Default processing with shared memory interaction
      float factor = shared_data[tid % 256];
      output[i] = uint8_t(float(val) * (1.0f + factor / 256.0f));
    }

    // Nested conditions for more coverage paths
    if (i % 16 == 0) {
      if (val & 0x01) {
        atomicXor((int *)&output[i], 0x55);
      } else {
        atomicOr((int *)&output[i], 0xAA);
      }
    }
  }

  __syncthreads();

  // Reduction operation using shared memory
  if (tid < 128) {
    shared_data[tid] += shared_data[tid + 128];
  }
  __syncthreads();

  if (tid < 64) {
    shared_data[tid] += shared_data[tid + 64];
  }
  __syncthreads();

  // Warp-level reduction (no sync needed within warp)
  if (tid < 32) {
    volatile float *s_data = shared_data;
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
  }

  // Write final result
  if (tid == 0 && bid == 0) {
    // Store block statistics at the end of input buffer if space available
    if (size > 4) {
      output[size - 4] = uint8_t(block_sum & 0xFF);
      output[size - 3] = uint8_t((block_sum >> 8) & 0xFF);
      output[size - 2] = uint8_t(block_max & 0xFF);
      output[size - 1] = uint8_t(shared_data[0]) & 0xFF;
    }
  }
}

extern "C" __global__ void vulnerable_kernel_entry(volatile uint8_t *input,
                                                     volatile uint8_t *output,
                                                     size_t size) {
  vulnerable_kernel(input, output, size);
}
