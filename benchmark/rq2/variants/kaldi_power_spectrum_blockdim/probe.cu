#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "power_spectrum_blockdim.cu"

namespace kaldi {

__global__ void power_spectrum_kernel_original(
    int row_length, const float *A_in, int32_t ldi, float *A_out, int32_t ldo,
    bool use_power) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  const float *Ar = A_in + block_id * ldi;
  float *Aw = A_out + block_id * ldo;

  int half_length = row_length / 2;
  for (int idx = thread_id; idx < half_length; idx += 256) {
    if (idx == 0) continue;

    float2 val = reinterpret_cast<const float2 *>(Ar)[idx];
    float ret = val.x * val.x + val.y * val.y;
    if (use_power) {
      Aw[idx] = ret;
    } else {
      Aw[idx] = sqrtf(ret);
    }
  }

  if (threadIdx.x == 0) {
    float real = Ar[0];
    float im = Ar[row_length];

    if (use_power) {
      Aw[0] = real * real;
      Aw[half_length] = im * im;
    } else {
      Aw[0] = fabs(real);
      Aw[half_length] = fabs(im);
    }
  }
}

}  // namespace kaldi

namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

void run_case(int row_length, int variant_block, bool use_power) {
  const int ldi = row_length + 2;
  const int ldo = row_length / 2 + 1;
  std::vector<float> input(ldi);
  for (int i = 0; i < ldi; ++i) {
    const int numerator = ((i * 37 + row_length) % 257) - 128;
    input[i] = static_cast<float>(numerator) / 16.0f;
  }

  float *device_input = nullptr;
  float *device_original = nullptr;
  float *device_variant = nullptr;
  check_cuda(cudaMalloc(&device_input, input.size() * sizeof(float)),
             "cudaMalloc input");
  check_cuda(cudaMalloc(&device_original, ldo * sizeof(float)),
             "cudaMalloc original output");
  check_cuda(cudaMalloc(&device_variant, ldo * sizeof(float)),
             "cudaMalloc variant output");
  check_cuda(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "copy input");
  check_cuda(cudaMemset(device_original, 0xa5, ldo * sizeof(float)),
             "initialize original output");
  check_cuda(cudaMemset(device_variant, 0x5a, ldo * sizeof(float)),
             "initialize variant output");

  kaldi::power_spectrum_kernel_original<<<1, 256>>>(
      row_length, device_input, ldi, device_original, ldo, use_power);
  check_cuda(cudaGetLastError(), "launch original kernel");
  kaldi::power_spectrum_kernel_blockdim_stride<<<1, variant_block>>>(
      row_length, device_input, ldi, device_variant, ldo, use_power);
  check_cuda(cudaGetLastError(), "launch blockDim-stride variant");
  check_cuda(cudaDeviceSynchronize(), "synchronize kernels");

  std::vector<float> original(ldo);
  std::vector<float> variant(ldo);
  check_cuda(cudaMemcpy(original.data(), device_original, ldo * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "copy original output");
  check_cuda(cudaMemcpy(variant.data(), device_variant, ldo * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "copy variant output");
  for (int i = 0; i < ldo; ++i) {
    if (std::memcmp(&original[i], &variant[i], sizeof(float)) != 0) {
      std::fprintf(stderr,
                   "bitwise mismatch row_length=%d block=%d use_power=%d "
                   "index=%d original=%a variant=%a\n",
                   row_length, variant_block, use_power, i, original[i],
                   variant[i]);
      std::exit(1);
    }
  }

  check_cuda(cudaFree(device_input), "cudaFree input");
  check_cuda(cudaFree(device_original), "cudaFree original output");
  check_cuda(cudaFree(device_variant), "cudaFree variant output");
}

}  // namespace

int main() {
  constexpr int kRowLengths[] = {32, 64, 128, 256};
  constexpr int kVariantBlocks[] = {32, 64, 128, 256};
  for (int row_length : kRowLengths) {
    for (int variant_block : kVariantBlocks) {
      run_case(row_length, variant_block, false);
      run_case(row_length, variant_block, true);
    }
  }
  std::puts("kaldi power-spectrum blockDim-stride probe passed");
  return 0;
}
