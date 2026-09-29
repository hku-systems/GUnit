#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kIlp = 4;

} // namespace

extern "C" __global__ void apex_adam_cuda_kernel(
    float* __restrict__ p,
    float* __restrict__ p_copy,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ g,
    float b1,
    float b2,
    float eps,
    float grad_scale,
    float step_size,
    size_t tsize,
    int mode,
    float decay,
    int write_p_copy)
{
    const int block_id = gridDim.x * blockIdx.y + blockIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int i = block_id * threads_per_block + thread_id;
    const int total_threads = gridDim.x * gridDim.y * threads_per_block;

    for (size_t j = i; j < tsize; j += total_threads) {
        const float scaled_grad = g[j] / grad_scale;
        m[j] = b1 * m[j] + (1.0f - b1) * scaled_grad;
        v[j] = b2 * v[j] + (1.0f - b2) * scaled_grad * scaled_grad;
        const float denom = mode == 0 ? sqrtf(v[j] + eps) : sqrtf(v[j]) + eps;
        const float update = m[j] / denom + decay * p[j];
        p[j] -= step_size * update;
        if (write_p_copy != 0) {
            p_copy[j] = p[j];
        }
    }
}

extern "C" __global__ void apex_maybe_cast_kernel(
    int overflow_flag,
    const float* p_in,
    float* p_out,
    size_t tsize)
{
    if (overflow_flag != 0) {
        return;
    }

    const int block_id = gridDim.x * blockIdx.y + blockIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int i = block_id * threads_per_block + thread_id;
    const int total_threads = gridDim.x * gridDim.y * threads_per_block;

    float input[kIlp];
    float output[kIlp];

    for (size_t start = 0; start < tsize; start += total_threads * kIlp) {
#pragma unroll
        for (int lane = 0; lane < kIlp; ++lane) {
            input[lane] = 0.0f;
            const size_t j = start + i + total_threads * lane;
            if (j < tsize) {
                input[lane] = p_in[j];
            }
        }

#pragma unroll
        for (int lane = 0; lane < kIlp; ++lane) {
            output[lane] = static_cast<float>(input[lane]);
        }

#pragma unroll
        for (int lane = 0; lane < kIlp; ++lane) {
            const size_t j = start + i + total_threads * lane;
            if (j < tsize) {
                p_out[j] = output[lane];
            }
        }
    }
}
