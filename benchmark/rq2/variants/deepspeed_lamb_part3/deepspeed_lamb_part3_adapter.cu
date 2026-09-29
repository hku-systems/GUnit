#include <cuda_runtime.h>

#include <cstddef>

extern "C" __global__ void deepspeed_lamb_cuda_kernel_part3(
    float* __restrict__ p,
    float* __restrict__ p_copy,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ g,
    float b1,
    float b2,
    float max_coeff,
    float min_coeff,
    float eps,
    float grad_scale,
    float step_size,
    size_t tsize,
    int mode,
    float decay,
    float* __restrict__ w_l2_i,
    float* __restrict__ u_l2_i,
    float* __restrict__ lamb_coeff_val,
    int write_p_copy)
{
    const int block_id = gridDim.x * blockIdx.y + blockIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int i = block_id * threads_per_block + thread_id;
    const int total_threads = gridDim.x * gridDim.y * threads_per_block;

    const float weight_norm = sqrtf(w_l2_i[0]);
    const float update_norm = sqrtf(u_l2_i[0]);
    float lamb_coeff = 1.0f;

    if (weight_norm != 0.0f && update_norm != 0.0f) {
        lamb_coeff = weight_norm / update_norm;
        if (lamb_coeff > max_coeff) {
            lamb_coeff = max_coeff;
        }
        if (lamb_coeff < min_coeff) {
            lamb_coeff = min_coeff;
        }
    }

    if (block_id == 0 && thread_id == 0) {
        lamb_coeff_val[0] = lamb_coeff;
    }

    for (size_t j = i; j < tsize; j += total_threads) {
        const float denom = mode == 0 ? sqrtf(v[j] + eps) : sqrtf(v[j]) + eps;
        const float update = m[j] / denom + decay * p[j];
        p[j] -= step_size * lamb_coeff * update;
        if (write_p_copy != 0) {
            p_copy[j] = p[j];
        }
    }
}
