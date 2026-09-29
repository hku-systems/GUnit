#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kExperts = 128;
constexpr int kExpertsPerThread = kExperts / kWarpSize;

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

template <bool use_limit>
__device__ __forceinline__ void softmax_warp_inplace(
    float (&values)[kExpertsPerThread], int limit, int lane) {
    float max_value = -INFINITY;

#pragma unroll
    for (int i = 0; i < kExpertsPerThread; ++i) {
        const bool active = !use_limit || lane + i * kWarpSize < limit;
        if (active) {
            max_value = fmaxf(max_value, values[i]);
        }
    }
    max_value = warp_reduce_max(max_value);

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kExpertsPerThread; ++i) {
        const bool active = !use_limit || lane + i * kWarpSize < limit;
        values[i] = active ? expf(values[i] - max_value) : 0.0f;
        sum += values[i];
    }
    const float inverse_sum = 1.0f / warp_reduce_sum(sum);

#pragma unroll
    for (int i = 0; i < kExpertsPerThread; ++i) {
        if (!use_limit || lane + i * kWarpSize < limit) {
            values[i] *= inverse_sum;
        }
    }
}

__device__ __forceinline__ void sigmoid_warp_inplace(
    float (&values)[kExpertsPerThread]) {
#pragma unroll
    for (int i = 0; i < kExpertsPerThread; ++i) {
        values[i] = 1.0f / (1.0f + expf(-values[i]));
    }
}

} // namespace

// Flat-ABI specialization of llama.cpp topk_moe_cuda<128, false>.
// The absent bias argument fixes the upstream optional pointer to nullptr.
extern "C" __global__ __launch_bounds__(128, 1)
void topk_moe_cuda_128_no_bias_adapter(const float* logits,
                                       float* weights,
                                       int32_t* ids,
                                       int n_rows,
                                       int n_expert_used,
                                       float clamp_val,
                                       float scale_val,
                                       bool use_sigmoid,
                                       bool with_norm,
                                       bool delayed_softmax) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += kExperts * row;
    weights += n_expert_used * row;
    ids += kExperts * row;

    float values[kExpertsPerThread];
#pragma unroll
    for (int i = 0; i < kExpertsPerThread; ++i) {
        values[i] = logits[threadIdx.x + i * kWarpSize];
    }

    const bool do_delayed_softmax = delayed_softmax && !with_norm;
    if (!do_delayed_softmax) {
        if (use_sigmoid) {
            sigmoid_warp_inplace(values);
        } else {
            softmax_warp_inplace<false>(values, kExperts, threadIdx.x);
        }
    }

#pragma unroll
    for (int i = 0; i < kExpertsPerThread; ++i) {
        if (__isnanf(values[i])) {
            values[i] = -FLT_MAX;
        }
    }

    float weight_sum = 0.0f;
    float output_weights[kExpertsPerThread] = {};

    for (int k = 0; k < n_expert_used; ++k) {
        float max_value = values[0];
        int max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < kExpertsPerThread; ++i) {
            const int expert = threadIdx.x + i * kWarpSize;
            if (values[i] > max_value) {
                max_value = values[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
            const float other_value = __shfl_xor_sync(0xffffffff, max_value, mask);
            const int other_expert = __shfl_xor_sync(0xffffffff, max_expert, mask);
            if (other_value > max_value ||
                (other_value == max_value && other_expert < max_expert)) {
                max_value = other_value;
                max_expert = other_expert;
            }
        }

        if ((max_expert & (kWarpSize - 1)) == threadIdx.x) {
            values[max_expert / kWarpSize] = -INFINITY;
            ids[k] = max_expert;
            if (with_norm) {
                weight_sum += max_value;
            }
        }
        if ((k & (kWarpSize - 1)) == threadIdx.x) {
            output_weights[k / kWarpSize] = max_value;
        }
    }

    if (with_norm) {
        const float inverse_sum = 1.0f / fmaxf(warp_reduce_sum(weight_sum), clamp_val);
#pragma unroll
        for (int i = 0; i < kExpertsPerThread; ++i) {
            output_weights[i] *= inverse_sum;
        }
    }

    if (do_delayed_softmax) {
        softmax_warp_inplace<true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < kExpertsPerThread; ++i) {
        const int index = threadIdx.x + i * kWarpSize;
        if (index < n_expert_used) {
            weights[index] = output_weights[i] * scale_val;
        }
    }
}
