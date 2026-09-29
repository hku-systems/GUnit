#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gpuntt/common/modular_arith.cuh>

namespace {

struct Modulus64Fields {
    Data64 value;
    Data64 bit;
    Data64 mu;
};

static_assert(sizeof(Data64) == 8);
static_assert(sizeof(Modulus64Fields) == sizeof(Modulus64));
static_assert(alignof(Modulus64Fields) == alignof(Modulus64));
static_assert(std::is_standard_layout_v<Modulus64>);
static_assert(std::is_trivially_copyable_v<Modulus64>);
static_assert(offsetof(Modulus64, value) == 0);
static_assert(offsetof(Modulus64, bit) == 8);
static_assert(offsetof(Modulus64, mu) == 16);

template <typename ModulusLike>
__device__ __forceinline__ Data64 add_mod(Data64 lhs, Data64 rhs,
                                          const ModulusLike& modulus)
{
    const Data64 sum = lhs + rhs;
    return (sum >= modulus.value) ? (sum - modulus.value) : sum;
}

template <typename ModulusLike>
__device__ __forceinline__ Data64 sub_mod(Data64 lhs, Data64 rhs,
                                          const ModulusLike& modulus)
{
    const Data64 difference = lhs + modulus.value - rhs;
    return (difference >= modulus.value) ? (difference - modulus.value)
                                         : difference;
}

__device__ __forceinline__ Data64 shifted_low(Data64 low, Data64 high,
                                               std::uint32_t shift)
{
    return (low >> shift) | (high << (64U - shift));
}

template <typename ModulusLike>
__device__ __forceinline__ Data64 reduce_mod(Data64 input,
                                             const ModulusLike& modulus)
{
    Data64 quotient = shifted_low(input, 0, modulus.bit - 2);
    Data64 quotient_high = __umul64hi(quotient, modulus.mu);
    quotient *= modulus.mu;
    quotient = shifted_low(quotient, quotient_high, modulus.bit + 3);
    const Data64 product = quotient * modulus.value;
    const Data64 reduced = input - product;
    return (reduced >= modulus.value) ? (reduced - modulus.value) : reduced;
}

template <typename ModulusLike>
__device__ __forceinline__ Data64 reduce_forced(Data64 input,
                                                const ModulusLike& modulus)
{
    while (input >= modulus.value) {
        input = reduce_mod(input, modulus);
    }
    return input;
}

template <typename ModulusLike>
__device__ __forceinline__ Data64 multiply_mod(Data64 lhs, Data64 rhs,
                                               const ModulusLike& modulus)
{
    Data64 product = lhs * rhs;
    Data64 product_high = __umul64hi(lhs, rhs);
    Data64 quotient =
        shifted_low(product, product_high, modulus.bit - 2);
    Data64 quotient_high = __umul64hi(quotient, modulus.mu);
    quotient *= modulus.mu;
    quotient = shifted_low(quotient, quotient_high, modulus.bit + 3);
    product -= quotient * modulus.value;
    return (product >= modulus.value) ? (product - modulus.value) : product;
}

} // namespace

// Faithful one-block extraction of HEonGPU's BFV decryption kernel. The two
// by-value Modulus64 arguments are flattened into their exact POD fields.
extern "C" __global__ void heongpu_decryption_tile_kernel(
    Data64* ct0, Data64* ct1, Data64* plain, Modulus64* modulus,
    Data64 plain_mod_value, Data64 plain_mod_bit, Data64 plain_mod_mu,
    Data64 gamma_value, Data64 gamma_bit, Data64 gamma_mu, Data64* Qi_t,
    Data64* Qi_gamma, Data64* Qi_inverse, Data64 mulq_inv_t,
    Data64 mulq_inv_gamma, Data64 inv_gamma, int n_power,
    int decomp_mod_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = 1 << n_power;
    if (idx >= n) {
        return;
    }

    const Modulus64Fields plain_mod = {
        plain_mod_value, plain_mod_bit, plain_mod_mu};
    const Modulus64Fields gamma = {gamma_value, gamma_bit, gamma_mu};
    Data64 sum_t = 0;
    Data64 sum_gamma = 0;

#pragma unroll
    for (int i = 0; i < decomp_mod_count; ++i) {
        const int location = idx + (i << n_power);
        Data64 mt = add_mod(ct0[location], ct1[location], modulus[i]);
        const Data64 gamma_in_q = reduce_forced(gamma.value, modulus[i]);

        mt = multiply_mod(mt, plain_mod.value, modulus[i]);
        mt = multiply_mod(mt, gamma_in_q, modulus[i]);
        mt = multiply_mod(mt, Qi_inverse[i], modulus[i]);

        Data64 mt_in_t = reduce_forced(mt, plain_mod);
        Data64 mt_in_gamma = reduce_forced(mt, gamma);
        mt_in_t = multiply_mod(mt_in_t, Qi_t[i], plain_mod);
        mt_in_gamma = multiply_mod(mt_in_gamma, Qi_gamma[i], gamma);
        sum_t = add_mod(sum_t, mt_in_t, plain_mod);
        sum_gamma = add_mod(sum_gamma, mt_in_gamma, gamma);
    }

    sum_t = multiply_mod(sum_t, mulq_inv_t, plain_mod);
    sum_gamma = multiply_mod(sum_gamma, mulq_inv_gamma, gamma);

    if (sum_gamma > (gamma.value >> 1)) {
        const Data64 gamma_in_t = reduce_forced(gamma.value, plain_mod);
        const Data64 sum_gamma_in_t = reduce_forced(sum_gamma, plain_mod);
        Data64 result = sub_mod(gamma_in_t, sum_gamma_in_t, plain_mod);
        result = add_mod(sum_t, result, plain_mod);
        plain[idx] = multiply_mod(result, inv_gamma, plain_mod);
    }
    else {
        const Data64 sum_t_reduced = reduce_forced(sum_t, plain_mod);
        const Data64 sum_gamma_in_t = reduce_forced(sum_gamma, plain_mod);
        const Data64 result =
            sub_mod(sum_t_reduced, sum_gamma_in_t, plain_mod);
        plain[idx] = multiply_mod(result, inv_gamma, plain_mod);
    }
}
