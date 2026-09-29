#include <cuda_runtime.h>

namespace {

constexpr float kEpsilon = 1e-8f;

template <int N>
__device__ __forceinline__ void llt(const float A[N][N], float L[N][N])
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            L[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = 0.0f;
            for (int k = 0; k < j; ++k) {
                s += L[i][k] * L[j][k];
            }

            if (i == j) {
                s = s > A[i][i] ? A[i][i] + kEpsilon : s;
                L[i][j] = sqrtf(A[i][i] - s);
            } else {
                L[i][j] = (A[i][j] - s) / L[j][j];
            }
        }
    }
}

template <int N>
__device__ __forceinline__ void llt_solve(const float L[N][N], float x[N])
{
    for (int i = 0; i < N; ++i) {
        float s = 0.0f;
        for (int j = 0; j < i; ++j) {
            s += L[i][j] * x[j];
        }
        x[i] = (x[i] - s) / L[i][i];
    }

    for (int i = N - 1; i >= 0; --i) {
        float s = 0.0f;
        for (int j = i + 1; j < N; ++j) {
            s += L[j][i] * x[j];
        }
        x[i] = (x[i] - s) / L[i][i];
    }
}

} // namespace

extern "C" __global__ void cholesky_solve6x6_forward_raw_kernel(
    const float* H_ptr, const float* b_ptr, float* x_ptr, int dim)
{
    const int m = threadIdx.x;
    if (m >= dim) {
        return;
    }

    float H[6][6];
    float L[6][6];
    float x[6];

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            H[i][j] = H_ptr[m + (6 * i + j) * dim];
        }
    }
    for (int i = 0; i < 6; ++i) {
        x[i] = b_ptr[m + i * dim];
    }

    llt<6>(H, L);
    llt_solve<6>(L, x);

    for (int i = 0; i < 6; ++i) {
        x_ptr[m + i * dim] = x[i];
    }
}
