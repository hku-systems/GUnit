#include <stdint.h>

template <int TileX, int TileY>
__device__ __forceinline__ void GemmTile(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

__global__ void reference_gemm_dispatch_kernel(
  uint32_t tile_x,
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  switch (tile_x) {
    case 8:
      if (blockDim.x != 8 || blockDim.y != 8 || blockDim.z != 1) {
        return;
      }
      GemmTile<8, 8>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      break;
    case 16:
      if (blockDim.x != 16 || blockDim.y != 16 || blockDim.z != 1) {
        return;
      }
      GemmTile<16, 16>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      break;
    case 32:
      if (blockDim.x != 32 || blockDim.y != 8 || blockDim.z != 1) {
        return;
      }
      GemmTile<32, 8>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      break;
    default:
      return;
  }
}
