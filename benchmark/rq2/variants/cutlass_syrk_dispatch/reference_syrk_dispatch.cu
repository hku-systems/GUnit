#include <stdint.h>

template <int TileX, int TileY>
__device__ __forceinline__ void SyrkTile(
  int N,
  int K,
  double alpha,
  double const *A,
  int lda,
  double beta,
  double *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < N && j < N && i >= j ) { // Since C is in Lower Fill Mode
    double accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * A[j + k * lda];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

__global__ void reference_syrk_dispatch_kernel(
  uint32_t tile_x,
  int N,
  int K,
  double alpha,
  double const *A,
  int lda,
  double beta,
  double *C,
  int ldc) {

  switch (tile_x) {
    case 8:
      if (blockDim.x != 8 || blockDim.y != 8 || blockDim.z != 1) {
        return;
      }
      SyrkTile<8, 8>(N, K, alpha, A, lda, beta, C, ldc);
      break;
    case 16:
      if (blockDim.x != 16 || blockDim.y != 16 || blockDim.z != 1) {
        return;
      }
      SyrkTile<16, 16>(N, K, alpha, A, lda, beta, C, ldc);
      break;
    case 32:
      if (blockDim.x != 32 || blockDim.y != 8 || blockDim.z != 1) {
        return;
      }
      SyrkTile<32, 8>(N, K, alpha, A, lda, beta, C, ldc);
      break;
    default:
      return;
  }
}
