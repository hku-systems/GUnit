#include <stdint.h>

template <int TileX, int TileY>
__device__ __forceinline__ void TrmmTile(
  int M,
  int N,
  double alpha,
  double const *A,
  int lda,
  double const *B,
  int ldb,
  double *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    double accumulator = 0;

    for (int k = 0; k < M; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb]; // Since A is in Left-Side Mode
    }

    C[i + j * ldc] = alpha * accumulator;
  }
}

__global__ void reference_trmm_dispatch_kernel(
  uint32_t tile_x,
  int M,
  int N,
  double alpha,
  double const *A,
  int lda,
  double const *B,
  int ldb,
  double *C,
  int ldc) {

  switch (tile_x) {
    case 8:
      if (blockDim.x != 8 || blockDim.y != 8 || blockDim.z != 1) {
        return;
      }
      TrmmTile<8, 8>(M, N, alpha, A, lda, B, ldb, C, ldc);
      break;
    case 16:
      if (blockDim.x != 16 || blockDim.y != 16 || blockDim.z != 1) {
        return;
      }
      TrmmTile<16, 16>(M, N, alpha, A, lda, B, ldb, C, ldc);
      break;
    case 32:
      if (blockDim.x != 32 || blockDim.y != 8 || blockDim.z != 1) {
        return;
      }
      TrmmTile<32, 8>(M, N, alpha, A, lda, B, ldb, C, ldc);
      break;
    default:
      return;
  }
}
