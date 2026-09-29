// Standalone extraction of CUTLASS's naive ReferenceGemm_kernel example.
extern "C" __global__ void rq1_cutlass_gemm(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    float* C,
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
