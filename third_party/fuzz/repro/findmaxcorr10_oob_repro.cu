// Minimal repro: FindMaxCorr10 negative-index OOB when numPts2 < 32.
// Build from the repository root:
//   /usr/local/cuda/bin/nvcc -std=c++14 -arch=sm_86 \
//     -I third_party/CudaSift -o /tmp/fmc10_repro \
//     third_party/fuzz/repro/findmaxcorr10_oob_repro.cu \
//     third_party/CudaSift/matching.cu third_party/CudaSift/cudaImage.cu
// Run with sanitizer:
//   compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/fmc10_repro

#include <cstdio>
#include <cuda_runtime.h>
#include "cudaSift.h"

__global__ void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2);

int main() {
  const int N1 = 32, N2 = 1;

  SiftPoint *d_sift1, *d_sift2;
  cudaMalloc(&d_sift1, N1 * sizeof(SiftPoint));
  cudaMalloc(&d_sift2, N2 * sizeof(SiftPoint));
  cudaMemset(d_sift1, 0, N1 * sizeof(SiftPoint));
  cudaMemset(d_sift2, 0, N2 * sizeof(SiftPoint));

  // Fill sift2 with a recognizable pattern so OOB reads are distinguishable.
  SiftPoint pt = {};
  pt.xpos = 42.0f;
  pt.ypos = 84.0f;
  for (int i = 0; i < 128; i++) pt.data[i] = 1.0f;
  cudaMemcpy(d_sift2, &pt, sizeof(SiftPoint), cudaMemcpyHostToDevice);

  // Fill sift1 descriptors so dot products are non-zero.
  SiftPoint pt1 = {};
  for (int i = 0; i < 128; i++) pt1.data[i] = 1.0f;
  for (int i = 0; i < N1; i++)
    cudaMemcpy(d_sift1 + i, &pt1, sizeof(SiftPoint), cudaMemcpyHostToDevice);

  // numPts2=1 < M7H(32): inner loop body never executes, index stays -1.
  // Output path reads sift2[-1] => OOB.
  FindMaxCorr10<<<1, dim3(32, 32)>>>(d_sift1, d_sift2, N1, N2);
  cudaDeviceSynchronize();

  SiftPoint result;
  cudaMemcpy(&result, d_sift1, sizeof(SiftPoint), cudaMemcpyDeviceToHost);
  printf("sift1[0].match      = %d\n", result.match);
  printf("sift1[0].match_xpos = %f\n", result.match_xpos);
  printf("sift1[0].match_ypos = %f\n", result.match_ypos);
  printf("sift1[0].score      = %f\n", result.score);

  cudaFree(d_sift1);
  cudaFree(d_sift2);
  return 0;
}
