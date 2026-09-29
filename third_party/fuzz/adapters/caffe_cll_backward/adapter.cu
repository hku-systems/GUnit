#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace caffe_cll_detail {

__host__ __device__ inline bool valid_shape(uint32_t batch, uint32_t channels) {
  return batch >= 1 && batch <= 32 && channels >= 1 && channels <= 256 &&
         static_cast<uint64_t>(batch) * channels <= 2048;
}

__host__ __device__ inline float backward_value(
    uint32_t i, uint32_t channels, float margin, int32_t legacy_version,
    float alpha, const float* y, const float* diff, const float* dist_sq) {
  const uint32_t n = i / channels;
  if (static_cast<int32_t>(y[n]) != 0) {
    return alpha * diff[i];
  }

  float mdist;
  float beta;
  if (legacy_version != 0) {
    mdist = margin - dist_sq[n];
    beta = -alpha;
  } else {
    const float dist = sqrtf(dist_sq[n]);
    mdist = margin - dist;
    beta = -alpha * mdist / (dist + 1.0e-4f) * diff[i];
  }
  return mdist > 0.0f ? beta : 0.0f;
}

}  // namespace caffe_cll_detail

extern "C" __global__ void caffe_cll_backward_f32(
    uint32_t batch, uint32_t channels, float margin,
    int32_t legacy_version, float alpha, const float* y, const float* diff,
    const float* dist_sq, float* bottom_diff) {
  if (!caffe_cll_detail::valid_shape(batch, channels)) {
    return;
  }
  const uint32_t count = batch * channels;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    bottom_diff[i] = caffe_cll_detail::backward_value(
        i, channels, margin, legacy_version, alpha, y, diff, dist_sq);
  }
}
