#include <cstdint>

// Negative control: faithful copy of Apex's fixed-geometry generic float forward kernel.
__global__ void index_mul_2d_float_orig(float* out, const float* in1, const float* in2,
                                        const int64_t* idx1, const int64_t size,
                                        const int64_t fea_dim) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int bidx = blockIdx.x;
  const int start_idx = bidx * blockDim.y + tidy;
  const int stride = blockDim.x;

  if (start_idx < size) {
    int64_t vec_idx1 = (idx1[start_idx] * fea_dim);
    int64_t vec_idx2 = (start_idx * fea_dim);

    for (int i = tidx; i < fea_dim; i += stride) {
      out[vec_idx2 + i] = in1[vec_idx1 + i] * in2[vec_idx2 + i];
    }
  }
}

// GEOMETRY-FREED variant: the vendor body already derives ownership from blockDim.
__global__ void index_mul_2d_float_vgeo(float* out, const float* in1, const float* in2,
                                        const int64_t* idx1, const int64_t size,
                                        const int64_t fea_dim) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int bidx = blockIdx.x;
  const int start_idx = bidx * blockDim.y + tidy;
  const int stride = blockDim.x;

  if (start_idx < size) {
    int64_t vec_idx1 = (idx1[start_idx] * fea_dim);
    int64_t vec_idx2 = (start_idx * fea_dim);

    for (int i = tidx; i < fea_dim; i += stride) {
      out[vec_idx2 + i] = in1[vec_idx1 + i] * in2[vec_idx2 + i];
    }
  }
}
