// Standalone float/float/float, PARALLEL_LOADS=2 extraction of PyTorch's NHWC
// BatchNorm forward body. The fixed early return remains before all statistics
// and affine-array reads.
extern "C" __global__ void rq1_pytorch_batchnorm(
    const float* __restrict__ input,
    const float* __restrict__ z,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    const float* __restrict__ weight,
    const float* __restrict__ shift,
    float* __restrict__ out,
    const int reduction_size,
    const int stride,
    const bool fuse_relu) {
  constexpr int PARALLEL_LOADS = 2;
  int inner_loop_stride = blockDim.y * gridDim.y;
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (c_offset >= stride || m_offset >= reduction_size) {
    return;
  }

  auto m_c = mean[c_offset];
  auto inv_std_c = static_cast<float>(inv_std[c_offset]);
  auto w_c = weight == nullptr ? float(1.0) : static_cast<float>(weight[c_offset]);
  auto s_c = shift == nullptr ? float(0.0) : static_cast<float>(shift[c_offset]);

  int loop_count =
      1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        auto tmp =
            w_c * (static_cast<float>(input[address_base]) - m_c) * inv_std_c +
            s_c;
        if (z != nullptr) {
          tmp += z[address_base];
        }
        out[address_base] =
            (fuse_relu && tmp <= float(0.0) ? float(0.0) : static_cast<float>(tmp));
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}
