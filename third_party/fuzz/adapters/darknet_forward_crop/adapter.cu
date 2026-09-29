#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace darknet_crop_detail {

__host__ __device__ inline bool valid_shape(
    uint32_t batch, uint32_t channels, uint32_t height, uint32_t width,
    uint32_t crop_height, uint32_t crop_width) {
  return batch >= 1 && batch <= 4 && channels >= 1 && channels <= 4 &&
         height >= 1 && height <= 32 && width >= 1 && width <= 32 &&
         crop_height >= 1 && crop_height <= height && crop_width >= 1 &&
         crop_width <= width;
}

__host__ __device__ inline float get_pixel(
    const float* image, uint32_t width, uint32_t height, int32_t x, int32_t y,
    uint32_t channel) {
  if (x < 0 || x >= static_cast<int32_t>(width) || y < 0 ||
      y >= static_cast<int32_t>(height)) {
    return 0.0f;
  }
  return image[x + width * (y + channel * height)];
}

__host__ __device__ inline float bilinear_interpolate(
    const float* image, uint32_t width, uint32_t height, float x, float y,
    uint32_t channel) {
  const int32_t ix = static_cast<int32_t>(floorf(x));
  const int32_t iy = static_cast<int32_t>(floorf(y));
  const float dx = x - ix;
  const float dy = y - iy;
  return (1.0f - dy) * (1.0f - dx) *
             get_pixel(image, width, height, ix, iy, channel) +
         dy * (1.0f - dx) *
             get_pixel(image, width, height, ix, iy + 1, channel) +
         (1.0f - dy) * dx *
             get_pixel(image, width, height, ix + 1, iy, channel) +
         dy * dx *
             get_pixel(image, width, height, ix + 1, iy + 1, channel);
}

__host__ __device__ inline float crop_value(
    const float* input, uint32_t id, uint32_t channels, uint32_t height,
    uint32_t width, uint32_t crop_height, uint32_t crop_width, int32_t train,
    int32_t flip, float angle, float r4, float r5, float r6, float r7) {
  const float cx = width / 2.0f;
  const float cy = height / 2.0f;

  uint32_t coordinate = id;
  const uint32_t j = coordinate % crop_width;
  coordinate /= crop_width;
  const uint32_t i = coordinate % crop_height;
  coordinate /= crop_height;
  const uint32_t channel = coordinate % channels;
  const uint32_t batch = coordinate / channels;

  float dw = (width - crop_width) * r4;
  float dh = (height - crop_height) * r5;
  int32_t selected_flip = flip && r6 > 0.5f;
  float selected_angle = 2.0f * angle * r7 - angle;
  if (!train) {
    dw = (width - crop_width) / 2.0f;
    dh = (height - crop_height) / 2.0f;
    selected_flip = 0;
    selected_angle = 0.0f;
  }

  input += static_cast<size_t>(width) * height * channels * batch;
  const float x = selected_flip ? width - dw - j - 1.0f : j + dw;
  const float y = i + dh;
  const float rx = cosf(selected_angle) * (x - cx) -
                       sinf(selected_angle) * (y - cy) +
                   cx;
  const float ry = sinf(selected_angle) * (x - cx) +
                       cosf(selected_angle) * (y - cy) +
                   cy;
  return bilinear_interpolate(input, width, height, rx, ry, channel);
}

}  // namespace darknet_crop_detail

extern "C" __global__ void darknet_forward_crop_orig512(
    float* input, float* output, uint32_t batch, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t crop_height,
    uint32_t crop_width, int32_t train, int32_t flip, float angle, float r4,
    float r5, float r6, float r7) {
  if (!darknet_crop_detail::valid_shape(batch, channels, height, width,
                                        crop_height, crop_width)) {
    return;
  }
  const uint32_t size = batch * channels * crop_height * crop_width;
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = darknet_crop_detail::crop_value(
        input, id, channels, height, width, crop_height, crop_width, train,
        flip, angle, r4, r5, r6, r7);
  }
}

extern "C" __global__ void darknet_forward_crop_vgeo(
    float* input, float* output, uint32_t batch, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t crop_height,
    uint32_t crop_width, int32_t train, int32_t flip, float angle, float r4,
    float r5, float r6, float r7) {
  if (!darknet_crop_detail::valid_shape(batch, channels, height, width,
                                        crop_height, crop_width)) {
    return;
  }
  const uint32_t size = batch * channels * crop_height * crop_width;
  for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < size;
       id += blockDim.x * gridDim.x) {
    output[id] = darknet_crop_detail::crop_value(
        input, id, channels, height, width, crop_height, crop_width, train,
        flip, angle, r4, r5, r6, r7);
  }
}
