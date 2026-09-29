#include <cuda_runtime.h>

template <typename T>
struct template_box {
  T* out;
  T value;
};

template <typename T>
__global__ void templated_entry_kernel(const template_box<T> box) {
  if (box.out != nullptr) {
    box.out[0] = box.value;
  }
}

template __global__ void templated_entry_kernel<int>(template_box<int>);
template __global__ void templated_entry_kernel<float>(template_box<float>);
