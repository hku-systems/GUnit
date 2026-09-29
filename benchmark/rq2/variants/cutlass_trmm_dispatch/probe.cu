#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

#define main cutlass_reference_trmm_example_main
#include "../../../../third_party/cutlass/examples/32_basic_trmm/basic_trmm.cu"
#undef main

#include "reference_trmm_dispatch.cu"

namespace {

constexpr int kGuardElements = 4;
constexpr double kCanary = 987654.5;
constexpr double kPaddingSentinel = -777.25;

template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(DeviceBuffer const &) = delete;
  DeviceBuffer &operator=(DeviceBuffer const &) = delete;

  ~DeviceBuffer() {
    if (data_) {
      cudaFree(data_);
    }
  }

  bool allocate(size_t count) {
    return cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)) == cudaSuccess;
  }

  T *get() const { return data_; }

private:
  T *data_ = nullptr;
};

bool SameBits(double lhs, double rhs) {
  return std::memcmp(&lhs, &rhs, sizeof(double)) == 0;
}

bool CheckCuda(cudaError_t status, char const *operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool CheckOutput(
  std::vector<double> const &reference,
  std::vector<double> const &dispatcher,
  int M,
  int N,
  int ldc) {

  if (reference.size() != dispatcher.size()) {
    return false;
  }
  for (int index = 0; index < kGuardElements; ++index) {
    size_t suffix = reference.size() - kGuardElements + index;
    if (!SameBits(reference[index], kCanary) ||
        !SameBits(dispatcher[index], kCanary) ||
        !SameBits(reference[suffix], kCanary) ||
        !SameBits(dispatcher[suffix], kCanary)) {
      std::fprintf(stderr, "C canary modified\n");
      return false;
    }
  }
  for (int j = 0; j < N; ++j) {
    for (int i = M; i < ldc; ++i) {
      size_t index = kGuardElements + i + j * ldc;
      if (!SameBits(reference[index], kPaddingSentinel) ||
          !SameBits(dispatcher[index], kPaddingSentinel)) {
        std::fprintf(stderr, "C padding modified at (%d, %d)\n", i, j);
        return false;
      }
    }
  }
  for (size_t index = 0; index < reference.size(); ++index) {
    if (!SameBits(reference[index], dispatcher[index])) {
      std::fprintf(stderr, "dispatcher mismatch at storage index %zu\n", index);
      return false;
    }
  }
  return true;
}

bool RunCase(int tile_x, int tile_y, int M, int N) {
  int lda = M + 3;
  int ldb = M + 4;
  int ldc = M + 5;
  size_t a_elements = static_cast<size_t>(lda) * M;
  size_t b_elements = static_cast<size_t>(ldb) * N;
  size_t c_elements = static_cast<size_t>(ldc) * N;
  size_t c_storage_elements = kGuardElements + c_elements + kGuardElements;

  std::vector<double> A(a_elements);
  std::vector<double> B(b_elements);
  for (size_t index = 0; index < A.size(); ++index) {
    A[index] = (static_cast<int>(index % 19) - 9) * 0.125;
  }
  for (size_t index = 0; index < B.size(); ++index) {
    B[index] = (static_cast<int>(index % 23) - 11) * 0.0625;
  }

  std::vector<double> initial(c_storage_elements, kPaddingSentinel);
  for (int index = 0; index < kGuardElements; ++index) {
    initial[index] = kCanary;
    initial[initial.size() - kGuardElements + index] = kCanary;
  }
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      initial[kGuardElements + i + j * ldc] = (i + 3 * j + 1) * 0.25;
    }
  }

  DeviceBuffer<double> device_A;
  DeviceBuffer<double> device_B;
  DeviceBuffer<double> reference_storage;
  DeviceBuffer<double> dispatcher_storage;
  if (!device_A.allocate(a_elements) ||
      !device_B.allocate(b_elements) ||
      !reference_storage.allocate(c_storage_elements) ||
      !dispatcher_storage.allocate(c_storage_elements)) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return false;
  }
  if (!CheckCuda(cudaMemcpy(device_A.get(), A.data(), A.size() * sizeof(double),
                            cudaMemcpyHostToDevice), "copy A") ||
      !CheckCuda(cudaMemcpy(device_B.get(), B.data(), B.size() * sizeof(double),
                            cudaMemcpyHostToDevice), "copy B") ||
      !CheckCuda(cudaMemcpy(reference_storage.get(), initial.data(),
                            initial.size() * sizeof(double), cudaMemcpyHostToDevice),
                 "copy reference C") ||
      !CheckCuda(cudaMemcpy(dispatcher_storage.get(), initial.data(),
                            initial.size() * sizeof(double), cudaMemcpyHostToDevice),
                 "copy dispatcher C")) {
    return false;
  }

  dim3 block(tile_x, tile_y, 1);
  dim3 grid(1, 1, 1);
  double *reference_C = reference_storage.get() + kGuardElements;
  double *dispatcher_C = dispatcher_storage.get() + kGuardElements;
  ReferenceTrmm_kernel<<<grid, block>>>(
    M, N, 1.25, device_A.get(), lda, device_B.get(), ldb, reference_C, ldc);
  reference_trmm_dispatch_kernel<<<grid, block>>>(
    static_cast<uint32_t>(tile_x), M, N, 1.25, device_A.get(), lda,
    device_B.get(), ldb, dispatcher_C, ldc);
  if (!CheckCuda(cudaGetLastError(), "launch kernels") ||
      !CheckCuda(cudaDeviceSynchronize(), "synchronize kernels")) {
    return false;
  }

  std::vector<double> reference(c_storage_elements);
  std::vector<double> dispatcher(c_storage_elements);
  if (!CheckCuda(cudaMemcpy(reference.data(), reference_storage.get(),
                            reference.size() * sizeof(double), cudaMemcpyDeviceToHost),
                 "copy reference result") ||
      !CheckCuda(cudaMemcpy(dispatcher.data(), dispatcher_storage.get(),
                            dispatcher.size() * sizeof(double), cudaMemcpyDeviceToHost),
                 "copy dispatcher result")) {
    return false;
  }
  return CheckOutput(reference, dispatcher, M, N, ldc);
}

} // namespace

int main() {
  constexpr std::array<std::array<int, 2>, 3> kTiles = {{{8, 8}, {16, 16}, {32, 8}}};
  constexpr std::array<std::array<int, 2>, 4> kCases = {{{1, 1}, {7, 5}, {8, 7}, {5, 8}}};

  for (auto const &tile : kTiles) {
    for (auto const &shape : kCases) {
      if (!RunCase(tile[0], tile[1], shape[0], shape[1])) {
        std::fprintf(stderr, "failed tile=(%d,%d), M=%d, N=%d\n",
                     tile[0], tile[1], shape[0], shape[1]);
        return 1;
      }
    }
  }
  std::puts("ReferenceTrmm dispatcher probe passed");
  return 0;
}
