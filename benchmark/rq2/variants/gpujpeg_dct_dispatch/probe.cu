#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../../../third_party/GPUJPEG/src/gpujpeg_dct_gpu.cu"
#include "gpujpeg_dct_dispatch.cu"

namespace {

constexpr int kGuardElements = 16;
constexpr int16_t kOutputCanary = 0x5a5a;

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    bool allocate(size_t count)
    {
        return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)) == cudaSuccess;
    }

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

bool check_cuda(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

template <typename T>
bool copy_to_device(DeviceBuffer<T>& device, const std::vector<T>& host, const char* operation)
{
    return check_cuda(cudaMemcpy(device.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), operation);
}

template <typename T>
bool copy_from_device(std::vector<T>& host, const DeviceBuffer<T>& device, const char* operation)
{
    return check_cuda(cudaMemcpy(host.data(), device.get(), host.size() * sizeof(T), cudaMemcpyDeviceToHost), operation);
}

template <typename T>
bool unchanged(const std::vector<T>& expected, const std::vector<T>& actual, const char* name)
{
    if (expected == actual) {
        return true;
    }
    std::fprintf(stderr, "%s or its canary was modified\n", name);
    return false;
}

bool check_output(const std::vector<int16_t>& reference, const std::vector<int16_t>& dispatcher)
{
    for (int index = 0; index < kGuardElements; ++index) {
        const size_t suffix = reference.size() - kGuardElements + index;
        if (reference[index] != kOutputCanary || dispatcher[index] != kOutputCanary ||
            reference[suffix] != kOutputCanary || dispatcher[suffix] != kOutputCanary) {
            std::fprintf(stderr, "output canary modified\n");
            return false;
        }
    }
    if (reference == dispatcher) {
        return true;
    }
    for (size_t index = 0; index < reference.size(); ++index) {
        if (reference[index] != dispatcher[index]) {
            std::fprintf(stderr, "dispatcher mismatch at output index %zu\n", index);
            break;
        }
    }
    return false;
}

bool run_case(uint32_t warp_count, int block_count_y, int pattern)
{
    constexpr int block_count_x = 4;
    constexpr unsigned int source_stride = 8 * block_count_x;
    constexpr int output_stride = 64 * block_count_x;

    const size_t source_elements = static_cast<size_t>(source_stride) * block_count_y * 8;
    const size_t output_elements = static_cast<size_t>(output_stride) * block_count_y;
    std::vector<uint8_t> source(kGuardElements + source_elements + kGuardElements, 0xa5);
    for (size_t index = 0; index < source_elements; ++index) {
        source[kGuardElements + index] = static_cast<uint8_t>(index * 37 + pattern * 53 + (index / source_stride) * 11);
    }
    std::vector<float> quant(kGuardElements + 64 + kGuardElements, 12345.25f);
    for (int index = 0; index < 64; ++index) {
        quant[kGuardElements + index] = 0.015625f * static_cast<float>(1 + (index + pattern * 7) % 23);
    }
    std::vector<int16_t> initial_output(kGuardElements + output_elements + kGuardElements, -22222);
    for (int index = 0; index < kGuardElements; ++index) {
        initial_output[index] = kOutputCanary;
        initial_output[initial_output.size() - kGuardElements + index] = kOutputCanary;
    }

    DeviceBuffer<uint8_t> reference_source;
    DeviceBuffer<uint8_t> dispatcher_source;
    DeviceBuffer<float> reference_quant;
    DeviceBuffer<float> dispatcher_quant;
    DeviceBuffer<int16_t> reference_output;
    DeviceBuffer<int16_t> dispatcher_output;
    if (!reference_source.allocate(source.size()) || !dispatcher_source.allocate(source.size()) ||
        !reference_quant.allocate(quant.size()) || !dispatcher_quant.allocate(quant.size()) ||
        !reference_output.allocate(initial_output.size()) || !dispatcher_output.allocate(initial_output.size()) ||
        !copy_to_device(reference_source, source, "copy reference source") ||
        !copy_to_device(dispatcher_source, source, "copy dispatcher source") ||
        !copy_to_device(reference_quant, quant, "copy reference quant table") ||
        !copy_to_device(dispatcher_quant, quant, "copy dispatcher quant table") ||
        !copy_to_device(reference_output, initial_output, "copy reference output") ||
        !copy_to_device(dispatcher_output, initial_output, "copy dispatcher output")) {
        return false;
    }

    gpujpeg_dct_gpu_kernel<4><<<dim3(1, (block_count_y + 3) / 4, 1), dim3(32, 4, 1)>>>(
        block_count_x, block_count_y, reference_source.get() + kGuardElements, source_stride,
        reference_output.get() + kGuardElements, output_stride, reference_quant.get() + kGuardElements);
    gpujpeg_dct_dispatch_kernel<<<dim3(1, 1, 1), dim3(32, warp_count, 1)>>>(
        warp_count, block_count_x, block_count_y, dispatcher_source.get() + kGuardElements, source_stride,
        dispatcher_output.get() + kGuardElements, output_stride, dispatcher_quant.get() + kGuardElements);
    if (!check_cuda(cudaGetLastError(), "launch kernels") ||
        !check_cuda(cudaDeviceSynchronize(), "synchronize kernels")) {
        return false;
    }

    std::vector<uint8_t> reference_source_after(source.size());
    std::vector<uint8_t> dispatcher_source_after(source.size());
    std::vector<float> reference_quant_after(quant.size());
    std::vector<float> dispatcher_quant_after(quant.size());
    std::vector<int16_t> reference_output_after(initial_output.size());
    std::vector<int16_t> dispatcher_output_after(initial_output.size());
    if (!copy_from_device(reference_source_after, reference_source, "copy reference source result") ||
        !copy_from_device(dispatcher_source_after, dispatcher_source, "copy dispatcher source result") ||
        !copy_from_device(reference_quant_after, reference_quant, "copy reference quant result") ||
        !copy_from_device(dispatcher_quant_after, dispatcher_quant, "copy dispatcher quant result") ||
        !copy_from_device(reference_output_after, reference_output, "copy reference output result") ||
        !copy_from_device(dispatcher_output_after, dispatcher_output, "copy dispatcher output result")) {
        return false;
    }
    return unchanged(source, reference_source_after, "reference source") &&
           unchanged(source, dispatcher_source_after, "dispatcher source") &&
           unchanged(quant, reference_quant_after, "reference quant table") &&
           unchanged(quant, dispatcher_quant_after, "dispatcher quant table") &&
           check_output(reference_output_after, dispatcher_output_after);
}

} // namespace

int main()
{
    constexpr std::array<uint32_t, 3> warp_counts = {2, 4, 8};
    for (uint32_t warp_count : warp_counts) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            for (int block_count_y = 1; block_count_y <= static_cast<int>(warp_count); ++block_count_y) {
                if (!run_case(warp_count, block_count_y, pattern)) {
                    std::fprintf(stderr, "failed warp_count=%u, block_count_y=%d, pattern=%d\n",
                                 warp_count, block_count_y, pattern);
                    return 1;
                }
            }
        }
    }
    std::puts("GPUJPEG DCT dispatcher probe passed");
    return 0;
}
