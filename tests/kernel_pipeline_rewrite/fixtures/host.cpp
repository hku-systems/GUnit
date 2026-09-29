#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static bool read_text_file(const char *path, std::string &out) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

static bool read_binary_file(const char *path, std::vector<unsigned char> &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

static bool read_pointer_patches(const char *path, std::vector<std::pair<size_t, size_t>> &out) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  size_t pointer_offset = 0;
  size_t target_offset = 0;
  while (in >> pointer_offset >> target_offset) {
    out.emplace_back(pointer_offset, target_offset);
  }
  return true;
}

int main() {
  CUdevice dev;
  CUcontext ctx;
  CUmodule mod;
  CUfunction fn;
  if (cuInit(0) != CUDA_SUCCESS) return 10;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 11;
  if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) return 12;

  std::string ptx_text;
  if (!read_text_file("linked.ptx", ptx_text)) return 13;
  if (cuModuleLoadData(&mod, ptx_text.c_str()) != CUDA_SUCCESS) return 13;
  if (cuModuleGetFunction(&fn, mod, "run_generated") != CUDA_SUCCESS) return 14;

  std::vector<unsigned char> payload;
  std::vector<unsigned char> expected;
  std::vector<std::pair<size_t, size_t>> pointer_patches;
  if (!read_binary_file("input_payload.bin", payload)) return 21;
  if (!read_binary_file("expected_payload.bin", expected)) return 22;
  read_pointer_patches("pointer_patches.txt", pointer_patches);
  if (payload.size() != expected.size()) return 23;
  size_t payload_size = payload.size();
  const std::vector<unsigned char> original_payload = payload;

  CUdeviceptr d_payload = 0;
  if (cuMemAlloc(&d_payload, payload_size) != CUDA_SUCCESS) return 15;
  for (const auto &patch : pointer_patches) {
    if (patch.first + sizeof(uintptr_t) > payload.size() || patch.second >= payload.size()) return 24;
    uintptr_t ptr_value = static_cast<uintptr_t>(d_payload + patch.second);
    for (size_t i = 0; i < sizeof(uintptr_t); ++i) {
      payload[patch.first + i] = static_cast<unsigned char>((ptr_value >> (8 * i)) & 0xFFu);
    }
  }
  if (cuMemcpyHtoD(d_payload, payload.data(), payload_size) != CUDA_SUCCESS) return 16;
  size_t size = payload_size;
  void *args[] = {&d_payload, &size};
  if (cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, args, nullptr) != CUDA_SUCCESS) return 17;
  if (cuCtxSynchronize() != CUDA_SUCCESS) return 18;
  if (cuMemcpyDtoH(payload.data(), d_payload, payload_size) != CUDA_SUCCESS) return 19;

  for (const auto &patch : pointer_patches) {
    for (size_t i = 0; i < sizeof(uintptr_t); ++i) {
      payload[patch.first + i] = original_payload[patch.first + i];
    }
  }

  cuMemFree(d_payload);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);

  for (size_t i = 0; i < payload.size(); ++i) {
    if (payload[i] != expected[i]) {
      std::printf("payload mismatch at byte %zu: got=%u expected=%u\n",
                  i,
                  static_cast<unsigned>(payload[i]),
                  static_cast<unsigned>(expected[i]));
      return 20;
    }
  }

  std::printf("payload matched (%zu bytes)\n", payload.size());
  return 0;
}
