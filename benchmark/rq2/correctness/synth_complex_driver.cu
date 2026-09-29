#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

extern "C" __global__ void rq1_synth_complex(
    const unsigned char* input, unsigned int* output, unsigned int n);

namespace {

constexpr unsigned int kInputCapacity = 4096;
constexpr unsigned int kOutputCapacity = 256;
constexpr unsigned int kSentinel = 0xa5a5a5a5u;

struct MagicChain {
  unsigned int base;
  unsigned int salt;
  std::array<unsigned int, 4> targets;
};

constexpr std::array<MagicChain, 24> kMagicChains = {{
    {0u, 0x5a17u, {0x1u, 0x7u, 0xau, 0x12u}},
    {4u, 0xc3d2u, {0xdu, 0x2u, 0x4u, 0x03u}},
    {8u, 0x91e4u, {0x6u, 0xbu, 0x1u, 0x1cu}},
    {12u, 0x2b6du, {0x8u, 0x0u, 0xeu, 0x09u}},
    {16u, 0x7f08u, {0x3u, 0xcu, 0x5u, 0x16u}},
    {20u, 0xa465u, {0xfu, 0x4u, 0x9u, 0x01u}},
    {24u, 0x1d3bu, {0x2u, 0xeu, 0x7u, 0x1au}},
    {28u, 0xe290u, {0xau, 0x6u, 0x0u, 0x0du}},
    {32u, 0x48c7u, {0x5u, 0x9u, 0xcu, 0x14u}},
    {36u, 0xb51au, {0xcu, 0x3u, 0x2u, 0x07u}},
    {40u, 0x63f4u, {0x0u, 0xdu, 0x8u, 0x1eu}},
    {44u, 0xd82eu, {0x7u, 0x5u, 0xfu, 0x0bu}},
    {48u, 0x0fa9u, {0xeu, 0x1u, 0x6u, 0x18u}},
    {52u, 0x956cu, {0x4u, 0xau, 0x3u, 0x05u}},
    {56u, 0x3e71u, {0xbu, 0x8u, 0xdu, 0x10u}},
    {60u, 0xac04u, {0x9u, 0xfu, 0xbu, 0x02u}},
    {64u, 0x729bu, {0x1u, 0x6u, 0x4u, 0x1du}},
    {68u, 0xe15du, {0xdu, 0xcu, 0xau, 0x08u}},
    {72u, 0x4c86u, {0x6u, 0x2u, 0x1u, 0x15u}},
    {76u, 0xb307u, {0x8u, 0xbu, 0xeu, 0x04u}},
    {80u, 0x68dau, {0x3u, 0x0u, 0x5u, 0x1bu}},
    {84u, 0xd4f1u, {0xfu, 0x9u, 0x9u, 0x0eu}},
    {88u, 0x237cu, {0x2u, 0x3u, 0x7u, 0x13u}},
    {92u, 0x9e50u, {0xau, 0xdu, 0x0u, 0x06u}},
}};

bool check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

unsigned int rotl(unsigned int value, unsigned int shift) {
  return (value << shift) | (value >> (32u - shift));
}

void apply_magic_chains(const std::vector<unsigned char>& input,
                        unsigned int& acc) {
  constexpr std::array<unsigned int, 4> kMasks = {
      0x0fu, 0x0fu, 0x0fu, 0x1fu};
  for (const MagicChain& chain : kMagicChains) {
    for (unsigned int depth = 0; depth < 4u; ++depth) {
      const unsigned int byte = input[chain.base + depth];
      const unsigned int salt_bits =
          (chain.salt >> (depth * 4u)) & kMasks[depth];
      if (((byte ^ salt_bits) & kMasks[depth]) != chain.targets[depth]) {
        break;
      }
      switch (depth) {
        case 0u:
          acc = rotl(acc ^ (byte + chain.salt + 0x00009e37u), 5u) +
                0x7f4a7c15u;
          break;
        case 1u:
          acc = rotl(acc + (byte ^ chain.salt ^ 0x0000b529u), 7u) ^
                0x68e31da4u;
          break;
        case 2u:
          acc = rotl(acc ^ (byte + chain.salt + 0x0001c2b3u), 11u) +
                0x1b873593u;
          break;
        default:
          acc = rotl(acc + (byte ^ chain.salt ^ 0x00027d4du), 13u) ^
                (0xd00d0000u | chain.base);
          break;
      }
    }
  }
}

unsigned int reference_output(const std::vector<unsigned char>& input,
                              unsigned int n, unsigned int block,
                              unsigned int tid) {
  unsigned int acc = 0x811c9dc5u ^ n ^ (block << 16u) ^ tid;
  for (unsigned int i = tid; i < n; i += block) {
    const unsigned int value = input[i];
    const unsigned int displaced = i + block;
    const unsigned int neighbor =
        displaced < n ? input[displaced] : input[i];
    const unsigned int left = acc ^ (value * 0x45d9f3bu);
    const unsigned int right = acc + neighbor + 0x9e3779b9u;
    const bool paired_condition = ((value + i) & 1u) != 0u;
    const unsigned int select_value = paired_condition ? left : right;
    const unsigned int branch_value = paired_condition ? left : right;
    acc = rotl(acc + select_value, 9u) ^ branch_value;

    switch ((value ^ neighbor ^ (acc >> 11u)) & 7u) {
      case 0u:
        acc += 0x243f6a88u;
        break;
      case 1u:
        acc ^= 0x85a308d3u;
        break;
      case 2u:
        acc = rotl(acc, 3u) + 0x13198a2eu;
        break;
      case 3u:
        acc = acc * 33u + 0x03707344u;
        break;
      case 4u:
        acc ^= acc >> 7u;
        break;
      case 5u:
        acc += value * 257u;
        break;
      case 6u:
        acc = rotl(acc ^ neighbor, 17u);
        break;
      default:
        acc += (value << 8u) | neighbor;
        break;
    }
    const unsigned int byte_guard = (value ^ (tid * 13u)) & 0xffu;
    if (byte_guard == 0xa5u) {
      acc ^= 0x5a17c3d2u;
    } else if (byte_guard == 0x3cu) {
      acc += 0x1234abcdu;
    } else if ((byte_guard ^ 0x5au) == 0x17u) {
      acc = rotl(acc, 19u);
    } else {
      acc += byte_guard;
    }
  }
  if (tid == 0u) {
    apply_magic_chains(input, acc);
  }
  return acc;
}

std::vector<unsigned char> make_input(bool hit_all_magic) {
  std::vector<unsigned char> input(kInputCapacity);
  for (unsigned int index = 0; index < kInputCapacity; ++index) {
    input[index] = static_cast<unsigned char>(
        (index * 73u + (index >> 2u) * 29u + 0x5bu) & 0xffu);
  }
  if (hit_all_magic) {
    constexpr std::array<unsigned int, 4> kMasks = {
        0x0fu, 0x0fu, 0x0fu, 0x1fu};
    for (const MagicChain& chain : kMagicChains) {
      for (unsigned int depth = 0; depth < 4u; ++depth) {
        const unsigned int salt_bits =
            (chain.salt >> (depth * 4u)) & kMasks[depth];
        input[chain.base + depth] = static_cast<unsigned char>(
            chain.targets[depth] ^ salt_bits);
      }
    }
  }
  return input;
}

bool run_case(int block, unsigned int n, bool hit_all_magic) {
  const std::vector<unsigned char> input = make_input(hit_all_magic);
  const std::vector<unsigned int> initial_output(kOutputCapacity, kSentinel);

  unsigned char* device_input = nullptr;
  unsigned int* device_output = nullptr;
  bool ok = check_cuda(cudaMalloc(&device_input, input.size()), "cudaMalloc input") &&
            check_cuda(cudaMalloc(&device_output,
                                  initial_output.size() * sizeof(unsigned int)),
                       "cudaMalloc output");
  if (ok) {
    ok = check_cuda(cudaMemcpy(device_input, input.data(), input.size(),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy input H2D") &&
         check_cuda(cudaMemcpy(device_output, initial_output.data(),
                               initial_output.size() * sizeof(unsigned int),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy output H2D");
  }
  if (ok) {
    rq1_synth_complex<<<1, block>>>(device_input, device_output, n);
    ok = check_cuda(cudaGetLastError(), "synth_complex launch") &&
         check_cuda(cudaDeviceSynchronize(), "synth_complex synchronize");
  }

  std::vector<unsigned int> actual(kOutputCapacity);
  if (ok) {
    ok = check_cuda(cudaMemcpy(actual.data(), device_output,
                               actual.size() * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy output D2H");
  }
  if (device_output != nullptr) {
    ok = check_cuda(cudaFree(device_output), "cudaFree output") && ok;
  }
  if (device_input != nullptr) {
    ok = check_cuda(cudaFree(device_input), "cudaFree input") && ok;
  }
  if (!ok) {
    return false;
  }

  for (int tid = 0; tid < block; ++tid) {
    const unsigned int expected =
        reference_output(input, n, static_cast<unsigned int>(block), tid);
    if (actual[tid] != expected) {
      std::fprintf(stderr,
                   "synth_complex mismatch block=%d n=%u magic=%d index=%d "
                   "expected=0x%08x actual=0x%08x\n",
                   block, n, hit_all_magic ? 1 : 0, tid, expected, actual[tid]);
      return false;
    }
  }
  for (unsigned int tid = static_cast<unsigned int>(block);
       tid < kOutputCapacity; ++tid) {
    if (actual[tid] != kSentinel) {
      std::fprintf(stderr,
                   "synth_complex sentinel corrupted block=%d n=%u magic=%d "
                   "index=%u actual=0x%08x\n",
                   block, n, hit_all_magic ? 1 : 0, tid, actual[tid]);
      return false;
    }
  }
  return true;
}

std::set<unsigned int> case_sizes(int block) {
  return {
      128u,
      129u,
      255u,
      256u,
      257u,
      static_cast<unsigned int>(block + 127),
      1024u,
      4095u,
      4096u,
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "expected at least one block candidate\n");
    return 2;
  }

  int cases = 0;
  for (int arg = 1; arg < argc; ++arg) {
    const int block = std::atoi(argv[arg]);
    if (block <= 0 || block > static_cast<int>(kOutputCapacity)) {
      std::fprintf(stderr, "invalid block candidate %s\n", argv[arg]);
      return 2;
    }
    for (unsigned int n : case_sizes(block)) {
      for (bool hit_all_magic : {false, true}) {
        if (!run_case(block, n, hit_all_magic)) {
          return 1;
        }
        ++cases;
      }
    }
  }
  std::printf("synth_complex correctness passed: %d cases\n", cases);
  return 0;
}
