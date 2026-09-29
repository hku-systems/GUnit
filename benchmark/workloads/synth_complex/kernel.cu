static __device__ __forceinline__ unsigned int synth_rotl(unsigned int value,
                                                          unsigned int shift) {
  return (value << shift) | (value >> (32u - shift));
}

static __device__ __forceinline__ unsigned int synth_select_form(
    bool condition, unsigned int true_value, unsigned int false_value) {
  return condition ? true_value : false_value;
}

static __device__ __forceinline__ unsigned int synth_if_form(
    bool condition, unsigned int true_value, unsigned int false_value,
    volatile unsigned int* branch_sink) {
  unsigned int result;
  if (condition) {
    result = true_value;
    branch_sink[0] = result;
  } else {
    result = false_value;
  }
  return result;
}

// Each chain has cumulative hit probabilities 1/16, 1/256, 1/4096, and
// 1/131072 under uniform bytes. Conditional volatile stores keep the guarded
// paths as branches through the O3 capture pipeline; the final store below
// makes the externally visible output independent of those temporary stores.
#define SYNTH_MAGIC_CHAIN(BASE, SALT, T0, T1, T2, T3)                    \
  do {                                                                  \
    const unsigned int synth_b0 = input[(BASE) + 0u];                   \
    if (((synth_b0 ^ (((SALT) >> 0u) & 0x0fu)) & 0x0fu) == (T0)) {     \
      acc = synth_rotl(acc ^ (synth_b0 + (SALT) + 0x00009e37u), 5u) +  \
            0x7f4a7c15u;                                               \
      branch_sink[0] = acc;                                             \
      const unsigned int synth_b1 = input[(BASE) + 1u];                 \
      if (((synth_b1 ^ (((SALT) >> 4u) & 0x0fu)) & 0x0fu) == (T1)) {   \
        acc = synth_rotl(acc + (synth_b1 ^ (SALT) ^ 0x0000b529u), 7u) ^ \
              0x68e31da4u;                                             \
        branch_sink[0] = acc;                                           \
        const unsigned int synth_b2 = input[(BASE) + 2u];               \
        if (((synth_b2 ^ (((SALT) >> 8u) & 0x0fu)) & 0x0fu) == (T2)) { \
          acc = synth_rotl(acc ^ (synth_b2 + (SALT) + 0x0001c2b3u),    \
                           11u) +                                      \
                0x1b873593u;                                           \
          branch_sink[0] = acc;                                         \
          const unsigned int synth_b3 = input[(BASE) + 3u];             \
          if (((synth_b3 ^ (((SALT) >> 12u) & 0x1fu)) & 0x1fu) ==      \
              (T3)) {                                                   \
            acc = synth_rotl(acc + (synth_b3 ^ (SALT) ^ 0x00027d4du),  \
                             13u) ^                                    \
                  (0xd00d0000u | (BASE));                              \
            branch_sink[0] = acc;                                       \
          }                                                             \
        }                                                               \
      }                                                                 \
    }                                                                   \
  } while (0)

extern "C" __global__ void rq1_synth_complex(
    const unsigned char* input, unsigned int* output,
    const unsigned int n) {
  const unsigned int tid = threadIdx.x;
  volatile unsigned int* branch_sink = output;
  unsigned int acc = 0x811c9dc5u ^ n ^ (blockDim.x << 16u) ^ tid;

  // The two payload sites deliberately depend on blockDim.x. Across logical
  // block widths, sectors are assigned to different logical warps and the
  // displaced site also changes the addresses that are reached.
  for (unsigned int i = tid; i < n; i += blockDim.x) {
    const unsigned int value = input[i];
    const unsigned int displaced = i + blockDim.x;
    unsigned int neighbor;
    if (displaced < n) {
      neighbor = input[displaced];
    } else {
      neighbor = input[i];
    }

    const unsigned int left = acc ^ (value * 0x45d9f3bu);
    const unsigned int right = acc + neighbor + 0x9e3779b9u;

    // Controlled WS4 carrier: identical predicate and arms, expressed once
    // as a side-effect-free select and once as explicit control flow. The
    // force-inlined helpers keep both representations inside the instrumented
    // entry function; the volatile store preserves the if-form's real CFG.
    const bool paired_condition = ((value + i) & 1u) != 0u;
    const unsigned int select_value =
        synth_select_form(paired_condition, left, right);
    const unsigned int branch_value =
        synth_if_form(paired_condition, left, right, branch_sink + tid);
    acc = synth_rotl(acc + select_value, 9u) ^ branch_value;

    switch ((value ^ neighbor ^ (acc >> 11u)) & 7u) {
      case 0u:
        acc += 0x243f6a88u;
        break;
      case 1u:
        acc ^= 0x85a308d3u;
        break;
      case 2u:
        acc = synth_rotl(acc, 3u) + 0x13198a2eu;
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
        acc = synth_rotl(acc ^ neighbor, 17u);
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
      acc = synth_rotl(acc, 19u);
    } else {
      acc += byte_guard;
    }
  }

  if (tid == 0u) {
    SYNTH_MAGIC_CHAIN(0u, 0x5a17u, 0x1u, 0x7u, 0xau, 0x12u);
    SYNTH_MAGIC_CHAIN(4u, 0xc3d2u, 0xdu, 0x2u, 0x4u, 0x03u);
    SYNTH_MAGIC_CHAIN(8u, 0x91e4u, 0x6u, 0xbu, 0x1u, 0x1cu);
    SYNTH_MAGIC_CHAIN(12u, 0x2b6du, 0x8u, 0x0u, 0xeu, 0x09u);
    SYNTH_MAGIC_CHAIN(16u, 0x7f08u, 0x3u, 0xcu, 0x5u, 0x16u);
    SYNTH_MAGIC_CHAIN(20u, 0xa465u, 0xfu, 0x4u, 0x9u, 0x01u);
    SYNTH_MAGIC_CHAIN(24u, 0x1d3bu, 0x2u, 0xeu, 0x7u, 0x1au);
    SYNTH_MAGIC_CHAIN(28u, 0xe290u, 0xau, 0x6u, 0x0u, 0x0du);
    SYNTH_MAGIC_CHAIN(32u, 0x48c7u, 0x5u, 0x9u, 0xcu, 0x14u);
    SYNTH_MAGIC_CHAIN(36u, 0xb51au, 0xcu, 0x3u, 0x2u, 0x07u);
    SYNTH_MAGIC_CHAIN(40u, 0x63f4u, 0x0u, 0xdu, 0x8u, 0x1eu);
    SYNTH_MAGIC_CHAIN(44u, 0xd82eu, 0x7u, 0x5u, 0xfu, 0x0bu);
    SYNTH_MAGIC_CHAIN(48u, 0x0fa9u, 0xeu, 0x1u, 0x6u, 0x18u);
    SYNTH_MAGIC_CHAIN(52u, 0x956cu, 0x4u, 0xau, 0x3u, 0x05u);
    SYNTH_MAGIC_CHAIN(56u, 0x3e71u, 0xbu, 0x8u, 0xdu, 0x10u);
    SYNTH_MAGIC_CHAIN(60u, 0xac04u, 0x9u, 0xfu, 0xbu, 0x02u);
    SYNTH_MAGIC_CHAIN(64u, 0x729bu, 0x1u, 0x6u, 0x4u, 0x1du);
    SYNTH_MAGIC_CHAIN(68u, 0xe15du, 0xdu, 0xcu, 0xau, 0x08u);
    SYNTH_MAGIC_CHAIN(72u, 0x4c86u, 0x6u, 0x2u, 0x1u, 0x15u);
    SYNTH_MAGIC_CHAIN(76u, 0xb307u, 0x8u, 0xbu, 0xeu, 0x04u);
    SYNTH_MAGIC_CHAIN(80u, 0x68dau, 0x3u, 0x0u, 0x5u, 0x1bu);
    SYNTH_MAGIC_CHAIN(84u, 0xd4f1u, 0xfu, 0x9u, 0x9u, 0x0eu);
    SYNTH_MAGIC_CHAIN(88u, 0x237cu, 0x2u, 0x3u, 0x7u, 0x13u);
    SYNTH_MAGIC_CHAIN(92u, 0x9e50u, 0xau, 0xdu, 0x0u, 0x06u);
  }

  output[tid] = acc;
}

#undef SYNTH_MAGIC_CHAIN
