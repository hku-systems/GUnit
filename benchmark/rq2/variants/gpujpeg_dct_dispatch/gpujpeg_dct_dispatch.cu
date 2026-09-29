#include <cuda_runtime.h>

#include <stdint.h>

template <typename T>
__device__ static inline void gpujpeg_dct_dispatch_1d(
    const T in0,
    const T in1,
    const T in2,
    const T in3,
    const T in4,
    const T in5,
    const T in6,
    const T in7,
    T& out0,
    T& out1,
    T& out2,
    T& out3,
    T& out4,
    T& out5,
    T& out6,
    T& out7,
    const float level_shift_8 = 0.0f)
{
    const float diff0 = in0 + in7;
    const float diff1 = in1 + in6;
    const float diff2 = in2 + in5;
    const float diff3 = in3 + in4;
    const float diff4 = in3 - in4;
    const float diff5 = in2 - in5;
    const float diff6 = in1 - in6;
    const float diff7 = in0 - in7;

    const float even0 = diff0 + diff3;
    const float even1 = diff1 + diff2;
    const float even2 = diff1 - diff2;
    const float even3 = diff0 - diff3;
    const float even_diff = even2 + even3;

    const float odd0 = diff4 + diff5;
    const float odd1 = diff5 + diff6;
    const float odd2 = diff6 + diff7;
    const float odd_diff5 = (odd0 - odd2) * 0.382683433f;
    const float odd_diff4 = 1.306562965f * odd2 + odd_diff5;
    const float odd_diff3 = diff7 - odd1 * 0.707106781f;
    const float odd_diff2 = 0.541196100f * odd0 + odd_diff5;
    const float odd_diff1 = diff7 + odd1 * 0.707106781f;

    out0 = even0 + even1 + level_shift_8;
    out1 = odd_diff1 + odd_diff4;
    out2 = even3 + even_diff * 0.707106781f;
    out3 = odd_diff3 - odd_diff2;
    out4 = even0 - even1;
    out5 = odd_diff3 + odd_diff2;
    out6 = even3 - even_diff * 0.707106781f;
    out7 = odd_diff1 - odd_diff4;
}

template <int WARP_COUNT>
__device__ __forceinline__ void gpujpeg_dct_dispatch_body(
    int block_count_x,
    int block_count_y,
    uint8_t* source,
    const unsigned int source_stride,
    int16_t* output,
    int output_stride,
    const float* const quant_table,
    float* s_transposition_all)
{
    const int block_idx_x = threadIdx.x >> 3;
    const int block_idx_y = threadIdx.y;

    const int block_offset_x = blockIdx.x * 4;
    const int block_offset_y = blockIdx.y * WARP_COUNT;

    const bool processing = block_offset_x + block_idx_x < block_count_x &&
                            block_offset_y + block_idx_y < block_count_y;
    if (!processing) {
        return;
    }

    const int dct_idx = threadIdx.x & 7;

    typedef float dct_t;
    enum {
        SHARED_STRIDE = ((32 * sizeof(dct_t)) | 4) / sizeof(dct_t),
        SHARED_SIZE_WARP = SHARED_STRIDE * 8,
    };

    dct_t* const s_transposition =
        s_transposition_all + block_idx_y * SHARED_SIZE_WARP + block_idx_x * 8;

    const int in_x = (block_offset_x + block_idx_x) * 8 + dct_idx;
    const int in_y = (block_offset_y + block_idx_y) * 8;
    const int in_offset = in_x + in_y * source_stride;
    const uint8_t* in = source + in_offset;

    dct_t src0 = *in;
    in += source_stride;
    dct_t src1 = *in;
    in += source_stride;
    dct_t src2 = *in;
    in += source_stride;
    dct_t src3 = *in;
    in += source_stride;
    dct_t src4 = *in;
    in += source_stride;
    dct_t src5 = *in;
    in += source_stride;
    dct_t src6 = *in;
    in += source_stride;
    dct_t src7 = *in;

    dct_t* const s_dest = s_transposition + dct_idx;
    gpujpeg_dct_dispatch_1d(
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        src6,
        src7,
        s_dest[SHARED_STRIDE * 0],
        s_dest[SHARED_STRIDE * 1],
        s_dest[SHARED_STRIDE * 2],
        s_dest[SHARED_STRIDE * 3],
        s_dest[SHARED_STRIDE * 4],
        s_dest[SHARED_STRIDE * 5],
        s_dest[SHARED_STRIDE * 6],
        s_dest[SHARED_STRIDE * 7],
        -1024.0f);

    volatile dct_t* s_src = s_transposition + SHARED_STRIDE * dct_idx;
    dct_t dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7;
    gpujpeg_dct_dispatch_1d(
        s_src[0],
        s_src[1],
        s_src[2],
        s_src[3],
        s_src[4],
        s_src[5],
        s_src[6],
        s_src[7],
        dct0,
        dct1,
        dct2,
        dct3,
        dct4,
        dct5,
        dct6,
        dct7);

    const float* const quantization_row = quant_table + dct_idx;
    const int out0 = rintf(dct0 * quantization_row[0 * 8]);
    const int out1 = rintf(dct1 * quantization_row[1 * 8]);
    const int out2 = rintf(dct2 * quantization_row[2 * 8]);
    const int out3 = rintf(dct3 * quantization_row[3 * 8]);
    const int out4 = rintf(dct4 * quantization_row[4 * 8]);
    const int out5 = rintf(dct5 * quantization_row[5 * 8]);
    const int out6 = rintf(dct6 * quantization_row[6 * 8]);
    const int out7 = rintf(dct7 * quantization_row[7 * 8]);

    const int out_x = (block_offset_x + block_idx_x) * 64;
    const int out_y = (block_offset_y + block_idx_y) * output_stride;
    ((uint4*)(output + out_x + out_y))[dct_idx] = make_uint4(
        (out0 & 0xFFFF) + (out1 << 16),
        (out2 & 0xFFFF) + (out3 << 16),
        (out4 & 0xFFFF) + (out5 << 16),
        (out6 & 0xFFFF) + (out7 << 16));
}

__global__ void gpujpeg_dct_dispatch_kernel(
    uint32_t warp_count,
    int block_count_x,
    int block_count_y,
    uint8_t* source,
    unsigned int source_stride,
    int16_t* output,
    int output_stride,
    const float* quant_table)
{
    enum {
        SHARED_STRIDE = ((32 * sizeof(float)) | 4) / sizeof(float),
        SHARED_SIZE_WARP = SHARED_STRIDE * 8,
        MAX_WARP_COUNT = 8,
    };
    __shared__ float s_transposition_all[SHARED_SIZE_WARP * MAX_WARP_COUNT];

    switch (warp_count) {
    case 2:
        if (blockDim.x != 32 || blockDim.y != 2 || blockDim.z != 1) {
            return;
        }
        gpujpeg_dct_dispatch_body<2>(block_count_x, block_count_y, source, source_stride,
                                     output, output_stride, quant_table, s_transposition_all);
        break;
    case 4:
        if (blockDim.x != 32 || blockDim.y != 4 || blockDim.z != 1) {
            return;
        }
        gpujpeg_dct_dispatch_body<4>(block_count_x, block_count_y, source, source_stride,
                                     output, output_stride, quant_table, s_transposition_all);
        break;
    case 8:
        if (blockDim.x != 32 || blockDim.y != 8 || blockDim.z != 1) {
            return;
        }
        gpujpeg_dct_dispatch_body<8>(block_count_x, block_count_y, source, source_stride,
                                     output, output_stride, quant_table, s_transposition_all);
        break;
    default:
        return;
    }
}
