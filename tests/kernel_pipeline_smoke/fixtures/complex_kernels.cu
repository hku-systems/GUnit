#include "common_helpers.cuh"
#include "fixture_api.cuh"
#include <cstdio>
#include <cuda_runtime.h>

#define DEFINE_SIMPLE_KERNEL(NAME, TYPE) \
extern "C" __global__ void NAME(TYPE* out, TYPE value) { \
  out[0] = value; \
}

#define DECLARE_PROTO(NAME) extern "C" __global__ void NAME(int* out, int x)

__device__ int helper_twice(int x) {
  return x * 2;
}

extern "C" __global__ void direct_kernel(int* out, int x) {
  out[0] = helper_twice(x);
}

DEFINE_SIMPLE_KERNEL(macro_defined_kernel, int)

DECLARE_PROTO(macro_decl_kernel) {
  out[0] = x + 1;
}

template <typename T>
__global__ void templated_kernel(T* out, T x) {
  out[0] = x;
}

template __global__ void templated_kernel<int>(int*, int);

struct record_pair {
  int x;
  float y;
};

typedef int typedef_scalar_int;
typedef size_t typedef_size_count;
typedef int* typedef_int_ptr;
typedef record_pair typedef_record_alias;
typedef record_pair* typedef_record_ptr;
typedef struct {
  int x;
  int y;
} typedef_anonymous_alias;

enum Mode {
  ModeA = 0,
  ModeB = 1,
};

struct enum_inner {
  int x;
  float y;
};

struct enum_flags {
  unsigned int low : 3;
  unsigned int high : 5;
};

struct enum_outer {
  Mode mode;
  enum_inner inner;
  enum_flags flags;
};

template <int Rank_>
struct static_rank_coord {
  static int const kRank = Rank_;
  int idx[kRank];
};

template <typename Coord>
struct rank_box {
  Coord coord;
  int extent[Coord::kRank];
};

struct rank_wrapper {
  rank_box<static_rank_coord<4>> value;
};

struct layout_plain_val {
  int a;
  float b;
  unsigned int c;
};

struct layout_scalar_pair {
  int x;
  float y;
};

struct layout_outer_struct {
  layout_scalar_pair inner;
  int tail;
};

struct layout_ptr_leaf {
  int count;
  float* data;
};

struct layout_struct_array_val {
  layout_scalar_pair items[2];
  int tail;
};

struct layout_struct_array_ptr {
  layout_ptr_leaf items[2];
  int tail;
};

struct layout_nested_item {
  layout_scalar_pair inner;
  int scale;
};

struct layout_struct_array_nested {
  layout_nested_item items[2];
  int tail;
};

struct layout_nested_ptr_item {
  layout_ptr_leaf inner;
  int scale;
};

struct layout_struct_array_nested_ptr {
  layout_nested_ptr_item items[2];
  int tail;
};

struct layout_anonymous_ptr {
  int prefix;
  struct {
    int value;
    int* ptr;
  };
  int tail;
};

struct NestedParams {
  int inner_count;
  float* lanes[4];

  union Choice {
    int* int_ptr;
    float* float_ptr;
    int tag;
  } choice;

  struct {
    int value;
    int* ptr;
  };
};

struct Kind {
  int dummy;
  int selector;
};

struct KernelParams {
  int count;
  float weights[4];
  float* data;
  Kind* kind;
  NestedParams nested;
};

struct layout_base_params {
  int base_count;
  float* base_data;
};

struct layout_derived_params : public layout_base_params {
  int derived_tail;
  float* derived_data;
};

struct material_public_methods {
  int count;
  float* data;

  __host__ __device__ int value() const {
    return count;
  }
};

class material_private_field {
public:
  int visible;

private:
  int hidden;
};

struct material_virtual_method {
  int value;

  __host__ __device__ virtual int get() const {
    return value;
  }
};

struct material_reference_field {
  int& ref;
};

struct material_const_field {
  const int value;
};

extern "C" __global__ void record_ptr_copy(record_pair* out, const record_pair* in) {
  out[0] = in[0];
}

extern "C" __global__ void typedef_scalar_kernel(unsigned int* out,
                                                  typedef_scalar_int value,
                                                  typedef_size_count count) {
  out[0] = static_cast<unsigned int>(value) + static_cast<unsigned int>(count);
}

extern "C" __global__ void typedef_int_ptr_kernel(typedef_int_ptr values, unsigned int* out) {
  out[0] = values ? static_cast<unsigned int>(values[0]) : 0;
}

extern "C" __global__ void typedef_record_alias_kernel(unsigned int* out, typedef_record_alias cfg) {
  out[0] = static_cast<unsigned int>(cfg.x) + static_cast<unsigned int>(cfg.y);
}

extern "C" __global__ void typedef_record_ptr_kernel(typedef_record_ptr records, unsigned int* out) {
  out[0] = records ? static_cast<unsigned int>(records[0].x) + static_cast<unsigned int>(records[0].y) : 0;
}

extern "C" __global__ void typedef_anonymous_alias_kernel(unsigned int* out, typedef_anonymous_alias cfg) {
  out[0] = static_cast<unsigned int>(cfg.x + cfg.y);
}

extern "C" __global__ void rank_wrapper_copy(rank_wrapper* out, const rank_wrapper* in) {
  out[0] = in[0];
}

extern "C" __global__ void uint3_write_sum(unsigned int* out, uint3 dims) {
  out[0] = dims.x + dims.y + dims.z;
}

extern "C" __global__ void struct_enum_kernel(unsigned int* out, enum_outer cfg) {
  out[0] = static_cast<unsigned int>(cfg.mode) + static_cast<unsigned int>(cfg.inner.x) +
           static_cast<unsigned int>(cfg.inner.y) + cfg.flags.low + cfg.flags.high;
}

extern "C" __global__ void layout_plain_val_kernel(unsigned int* out, layout_plain_val cfg) {
  out[0] = static_cast<unsigned int>(cfg.a) + static_cast<unsigned int>(cfg.b) + cfg.c;
}

extern "C" __global__ void layout_outer_struct_kernel(unsigned int* out, layout_outer_struct cfg) {
  out[0] = static_cast<unsigned int>(cfg.inner.x) + static_cast<unsigned int>(cfg.inner.y) +
           static_cast<unsigned int>(cfg.tail);
}

extern "C" __global__ void layout_struct_array_val_kernel(unsigned int* out, layout_struct_array_val cfg) {
  out[0] = static_cast<unsigned int>(cfg.items[0].x) + static_cast<unsigned int>(cfg.items[1].y) +
           static_cast<unsigned int>(cfg.tail);
}

extern "C" __global__ void layout_struct_array_ptr_kernel(unsigned int* out, layout_struct_array_ptr cfg) {
  float first = cfg.items[0].data ? cfg.items[0].data[0] : 0.0f;
  out[0] = static_cast<unsigned int>(cfg.items[0].count) + static_cast<unsigned int>(first) +
           static_cast<unsigned int>(cfg.tail);
}

extern "C" __global__ void layout_struct_array_nested_kernel(unsigned int* out, layout_struct_array_nested cfg) {
  out[0] = static_cast<unsigned int>(cfg.items[0].inner.x) +
           static_cast<unsigned int>(cfg.items[1].inner.y) +
           static_cast<unsigned int>(cfg.items[0].scale) +
           static_cast<unsigned int>(cfg.tail);
}

extern "C" __global__ void layout_struct_array_nested_ptr_kernel(unsigned int* out, layout_struct_array_nested_ptr cfg) {
  float first = cfg.items[1].inner.data ? cfg.items[1].inner.data[0] : 0.0f;
  out[0] = static_cast<unsigned int>(cfg.items[1].inner.count) + static_cast<unsigned int>(first) +
           static_cast<unsigned int>(cfg.items[1].scale) + static_cast<unsigned int>(cfg.tail);
}

extern "C" __global__ void layout_anonymous_ptr_kernel(unsigned int* out, layout_anonymous_ptr cfg) {
  int pointed = cfg.ptr ? cfg.ptr[0] : 0;
  out[0] = static_cast<unsigned int>(cfg.prefix + cfg.value + pointed + cfg.tail);
}

extern "C" __global__ void complex_args_kernel(int seed, KernelParams params, float* output) {
  int pointed = params.nested.ptr ? params.nested.ptr[0] : 0;
  float lane = params.nested.lanes[0] ? params.nested.lanes[0][0] : 0.0f;
  float data = params.data ? params.data[0] : 0.0f;
  output[0] = static_cast<float>(seed + params.count + params.nested.inner_count +
                                 params.nested.value + pointed) +
              params.weights[0] + lane + data;
}

extern "C" __global__ void layout_derived_params_kernel(unsigned int* out, layout_derived_params cfg) {
  float base = cfg.base_data ? cfg.base_data[0] : 0.0f;
  float derived = cfg.derived_data ? cfg.derived_data[0] : 0.0f;
  out[0] = static_cast<unsigned int>(cfg.base_count + cfg.derived_tail + base + derived);
}

extern "C" __global__ void material_public_methods_kernel(unsigned int* out, material_public_methods cfg) {
  float first = cfg.data ? cfg.data[0] : 0.0f;
  out[0] = static_cast<unsigned int>(cfg.value() + first);
}

extern "C" __global__ void material_private_field_kernel(unsigned int* out, material_private_field cfg) {
  out[0] = static_cast<unsigned int>(cfg.visible);
}

extern "C" __global__ void material_virtual_method_kernel(unsigned int* out, material_virtual_method cfg) {
  out[0] = static_cast<unsigned int>(cfg.value);
}

extern "C" __global__ void material_reference_field_kernel(unsigned int* out, material_reference_field cfg) {
  out[0] = static_cast<unsigned int>(cfg.ref);
}

extern "C" __global__ void material_const_field_kernel(unsigned int* out, material_const_field cfg) {
  out[0] = static_cast<unsigned int>(cfg.value);
}

/* Non-extern "C" kernels using the shared helper */
__global__ void clamped_write(int* out, int value, int lo, int hi) {
  out[0] = clamp_int(value, lo, hi);
}

__global__ void scaled_write(float* out, float value, float factor) {
  out[0] = scale_float(value, factor);
}

/* Non-extern "C" kernel: bitwise blend of two ints via mask */
__global__ void bitwise_blend(int* out, int a, int b, int mask) {
  out[0] = bitwise_blend_impl(a, b, mask);
}

/* Non-extern "C" kernel: reduce a pair of ints to their average */
__global__ void reduce_pair(int* out, int a, int b) {
  out[0] = reduce_pair_impl(a, b);
}

/* ---- Host-callable test wrappers ---- */

static bool check_cuda(cudaError_t err, const char* ctx) {
  if (err == cudaSuccess) return true;
  printf("  CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
  return false;
}

extern "C" int test_direct_kernel(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  direct_kernel<<<1,1>>>(d_out, 21);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 42) { printf("  FAIL direct_kernel: expected 42 got %d\n", h_out); return 1; }
  printf("  PASS direct_kernel\n");
  return 0;
}

extern "C" int test_macro_defined_kernel(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  macro_defined_kernel<<<1,1>>>(d_out, 99);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 99) { printf("  FAIL macro_defined_kernel: expected 99 got %d\n", h_out); return 1; }
  printf("  PASS macro_defined_kernel\n");
  return 0;
}

extern "C" int test_macro_decl_kernel(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  macro_decl_kernel<<<1,1>>>(d_out, 10);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 11) { printf("  FAIL macro_decl_kernel: expected 11 got %d\n", h_out); return 1; }
  printf("  PASS macro_decl_kernel\n");
  return 0;
}

extern "C" int test_templated_kernel(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  templated_kernel<int><<<1,1>>>(d_out, 77);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 77) { printf("  FAIL templated_kernel: expected 77 got %d\n", h_out); return 1; }
  printf("  PASS templated_kernel\n");
  return 0;
}

extern "C" int test_record_ptr_copy(void) {
  record_pair h_in{17, 2.5f};
  record_pair h_out{0, 0.0f};
  record_pair *d_in = nullptr;
  record_pair *d_out = nullptr;
  if (!check_cuda(cudaMalloc(&d_in, sizeof(record_pair)), "malloc in")) return 1;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(record_pair)), "malloc out")) {
    cudaFree(d_in);
    return 1;
  }
  if (!check_cuda(cudaMemcpy(d_in, &h_in, sizeof(record_pair), cudaMemcpyHostToDevice), "copy in")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return 1;
  }
  record_ptr_copy<<<1,1>>>(d_out, d_in);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) {
    cudaFree(d_in);
    cudaFree(d_out);
    return 1;
  }
  cudaMemcpy(&h_out, d_out, sizeof(record_pair), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);
  if (h_out.x != h_in.x || h_out.y < h_in.y - 0.01f || h_out.y > h_in.y + 0.01f) {
    printf("  FAIL record_ptr_copy: expected {%d, %.2f} got {%d, %.2f}\n", h_in.x, h_in.y, h_out.x, h_out.y);
    return 1;
  }
  printf("  PASS record_ptr_copy\n");
  return 0;
}

extern "C" int test_uint3_write_sum(void) {
  unsigned int* d_out;
  unsigned int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(unsigned int)), "malloc")) return 1;
  uint3 dims = make_uint3(3u, 4u, 5u);
  uint3_write_sum<<<1,1>>>(d_out, dims);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 12u) { printf("  FAIL uint3_write_sum: expected 12 got %u\n", h_out); return 1; }
  printf("  PASS uint3_write_sum\n");
  return 0;
}

extern "C" int test_struct_enum_kernel(void) {
  unsigned int *d_out;
  unsigned int h_out = 0;
  enum_outer cfg;
  cfg.mode = ModeB;
  cfg.inner.x = 3;
  cfg.inner.y = 4.0f;
  cfg.flags.low = 2;
  cfg.flags.high = 5;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(unsigned int)), "malloc")) return 1;
  struct_enum_kernel<<<1,1>>>(d_out, cfg);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 15u) { printf("  FAIL struct_enum_kernel: expected 15 got %u\n", h_out); return 1; }
  printf("  PASS struct_enum_kernel\n");
  return 0;
}

extern "C" int test_clamped_write(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  clamped_write<<<1,1>>>(d_out, 100, 0, 50);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 50) { printf("  FAIL clamped_write: expected 50 got %d\n", h_out); return 1; }
  printf("  PASS clamped_write\n");
  return 0;
}

extern "C" int test_scaled_write(void) {
  float *d_out;
  float h_out = 0.0f;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(float)), "malloc")) return 1;
  scaled_write<<<1,1>>>(d_out, 3.0f, 2.5f);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  float expected = 7.5f;
  if (h_out < expected - 0.01f || h_out > expected + 0.01f) {
    printf("  FAIL scaled_write: expected %.2f got %.2f\n", expected, h_out);
    return 1;
  }
  printf("  PASS scaled_write\n");
  return 0;
}

extern "C" int test_bitwise_blend(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  /* a=0xFF00, b=0x00FF, mask=0xF0F0 -> (0xFF00 & 0xF0F0) | (0x00FF & 0x0F0F)
     = 0xF000 | 0x000F = 0xF00F */
  bitwise_blend<<<1,1>>>(d_out, 0xFF00, 0x00FF, 0xF0F0);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 0xF00F) { printf("  FAIL bitwise_blend: expected 0xF00F got 0x%X\n", h_out); return 1; }
  printf("  PASS bitwise_blend\n");
  return 0;
}

extern "C" int test_reduce_pair(void) {
  int *d_out;
  int h_out = 0;
  if (!check_cuda(cudaMalloc(&d_out, sizeof(int)), "malloc")) return 1;
  reduce_pair<<<1,1>>>(d_out, 30, 50);
  if (!check_cuda(cudaDeviceSynchronize(), "sync")) { cudaFree(d_out); return 1; }
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  if (h_out != 40) { printf("  FAIL reduce_pair: expected 40 got %d\n", h_out); return 1; }
  printf("  PASS reduce_pair\n");
  return 0;
}
