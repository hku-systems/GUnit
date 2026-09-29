# Manifest Example: Nested Pointer-Bearing Args

This example records a deliberately complex CUDA kernel signature and the
corresponding `manifest.json` shape. The manifest intentionally does not store a
`contains_pointer` field. Pointer containment is derived from the top-level
`kind` and the recursive `type_layout` tree.

The layout values below assume a normal 64-bit CUDA C++ ABI:

- pointer size: 8 bytes
- pointer alignment: 8 bytes
- `int` alignment: 4 bytes
- `float` alignment: 4 bytes

The real Phase 1 pipeline should capture these values from Clang instead of
hard-coding them.

## Kernel Source

```cpp
struct NestedParams {
  int inner_count;
  float *lanes[4];

  union Choice {
    int *int_ptr;
    float *float_ptr;
    int tag;
  } choice;

  struct {
    int value;
    int *ptr;
  };
};

struct Kind {
  int dummy;
  int selector;
};

struct KernelParams {
  int count;
  float weights[4];
  float *data;
  Kind *kind;
  NestedParams nested;
};

__global__ void complex_args_kernel(
    int seed,
    KernelParams params,
    float *output) {
  // Body omitted. This file only documents the manifest shape.
}
```

## Expected Manifest

```json
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "complex_args_kernel",
      "display_name": "complex_args_kernel",
      "args": [
        {
          "index": 0,
          "name": "seed",
          "type": "int",
          "kind": "scalar",
          "size_bytes": 4,
          "align_bytes": 4
        },
        {
          "index": 1,
          "name": "params",
          "type": "KernelParams",
          "kind": "opaque_with_ptr",
          "size_bytes": 104,
          "align_bytes": 8,
          "type_info": {
            "kind": "struct",
            "qualified_name": "KernelParams",
            "definition": {
              "status": "available"
            }
          },
          "type_layout": {
            "layout_status": "complete",
            "fields": [
              {
                "index": "params.count",
                "name": "count",
                "type": "int",
                "kind": "scalar",
                "size_bytes": 4,
                "align_bytes": 4
              },
              {
                "index": "params.weights",
                "name": "weights",
                "type": "float[4]",
                "kind": "opaque_val",
                "size_bytes": 16,
                "align_bytes": 4,
                "layout_status": "complete",
                "element_count": 4,
                "element": {
                  "index": "params.weights[]",
                  "name": "$element",
                  "type": "float",
                  "kind": "scalar",
                  "size_bytes": 4,
                  "align_bytes": 4
                }
              },
              {
                "index": "params.data",
                "name": "data",
                "type": "float *",
                "kind": "pointer",
                "pointer_role": "payload_buffer",
                "size_bytes": 8,
                "align_bytes": 8,
                "pointee_layout": {
                  "index": "params.data.*",
                  "name": "$pointee",
                  "type": "float",
                  "kind": "scalar",
                  "size_bytes": 4,
                  "align_bytes": 4
                }
              },
              {
                "index": "params.kind",
                "name": "kind",
                "type": "Kind *",
                "kind": "pointer",
                "pointer_role": "payload_buffer",
                "size_bytes": 8,
                "align_bytes": 8,
                "pointee_layout": {
                  "index": "params.kind.*",
                  "name": "$pointee",
                  "type": "Kind",
                  "kind": "opaque_val",
                  "size_bytes": 8,
                  "align_bytes": 4,
                  "layout_status": "complete",
                  "type_info": {
                    "kind": "struct",
                    "qualified_name": "Kind",
                    "definition": {
                      "status": "available"
                    }
                  },
                  "fields": [
                    {
                      "index": "params.kind.*.dummy",
                      "name": "dummy",
                      "type": "int",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4
                    },
                    {
                      "index": "params.kind.*.selector",
                      "name": "selector",
                      "type": "int",
                      "kind": "scalar",
                      "size_bytes": 4,
                      "align_bytes": 4
                    }
                  ]
                }
              },
              {
                "index": "params.nested",
                "name": "nested",
                "type": "NestedParams",
                "kind": "opaque_with_ptr",
                "size_bytes": 64,
                "align_bytes": 8,
                "layout_status": "complete",
                "type_info": {
                  "kind": "struct",
                  "qualified_name": "NestedParams",
                  "definition": {
                    "status": "available"
                  }
                },
                "fields": [
                  {
                    "index": "params.nested.inner_count",
                    "name": "inner_count",
                    "type": "int",
                    "kind": "scalar",
                    "size_bytes": 4,
                    "align_bytes": 4
                  },
                  {
                    "index": "params.nested.lanes",
                    "name": "lanes",
                    "type": "float *[4]",
                    "kind": "opaque_with_ptr",
                    "size_bytes": 32,
                    "align_bytes": 8,
                    "layout_status": "complete",
                    "element_count": 4,
                    "element": {
                      "index": "params.nested.lanes[]",
                      "name": "$element",
                      "type": "float *",
                      "kind": "pointer",
                      "pointer_role": "payload_buffer",
                      "size_bytes": 8,
                      "align_bytes": 8,
                      "pointee_layout": {
                        "index": "params.nested.lanes[].*",
                        "name": "$pointee",
                        "type": "float",
                        "kind": "scalar",
                        "size_bytes": 4,
                        "align_bytes": 4
                      }
                    }
                  },
                  {
                    "index": "params.nested.choice",
                    "name": "choice",
                    "type": "NestedParams::Choice",
                    "kind": "opaque_with_ptr",
                    "size_bytes": 8,
                    "align_bytes": 8,
                    "layout_status": "complete",
                    "type_info": {
                      "kind": "union",
                      "qualified_name": "NestedParams::Choice",
                      "definition": {
                        "status": "available"
                      }
                    },
                    "fields": [
                      {
                        "index": "params.nested.choice.int_ptr",
                        "name": "int_ptr",
                        "type": "int *",
                        "kind": "pointer",
                        "pointer_role": "payload_buffer",
                        "size_bytes": 8,
                        "align_bytes": 8,
                        "pointee_layout": {
                          "index": "params.nested.choice.int_ptr.*",
                          "name": "$pointee",
                          "type": "int",
                          "kind": "scalar",
                          "size_bytes": 4,
                          "align_bytes": 4
                        }
                      },
                      {
                        "index": "params.nested.choice.float_ptr",
                        "name": "float_ptr",
                        "type": "float *",
                        "kind": "pointer",
                        "pointer_role": "payload_buffer",
                        "size_bytes": 8,
                        "align_bytes": 8,
                        "pointee_layout": {
                          "index": "params.nested.choice.float_ptr.*",
                          "name": "$pointee",
                          "type": "float",
                          "kind": "scalar",
                          "size_bytes": 4,
                          "align_bytes": 4
                        }
                      },
                      {
                        "index": "params.nested.choice.tag",
                        "name": "tag",
                        "type": "int",
                        "kind": "scalar",
                        "size_bytes": 4,
                        "align_bytes": 4
                      }
                    ]
                  },
                  {
                    "index": "params.nested.anon3",
                    "name": "anon3",
                    "type": "NestedParams::(anonymous struct)",
                    "kind": "opaque_with_ptr",
                    "size_bytes": 16,
                    "align_bytes": 8,
                    "layout_status": "complete",
                    "type_info": {
                      "kind": "struct",
                      "definition": {
                        "status": "available"
                      }
                    },
                    "fields": [
                      {
                        "index": "params.nested.anon3.value",
                        "name": "value",
                        "type": "int",
                        "kind": "scalar",
                        "size_bytes": 4,
                        "align_bytes": 4
                      },
                      {
                        "index": "params.nested.anon3.ptr",
                        "name": "ptr",
                        "type": "int *",
                        "kind": "pointer",
                        "pointer_role": "payload_buffer",
                        "size_bytes": 8,
                        "align_bytes": 8,
                        "pointee_layout": {
                          "index": "params.nested.anon3.ptr.*",
                          "name": "$pointee",
                          "type": "int",
                          "kind": "scalar",
                          "size_bytes": 4,
                          "align_bytes": 4
                        }
                      }
                    ]
                  }
                ]
              }
            ]
          }
        },
        {
          "index": 2,
          "name": "output",
          "type": "float *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "size_bytes": 8,
          "align_bytes": 8,
          "pointee_layout": {
            "index": "output.*",
            "name": "$pointee",
            "type": "float",
            "kind": "scalar",
            "size_bytes": 4,
            "align_bytes": 4
          }
        }
      ],
      "constraints": [
        {
          "kind": "count_fits_buffer",
          "count_arg": 1,
          "count_path": [
            "count"
          ],
          "buffer_arg": 1,
          "buffer_path": [
            "data"
          ],
          "elem_size_bytes": 4
        },
        {
          "kind": "count_fits_buffer",
          "count_arg": 1,
          "count_path": [
            "nested",
            "inner_count"
          ],
          "buffer_arg": 2,
          "elem_size_bytes": 4
        },
        {
          "kind": "scalar_compare_const",
          "scalar_arg": 0,
          "op": ">=",
          "value": 0
        },
        {
          "kind": "scalar_compare_const",
          "scalar_arg": 1,
          "scalar_path": [
            "nested",
            "inner_count"
          ],
          "op": ">=",
          "value": 0
        },
        {
          "kind": "scalar_compare_scalar",
          "lhs_arg": 1,
          "lhs_path": [
            "nested",
            "choice",
            "tag"
          ],
          "op": "<=",
          "rhs_arg": 1,
          "rhs_path": [
            "count"
          ]
        }
      ],
      "others": {
        "source_file": "manifest_example.cu",
        "source_line": 18,
        "source_column": 17,
        "type_shim_header": "type_shim.v1.cuh",
        "type_shim_status": "ok"
      }
    }
  ]
}
```

## Notes

- `contains_pointer` is not stored. The top-level `params` arg has
  `kind=opaque_with_ptr`, and concrete pointer leaves are visible in
  `type_layout`.
- `params.weights` is `opaque_val` because `float[4]` is inline value storage
  with scalar elements.
- Layout field offsets are omitted from this example. The input envelope is
  ordered by manifest traversal and uses `size_bytes` plus pointer payload
  segments, so ABI layout position is not the envelope source of truth.
- The encoder/fuzzer must insert canonical padding so each scalar/object segment
  and each payload pointer target starts at the alignment checked by the
  generated decoder. For payload pointers, the decoder skips the same canonical
  padding before reading the `u64` length field, then validates the payload
  pointer alignment after the length field.
- `params.nested.lanes` is `opaque_with_ptr` because the inline array contains
  pointer elements.
- `params.nested.choice` is represented as `opaque_with_ptr` with
  `type_info.kind=union`. Phase 2 should reject recursive materialization
  for this union until active-member/discriminator support exists.
- The two `count_fits_buffer` constraints demonstrate top-level and nested
  field paths under the current schema. A future path-ref schema can collapse
  `count_arg`/`count_path` and `buffer_arg`/`buffer_path` into unified
  `{arg, path}` references.
- The scalar comparison constraints express `params.nested.inner_count >= 0`
  and `params.nested.choice.tag <= params.count`.
- Constraint paths are relative JSON segment arrays. For example,
  `["nested", "choice", "tag"]` means `params.nested.choice.tag` when
  paired with `lhs_arg: 1`; omitting the path means the constraint references
  the top-level arg itself, as in `seed >= 0`. For fixed-size arrays, a
  decimal string segment selects an element; for example
  `["items", "1", "inner", "data"]` means `items[1].inner.data`.
- Anonymous aggregate layout nodes such as `params.nested.anon3` give the
  manifest a stable layout path. The C++ access path is still transparent for
  anonymous members, so generated C++ may access `params.nested.value` and
  `params.nested.ptr` directly.


## Phase 2 Decode/Invoke Shape

The exact manifest above is intentionally complex enough to include two current
Phase 2 blockers:

1. `count_fits_buffer` with `count_path` / `buffer_path` is schema-level
   documentation here. The current Phase 2 contract parser supports
   `scalar_le_buffer_len(unit=bytes)` and scalar comparison constraints, but
   does not yet lower `count_fits_buffer`.
2. `params.nested.choice` is a pointer-bearing union. Current Phase 2 rejects
   this until active-member/discriminator support exists.

Current strict outcomes are therefore:

```text
with constraints[] as written:
  phase2_status: failed
  failure_reason: phase2_input_invalid
  failure_detail: constraint_not_supported

with constraints[] omitted or after count_fits_buffer support lands:
  phase2_status: failed
  failure_reason: phase2_input_invalid
  failure_detail: params.nested.choice:union_not_supported
```

To make the generated code concrete, the snippets below use the same manifest
projected to the currently supported Phase 2 subset:

- remove the `params.nested.choice` union field from `params.type_layout`;
- replace `constraints[]` with the currently supported top-level constraint:

```json
"constraints": [
  {
    "kind": "scalar_le_buffer_len",
    "scalar_arg": 0,
    "buffer_arg": 2,
    "unit": "bytes"
  },
  {
    "kind": "scalar_compare_const",
    "scalar_arg": 0,
    "op": ">=",
    "value": 0
  },
  {
    "kind": "scalar_compare_const",
    "scalar_arg": 1,
    "scalar_path": [
      "nested",
      "inner_count"
    ],
    "op": ">=",
    "value": 0
  }
]
```

That supported projection keeps the required three top-level args (`int`,
`struct`, `ptr`) and keeps the recursive struct fields `int`, `float[4]`,
`float *`, nested `struct`, and nested `float *[4]`. It omits only the union
field that Phase 2 must reject today.

The generated decode header uses `RAPID_DECODE_ASSERT` for debug decode
checks. These assertions are enabled when `RAPID_DECODE_DEBUG_ASSERT` is defined
and `NDEBUG` is not defined; otherwise they compile to `((void)0)`. Constraint
assertions use the same macro and are inserted after all referenced values have
been decoded.
Local payload length variable names are derived from the manifest field path.
Array element placeholders use the concrete element index, so
`params.nested.lanes[]` becomes names such as `params_nested_lanes_0_len_u64`
without repeating parent prefixes.

### Example `gen/fuzzer_decode.v1.cuh`

```cpp
#ifndef __PHASE2_DECODE_COMPLEX_ARGS_KERNEL_CUH__
#define __PHASE2_DECODE_COMPLEX_ARGS_KERNEL_CUH__

#include <cstddef>
#include <cstdint>
#include <assert.h>
#include <type_traits>
#include "type_shim.v1.cuh"

struct DecodedKernelArgs {
  int seed;
  KernelParams params;
  float * output;
};

#ifndef RAPID_DECODE_ASSERT
#if !defined(NDEBUG) && defined(RAPID_DECODE_DEBUG_ASSERT)
#define RAPID_DECODE_ASSERT(cond) assert(cond)
#else
#define RAPID_DECODE_ASSERT(cond) ((void)0)
#endif
#endif

template <typename T, bool IsEnum = std::is_enum<T>::value>
struct RapidScalarStorage {
  using type = T;
};

template <typename T>
struct RapidScalarStorage<T, true> {
  using type = typename std::underlying_type<T>::type;
};

template <typename T>
__device__ inline T read_scalar_le(volatile uint8_t *data, size_t &offset) {
  static_assert(std::is_integral<T>::value || std::is_enum<T>::value, "read_scalar_le only supports integral or enum types");
  using Storage = typename RapidScalarStorage<T>::type;
  Storage value = 0;
  for (size_t i = 0; i < sizeof(Storage); ++i) {
    value |= static_cast<Storage>(static_cast<unsigned long long>(data[offset + i]) << (8 * i));
  }
  offset += sizeof(Storage);
  return static_cast<T>(value);
}

__device__ inline float read_f32_le(volatile uint8_t *data, size_t &offset) {
  union { uint32_t u; float f; } bits{read_scalar_le<uint32_t>(data, offset)};
  return bits.f;
}

template <typename T>
__device__ inline T read_object_bytes(volatile uint8_t *data, size_t &offset) {
  static_assert(std::is_trivially_copyable<T>::value, "read_object_bytes requires trivially copyable type");
  static_assert(std::is_standard_layout<T>::value, "read_object_bytes requires standard layout type");
  T value{};
  uint8_t *dst = reinterpret_cast<uint8_t *>(&value);
  for (size_t i = 0; i < sizeof(T); ++i) {
    dst[i] = data[offset + i];
  }
  offset += sizeof(T);
  return value;
}

__device__ inline double read_f64_le(volatile uint8_t *data, size_t &offset) {
  union { uint64_t u; double d; } bits{read_scalar_le<uint64_t>(data, offset)};
  return bits.d;
}

__device__ inline DecodedKernelArgs fuzzer_decode_v1(volatile uint8_t *data, size_t full_size) {
  RAPID_DECODE_ASSERT(data != nullptr);

  DecodedKernelArgs decoded{};
  size_t offset = 0;
  // decode value arg `seed`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(int));
  decoded.seed = read_scalar_le<int>(data, offset);
  // decode field-aware aggregate arg `params`
  // decode scalar field `params.count`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(int));
  decoded.params.count = read_scalar_le<int>(data, offset);
  // decode scalar field `params.weights[]`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(float));
  decoded.params.weights[0] = read_f32_le(data, offset);
  // decode scalar field `params.weights[]`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(float));
  decoded.params.weights[1] = read_f32_le(data, offset);
  // decode scalar field `params.weights[]`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(float));
  decoded.params.weights[2] = read_f32_le(data, offset);
  // decode scalar field `params.weights[]`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(float));
  decoded.params.weights[3] = read_f32_le(data, offset);
  // decode payload-buffer `params_data`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_data_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_data_len_u64 <= full_size && offset <= full_size - params_data_len_u64);
  decoded.params.data = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.data) % 8) == 0);
  offset += static_cast<size_t>(params_data_len_u64);
  // decode payload-buffer `params_kind`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_kind_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_kind_len_u64 <= full_size && offset <= full_size - params_kind_len_u64);
  decoded.params.kind = reinterpret_cast<Kind *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.kind) % 8) == 0);
  offset += static_cast<size_t>(params_kind_len_u64);
  // decode scalar field `params.nested.inner_count`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(int));
  decoded.params.nested.inner_count = read_scalar_le<int>(data, offset);
  // decode payload-buffer `params_nested_lanes_0`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_nested_lanes_0_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_nested_lanes_0_len_u64 <= full_size && offset <= full_size - params_nested_lanes_0_len_u64);
  decoded.params.nested.lanes[0] = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.lanes[0]) % 8) == 0);
  offset += static_cast<size_t>(params_nested_lanes_0_len_u64);
  // decode payload-buffer `params_nested_lanes_1`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_nested_lanes_1_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_nested_lanes_1_len_u64 <= full_size && offset <= full_size - params_nested_lanes_1_len_u64);
  decoded.params.nested.lanes[1] = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.lanes[1]) % 8) == 0);
  offset += static_cast<size_t>(params_nested_lanes_1_len_u64);
  // decode payload-buffer `params_nested_lanes_2`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_nested_lanes_2_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_nested_lanes_2_len_u64 <= full_size && offset <= full_size - params_nested_lanes_2_len_u64);
  decoded.params.nested.lanes[2] = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.lanes[2]) % 8) == 0);
  offset += static_cast<size_t>(params_nested_lanes_2_len_u64);
  // decode payload-buffer `params_nested_lanes_3`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_nested_lanes_3_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_nested_lanes_3_len_u64 <= full_size && offset <= full_size - params_nested_lanes_3_len_u64);
  decoded.params.nested.lanes[3] = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.lanes[3]) % 8) == 0);
  offset += static_cast<size_t>(params_nested_lanes_3_len_u64);
  // params.nested.choice is a union and is rejected by the current supported projection.
  // decode scalar field `params.nested.anon3.value`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(int));
  decoded.params.nested.value = read_scalar_le<int>(data, offset);
  // decode payload-buffer `params_nested_anon3_ptr`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_nested_anon3_ptr_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_nested_anon3_ptr_len_u64 <= full_size && offset <= full_size - params_nested_anon3_ptr_len_u64);
  decoded.params.nested.ptr = reinterpret_cast<int *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.ptr) % 8) == 0);
  offset += static_cast<size_t>(params_nested_anon3_ptr_len_u64);
  // decode payload-buffer `output`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t output_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(output_len_u64 <= full_size && offset <= full_size - output_len_u64);
  decoded.output = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.output) % 8) == 0);
  offset += static_cast<size_t>(output_len_u64);
  RAPID_DECODE_ASSERT(decoded.seed <= output_len_u64);
  RAPID_DECODE_ASSERT(decoded.seed >= 0);
  RAPID_DECODE_ASSERT(decoded.params.nested.inner_count >= 0);
  return decoded;
}

#endif // __PHASE2_DECODE_COMPLEX_ARGS_KERNEL_CUH__
```

The last three assertions are generated constraint checks for the supported
projection's `scalar_compare_const` and `scalar_le_buffer_len` constraints:

```cpp
RAPID_DECODE_ASSERT(decoded.seed <= output_len_u64);
RAPID_DECODE_ASSERT(decoded.seed >= 0);
RAPID_DECODE_ASSERT(decoded.params.nested.inner_count >= 0);
```

For the full manifest's future `count_fits_buffer` constraints, the intended
assertions would be field-aware equivalents once that parser support exists:

```cpp
RAPID_DECODE_ASSERT(decoded.params.count <= params_data_len_u64 / 4);
RAPID_DECODE_ASSERT(decoded.params.nested.inner_count <= output_len_u64 / 4);
```

### Example `gen/fuzzer_invoke.v1.cuh`

```cpp
#ifndef __PHASE2_INVOKE_COMPLEX_ARGS_KERNEL_CUH__
#define __PHASE2_INVOKE_COMPLEX_ARGS_KERNEL_CUH__

#include "fuzzer_decode.v1.cuh"
#include "type_shim.v1.cuh"

extern "C" __device__ void __rapid_entry__complex_args_kernel(int seed, KernelParams params, float * output);

__device__ inline void fuzzer_invoke_v1(const DecodedKernelArgs &decoded) {
  __rapid_entry__complex_args_kernel(decoded.seed, decoded.params, decoded.output);
}

#endif // __PHASE2_INVOKE_COMPLEX_ARGS_KERNEL_CUH__
```

## Theoretical Full Decode/Invoke With Union Support

This section is intentionally theoretical. It shows the decode/invoke shape for
the full manifest above assuming Phase 2 has added both features that are
currently missing:

- field-aware `count_fits_buffer` constraints;
- active-member/discriminator support for pointer-bearing unions.

A C/C++ union cannot have all members materialized at once because every member
shares the same storage. The theoretical envelope therefore needs one extra
union selector before the union payload. This example uses:

```cpp
enum ChoiceActiveMember : uint8_t {
  kChoiceIntPtr = 0,
  kChoiceFloatPtr = 1,
  kChoiceTag = 2,
};
```

The selector is not part of the kernel ABI. It is decode-only envelope metadata
used to decide which union member to materialize.

### Theoretical `gen/fuzzer_decode.v1.cuh`

Only the body after `decoded.params.nested.lanes[3]` differs from the supported
projection shown above. The helper functions, `DecodedKernelArgs`, and
`RAPID_DECODE_ASSERT` macro are the same as in the current generated header.

```cpp
  // params.nested.choice: theoretical active-member union decode.
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint8_t));
  uint8_t params_nested_choice_active = read_scalar_le<uint8_t>(data, offset);
  RAPID_DECODE_ASSERT(params_nested_choice_active <= kChoiceTag);

  switch (params_nested_choice_active) {
    case kChoiceIntPtr: {
      RAPID_DECODE_ASSERT((offset % 8) == 0);
      RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
      uint64_t params_nested_choice_int_ptr_len_u64 = read_scalar_le<uint64_t>(data, offset);
      RAPID_DECODE_ASSERT((offset % 8) == 0);
      RAPID_DECODE_ASSERT(params_nested_choice_int_ptr_len_u64 <= full_size &&
                          offset <= full_size - params_nested_choice_int_ptr_len_u64);
      decoded.params.nested.choice.int_ptr = reinterpret_cast<int *>(
          const_cast<uint8_t *>(data + offset));
      RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.choice.int_ptr) % 8) == 0);
      offset += static_cast<size_t>(params_nested_choice_int_ptr_len_u64);
      break;
    }
    case kChoiceFloatPtr: {
      RAPID_DECODE_ASSERT((offset % 8) == 0);
      RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
      uint64_t params_nested_choice_float_ptr_len_u64 = read_scalar_le<uint64_t>(data, offset);
      RAPID_DECODE_ASSERT((offset % 8) == 0);
      RAPID_DECODE_ASSERT(params_nested_choice_float_ptr_len_u64 <= full_size &&
                          offset <= full_size - params_nested_choice_float_ptr_len_u64);
      decoded.params.nested.choice.float_ptr = reinterpret_cast<float *>(
          const_cast<uint8_t *>(data + offset));
      RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.choice.float_ptr) % 8) == 0);
      offset += static_cast<size_t>(params_nested_choice_float_ptr_len_u64);
      break;
    }
    case kChoiceTag: {
      RAPID_DECODE_ASSERT((offset % 4) == 0);
      RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(int));
      decoded.params.nested.choice.tag = read_scalar_le<int>(data, offset);
      RAPID_DECODE_ASSERT(decoded.params.nested.choice.tag <= decoded.params.count);
      break;
    }
    default: {
      RAPID_DECODE_ASSERT(false);
      break;
    }
  }

  // decode scalar field `params.nested.anon3.value`
  RAPID_DECODE_ASSERT((offset % 4) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(int));
  decoded.params.nested.value = read_scalar_le<int>(data, offset);
  // decode payload-buffer `params_nested_anon3_ptr`
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t params_nested_anon3_ptr_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(params_nested_anon3_ptr_len_u64 <= full_size &&
                      offset <= full_size - params_nested_anon3_ptr_len_u64);
  decoded.params.nested.ptr = reinterpret_cast<int *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.params.nested.ptr) % 8) == 0);
  offset += static_cast<size_t>(params_nested_anon3_ptr_len_u64);

  // arg2: top-level payload pointer output
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(offset <= full_size && full_size - offset >= sizeof(uint64_t));
  uint64_t output_len_u64 = read_scalar_le<uint64_t>(data, offset);
  RAPID_DECODE_ASSERT((offset % 8) == 0);
  RAPID_DECODE_ASSERT(output_len_u64 <= full_size && offset <= full_size - output_len_u64);
  decoded.output = reinterpret_cast<float *>(const_cast<uint8_t *>(data + offset));
  RAPID_DECODE_ASSERT((reinterpret_cast<uintptr_t>(decoded.output) % 8) == 0);
  offset += static_cast<size_t>(output_len_u64);

  // Theoretical field-aware count_fits_buffer constraints for the full manifest.
  RAPID_DECODE_ASSERT(decoded.params.count <= params_data_len_u64 / 4);
  RAPID_DECODE_ASSERT(decoded.params.nested.inner_count <= output_len_u64 / 4);
  RAPID_DECODE_ASSERT(decoded.seed >= 0);
  RAPID_DECODE_ASSERT(decoded.params.nested.inner_count >= 0);

  return decoded;
```

The important union rule is that the active selector chooses exactly one union
member. For pointer members, decode materializes one payload pointer and writes
that pointer into the union storage. For the scalar `tag` member, decode writes
only the scalar bytes. It must not attempt to decode `int_ptr`, `float_ptr`, and
`tag` in sequence, because those writes alias the same union storage. Any
constraint that reads a union member must also be guarded by the same active
selector; in the example, `choice.tag <= count` is checked only in the
`kChoiceTag` branch.

### Theoretical `gen/fuzzer_invoke.v1.cuh`

Invoke does not need special union handling. Once decode has materialized the
active union member inside `decoded.params`, invoke forwards the already-decoded
kernel arguments exactly like the current supported projection:

```cpp
#ifndef __PHASE2_INVOKE_COMPLEX_ARGS_KERNEL_CUH__
#define __PHASE2_INVOKE_COMPLEX_ARGS_KERNEL_CUH__

#include "fuzzer_decode.v1.cuh"
#include "type_shim.v1.cuh"

extern "C" __device__ void __rapid_entry__complex_args_kernel(int seed, KernelParams params, float * output);

__device__ inline void fuzzer_invoke_v1(const DecodedKernelArgs &decoded) {
  __rapid_entry__complex_args_kernel(decoded.seed, decoded.params, decoded.output);
}

#endif // __PHASE2_INVOKE_COMPLEX_ARGS_KERNEL_CUH__
```
