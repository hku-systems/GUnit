/*
 * fixture_runner.cu -- Top-level runner that calls all kernel test wrappers
 * from complex_kernels.cu and math_kernels.cu, validates outputs, and prints
 * an overall PASS/FAIL result.
 *
 * Build: see fixtures/Makefile target 'all'.
 * Run:   ./fixtures_runner
 */
#include <cstdio>
#include "fixture_api.cuh"

typedef int (*test_fn)(void);

struct test_entry {
  const char* name;
  test_fn     fn;
};

int main() {
  test_entry tests[] = {
    /* complex_kernels.cu */
    {"test_direct_kernel",        test_direct_kernel},
    {"test_macro_defined_kernel", test_macro_defined_kernel},
    {"test_macro_decl_kernel",    test_macro_decl_kernel},
    {"test_templated_kernel",     test_templated_kernel},
    {"test_record_ptr_copy",      test_record_ptr_copy},
    {"test_uint3_write_sum",      test_uint3_write_sum},
    {"test_struct_enum_kernel",   test_struct_enum_kernel},
    {"test_clamped_write",        test_clamped_write},
    {"test_scaled_write",         test_scaled_write},
    {"test_bitwise_blend",        test_bitwise_blend},
    {"test_reduce_pair",          test_reduce_pair},
    /* math_kernels.cu */
    {"test_add_kernel",           test_add_kernel},
    {"test_mul_kernel",           test_mul_kernel},
    {"test_clamped_add",          test_clamped_add},
    {"test_fused_mul_add",        test_fused_mul_add},
    /* ns_kernels.cu */
    {"test_ns_add",               test_ns_add},
    {"test_ns_scale",             test_ns_scale},
    {"test_cuda_api_stress",      test_cuda_api_stress},
    /* cublas_host_api_smoke.cu */
    {"test_cublas_host_api_smoke", test_cublas_host_api_smoke},
  };

  int n = sizeof(tests) / sizeof(tests[0]);
  int passed = 0;
  int failed = 0;

  printf("Running %d kernel tests...\n", n);

  for (int i = 0; i < n; i++) {
    printf("[%d/%d] %s\n", i + 1, n, tests[i].name);
    int rc = tests[i].fn();
    if (rc == 0) {
      passed++;
    } else {
      failed++;
    }
  }

  printf("\nResults: %d/%d passed, %d failed\n", passed, n, failed);

  if (failed == 0) {
    printf("PASS\n");
    return 0;
  } else {
    printf("FAIL\n");
    return 1;
  }
}
