#ifndef FIXTURE_API_CUH
#define FIXTURE_API_CUH

/*
 * Host-callable test wrapper declarations.
 * Each wrapper launches the corresponding kernel(s) defined in its own
 * translation unit and validates outputs.  Returns 0 on success, non-zero
 * on failure.  Wrappers print diagnostic PASS/FAIL lines to stdout.
 *
 * Exported as plain C so the runner can call them across TUs without
 * requiring cross-TU kernel launches.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* complex_kernels.cu wrappers */
int test_direct_kernel(void);
int test_macro_defined_kernel(void);
int test_macro_decl_kernel(void);
int test_templated_kernel(void);
int test_record_ptr_copy(void);
int test_uint3_write_sum(void);
int test_struct_enum_kernel(void);
int test_clamped_write(void);
int test_scaled_write(void);
int test_bitwise_blend(void);
int test_reduce_pair(void);

/* math_kernels.cu wrappers */
int test_add_kernel(void);
int test_mul_kernel(void);
int test_clamped_add(void);
int test_fused_mul_add(void);

/* ns_kernels.cu wrappers */
int test_ns_add(void);
int test_ns_scale(void);
int test_cuda_api_stress(void);

/* cublas_host_api_smoke.cu wrappers */
int test_cublas_host_api_smoke(void);

#ifdef __cplusplus
}
#endif

#endif /* FIXTURE_API_CUH */
