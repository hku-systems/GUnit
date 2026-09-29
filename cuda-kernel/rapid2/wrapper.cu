#include "rapid_target_layout.v1.h"
#include "config.h"

#if RAPID_ENABLE_INNER_FEEDBACK
#include "coverage/coverage.cuh"
#endif

#ifndef FUZZER_INVOKE_HEADER
#error "FUZZER_INVOKE_HEADER must point to a generated invoke header"
#endif

#include "combined_input.cuh"
#include FUZZER_INVOKE_HEADER
#include "pipelined_kernel_impl.cuh"

extern "C" __global__ void rapid2_persistent_kernel(
    CombinedInput *d_combined0, CombinedInput *d_combined1,
    CombinedInput *d_combined2, CombinedInput *d_combined3,
    uint8_t *d_coverage0, uint8_t *d_coverage1, uint8_t *d_coverage2,
    uint8_t *d_coverage3, RapidKernelContext *d_context0,
    RapidKernelContext *d_context1, RapidKernelContext *d_context2,
    RapidKernelContext *d_context3, volatile uint64_t *ready0,
    volatile uint64_t *ready1, volatile uint64_t *ready2,
    volatile uint64_t *ready3, volatile uint64_t *done0,
    volatile uint64_t *done1, volatile uint64_t *done2,
    volatile uint64_t *done3,
    volatile bool *should_exit, int kernel_id) {
  double_buffered_persistent_kernel_impl(
      d_combined0, d_combined1, d_combined2, d_combined3, d_coverage0,
      d_coverage1, d_coverage2, d_coverage3, d_context0, d_context1,
      d_context2, d_context3, ready0, ready1, ready2, ready3, done0, done1,
      done2, done3, should_exit, kernel_id);
}
