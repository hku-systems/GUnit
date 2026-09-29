#include <cstddef>
#include <cstdint>

#ifndef PHASE2_INVOKE_HEADER
#error "PHASE2_INVOKE_HEADER must be provided by the runtime smoke build"
#endif

#include PHASE2_INVOKE_HEADER

extern "C" __global__ void run_generated(volatile uint8_t *data, size_t size) {
  DecodedKernelArgs decoded = fuzzer_decode_v1(data, size);
  fuzzer_invoke_v1(decoded);
}
