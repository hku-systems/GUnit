#include "rapid_target_layout.v1.h"
#include "input/task_envelope.cuh"

#include <cstddef>
#include <cstdint>

#ifndef FUZZER_INVOKE_HEADER
#error "FUZZER_INVOKE_HEADER must point to a generated invoke header"
#endif

#include FUZZER_INVOKE_HEADER

extern "C" __global__ void cufuzz_wrapper(RapidTaskEnvelopeHeader *envelope,
                                           RapidKernelContext *context) {
  volatile uint8_t *data = rapid_task_payload(envelope);
  const size_t size = static_cast<size_t>(envelope->payload_size);
  DecodedKernelArgs decoded = fuzzer_decode_v1(data, size);
  fuzzer_invoke_v1(decoded, context);
}
