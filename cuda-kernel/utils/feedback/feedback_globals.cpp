#include "feedback/feedback_constants.h"

extern "C" {
__attribute__((visibility("default"), aligned(4096))) uint8_t
    libafl_simt_memcov_bits[SIMT_MEMCOV_STORAGE_SIZE] = {0};
}
