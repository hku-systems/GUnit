#include "coverage/coverage_constants.h"

extern "C" {
__attribute__((visibility("default"), aligned(4096))) uint8_t
    libafl_cov_map[MAP_SIZE] = {0};
}
