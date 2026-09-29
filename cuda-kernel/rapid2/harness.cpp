#include <cstdint>

#include "rapid_target_layout.v1.h"
#include "coverage_constants.h"

extern "C" {
__attribute__((visibility("default"), aligned(4096))) uint8_t
    libafl_cov_map[MAP_SIZE] = {0};
}

#include "kernel_backend.cuh"
#include "libafl_interface_shared.cuh"
