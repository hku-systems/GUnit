#ifndef __UTILS_COVERAGE_CONSTANTS_H__
#define __UTILS_COVERAGE_CONSTANTS_H__

#include <stdint.h>

// Keep consistent with EDGES_MAP_SIZE in Rust code
#define MAP_SIZE 65536

// Global host-visible coverage map array.
//
// This is exported for the Rust fuzzer via dlsym(), so keep the default
// visibility and stable alignment.
extern "C" {
__attribute__((visibility("default"), aligned(4096))) extern uint8_t
    libafl_cov_map[MAP_SIZE];
}

#endif /* __UTILS_COVERAGE_CONSTANTS_H__ */
