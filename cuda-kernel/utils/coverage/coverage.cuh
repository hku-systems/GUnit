#ifndef __UTILS_COVERAGE_CUH__
#define __UTILS_COVERAGE_CUH__

#include "coverage_constants.h"

#include <cuda_runtime.h>
#include <cstdio>

// Global host-visible coverage map pointer
// Global device-side fallback coverage map and task-local coverage binding.
#if defined(RAPID_DEFINE_DEVICE_COVERAGE_MAP)
__device__ __align__(sizeof(uint32_t)) uint8_t device_cov_map[MAP_SIZE];
__device__ uint8_t *rapid_bound_cov_map;
#else
extern __device__ __align__(sizeof(uint32_t)) uint8_t device_cov_map[MAP_SIZE];
extern __device__ uint8_t *rapid_bound_cov_map;
#endif

// Initialize edges (host)
extern "C" void init_edges(cudaStream_t copy_stream);
extern "C" void get_edges(cudaStream_t copy_stream);
// Record edge coverage (device)
__device__ void __libafl_edge(uint16_t _index, uint8_t *_cov_map);

#include "coverage_helper.cuh"

// Macro to record edge coverage (device)
// Uses the task-bound global output map. RAPID2 binds this to the active
// double-buffer slot; origin/rapid bind it to the module global map.
#define __LIBAFL_EDGE(_index) (__libafl_edge(_index, rapid_current_coverage_map()))

#endif /* __UTILS_COVERAGE_CUH__ */
