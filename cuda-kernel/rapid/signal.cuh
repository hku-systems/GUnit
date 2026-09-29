#ifndef __RAPID_SIGNAL_CUH__
#define __RAPID_SIGNAL_CUH__

#include <cuda_runtime.h>

#include <cstdio>

#include "cuda_utils.cuh"

struct Semaphore {
  volatile int *data_ready_sem;
  volatile int *data_processed_sem;
  volatile bool *run;
};

// Device-side semaphore operations
__device__ void gpu_wait(Semaphore semaphore);
__device__ void gpu_signal(Semaphore semaphore);

// Host-side semaphore operations
void cpu_wait(int *d_sem, cudaStream_t *stream);
void cpu_signal(int *d_sem, cudaStream_t *stream);

// Kernel control
void stop_persistent_kernel(bool *d_run, int *d_data_ready_sem,
                            cudaStream_t *stream);

#include "signal_helper.cuh"

#endif // __RAPID_SIGNAL_CUH__
