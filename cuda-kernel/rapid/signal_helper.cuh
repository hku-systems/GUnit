#ifndef __RAPID_SIGNAL_HELPER_CUH__
#define __RAPID_SIGNAL_HELPER_CUH__
#ifndef __RAPID_SIGNAL_CUH__
#error "signal_helper.cuh should only be included from signal.cuh"
#endif

// Device-side semaphore wait operation
__device__ void gpu_wait(Semaphore semaphore) {
  // Spin-wait until the semaphore is set to 1 by the CPU
  volatile int *data_ready = semaphore.data_ready_sem;
  while (threadIdx.x == 0 && !(*data_ready)) {
    // Busy-wait loop
  }
  // All threads in the block wait here until thread 0 has observed the
  // change in the semaphore.
  __syncthreads();
}

// Device-side semaphore signal operation
__device__ void gpu_signal(Semaphore semaphore) {
  // Atomically set the semaphore to 1
  int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_tid == 0) {
    int* data_ready = const_cast<int*>(semaphore.data_ready_sem);
    int* data_processed = const_cast<int*>(semaphore.data_processed_sem);
    // Publish all output and feedback writes before the host-visible
    // completion flag. The host treats data_processed as the message-passing
    // publication point for this envelope.
    __threadfence_system();
    atomicExch(data_ready, 0);
    atomicExch(data_processed, 1);
  }
}

// Host-side semaphore wait operation
void cpu_wait(int *d_sem, cudaStream_t *stream) {
  int h_sem = 0;
  do {
    CUDA_CHECK(cudaMemcpyAsync(&h_sem, d_sem, sizeof(int),
                               cudaMemcpyDeviceToHost, *stream));
  } while (h_sem == 0);
  // Reset semaphore on device
  h_sem = 0;
  CUDA_CHECK(cudaMemcpyAsync(d_sem, &h_sem, sizeof(int), cudaMemcpyHostToDevice,
                             *stream));
  cudaStreamSynchronize(*stream);
}

// Host-side semaphore signal operation
void cpu_signal(int *d_sem, cudaStream_t *stream) {
  int h_sem = 1;
  CUDA_CHECK(cudaMemcpyAsync(d_sem, &h_sem, sizeof(int), cudaMemcpyHostToDevice,
                             *stream));
  cudaStreamSynchronize(*stream);
}

void stop_persistent_kernel(bool *d_run, int *d_data_ready_sem,
                            cudaStream_t *stream) {
  printf("\nCPU: Stopping the kernel...\n");
  bool h_run = false;
  CUDA_CHECK(cudaMemcpyAsync(d_run, &h_run, sizeof(bool),
                             cudaMemcpyHostToDevice, *stream));
  cudaStreamSynchronize(*stream);
  // Signal one last time to exit the loop
  cpu_signal(d_data_ready_sem, stream);
}

#endif // __RAPID_SIGNAL_HELPER_CUH__
