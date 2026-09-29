#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                               \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

#define VDIM_KERNEL __attribute__((annotate("vdim")))

extern "C" void vdim_set(dim3 *vgrid, dim3 *vblock);

extern "C" __global__ VDIM_KERNEL void vdim_kernel(const float *in, float *out,
                                                   int accumN, int iters) {
  int bx = (int)blockIdx.x;
  int by = (int)blockIdx.y;
  int tx = (int)threadIdx.x;
  int ty = (int)threadIdx.y;

  int block_linear = by * (int)gridDim.x + bx;
  int thread_linear = ty * (int)blockDim.x + tx;
  int linear = block_linear * ((int)blockDim.x * (int)blockDim.y) + thread_linear;

  float acc = 0.0f;
  for (int i = 0; i < iters; ++i) {
    int idx = linear + by * (int)gridDim.x + i * accumN;
    acc += in[idx] * (1.0f + 0.01f * (bx + by) + 0.001f * (tx + ty));
  }
  out[linear] = acc;
}

static float ref_value(int bx, int by, int tx, int ty, dim3 vgrid, dim3 vblock,
                       const float *in, int accumN, int iters) {
  int block_linear = by * (int)vgrid.x + bx;
  int thread_linear = ty * (int)vblock.x + tx;
  int linear = block_linear * ((int)vblock.x * (int)vblock.y) + thread_linear;
  float acc = 0.0f;
  for (int i = 0; i < iters; ++i) {
    int idx = linear + by * (int)vgrid.x + i * accumN;
    acc += in[idx] * (1.0f + 0.01f * (bx + by) + 0.001f * (tx + ty));
  }
  return acc;
}

static void run_case(dim3 pgrid, dim3 pblock, dim3 vgrid, dim3 vblock,
                     int accumN, int iters) {
  if (vgrid.x > pgrid.x || vgrid.y > pgrid.y || vgrid.z > pgrid.z ||
      vblock.x > pblock.x || vblock.y > pblock.y || vblock.z > pblock.z) {
    std::fprintf(stderr, "Invalid vdim > physical dim\n");
    std::exit(1);
  }

  int vthreads = (int)(vgrid.x * vgrid.y * vblock.x * vblock.y);
  int extra = (int)(vgrid.x * vgrid.y);
  int input_size = vthreads + extra + (iters - 1) * accumN;

  std::vector<float> h_in(input_size, 1.0f);
  std::vector<float> h_out(vthreads, -1.0f);
  std::vector<float> h_ref(vthreads, -1.0f);

  for (unsigned by = 0; by < vgrid.y; ++by) {
    for (unsigned bx = 0; bx < vgrid.x; ++bx) {
      for (unsigned ty = 0; ty < vblock.y; ++ty) {
        for (unsigned tx = 0; tx < vblock.x; ++tx) {
          int block_linear = (int)(by * vgrid.x + bx);
          int thread_linear = (int)(ty * vblock.x + tx);
          int linear = block_linear * ((int)vblock.x * (int)vblock.y) + thread_linear;
          h_ref[linear] = ref_value((int)bx, (int)by, (int)tx, (int)ty,
                                    vgrid, vblock, h_in.data(), accumN, iters);
        }
      }
    }
  }

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, h_out.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_out, h_out.data(), h_out.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 *d_vgrid = nullptr;
  dim3 *d_vblock = nullptr;
  CUDA_CHECK(cudaMalloc(&d_vgrid, sizeof(dim3)));
  CUDA_CHECK(cudaMalloc(&d_vblock, sizeof(dim3)));
  CUDA_CHECK(cudaMemcpy(d_vgrid, &vgrid, sizeof(dim3),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vblock, &vblock, sizeof(dim3),
                        cudaMemcpyHostToDevice));

  vdim_set(d_vgrid, d_vblock);
  vdim_kernel<<<pgrid, pblock>>>(d_in, d_out, accumN, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_abs_diff = 0.0f;
  for (int i = 0; i < vthreads; ++i) {
    float diff = h_out[i] - h_ref[i];
    if (diff < 0.0f)
      diff = -diff;
    if (diff > max_abs_diff)
      max_abs_diff = diff;
  }

  std::printf("vdim=(%u,%u) vblock=(%u,%u) max_abs_diff=%f\n",
              vgrid.x, vgrid.y, vblock.x, vblock.y, max_abs_diff);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_vgrid));
  CUDA_CHECK(cudaFree(d_vblock));
}

int main() {
  dim3 pgrid(8, 4, 1);
  dim3 pblock(32, 2, 1);
  int accumN = 4;
  int iters = 8;

  run_case(pgrid, pblock, dim3(8, 4, 1), dim3(32, 2, 1), accumN, iters);
  run_case(pgrid, pblock, dim3(8, 2, 1), dim3(32, 1, 1), accumN, iters);
  run_case(pgrid, pblock, dim3(4, 2, 1), dim3(32, 1, 1), accumN, iters);
  return 0;
}
