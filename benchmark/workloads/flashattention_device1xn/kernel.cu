// Fixed-d=16 standalone extraction of the FlashAttention device1xN
// row-validity path. It preserves the upstream row mapping and fixed condition
// while avoiding a conditional block-wide barrier.
extern "C" __global__ void rq1_flashattention_device1xn(
    const float* input, float* output) {
  constexpr int THREADS = 128;
  constexpr int THREADS_PER_ROW = 4;
  constexpr int ROWS = 16;
  constexpr int ROWS_PER_STG = 16;
  constexpr int STGS_PER_LOOP = 1;
  const int tidx = threadIdx.x;

  int rows[STGS_PER_LOOP];
  for (int jj = 0; jj < STGS_PER_LOOP; jj++) {
    rows[jj] = tidx / THREADS_PER_ROW + jj * ROWS_PER_STG;
  }
  bool o_rows_are_valid =
      (THREADS <= THREADS_PER_ROW * ROWS) ||
      (tidx / THREADS_PER_ROW < ROWS);

  float value = 0.0f;
  if (o_rows_are_valid) {
    const int lane = tidx % THREADS_PER_ROW;
    const int row_offset = rows[0] * THREADS_PER_ROW;
    value = input[row_offset + lane];
    value = value > 0.0f ? value : 0.0f;
    value *= 1.0f / 16.0f;
  }
  output[tidx] = value;
}
