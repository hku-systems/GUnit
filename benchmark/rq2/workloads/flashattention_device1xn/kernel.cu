// RQ2 VConfig adaptation of the fixed-d=16 FlashAttention device1xN
// row-validity extraction. ROWS_PER_STG preserves the one-stage upstream
// mapping; with jj=0 it contributes no row offset.
extern "C" __global__ void rq1_flashattention_device1xn(
    const float* input, float* output, const unsigned int n) {
  constexpr unsigned int THREADS_PER_ROW = 4;
  constexpr unsigned int ROWS_PER_STG = 16;
  constexpr unsigned int STGS_PER_LOOP = 1;
  const unsigned int tidx = threadIdx.x;

  unsigned int rows[STGS_PER_LOOP];
  for (unsigned int jj = 0; jj < STGS_PER_LOOP; ++jj) {
    rows[jj] = tidx / THREADS_PER_ROW + jj * ROWS_PER_STG;
  }
  const unsigned int row_count =
      (n + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
  const bool o_rows_are_valid =
      rows[0] < row_count && tidx < n;

  float value = 0.0f;
  if (o_rows_are_valid) {
    const unsigned int lane = tidx % THREADS_PER_ROW;
    const unsigned int row_offset = rows[0] * THREADS_PER_ROW;
    value = input[row_offset + lane];
    value = value > 0.0f ? value : 0.0f;
    value *= 1.0f / 16.0f;
  }
  output[tidx] = value;
}
