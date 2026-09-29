# Bug: shoc_radix_sort scanLSB shared-memory underflow

VConfig mutation exposed the first confirmed kernel bug found by a mutating
campaign. `benchmark/workloads/shoc_radix_sort/kernel.cu` unconditionally reads
`s_data[idx - 64]` in `scanLSB()`. With `blockDim.x == 32`, `idx` is 32 through
63, so the read indexes shared memory at -32 through -1 and the launch fails
with `CUDA_ERROR_ILLEGAL_ADDRESS`. All 96 archived crash inputs select logical
block `(32, 1, 1)` and are stored under:

`benchmark/rq4/results/profile-mutating-30s-seed1-r1-w2-feedback-v3-e9f8d3e/raw/shoc_radix_sort-libafl-plus-r1.workdir/crashes/`

The original kernel remains unchanged as bug evidence. The derived fix and its
correctness probe are in
`benchmark/rq2/variants/shoc_radix_sort_fix/`; the distance-64 read uses a
zero contribution when `idx < 64`. The declared 256-thread candidate also adds
the required distance-128 scan stage and shared scatter capacity.

The external variant name is `shoc_radix_sort_fix`. Its CUDA entry symbol, and
therefore generated kernel IDs such as
`rq1_shoc_radix_sort_scanfix__<hash>`, intentionally retain the pipeline name
to avoid requiring a Phase2 rebuild.
