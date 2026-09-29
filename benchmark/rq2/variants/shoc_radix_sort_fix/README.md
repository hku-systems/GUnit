# shoc_radix_sort_fix

Derived fix variant for the SHOC radix-sort block kernel
(`rq1_shoc_radix_sort`, extracted in
`benchmark/workloads/shoc_radix_sort/`).

## Bug

`scanLSB()` in the original kernel unconditionally reads `s_data[idx - 64]`.
With `blockDim.x == 32`, `idx` is 32..63, so the read indexes shared memory
at -32..-1 and the launch fails with `CUDA_ERROR_ILLEGAL_ADDRESS`. The bug
was exposed by a mutating campaign through VConfig mutation; all archived
crash inputs select logical block `(32, 1, 1)`. Full report:
[`../../reports/SHOC_RADIX_SORT_SCAN_UNDERFLOW.md`](../../reports/SHOC_RADIX_SORT_SCAN_UNDERFLOW.md).

The original kernel in `benchmark/workloads/shoc_radix_sort/kernel.cu` stays
unchanged as bug evidence and negative control.

## Files

- `kernel.cu` — the fixed entry `rq1_shoc_radix_sort_scanfix`:
  the distance-64 scan read contributes zero when `idx < 64`; a guarded
  distance-128 stage is added so the scan stays complete at 256 threads; the
  shared buffer grows 512 → 1024 words to hold the four-elements-per-thread
  scatter layout through `blockDim.x == 256`. The `rq1_` symbol prefix is
  kept intentionally to avoid requiring a Phase2 rebuild.
- `constraints.json` — RQ2-style VConfig contract with block candidates
  32/64/128/256 and VConfig mutation enabled.
- `probe.cu` — correctness probe. This is **not** the fuzz target; it is the
  executable evidence cited by `constraints.json` (`correctness_probe`).
- `LICENSE.upstream` — SHOC license text inherited from the source extraction
  in `benchmark/workloads/shoc_radix_sort/`.

## Build and run the probe

Run from the repository root (adjust `-arch` for the host GPU):

```bash
nvcc -std=c++17 -O2 -arch=sm_86 \
  benchmark/rq2/variants/shoc_radix_sort_fix/probe.cu \
  benchmark/workloads/shoc_radix_sort/kernel.cu \
  benchmark/rq2/variants/shoc_radix_sort_fix/kernel.cu \
  -o /tmp/shoc_radix_sort_fix_probe

# Validates the original kernel at block 128 and the fixed kernel at
# block 32/64/128/256 against block-local CPU stable-radix references.
/tmp/shoc_radix_sort_fix_probe
# expected: shoc radix-sort fix probe: all outputs match CPU references

# Opt-in: reproduces the original block-32 shared-memory illegal address.
/tmp/shoc_radix_sort_fix_probe --reproduce-original-block32
# expected: reproduced original block-32 shared-memory illegal address
```

Both modes were verified on sm_86 (RTX 3090).

## Using the fixed kernel in campaigns

RQ1/RQ2/RQ4 are throughput/coverage experiments over their own catalogs; none
of them automatically picks up this fix:

- **RQ1** (`benchmark/rq1/`) and **RQ4** (`benchmark/rq4/`) build from
  `benchmark/workloads/` with VConfig off and block 128 only. The buggy
  geometry is unreachable there, and the fix is bit-identical to the original
  at block 128, so their results are unaffected by the bug.
- **RQ2 formal catalog** (`benchmark/rq2/catalog.json`) still points
  `shoc_radix_sort` at the original kernel with logical block candidates
  64/96/128 — all safe geometries for the original scan.
- Derived-variant campaigns (the other entries in `benchmark/rq2/variants/`)
  ran through the third-party-fuzz pipeline as `*_variants` projects (see
  `build/e2e/third-party-fuzz/20260802-rq2-wave3/`), not through
  `benchmark.rq2.build`/`campaign`. This variant is not yet registered in any
  campaign ledger.

To run RQ2-style coverage campaigns on the fixed kernel, either:

1. add a `shoc_radix_sort_fix` entry to `benchmark/rq2/catalog.json` whose
   `kernel`/`constraints` point at this directory (plus a provenance record,
   like `synth_complex`'s `synthetic_provenance`), then use
   `benchmark.rq2.build` and `benchmark.rq2.campaign` as documented in
   `benchmark/rq2/README.md`; or
2. register it as a `*_variants` project in a third-party-fuzz campaign
   ledger and use `scripts.third_party_fuzz.cli` (`phase2` → `collect` →
   `build` → `coverage-matrix`), matching how the other derived variants were
   measured.
