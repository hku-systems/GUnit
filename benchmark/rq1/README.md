# RQ1 end-to-end performance benchmark

RQ1 measures fixed-input throughput for the 12 workloads selected by
`suite.json`. Workload identities and source metadata come from the shared
`benchmark/workloads/` catalog; `workloads` is a compatibility symlink to that
catalog.

## Prerequisites

- A CUDA-capable GPU and a CUDA toolkit matching the selected architecture
  (for example, `/usr/local/cuda` and `sm_86`).
- The repository-local Python environment at `.venv/`.
- Release fuzzer binaries:

  ```bash
  cargo build --manifest-path cuda-fuzzer/Cargo.toml --release --bins
  ```

- Initialized third-party submodules and a Phase 2 kernel directory for each
  imported workload. Each mapped directory must contain `manifest.json` and
  `phase2/`.
- A clean source/provenance gate:

  ```bash
  .venv/bin/python benchmark/rq1/verify.py
  ```

No workload contains hand-written feedback. CuFuzz-style and plain LibAFL
disable device feedback; LibAFL+, `rapid`, and `rapid2` use the repository's
LLVM feedback instrumentation.

## Historical results archive

RQ1 campaign results are no longer tracked in git (`/benchmark/rq1/results/`
is ignored). The historical six-workload archive collected on 2026-07-29 at
RAPID commit `f7587ac` (`REPORT.md`, `summary.csv`, `trials.jsonl`,
`environment.json`, `kernel_manifest.csv`, `logs/`) and the 2026-08-07
twelve-workload paper dataset (`results/rq1-paper-throughput-20260807.csv`)
were removed from the working tree; both remain recoverable from git history
at commit `d86fabc` and earlier. Regenerate current data with the
build/campaign/export flow below (or `benchmark/reproduce.sh full`).

## Build topology

RQ1 uses two build roots because the sources enter Phase 2 through different
paths. The roots are consumed separately by `runtime_verify.py` and
`campaign.py`, then merged by `export_paper.py`.

| Build root | Workloads | Producer |
| --- | --- | --- |
| standalone | `shoc_reduction`, `shoc_radix_sort`, `shoc_scan`, `cutlass_gemm`, `flashattention_device1xn`, `pytorch_batchnorm` | `build.py` captures, rewrites, and builds the checked-in sources |
| imported | `apex_maybe_cast`, `llama_upscale_f32_bilinear`, `gpurir_generate_rir`, `apex_index_mul_2d_vgeo`, `cuda_samples_inverse_cnd`, `synth_complex` | `import_phase2.py` copies verified host-local Phase 2 artifacts and builds the same backend matrix |

Both producers emit one `build_report.json` per workload. All backends for a
workload share the manifest, generated decode/invoke code, VConfig contract,
input layout, and payload size.

The Phase 2 map is a JSON object whose values are kernel directories, not the
`phase2/` subdirectories themselves:

```json
{
  "apex_maybe_cast": "/abs/path/to/apex_maybe_cast/kernel__id",
  "llama_upscale_f32_bilinear": "/abs/path/to/llama_upscale/kernel__id",
  "gpurir_generate_rir": "/abs/path/to/gpurir_generate_rir/kernel__id",
  "apex_index_mul_2d_vgeo": "/abs/path/to/apex_index_mul/kernel__id",
  "cuda_samples_inverse_cnd": "/abs/path/to/inverse_cnd/kernel__id",
  "synth_complex": "/abs/path/to/synth_complex/kernel__id"
}
```

## Run

Run all commands from the repository root. Replace the run IDs, GPU, CUDA path,
architecture, and Phase 2 map with values for the host.

### 1. Build the two roots

Build the six standalone workloads:

```bash
.venv/bin/python -m benchmark.rq1.build \
  --run-id standalone-sm86-YYYYMMDD \
  --cuda-path /usr/local/cuda \
  --cuda-arch sm_86 \
  --build-profile release
```

Build the six imported workloads:

```bash
.venv/bin/python -m benchmark.rq1.import_phase2 \
  --phase2-map /abs/path/to/rq1-phase2-map.json \
  --run-id imported-sm86-YYYYMMDD \
  --cuda-path /usr/local/cuda \
  --cuda-arch sm_86 \
  --fuzzer cuda-fuzzer/target/release/fuzzer \
  --workload apex_maybe_cast \
  --workload llama_upscale_f32_bilinear \
  --workload gpurir_generate_rir \
  --workload apex_index_mul_2d_vgeo \
  --workload cuda_samples_inverse_cnd \
  --workload synth_complex
```

These commands create:

```text
benchmark/rq1/build/standalone-sm86-YYYYMMDD/
benchmark/rq1/build/imported-sm86-YYYYMMDD/
```

Build run IDs are immutable: both producers reject an existing run directory.
Choose a new ID instead of overwriting an earlier build.

### 2. Verify runtime equivalence

Run the gate once per build root. This is a GPU operation and must complete
before collecting throughput:

```bash
.venv/bin/python -m benchmark.rq1.runtime_verify \
  --build-root benchmark/rq1/build/standalone-sm86-YYYYMMDD \
  --gpu-device 0 --runs 10 --rapid-window-size 4 \
  --fuzzer cuda-fuzzer/target/release/fuzzer

.venv/bin/python -m benchmark.rq1.runtime_verify \
  --build-root benchmark/rq1/build/imported-sm86-YYYYMMDD \
  --gpu-device 0 --runs 10 --rapid-window-size 4 \
  --fuzzer cuda-fuzzer/target/release/fuzzer
```

The window must be in `1..=32`. Each workload receives
`runtime_verification.json`; the build root receives
`runtime_verification_summary.json`.

### 3. Run the throughput campaigns

Use the same protocol for both roots. `--window-size 4` produces the `W=4`
rows required by the paper exporter; the campaign also includes the `W=1`
ordered configurations.

```bash
.venv/bin/python -m benchmark.rq1.campaign \
  --build-root benchmark/rq1/build/standalone-sm86-YYYYMMDD \
  --output benchmark/rq1/results/standalone-sm86-2r-30s \
  --repetitions 2 --warmup-runs 100 --benchmark-seconds 30 \
  --window-size 4 --gpu-device 0 --timeout 900 \
  --fuzzer cuda-fuzzer/target/release/fuzzer \
  --fuzzer-async cuda-fuzzer/target/release/fuzzer_async

.venv/bin/python -m benchmark.rq1.campaign \
  --build-root benchmark/rq1/build/imported-sm86-YYYYMMDD \
  --output benchmark/rq1/results/imported-sm86-2r-30s \
  --repetitions 2 --warmup-runs 100 --benchmark-seconds 30 \
  --window-size 4 --gpu-device 0 --timeout 900 \
  --fuzzer cuda-fuzzer/target/release/fuzzer \
  --fuzzer-async cuda-fuzzer/target/release/fuzzer_async
```

Reusing the same `--output` resumes at the next unrecorded
`(workload, configuration, repetition)` identity. Entries already present in
`trials.jsonl` or `failures.jsonl` are skipped; recorded timeouts and non-zero
exits are not automatically retried. Each invocation regenerates `summary.csv`
from completed trials and warns about configurations with fewer than the
requested repetitions.

### 4. Merge and export paper data

Pass both campaign result directories with repeated `--input` options:

```bash
.venv/bin/python -m benchmark.rq1.export_paper \
  --input benchmark/rq1/results/standalone-sm86-2r-30s \
  --input benchmark/rq1/results/imported-sm86-2r-30s \
  --repetitions 2 \
  --output benchmark/rq1/results/rq1-paper-throughput.csv
```

The exporter rejects missing/duplicate workloads, missing required backends,
and repetition counts that differ from `--repetitions`.

## Required parameters

| Tool | Required for this workflow | Meaning |
| --- | --- | --- |
| `build.py` | `--run-id` | New directory below `benchmark/rq1/build/` |
| `import_phase2.py` | `--phase2-map`, `--run-id`, six repeated `--workload` values | Host-local Phase 2 inputs and imported subset |
| `runtime_verify.py` | `--build-root` | One of the two completed RQ1 build roots |
| `campaign.py` | `--build-root`, `--output`, `--repetitions`, `--warmup-runs`, `--benchmark-seconds` | Input artifacts, resumable result root, and measurement protocol |
| `export_paper.py` | repeated `--input`, `--output` | Campaign summaries to merge and destination CSV |

`--cuda-path` defaults to `CUDA_PATH` or `/usr/local/cuda`; `--cuda-arch`
defaults to `CUDA_ARCH` or `sm_86`. GPU selectors default to device `0`.
Generated build products and campaign results are ignored experiment artifacts;
archive their build reports, runtime reports, environment, JSONL, logs, summaries,
and source/provenance metadata together.

## Offline source reconstruction

From a standalone workload directory:

```bash
patch --batch --forward upstream.source -o reconstructed.cu -i extraction.patch
```

`reconstructed.cu` must be byte-identical to `kernel.cu` and match the hash in
`provenance.json`.
