# Third-Party CUDA Targets

This directory contains upstream projects used to exercise RAPID against
real-world CUDA kernels. The projects remain independent Git submodules; the
superproject records the exact revision used by each test campaign.

The four targets below were selected for the first bug-finding campaign. They
combine relatively focused CUDA implementations with useful boundary,
correctness, and memory-safety test surfaces.

## Selected targets

| Priority | Path | Upstream | Initial revision | Primary test focus |
| --- | --- | --- | --- | --- |
| 1 | `gpuRIR/` | <https://github.com/DavidDiazGuerra/gpuRIR> | `fd8af43a4a113d3c2c05f0085a0119ecb1f1a484` | Dimensions, zero-distance inputs, NaN/Inf propagation, integer overflow, mixed precision, and out-of-bounds accesses |
| 2 | `CudaSift/` | <https://github.com/Celebrandil/CudaSift> | `ed5ef54c67354959e3d5fa829d30de5b278b4d62` | Tiny and odd image sizes, pitch and capacity boundaries, empty feature sets, matching counts, and OpenCV differential checks |
| 3 | `phantom-fhe/` | <https://github.com/encryptorion-lab/phantom-fhe> | `1f4a198443b3af77118e51f53d5b8f332154b875` | FHE encode/decode, FFT/NTT round trips, modulus and slot boundaries, Debug/Release differences, and stream interactions |
| 4 | `lietorch/` | <https://github.com/princeton-vl/lietorch> | `e7df86554156b36846008d8ddbcc4d8521a16554` | PyTorch tensor shape, dtype, batch, spatial-size, and radius validation; forward/backward and gradient consistency |

The revision in this table documents the initial import. Git always uses the
gitlink recorded by the current RAPID commit as the authoritative revision.

## Shallow initialization

The submodules are marked with `shallow = true` in the repository's
`.gitmodules`. Initialize only these targets with:

```bash
git submodule update --init --recommend-shallow --depth 1 \
  third_party/gpuRIR \
  third_party/CudaSift \
  third_party/phantom-fhe \
  third_party/lietorch
```

For a fresh clone, Git can also initialize every configured submodule shallowly:

```bash
git clone --recurse-submodules --shallow-submodules <rapid-repository-url>
```

Do not use `git submodule update --remote` for reproducible campaigns. Updating
an upstream target should be an explicit RAPID change that records a new
gitlink and documents any resulting build, kernel-discovery, or behavior
changes.

## Campaign order

1. Start with `gpuRIR` because its kernels mostly use direct pointer and scalar
   arguments and expose a compact, high-value dimension and numerical boundary
   surface.
2. Continue with `CudaSift`, prioritizing block-boundary image dimensions,
   pitch handling, feature capacity, and CPU/OpenCV differential checks.
3. Use the simpler `phantom-fhe` kernels, such as encode/decode and focused
   FFT/NTT operations, before attempting context-heavy FHE operations.
4. Begin `lietorch` at the public PyTorch extension boundary with shape and
   gradient fuzzing. Its `PackedTensorAccessor` kernel signatures are less
   direct for the current RAPID Phase 2 path.

For RAPID ingestion, capture each project's real CUDA build separately, run
Phase 1 filtering first, and carry only supported kernels into Phase 2 and
Phase 3. Unsupported template or aggregate signatures should remain explicit
support-boundary results rather than being forced through code generation.

## RAPID fuzz support data

RAPID-owned support data for these vendored targets lives under
`third_party/fuzz/`:

- `third_party/fuzz/kernel_constraints/` contains manifest override registries
  derived from project call sites, allocation expressions, launch constants,
  and documentation comments.
- `third_party/fuzz/support/` contains per-kernel run/skip decisions and the
  evidence used by third-party fuzz campaign reports.

The Python tooling that applies these files remains in `scripts/` because it is
RAPID infrastructure code. The JSON data is kept here so third-party campaign
inputs stay colocated with the vendored projects they describe.

## Upstream ownership

The source code and licenses inside each submodule belong to their respective
upstream projects. Keep RAPID-specific harnesses, manifests, seeds, and result
artifacts outside the submodule directories whenever possible. If an upstream
fix is required, develop it on a dedicated branch or fork and update the
submodule gitlink only after the change is reproducible.
