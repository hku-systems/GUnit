# Kernel-Smoke Pipeline

`scripts/kernel-smoke/` captures real CUDA build commands and turns them into per-kernel artifacts:

- `kernel.bc`
- `kernel.ptx`
- `manifest.json`
- `metadata.json`

## Core Capability

- Works with existing Make/CMake builds via compiler wrapper injection.
- Artifact discovery only: PTX/IR entry discovery + Clang helper metadata extraction (exact PTX entry matching; no source regex supplementation).
- Variant-aware processing (same source built with different flags/arch gets separate variants).
- Capture runs are stream-by-variant by default (discover -> emit -> manifest per variant).
- Variant-level resume in capture runs (skip completed variants on `--resume`).
- Parallel variant processing via `--jobs`.
- Per-kernel extraction from module BC and PTX emission.
- Resumable runs via variant progress tracking.
- Optional wall-clock profiling via `--profile` + `profile.json`.

## Quick E2E (Make)

```bash
python3 -m pip install -r scripts/kernel-smoke/requirements.txt

chmod +x scripts/kernel-smoke/rapid-wrap scripts/kernel-smoke/compiler-dispatch.sh

RAPID_CAPTURE_DIR="$(pwd)/build/kernel-smoke/capture" \
make CC="$(pwd)/scripts/kernel-smoke/gcc" \
     CXX="$(pwd)/scripts/kernel-smoke/g++" \
     NVCC="$(pwd)/scripts/kernel-smoke/nvcc"

python3 scripts/kernel-smoke/cli.py run \
  --capture-dir "$(pwd)/build/kernel-smoke/capture" \
  --out-root build/kernel-smoke \
  --run-id run-001 \
  --target-lib your_target \
  --mode artifact \
  --jobs 4
```

- `--run-id`: run identifier; output goes to `<out-root>/<run-id>/`; required for `--resume`.
- `--target-lib`: logical library name written into manifest/metadata.
- `--mode artifact`: PTX/IR entry discovery + helper-backed instantiated metadata extraction. This is the only supported Phase 1 run mode.
- `--jobs`: parallel workers for capture variant processing.
- `--profile`: write run/stage/variant wall-clock timing to `profile.json`.

Resume:

```bash
python3 scripts/kernel-smoke/cli.py run \
  --capture-dir "$(pwd)/build/kernel-smoke/capture" \
  --out-root build/kernel-smoke \
  --run-id run-001 \
  --target-lib your_target \
  --mode artifact \
  --jobs 4 \
  --resume
```

## How It Works (Pipeline)

1. **capture**
   - symlinked compiler wrappers (`gcc/g++/clang/clang++/nvcc`) -> `compiler-dispatch.sh` -> `rapid-wrap`
   - Records compiler invocations to `commands.jsonl` without changing compile behavior.
2. **discover**
   - `pipeline/capture_db.py` builds variant records.
   - Per variant, discovery replays to variant-level `module.bc`, generates `module.ptx`, extracts PTX `.entry` symbols, then calls a dedicated Clang helper to resolve each entry to instantiated `args / line / qualified_name` metadata.
3. **emit_bc** (streamed per variant)
   - `pipeline/emit_bc.py` + `adapters/replay.py`
    - Replays device compile to module `.bc`, extracts per-kernel `.bc` via `llvm-extract --recursive`, and generates per-kernel `.ptx`.
4. **manifest** (streamed per variant)
   - `pipeline/manifest.py`
   - Writes per-kernel manifest/metadata and output hashes.
5. **summary**
   - `cli.py`
   - Writes run index and reconciliation summary.

## File Map (`scripts/kernel-smoke/`)

- `rapid-wrap`: compiler wrapper executable.
- `compiler-dispatch.sh`: shared shell dispatcher used by symlinked compiler entrypoints (`gcc/g++/clang/clang++/nvcc`).
- `cli.py`: pipeline orchestrator (`run`, `wrap`).
- `print_type_defs.py`: inspect a kernel manifest and print declaration/definition/include hints for argument types.

`adapters/`
- `capture_wrapper.py`: execute+capture command records.
- `replay.py`: device BC replay; preprocess replay helpers are retained for diagnostics, not as the Phase 1 artifact discovery path.
- `toolchain.py`: toolchain version probing for metadata.

`pipeline/`
- `capture_db.py`: load/filter CUDA records and compute `variant_id`.
- `emit_bc.py`: module cache, per-kernel BC/PTX extraction, IR reconciliation helpers.
- `manifest.py`: write manifest/metadata.

`tooling/`
- `clang_entry_metadata.cc`: standalone Clang C++ helper used by artifact mode to map PTX entry symbols to instantiated metadata.

`utils/`
- `demangle.py`: demangle + qualified-name normalization helpers for display/debug.
- `profiling.py`: optional wall-clock profiling helpers (`--profile`).
- `clang_helper.py`: builds/invokes the standalone Clang helper.
- `compile_args.py`: normalizes captured compile commands for the standalone Clang helper.
- `kernel_id.py`: stable artifact kernel id generation.

## Output Layout

Run directory contains two classes of files:

- Contract outputs: stable files consumers should read.
- Internal resume/cache files: implementation details for restart/perf.

```text
build/kernel-smoke/<run_id>/
├── variant_progress.json            # internal: completed variant ids for capture resume
├── discover.json                    # contract: discovered kernels + provenance/capture stats + emit/manifest results
├── index.json                       # contract: kernel index (kernel_id -> artifact dir)
├── profile.json                     # optional contract-ish output: wall-clock timings when --profile is enabled
├── summary.json                     # contract: run-level counts + reconciliation summary
├── modules/<source_variant_hash>.bc # internal: per-variant module cache for extract/reconcile
├── modules/ptx/<source_variant_hash>.ptx # internal: module PTX cache
└── kernels/<kernel_id>/             # contract: per-kernel artifact directory
    ├── kernel.bc                    # contract: extracted single-kernel LLVM bitcode
    ├── kernel.ptx                   # contract: PTX emitted from kernel.bc
    ├── manifest.json                # contract: kernel signature/schema for downstream use
    └── metadata.json                # contract: build status, failure reason, output hashes
```

## Naming in Manifest

- `symbol_name`: exact linkage symbol used for extraction (mangled for C++ kernels).
- `display_name`: human-readable symbol name for display/debug.
- For `extern "C"` kernels, they are usually equal.

In artifact mode, metadata is now resolved per PTX `.entry` symbol:

- `symbol_name` stays the exact PTX/linkage symbol.
- `display_name`, `line`, and `args` come from the helper-backed Clang specialization lookup.

Each `args[]` entry includes `kind` plus the top-level parameter ABI layout
fields `size_bytes` (`sizeof(arg_type)`) and `align_bytes` (`alignof(arg_type)`).
`kind` is the downstream encoding category:

- `pointer`: a real pointer leaf. It must carry `pointer_role` and `pointee_layout`; `pointer_role=payload_buffer` is encoded by downstream `arg-pack-v1` as optional canonical padding plus `[len:u64][bytes + filler : len]`. `pointee_layout` describes one dereference for downstream sizing/mutation policy and future pointer-to-struct flows; it does not make the pointer node an inline aggregate.
- `scalar`: encoded as exactly `size_bytes` little-endian bytes.
- `opaque_val`: pointer-free inline aggregate/value storage, encoded as exactly `size_bytes` object bytes.
- `opaque_with_ptr`: inline aggregate/value storage that contains pointer fields or cannot be proven pointer-free; downstream must open `type_layout` and materialize fields recursively.

`size_bytes` always describes the top-level ABI type size. For pointer args it
is the pointer width, not the pointee buffer length.

In addition to `args[].type`, artifact manifests may include declaration-backed
type metadata under `args[].type_info`:

- `kind`: the C/C++ declaration form when the arg type resolves to a record/enum/etc.: `struct`, `class`, `union`, `enum`, etc.
- optional `qualified_name`, `usr`
- `decl_loc`
- `definition.status` and `definition.loc`

The node-level `type` field is the C/C++ spelling consumed by the current
codegen path. It may preserve typedef aliases such as `IntPtr`, `size_t`, or
`AnonymousAlias`; it is not the semantic source of truth for pointer/scalar/
aggregate classification. Downstream phases must use `kind`, `pointee_layout`,
`type_layout`, `size_bytes`, and `align_bytes` for semantic decisions.
`type_info` exists to preserve declaration identity and source locations, not
to duplicate `type`.

### Manifest Semantic Limits

Current pipeline behavior:

- `args[].kind`, `args[].size_bytes`, and `args[].align_bytes`
  - Artifact mode gets these from the Clang helper using semantic `QualType`.
  - `manifest.py` requires these fields before writing `manifest.json`.
  - `manifest.py` must not infer missing `kind` from source spelling. A missing
    kind is a malformed artifact and must fail fast.
- `args[].type`
  - Preserved for generated C/C++ signatures and storage declarations.
  - Must not be used by Phase 2/3 to infer whether an argument is a pointer,
    scalar, builtin type, or aggregate. For example, `typedef int *IntPtr`
    should be classified by `kind=pointer` and `pointee_layout.kind=scalar`,
    not by checking whether `type` ends with `*`.
- `args[].domain`
  - `manifest.py` preserves a provided domain.
  - The Clang helper can generate enum domains when the type resolves to a complete enum definition.
  - Supported schema domain kinds are `int_range`, `float_range`, `enum`, and `bytes`.
  - Domains describe value space only; ABI/layout details such as enum storage size stay on the owning arg/field (`size_bytes`, `align_bytes`, `type`) rather than inside `domain`.
- `args[].type_layout`
  - `manifest.py` preserves a provided layout.
  - The Clang helper emits a conservative recursive layout for records/unions/constant arrays when available.
  - Layout roots only carry `layout_status` plus `fields` or `element/element_count`; the top-level arg already carries `type/kind/size_bytes/align_bytes`.
  - Nested layout nodes use the same `kind` values as top-level args: `scalar`, `pointer`, `opaque_val`, `opaque_with_ptr`.
  - Pointer args and pointer layout nodes must carry `pointee_layout`. For `T *`, the pointee node describes `T`; when `T` is a record, `pointee_layout.kind` may be `opaque_val` or `opaque_with_ptr`.
  - Nested layout nodes carry `index`, a human-readable recursive manifest index such as `params.inner.buf` or `params.inner[].buf`. Array element templates keep `"name": "$element"` and use `[]` in `index`.
- `materialization_status`, `materialization_reason_codes`, and `materialization_blockers`
  - These fields are optional facts emitted on args, `type_layout` roots, nested layout nodes, and `kernels[].others`.
  - Absence means Phase 1 did not find a materialization blocker for that scope; `materialization_status="unsafe"` means generated field-wise decode may not be able to construct or assign the object safely.
  - Member functions are ignored for layout/materialization unless they imply an object-model blocker such as a virtual method/vptr.
  - Public data fields can be drilled down. Private/protected data fields are recorded as `private_data_field` / `protected_data_field`; virtual/vptr records as `virtual_method_or_vptr`; non-trivially-copyable records as `non_trivially_copyable`; reference fields as `reference_field`; const assignment blockers as `const_assignment_blocker`.
  - `type_shim_status` remains a separate shim-generation fact. Do not use `scoped_type_shim_unsupported` for private fields, virtual methods, reference fields, const fields, or other object materialization blockers.
- `kernels[].constraints`
  - `manifest.py` preserves provided kernel-level constraints.
  - Supported schema constraint kinds are `scalar_le_buffer_len`, `scalar_compare_const`, `scalar_compare_scalar`, and `count_fits_buffer`.
  - Phase 2 currently lowers `scalar_le_buffer_len(unit=bytes)`, `scalar_le_buffer_len(unit=elements)`, `scalar_compare_const`, `scalar_compare_scalar`, and `count_fits_buffer`.
  - Unsupported constraint kinds must fail fast during Phase 2 contract parsing; they must not be silently ignored.
  - Field paths in constraints are relative JSON segment arrays such as `["nested", "choice", "tag"]`; omitting the path references the top-level arg itself.

Not generated automatically:

- Scalar `int_range` / `float_range` domains from arbitrary C++ code.
- Pointer `bytes` domains such as `max_len` from arbitrary C++ code.
- Cross-argument constraints such as `size <= input.len` from names like `size`, `n`, `input`, or `output`.

These are intentionally not inferred from names: they usually require source
annotations, sidecar configuration, a later analysis pass, or an explicit fuzzing
policy. Guessing them in Phase 1 would make the manifest look more precise than
the information actually available from C++ type metadata.

Pointer fields inside `type_layout` are real pointer leaves in an aggregate.
If their role is `payload_buffer`, they use the same input envelope as a
top-level payload pointer. Their `pointee_layout` records the layout behind one
dereference, but does not change the pointer leaf's own envelope segment.
`opaque_with_ptr` is therefore a request for
field-aware materialization, not a copy-safe black box.

The recursive `type_layout` is conservative. It includes a bounded expansion of
records/unions/constant arrays and bitfield metadata (`bit_offset` / `bit_width`)
when available. If larger real-world manifests become a concern, add explicit
generation caps (for example max depth, max fields, or max serialized bytes) and
mark truncated layouts as `partial`.

### Phase 3 Manifest Prework Status

The Phase 1 manifest output is baseline-ready for Phase 3 manifest-driven
encoding:

- `args[].kind`, `args[].size_bytes`, and `args[].align_bytes` are mandatory
  manifest fields.
- Artifact mode derives those fields from Clang semantic types rather than from
  source-text heuristics.
- `manifest.py` preserves optional `domain`, `type_layout`, `type_info`, and
  kernel-level `constraints` when provided by upstream stages.
- Enum `domain` values are generated when complete enum definitions are
  available, and domains describe value space only. ABI details stay on the
  owning arg/field via `type`, `size_bytes`, and `align_bytes`.
- Aggregate parameters classified as `opaque_val` can be consumed using the
  top-level `size_bytes`; aggregate parameters classified as `opaque_with_ptr`
  require recursive `type_layout` materialization before entering the fuzz loop.

Deferred work that should not block Phase 3 v1:

- Inferring scalar range domains, pointer buffer length domains, or cross-arg
  constraints from parameter names.
- Supporting non-payload nested pointer roles such as `derived_pointer` and
  `external_device_pointer`.
- Relying on complete recursive struct semantics for `opaque_val`; pointer-free
  aggregates can still use the baseline `pointer` / `scalar` / `opaque_val`
  encoder.

### Constraint Placement

- Single-argument or single-field value restrictions live in `domain`.
  - Examples: scalar min/max, enum values, pointer buffer min/max length.
- Cross-argument or cross-field relationships live in kernel-level `constraints`.
  - Examples: `size <= input.len`, `scalar_a == scalar_b`, `count * elem_size <= buffer.len`.
  - Payload-buffer length equality such as `input.len == output.len` is not a
    v1 Phase 2 constraint. Add a future generic payload-length comparison only
    when payload length expressions can appear on both sides of a predicate.

This split keeps ownership clear: `domain` can be evaluated from one value,
while `constraints` are evaluated after all args have been decoded/normalized.

### Inspecting Type Definitions

Use `print_type_defs.py` to inspect `manifest.json` argument types and recover the best-known
definition/include hints:

```bash
python3 scripts/kernel-smoke/print_type_defs.py \
  build/kernel-smoke/<run_id>/kernels/<kernel_id>/manifest.json

python3 scripts/kernel-smoke/print_type_defs.py --json \
  build/kernel-smoke/<run_id>/kernels/<kernel_id>/manifest.json
```

The text output prints, when available:

- the resolved declaration kind/name
- the definition file:line:column
- the include root inferred from capture-time `-I` / `-isystem`
- a suggested `#include <...>` or `#include "..."`

## Notes

- Capture mode is recommended for real projects.
- `nvcc` has no AST dump API; AST discovery in nvcc builds uses clang replay.
- In capture mode, processing is stream-by-variant by default; there is no non-stream capture path.
- Artifact mode demangling requires Python package `cxxfilt` (install via `scripts/kernel-smoke/requirements.txt`).
- Artifact mode also requires the standalone Clang helper build prerequisites. Helper discovery now prefers `KSMOKE_CLANG_HELPER_BIN`, then `clang++`/`llvm-config` from `PATH` (with a fallback scan of common LLVM install roots).
- You can either call `rapid-wrap <real-compiler> ...` directly or point build systems at the symlinked wrappers in `scripts/kernel-smoke/` (for example `scripts/kernel-smoke/nvcc`).
