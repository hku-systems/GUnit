# RAPID End-to-End Usage

本文档说明如何把一个已有 CUDA 项目转换成 RAPID/LibAFL 可 fuzz 的目标，并进行端到端测试。

核心原则：正式流程必须基于真实项目构建命令的 capture/replay。不要把 `rg "__global__"` 或单文件裸编译当作 kernel discovery 的真源。很多真实项目通过宏、模板实例化、CMake/Python 生成参数、nvcc response file、架构 flag 和 include path 生成 kernel；这些信息只有真实 build command 才完整。

## 0. 总览

完整链路如下：

```text
CUDA project build
  -> compiler wrapper capture
  -> commands.jsonl
  -> Phase 1 kernel-smoke artifact mode
  -> per-kernel kernel.bc / kernel.ptx / manifest.json / metadata.json
  -> Phase 2 kernel rewrite + decode/invoke generation
  -> origin / rapid / rapid2 backend .so
  -> cuda-fuzzer fuzzer / fuzzer_async
```

每一层的输入和输出：

```text
Capture input:
  real Make/CMake/Ninja/Python-extension CUDA build

Capture output:
  <capture-dir>/commands.jsonl

Phase 1 input:
  commands.jsonl

Phase 1 output:
  <phase1-out>/<run-id>/index.json
  <phase1-out>/<run-id>/summary.json
  <phase1-out>/<run-id>/discover.json
  <phase1-out>/<run-id>/kernels/<kernel-id>/kernel.bc
  <phase1-out>/<run-id>/kernels/<kernel-id>/kernel.ptx
  <phase1-out>/<run-id>/kernels/<kernel-id>/manifest.json
  <phase1-out>/<run-id>/kernels/<kernel-id>/metadata.json

Phase 2 input:
  Phase 1 run dir

Phase 2 output:
  <run-dir>/rewrite_summary.json
  <run-dir>/kernels/<kernel-id>/phase2/kernel.device.bc
  <run-dir>/kernels/<kernel-id>/phase2/build_spec.json
  <run-dir>/kernels/<kernel-id>/phase2/gen/rapid_target_layout.v1.h
  <run-dir>/kernels/<kernel-id>/phase2/gen/fuzzer_decode.v1.cuh
  <run-dir>/kernels/<kernel-id>/phase2/gen/fuzzer_invoke.v1.cuh
  <run-dir>/kernels/<kernel-id>/phase2/metadata.phase2.json

Backend output:
  libphase2_origin_target.so
  libphase2_rapid_target.so
  librapid2_target.so
```

## 1. 准备环境

从仓库根目录开始：

```bash
cd /path/to/rapid
```

优先使用 repo-local Python 环境：

```bash
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  python3 -m venv .venv
  PY=.venv/bin/python
  "$PY" -m pip install --upgrade pip setuptools wheel
fi

"$PY" -m pip install -r scripts/kernel-smoke/requirements.txt
chmod +x scripts/kernel-smoke/rapid-wrap scripts/kernel-smoke/compiler-dispatch.sh
```

确认 CUDA/GPU 对当前执行环境可见：

```bash
nvidia-smi
```

如果在 sandbox 中看不到 GPU，但宿主机可见，需要在能访问 NVIDIA driver 的执行路径中运行 GPU 相关阶段。Phase 1/Phase 2 的很多静态步骤不一定需要运行 kernel，但 backend build 和 fuzzer runtime 需要 CUDA 工具链，fuzzer 执行还需要可用 GPU。

常用环境变量：

```bash
export CUDA_PATH=/usr/local/cuda
export KSMOKE_CUDA_ARCH=sm_86
```

`KSMOKE_CUDA_ARCH` 可按机器调整，例如 `sm_80`、`sm_86`、`sm_90`。

## 2. Capture 真实 CUDA build commands

### Make 项目

让项目原本的 build 系统使用 RAPID wrapper。wrapper 会原样执行编译命令，同时把命令追加写入 `commands.jsonl`。

```bash
PROJECT_NAME=my_cuda_project
CAPTURE_DIR="$(pwd)/build/e2e/${PROJECT_NAME}/capture"
PROJECT_DIR=/path/to/cuda/project

rm -rf "build/e2e/${PROJECT_NAME}"
mkdir -p "$CAPTURE_DIR"

RAPID_CAPTURE_DIR="$CAPTURE_DIR" \
make -C "$PROJECT_DIR" \
  CC="$(pwd)/scripts/kernel-smoke/gcc" \
  CXX="$(pwd)/scripts/kernel-smoke/g++" \
  NVCC="$(pwd)/scripts/kernel-smoke/nvcc" \
  -j
```

### CMake/Ninja 项目

CMake 项目通常需要在 configure 阶段指定 wrapped compiler，并在 build 阶段继续传入同一个 `RAPID_CAPTURE_DIR`。

```bash
PROJECT_NAME=my_cuda_project
PROJECT_DIR=/path/to/cuda/project
BUILD_DIR="$(pwd)/build/e2e/${PROJECT_NAME}/project-build"
CAPTURE_DIR="$(pwd)/build/e2e/${PROJECT_NAME}/capture"

rm -rf "$(pwd)/build/e2e/${PROJECT_NAME}"
mkdir -p "$CAPTURE_DIR"

RAPID_CAPTURE_DIR="$CAPTURE_DIR" \
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(pwd)/scripts/kernel-smoke/gcc" \
  -DCMAKE_CXX_COMPILER="$(pwd)/scripts/kernel-smoke/g++" \
  -DCMAKE_CUDA_COMPILER="$(pwd)/scripts/kernel-smoke/nvcc"

RAPID_CAPTURE_DIR="$CAPTURE_DIR" \
cmake --build "$BUILD_DIR" -j
```

如果项目不使用 CMake 的 CUDA language，而是通过 clang++ 编译 `.cu`，仍然要确保它调用的是 `scripts/kernel-smoke/clang++` 或 `scripts/kernel-smoke/g++` wrapper。

### Python extension 项目

FlashAttention 一类项目通常通过 Python/PyTorch extension 构造 nvcc/clang command。原则仍然相同：把真实 extension build 放在 wrapper 环境下。

```bash
PROJECT_NAME=my_python_cuda_ext
PROJECT_DIR=/path/to/cuda/project
CAPTURE_DIR="$(pwd)/build/e2e/${PROJECT_NAME}/capture"

rm -rf "$(pwd)/build/e2e/${PROJECT_NAME}"
mkdir -p "$CAPTURE_DIR"

RAPID_CAPTURE_DIR="$CAPTURE_DIR" \
VIRTUAL_ENV="$(pwd)/.venv" \
PATH="$(pwd)/scripts/kernel-smoke:$(pwd)/.venv/bin:$PATH" \
PYTHON="$(pwd)/.venv/bin/python" \
CC="$(pwd)/scripts/kernel-smoke/gcc" \
CXX="$(pwd)/scripts/kernel-smoke/g++" \
NVCC="$(pwd)/scripts/kernel-smoke/nvcc" \
"$PY" setup.py build_ext --inplace
```

不同项目可能需要额外依赖，例如 PyTorch、CUTLASS submodule、TensorRT headers。不要手写近似 compile command；先让项目自己的 build 成功，再 capture 它的真实 CUDA 编译命令。

### 验证 capture

```bash
test -s "$CAPTURE_DIR/commands.jsonl"
wc -l "$CAPTURE_DIR/commands.jsonl"
head -3 "$CAPTURE_DIR/commands.jsonl"
```

`commands.jsonl` 中每行包含：

```json
{
  "argv": ["real-compiler", "..."],
  "compiler": "real-compiler",
  "cwd": "/build/working/directory",
  "env_subset": {"CC": "...", "CXX": "...", "NVCC": "..."},
  "exit_code": 0
}
```

`exit_code != 0` 的记录也会被保留；Phase 1 会优先选择同一 variant 中成功的记录。

## 3. Phase 1: 从 capture 生成 kernel artifacts

运行 kernel-smoke artifact mode：

```bash
PROJECT_NAME=my_cuda_project
CAPTURE_DIR="$(pwd)/build/e2e/${PROJECT_NAME}/capture"
PHASE1_OUT="$(pwd)/build/e2e/${PROJECT_NAME}/phase1"
RUN_ID="${PROJECT_NAME}"

"$PY" scripts/kernel-smoke/cli.py run \
  --capture-dir "$CAPTURE_DIR" \
  --out-root "$PHASE1_OUT" \
  --run-id "$RUN_ID" \
  --target-lib "$PROJECT_NAME" \
  --mode artifact \
  --jobs 4 \
  --profile
```

正式 Phase 1 的 kernel discovery 不是源码 regex：

- `pipeline/capture_db.py` 从 `commands.jsonl` 中筛选 replayable CUDA compile variants。
- `adapters/replay.py` 将 captured clang/nvcc command replay 成 device-only `module.bc`。
- Phase 1 生成 `module.ptx`，扫描 PTX `.entry` symbols，作为 kernel 存在性的真源。
- Clang helper 只负责把这些 entry 映射到实例化后的参数、行号和 display name。

检查 Phase 1 输出：

```bash
export RUN_DIR="$PHASE1_OUT/$RUN_ID"

test -f "$RUN_DIR/index.json"
test -f "$RUN_DIR/summary.json"
test -f "$RUN_DIR/discover.json"

python3 - <<'PY'
import json, os
run_dir = os.environ["RUN_DIR"]
summary = json.load(open(os.path.join(run_dir, "summary.json")))
index = json.load(open(os.path.join(run_dir, "index.json")))
print("summary counts:", summary.get("counts", {}))
print("kernel entries:", len(index.get("kernels", [])))
PY
```

Per-kernel artifacts live under:

```text
$RUN_DIR/kernels/<kernel-id>/
```

Each built kernel should have:

```text
kernel.bc
kernel.ptx
manifest.json
metadata.json
```

## 4. Phase 2: rewrite kernel and generate decode/invoke

Run Phase 2 on the Phase 1 run dir:

```bash
"$PY" scripts/kernel-rewrite/cli.py \
  --run-dir "$RUN_DIR"
```

Useful strict mode:

```bash
"$PY" scripts/kernel-rewrite/cli.py \
  --run-dir "$RUN_DIR" \
  --strict
```

Without `--strict`, unsupported kernels are expected to be recorded in metadata instead of aborting the whole run. Use this mode for large third-party projects so one unsupported kernel does not stop the sweep.

Inspect Phase 2 results:

```bash
test -f "$RUN_DIR/rewrite_summary.json"

python3 - <<'PY'
import json, os
run_dir = os.environ["RUN_DIR"]
summary = json.load(open(os.path.join(run_dir, "rewrite_summary.json")))
print(json.dumps(summary.get("counts", {}), indent=2, sort_keys=True))
PY
```

Find built kernels:

```bash
find "$RUN_DIR/kernels" -path '*/phase2/metadata.phase2.json' -print \
  | while read -r meta; do
      python3 - "$meta" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
if data.get("phase2_status") == "built":
    print(path)
PY
    done
```

Find unsupported kernels and reasons:

```bash
find "$RUN_DIR/kernels" -path '*/phase2/metadata.phase2.json' -print \
  | while read -r meta; do
      python3 - "$meta" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
if data.get("phase2_status") != "built":
    print(path, data.get("failure_reason"), data.get("failure_detail"))
PY
    done
```

Current expected unsupported cases include:

- `union_not_supported`
- `pointer_role_not_supported`
- `opaque_with_ptr_layout_incomplete`
- `scalar_size_unsupported`
- `scoped_type_shim_unsupported`
- unsafe materialization facts such as private fields, virtual/vptr records, references, non-trivially-copyable records, and const assignment blockers

These are support-boundary rejections, not runner crashes, as long as they are written to `metadata.phase2.json` and `rewrite_summary.json`.

## 5. Build backend shared libraries

Each built kernel has a `phase2/` directory. Build one or more backend types from that directory.
The default `--feedback-instrumentation enabled` mode runs the LLVM feedback
instrumentation pass after `llvm-link` and lowers `instrumented.bc` to PTX.
Use `--feedback-instrumentation disabled` for same-kernel overhead baselines on
any backend; ABI validation still runs, but the builder lowers `linked.bc`
directly and does not create `feedback_metadata.json`.

```bash
KERNEL_PHASE2_DIR="$RUN_DIR/kernels/<kernel-id>/phase2"
BACKENDS_DIR="$KERNEL_PHASE2_DIR/backends"

"$PY" cuda-kernel/origin/build.py \
  --phase2-dir "$KERNEL_PHASE2_DIR" \
  --out-dir "$BACKENDS_DIR/origin" \
  --cuda-path "${CUDA_PATH:-/usr/local/cuda}" \
  --cuda-arch "${KSMOKE_CUDA_ARCH:-sm_86}"

"$PY" cuda-kernel/rapid/build.py \
  --phase2-dir "$KERNEL_PHASE2_DIR" \
  --out-dir "$BACKENDS_DIR/rapid" \
  --cuda-path "${CUDA_PATH:-/usr/local/cuda}" \
  --cuda-arch "${KSMOKE_CUDA_ARCH:-sm_86}"

"$PY" cuda-kernel/rapid2/build.py \
  --phase2-dir "$KERNEL_PHASE2_DIR" \
  --out-dir "$BACKENDS_DIR/rapid2" \
  --cuda-path "${CUDA_PATH:-/usr/local/cuda}" \
  --cuda-arch "${KSMOKE_CUDA_ARCH:-sm_86}"
```

Build an uninstrumented RAPID2 control for performance comparison:

```bash
"$PY" cuda-kernel/rapid2/build.py \
  --phase2-dir "$KERNEL_PHASE2_DIR" \
  --out-dir "$BACKENDS_DIR/rapid2-uninstrumented" \
  --cuda-path "${CUDA_PATH:-/usr/local/cuda}" \
  --cuda-arch "${KSMOKE_CUDA_ARCH:-sm_86}" \
  --feedback-instrumentation disabled
```

Expected libraries:

```text
$BACKENDS_DIR/origin/libphase2_origin_target.so
$BACKENDS_DIR/rapid/libphase2_rapid_target.so
$BACKENDS_DIR/rapid2/librapid2_target.so
```

Batch helper for Phase3 smoke can run built kernels that already have backend
artifacts. It can build the Rust fuzzer binary with `--build-fuzzer`, but it
does not build `origin` / `rapid` / `rapid2` backend `.so` files; build those
first with the backend builders above.

```bash
"$PY" scripts/phase3_smoke.py \
  --run-dir "$RUN_DIR" \
  --backend rapid2 \
  --profile release \
  --build-fuzzer \
  --runs 100
```

`phase3_smoke.py` intentionally runs fixed-payload smoke mode. For each selected
kernel it calls the matching fuzzer binary with:

```text
--manifest <same-kernel manifest.json> --no-mutate --runs <N>
```

This is a single-process bounded path used to validate generated artifacts and
runtime integration, not a full broker/client fuzzing campaign.

Use `--dump-seeds` to validate manifest-driven seed generation without loading backend `.so` files:

```bash
"$PY" scripts/phase3_smoke.py \
  --run-dir "$RUN_DIR" \
  --backend rapid2 \
  --profile release \
  --build-fuzzer \
  --dump-seeds
```

## 6. Build fuzzers

Build Rust fuzzer binaries:

```bash
cargo build --manifest-path cuda-fuzzer/Cargo.toml --release
```

Expected binaries:

```text
cuda-fuzzer/target/release/fuzzer
cuda-fuzzer/target/release/fuzzer_async
```

`fuzzer` is used by both `origin` and `rapid`; `fuzzer_async` is paired with
`librapid2_target.so`. Their feedback semantics are:

```text
origin  -> exact input/run/feedback lockstep
rapid   -> enqueue-and-return with stable cumulative completed feedback
rapid2  -> asynchronous completion with exact task-ID feedback
```

All three backends compile the same generated
`phase2/gen/rapid_target_layout.v1.h` for the selected kernel. It fixes the
exact feedback payload-slot capacity in both host and device code. At runtime,
each task copies one internal transport envelope containing the current
`RapidVConfig`, payload size, and the unchanged manifest-driven payload bytes.
The backend-owned feedback bitset address is initialized once and is not part
of fuzz input.

RAPID intentionally allows a completed older task's novelty to be credited to
the input currently evaluated by `StdFuzzer`. Its worker never writes the
observer maps directly, and cumulative OR ensures completed CFG,
memory/index, and thread-activity novelty cannot disappear.

## 7. Run fuzzer on one backend

Pick the artifact manifest that belongs to the same kernel as the backend `.so`.

```bash
MANIFEST="$RUN_DIR/kernels/<kernel-id>/manifest.json"
RAPID2_SO="$RUN_DIR/kernels/<kernel-id>/phase2/backends/rapid2/librapid2_target.so"
ORIGIN_SO="$RUN_DIR/kernels/<kernel-id>/phase2/backends/origin/libphase2_origin_target.so"
RAPID_SO="$RUN_DIR/kernels/<kernel-id>/phase2/backends/rapid/libphase2_rapid_target.so"
```

Important invariant: the fuzzer must receive the `manifest.json` from the same
artifact kernel that produced the backend `.so`. Do not rely on a repository-root
manifest or another kernel's manifest.

### Bounded runtime smoke

Use bounded fixed-payload smoke first. This is the recommended quick check for
generated `.so` files because it is single-process and deterministic enough for
batch e2e validation.

For RAPID2:

```bash
RUST_LOG=info \
cuda-fuzzer/target/release/fuzzer_async \
  "$RAPID2_SO" \
  --manifest "$MANIFEST" \
  --no-mutate \
  --runs 100
```

For `origin`:

```bash
RUST_LOG=info \
cuda-fuzzer/target/release/fuzzer \
  "$ORIGIN_SO" \
  --manifest "$MANIFEST" \
  --no-mutate \
  --runs 100
```

For `rapid`:

```bash
RUST_LOG=info \
cuda-fuzzer/target/release/fuzzer \
  "$RAPID_SO" \
  --manifest "$MANIFEST" \
  --no-mutate \
  --runs 100
```

In this mode the fuzzer still enters the LibAFL loop and executes the generated
backend `.so`. `--no-mutate` does not skip the stage; it switches
`RapidInputMutator` into fixed mode. Each stage iteration rewrites the input to:

```text
normalize_rapid_input_v1(default_seed_rapid_input_v1())
```

The expected stats therefore show seed generation and normalization, but no
random structure mutation:

```text
seed_generation_count > 0
normalize_calls > 0
mutation_calls = 0
```

Use both `--no-mutate` and `--runs N` for this single-process bounded smoke path.
`--no-mutate` without `--runs` is not the same workflow.

### Full fuzzing

For real fuzzing, omit `--no-mutate`. Each mutation stage chooses one of five
equally weighted plans: payload structure, payload havoc, VConfig only, structure
plus VConfig, or havoc plus VConfig. The payload plans cover all mutable kernel
argument leaves, including scalars, pointer buffers, opaque values, and nested
fields. Havoc stacks one to eight value-level operations and can resize variable
buffers in `elem_size_bytes` units. With VConfig mutation disabled, only the two
payload plans are eligible. Every plan ends with canonical envelope normalization,
which repairs domains, cross-argument constraints, launch-dependent constraints,
and `payload_size` before submission.

The normal fuzzing mode uses LibAFL's broker/client restarting-manager
architecture. Start the same command twice: the first process becomes the broker,
and the second process becomes the fuzzing client. A broker-only run will show
heartbeats with `clients: 0` and will not execute the target until a client joins.

Example for RAPID2 full fuzzing:

```bash
# terminal 1: broker
RUST_LOG=info \
cuda-fuzzer/target/release/fuzzer_async \
  "$RAPID2_SO" \
  --manifest "$MANIFEST" \
  --runs 100000
```

```bash
# terminal 2: client
RUST_LOG=info \
cuda-fuzzer/target/release/fuzzer_async \
  "$RAPID2_SO" \
  --manifest "$MANIFEST" \
  --runs 100000
```

Example for `origin` full fuzzing:

```bash
# terminal 1: broker
RUST_LOG=info \
cuda-fuzzer/target/release/fuzzer \
  "$ORIGIN_SO" \
  --manifest "$MANIFEST" \
  --runs 100000
```

```bash
# terminal 2: client
RUST_LOG=info \
cuda-fuzzer/target/release/fuzzer \
  "$ORIGIN_SO" \
  --manifest "$MANIFEST" \
  --runs 100000
```

Use the `rapid` backend the same way as `origin`, replacing `$ORIGIN_SO` with
`$RAPID_SO`.

### Run all three broker/client campaigns

`tests/feedback_e2e/run_campaign.py` launches the required two processes for
each backend, stores isolated logs and corpus/crash directories, and writes a
machine-readable report. It intentionally omits `--no-mutate`.

```bash
.venv/bin/python tests/feedback_e2e/run_campaign.py \
  --run-dir "$RUN_DIR" \
  --backends origin,rapid,rapid2 \
  --runs 100000 \
  --observe-seconds 30 \
  --gpu-devices 0,1,2
```

When a run directory contains multiple built kernels, select one explicitly:

```bash
.venv/bin/python tests/feedback_e2e/run_campaign.py \
  --run-dir "$RUN_DIR" \
  --kernel-id '<kernel-id-from-index.json>' \
  --backends origin,rapid,rapid2 \
  --runs 100000 --observe-seconds 30 --gpu-devices 0,1,2
```

The report requires a joined client, positive executions, and positive
`cfg_sites`, `simt_memcov_bits`, and `logical_thread_bits` counters for every
backend. See `docs/feedback-validation.md` for a fresh builtin run and a
supported TensorRT `clipKernel` third-party run.

For `rapid`, `executions` is the number of returned submissions. Read
`rapid_completed` from the report for completed GPU work and use that count for
throughput comparisons. A time-bounded restarting-manager campaign may end on
a periodic snapshot with nonzero `rapid_pending`; the runner selects all RAPID
counters from the same highest-submission snapshot. RAPID's `TimeObserver`
measures enqueue/backpressure latency rather than task execution time.

### RAPID2 instrumentation overhead baseline

After building both RAPID2 variants from the same `phase2/` directory, run a
same-kernel broker/client A/B benchmark:

```bash
.venv/bin/python tests/feedback_e2e/run_rapid2_benchmark.py \
  --run-dir "$RUN_DIR" \
  --kernel-id '<kernel-id-from-index.json>' \
  --instrumented-build "$BACKENDS_DIR/rapid2/backend_build.json" \
  --uninstrumented-build "$BACKENDS_DIR/rapid2-uninstrumented/backend_build.json" \
  --runs 100000 \
  --observe-seconds 30 \
  --gpu-device 0
```

The runner rejects mismatched `kernel_id`, display name, or `phase2_dir`, and it
requires both campaigns to have a joined client and positive executions. Its
`report.json` records executions, observation seconds, executions/sec, and
`slowdown = 1 - instrumented_rate / uninstrumented_rate`.

## 8. Recommended third-party project sweep

For a large repository with multiple CUDA subprojects, run per project rather than one global directory at first. This isolates dependencies, flags, and build failures.

Suggested layout:

```text
build/e2e/
  tensorrt/
    capture/
    project-build/
    phase1/<run-id>/
  cutlass/
    capture/
    project-build/
    phase1/<run-id>/
  flash_attention/
    capture/
    project-build/
    phase1/<run-id>/
  llama_cpp/
    capture/
    project-build/
    phase1/<run-id>/
```

The denominator should be PTX entry kernels discovered from replayed artifacts, not source files containing `__global__`.

## 9. What not to do

Do not use this as the official third-party e2e discovery path:

```bash
rg -l --glob '*.cu' '__global__' third_party
```

That command is only a rough source inventory. It misses macro-generated kernels and does not preserve real compile flags, include paths, template instantiations, generated headers, response files, or project-specific arch settings.

Do not manually invent compile commands for a real project unless you are creating a tiny repro. A synthetic command like:

```bash
clang++ -x cuda file.cu -I some/include --cuda-gpu-arch=sm_86
```

can be useful for smoke debugging, but it is not equivalent to the project's build system.

`test_e2e_workspace/run_e2e_test.py` currently follows this synthetic single-file style for part of its Phase 1 setup. Treat it as a debugging helper unless it is extended to consume a real `commands.jsonl` or existing Phase 1 `run_dir`.

## 10. Troubleshooting

### No `commands.jsonl`

The build did not call the wrapper. Check:

- `RAPID_CAPTURE_DIR` is set in the same environment as the build command.
- `CC`, `CXX`, and `NVCC` point to `scripts/kernel-smoke/*` wrappers.
- For CMake, compiler variables were set during configure, not only during build.
- For Python extensions, `PATH` and `NVCC` point to the wrappers.

### `commands.jsonl` exists but Phase 1 finds zero variants

Check whether records are replayable CUDA compiles:

- Compiler basename should contain `nvcc` or `clang`.
- `argv` should contain a `.cu` source or `-x cuda`.
- Prefer records with `exit_code=0`.

### Phase 1 replay fails

Inspect:

```bash
$RUN_DIR/discover.json
$RUN_DIR/summary.json
```

Common causes:

- Missing project include paths because capture did not use the real build.
- Missing generated headers because the project build was not completed in the same build directory.
- Unsupported or incompatible CUDA toolkit path.
- nvcc response files or arch flags not visible from the captured cwd.

### Phase 2 rejects kernels

This is expected for unsupported signatures. Inspect:

```bash
$RUN_DIR/rewrite_summary.json
$RUN_DIR/kernels/<kernel-id>/phase2/metadata.phase2.json
```

Unsupported kernels should be excluded from backend/fuzzer runs until support is implemented. Do not force them through Phase 3.

### Backend build fails

Check:

```bash
$KERNEL_PHASE2_DIR/build_spec.json
$KERNEL_PHASE2_DIR/metadata.phase2.json
$BACKENDS_DIR/<backend>/backend_build.json
```

Make sure the kernel has `phase2_status="built"` before invoking backend builders.

### Fuzzer cannot load CUDA or reports no device

Check GPU visibility in the same execution environment used by the fuzzer:

```bash
nvidia-smi
```

If sandboxed commands cannot see the NVIDIA driver, run fuzzer/backend validation through an execution path with GPU access.

## 11. Minimal smoke on the in-repository builtin kernel

The repo includes a known-good builtin pipeline driver:

```bash
"$PY" cuda-kernel/builtin_phase_pipeline.py \
  --out-root /tmp/rapid-builtin-phase \
  --run-id builtin-kernel \
  --cuda-arch "${KSMOKE_CUDA_ARCH:-sm_86}" \
  --cuda-path "${CUDA_PATH:-/usr/local/cuda}"
```

It performs capture, Phase 1, Phase 2, and builds all three backends. Use this before large third-party sweeps when validating a fresh machine or toolchain.
