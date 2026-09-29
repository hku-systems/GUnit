# Kernel Rewrite Pipeline

`scripts/kernel-rewrite/` 对 `scripts/kernel-smoke/`（Phase 1）产物做后处理。

当前最小闭环能力：

- 消费 Phase 1 的 `run_dir`（`index.json` + per-kernel `manifest/metadata/bc`）；
- 为 built kernel 构建 rewrite plan；
- 执行 plan-driven `.ll` rewrite：从 `kernel.bc` 反汇编出 `.ll`，按函数级执行 `__global__ -> __device__` 改写，生成单一 rewritten `entry_symbol`，再回组装成 `kernel.device.bc`；
- 为当前支持子集生成可编译的 `phase2/gen/fuzzer_decode.v1.cuh`、`phase2/gen/fuzzer_invoke.v1.cuh`；
- 产出 `phase2/kernel.device.bc`、`phase2/build_spec.json`、`phase2/metadata.phase2.json`；
- 产出 run 级 `rewrite_summary.json`。

当前目录结构已按功能拆分：

- `contracts/`：统一 kernel contract 与 support gate
- `rewrite/`：rewrite plan + rewrite-exec
- `codegen/`：decode / invoke header 生成
- `common.py`：共享 IO / hash / artifact helper
- `phase2.py`：阶段编排入口

## Quick Run

```bash
python3 scripts/kernel-rewrite/cli.py \
  --run-dir build/kernel-smoke/<run_id>
```

可选：

- `--strict`：只要任一 kernel rewrite 失败，命令返回非 0。

## Input Contract

`--run-dir` 目录下需要至少包含：

- `index.json`
- `kernels/<kernel_id>/kernel.bc`
- `kernels/<kernel_id>/kernel.ptx`
- `kernels/<kernel_id>/manifest.json`
- `kernels/<kernel_id>/metadata.json`

仅当 `metadata.json` 中 `build_status == "built"` 时，kernel 执行 rewrite 流程。

## Output Layout

```text
build/kernel-smoke/<run_id>/
├── rewrite_summary.json
└── kernels/<kernel_id>/
    └── phase2/
        ├── kernel.device.bc
        ├── build_spec.json
        ├── gen/
        │   ├── fuzzer_decode.v1.cuh
        │   └── fuzzer_invoke.v1.cuh
        └── metadata.phase2.json
```

## Current Rewrite Strategy

- 当前 `rewrite_exec` 为 plan-driven `.ll` workflow：`kernel.bc -> llvm-dis -> 函数级 global->device rewrite -> llvm-as -> kernel.device.bc`。
- 当前 rewrite 至少为每个目标 kernel 产出：
  - `entry_symbol`：由原始 `__global__` kernel 改写得到的 rewritten device 符号，由 `fuzzer_invoke_v1` 调用；
  - `build_spec.json`：记录生成头/bitcode 的相对路径，以及 `shim_header`；
  - `metadata.phase2.json`：记录 `input_symbol -> entry_symbol` 的最小结果，以及 `phase2_status` / `failure_reason` / `failure_detail`；当可用时还会记录 `failure_context`，用于解释 unsupported 的具体触发对象。
- 当前 `codegen` 已支持基础标量、`payload_buffer` pointer、`opaque_val`
  对象拷贝，以及带完整 `type_layout` 的 `opaque_with_ptr` 递归
  field-aware decode/invoke。该支持边界和 `manifest.example.md` 中的
  supported projection 对齐。
- 当前 full manifest 仍可能因为 unsupported layout/type facts 失败，例如
  pointer-bearing union、或 Phase 1 暂不能为 scoped/namespace 类型生成
  `type_shim.v1.cuh`。这类失败应以具体 `failure_detail` 落盘，而不是
  生成误导性的 rewrite/codegen 产物；对 `scoped_type_shim_unsupported`
  这类粗粒度 reason，还应在 `failure_context` 中保留
  `type_shim_missing_dependencies` / `type_shim_system_headers` 等真因。
- 当前实现是单-kernel PoC，不代表复杂模板/kernel metadata/所有
  aggregate-by-value 场景已经全部支持。

## Target Rewrite Architecture

- 正式 rewrite backend 应以 `kernel.bc` 为输入真源；如果实现上需要 `bc -> ll -> bc` roundtrip，也应把 `.ll` 视为 LLVM IR 的可逆中间表示，而不是把 raw 文本替换当成长期编辑面。
- `phase2/kernel.device.bc` 的当前目标是至少包含一个 rewritten `entry_symbol`，它来自原始 `__global__` kernel 的 `__device__` 化改写。
- `gen/fuzzer_invoke.v1.cuh` 的合同是“调用 rewritten `entry_symbol`”。
- 后续如果 Phase 4/5 需要额外 glue 层，可以再引入 thunk；当前阶段先保持一层，避免过度设计。

## Current Support Gate

- Phase 2 当前会先构建统一的 kernel contract，并对参数类型做支持性判定。
- 当前优先支持：基础标量参数、`pointer_role=payload_buffer` pointer
  参数、`opaque_val` by-value 对象参数，以及完整 layout 下可递归下钻的
  `opaque_with_ptr` 参数。
- 当前明确不支持：reference、callable/function type、缺失 shim 的非
  builtin 参数、incomplete/opaque `opaque_with_ptr` layout、unsupported
  pointer role、pointer-bearing union。这类 kernel 会在 support gate 失败，
  并以 `phase2_input_invalid` 结束，避免生成误导性的 rewrite/codegen 产物。
- 当前 fixture 回归中，这条最小 lane 已能稳定产出 built/failed 的真实区分：支持子集进入 rewrite+codegen，不支持子集显式失败。

## Runtime Example

仓库内已有一个真实的运行级 smoke 例子：

- `tests/kernel_pipeline_rewrite/drivers/test_phase2_runtime_e2e.py`
- `tests/kernel_pipeline_rewrite/fixtures/`
- `tests/kernel_pipeline_rewrite/fixtures/cases/`

该测试会：

- 遍历 `tests/kernel_pipeline_rewrite/fixtures/cases/` 下的 checked-in runtime cases；
- 编译一个完整 CUDA wrapper 程序，其中 `__global__ run_generated(...)` 调用生成的 `fuzzer_decode_v1` / `fuzzer_invoke_v1`；
- 将 wrapper 的 device bitcode 与 `phase2/kernel.device.bc` 通过 `llvm-link` 合并；
- 用 `llc` 生成 PTX，再通过 CUDA driver API 加载并运行；
- 读取每个 case 自带的 `input_payload.bin` / `expected_payload.bin`，检查输出 payload 是否完全匹配预期。

这条测试用于证明当前支持子集已经具备“生成头 + 改写后 device 模块 + 完整程序运行”的最小闭环，而不是只停留在文件存在或单文件编译级别。当前实现下，`fuzzer_invoke_v1` 直接调用 rewritten `entry_symbol`。

同一条链路现在由测试侧的共享 helper 执行：

```bash
python3 tests/kernel_pipeline_rewrite/drivers/run_runtime_cases.py \
  --run-dir build/kernel-smoke/<run_id>
```

执行后会在：

- `<run_dir>/phase2_runtime/<kernel_id>/`

下保留完整中间产物与结果，包括：

- `wrapper.cu`
- `wrapper.bc`
- `linked.bc`
- `linked.ptx`
- `host.cpp`
- `host.out`
- `runtime_smoke.json`
