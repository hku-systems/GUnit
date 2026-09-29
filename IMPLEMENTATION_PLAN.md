# RAPID 全流程实现计划（Kernel 提取 + 统一 ABI + Feedback + Sanitizer + VCONFIG）

本文档把当前讨论的方案固化为可执行路线图，目标是把“给定大型 CUDA 库 -> 自动提取 kernel -> 生成 harness `.so` -> origin/rapid2 fuzz -> triage 复现”做成完整流水线。

## 1. 目标与原则

- 统一输入 ABI：fuzzer 只输入 `uint8_t* data, size_t size`。
- 多 kernel 兼容：支持参数类型、参数个数、名称各不相同的 `__global__` kernel。
- 双后端一致语义：origin（同步）与 rapid2（异步）使用同一输入封包与同一 decode 规则。
- 反馈可扩展：自动收集 `edge/basic-block bitmap + SIMT memory feature bitmap`；SIMT MemCov 将直接替换旧 coarse memory/index 语义，详见 [`docs/feedback.md`](docs/feedback.md)。
- sanitizer 内嵌执行：sanitizer 插桩进入 device 执行路径并按 task 归因，详见 [`docs/sanitizer.md`](docs/sanitizer.md)。

## 2. 总体架构

### 2.1 三层架构

- 发现层（Compile-time Discovery）
  - 在 LLVM/NVVM 阶段自动枚举 kernel，输出 `KernelManifest`。
- ABI 层（Unified Input Envelope）
  - 将 `bytes` 解码为“某个 kernel 的一次合法调用”。
- 执行层（Execution Backends）
  - origin 同步执行；rapid2 异步提交/轮询执行。

### 2.2 反馈与诊断运行模型

- Fast path：长期 fuzz 默认收集 edge/basic-block feedback 与 SIMT memory feature feedback。
- Device sanitizer path：sanitizer 作为目标 kernel/device IR 插桩进入 persistent execution，每个 task 产出结构化诊断。
- 外部工具（如 `compute-sanitizer`）只作为后续校准/复核工具，不是 RAPID2 persistent kernel 的主 sanitizer 路径。

## 3. 分阶段实施计划

## Phase 0：协议与基线冻结（先做，避免返工）

### 目标

- 定义并冻结三个协议：
  - `KernelManifest schema`（编译期提取结果，记录每个 kernel 参数类型信息）
  - `Input Envelope v1`（运行时输入，raw bytes）
  - `Phase 0 interface mapping`（origin/rapid/rapid2 对齐）
- 新增并冻结 `Decode/Invoke Stable Interface v1`：
  - 固定接口名：`fuzzer_decode_v1`、`fuzzer_invoke_v1`
  - 约束：手写 `.cu/.cuh` 只保留稳定壳层；可变参数展开逻辑只来自生成头文件
  - 生成文件路径约定：`gen/fuzzer_decode.v1.cuh`、`gen/fuzzer_invoke.v1.cuh`

### 产出

- `docs/input-envelope-v1.md`
- `docs/phase0-interface-mapping.md`
- `docs/kernel-manifest.schema.json`

### 验收门槛

- 同一输入在 origin 与 rapid2 的 decode 结果一致。
- payload 保持 raw bytes；`vdim` 不从 payload 读取。
- manifest 可覆盖提取出的所有 `__global__` kernel，并记录参数类型信息。

---

## Phase 1：Kernel 发现 + Manifest 元数据 + Rewrite Plan

本阶段的细化实施计划见：`docs/phase1-implementation-plan.md`。
本阶段流水线说明见：`docs/kernel-smoke-pipeline.md`。

### 目标

- 对目标库自动枚举全部 `__global__` kernel，产出可复用的 manifest 元数据与 rewrite plan。

### 实现

- 修改 LLVM 逻辑，在编译期完整枚举所有 `__global__` kernel（含模板/间接生成结果），并提取签名/元数据。
- 为每个发现的 `__global__` kernel 产出 rewrite plan（原始符号、目标 rewritten device symbol、验证条件、后续 injected params 预留位点）；实际 `kernel.bc` 上的 LLVM IR 改写在 Phase 2 执行。
- 默认交付以 manifest/rewrite-plan 为主；不要求本阶段产出 `.so` 或 decode/invoke 代码。
- 默认按每个 kernel 打包：`artifacts/<target-lib>/<kernel>/`，包含 manifest、构建元数据、产物哈希、rewrite plan。
- per-kernel 构建元数据遵循 `docs/artifact-metadata.schema.json`，用于可追踪性与可复现校验。
- `KernelManifest.args` 预留并记录对齐信息（`align_bytes`）；Phase 1 只负责产出该元数据，不要求 fuzzer 在本阶段动态消费。
- Phase 1 默认只验证发现结果、manifest 完整性与 rewrite plan 可验证性；device/harness 构建在 Phase 2。
- 预留 metadata 参数通道（如 vdim）的改写位点，但具体参数注入放到 Phase 5 实施。
- 可选后续：增加 PTX 分发包；规模上来后引入 shard（16~64 kernels/so）。
- Phase 1 只负责提取/发现、manifest 元数据与 rewrite plan 产出；不执行改写，不生成 decode/invoke 代码。

### 验收门槛

- 大库样例中“提取集”与“manifest+rewrite-plan 产出集”一致（或有明确白名单解释）。
- 每个提取 kernel 均有可解析参数 schema，并具备可执行 rewrite plan。

## Phase 2：LLVM IR rewrite + 稳定 decode/invoke 接口接入

本阶段的细化实施计划见：`docs/phase2-implementation-plan.md`。
本阶段流水线说明见：`docs/kernel-rewrite-pipeline.md`。

### 目标

- 在 CU 侧固化稳定接口（`fuzzer_decode_v1`、`fuzzer_invoke_v1`），并把随 kernel/schema 变化的逻辑收敛到生成文件。

### 实现

- `Input Envelope v1` 保持 raw bytes，不在 payload 内编码 `kernel_id`/`vdim`/schema 头。
- 由 Phase 2 的 fuzzer 编译器（可继续基于 LLVM）读取 Phase 1 的 `KernelManifest` 与 rewrite plan，以 `kernel.bc` 为输入真源执行 `__global__ -> __device__` 的 LLVM IR 改写，并生成/接入：
  - `gen/fuzzer_decode.v1.cuh`：
    - `DecodedKernelArgs` 数据结构；
    - raw bytes 到参数视图/强类型参数的解码逻辑；
    - 必要的版本与长度/布局防御检查。
  - `gen/fuzzer_invoke.v1.cuh`：
    - 稳定入口 `fuzzer_invoke_v1(...)`；
    - 基于 manifest 的参数展开，以及对 rewritten `entry_symbol` 的调用表达式；
    - 后续 metadata/context 也优先作为 `entry_symbol` 的额外参数接入；
    - 与目标签名匹配的编译期约束（静态检查）。
- 手写 `.cu/.cuh` 仅保留稳定壳层，不承载可变参数展开细节：
  - `DecodedKernelArgs decoded = fuzzer_decode_v1(input, size_or_size_ptr);`
  - `fuzzer_invoke_v1(decoded);`
- rapid2/origin/rapid 统一复用上述模式，保证同一输入得到同一 decode/invoke 语义。
- Phase 2 当前的调用关系为：`fuzzer_invoke_v1 -> entry_symbol`。
- 对齐注意事项：decode 需按参数 `align_bytes` 做最小安全检查；debug 诊断模式下可通过 `RAPID_DECODE_DEBUG_ASSERT` 开启断言检查，用于尽早发现 contract/codegen 问题，同时避免默认 fuzz 构建因 device assert 打坏 CUDA context。
- 当前阶段保持稳定接口 `fuzzer_decode_v1` / `fuzzer_invoke_v1` 不变；不额外扩展错误返回契约，优先保证流水线先跑通。

### 验收门槛

- 每个 kernel 至少 1 个 smoke 输入可完成 decode + 调用。
- 非法输入不导致进程级崩溃或长期卡死（暂不要求统一错误码）。

---

## Phase 3：Fuzzer 侧 Manifest 驱动输入编码

### 目标

- 让 fuzzer 在变异后输出“可被 `fuzzer_decode_v1` 解析”的输入，而不是任意裸 bytes。
- 输入编码与 `KernelManifest` 对齐，保证“mutation 结果 -> decode -> args”链路稳定可执行。

### 实现

- fuzzer 启动时加载目标 `KernelManifest`（当前可先单 kernel，后续扩展多 kernel）。
- 以 manifest 的 `args` 顺序生成/规范化输入编码（`arg-pack-v1`）：
  - `uint8_t*` 按 `[len:u64][pad_to_align][bytes:len]` 编码；
  - 参数起始偏移按 `align_bytes` 做 padding 对齐（默认 1，padding byte=0）；
  - 标量按 little-endian 编码；
  - 必要边界与长度一致性检查（避免 decode 失败/越界）。
- 在变异链中加入“格式归一化/修复”步骤，确保提交到 harness 的输入保持可解析。
- 对历史 corpus 允许兼容读取并迁移到 `arg-pack-v1`。

### 验收门槛

- 提交到 harness 的输入中，绝大多数（目标 100%）可被 `fuzzer_decode_v1` 成功解析。
- 同一 manifest 下，origin/rapid/rapid2 的 decode 语义一致。
- 输入解码失败时可追踪（日志/计数），且不导致进程级崩溃。

---

## Phase 1+2+3 优先交付（里程碑）

- 优先目标：先交付“可运行自动化流水线”，输入一个目标库后可自动完成：
  1. 提取全部 `__global__` kernel；
  2. 生成每个 kernel 的 manifest + 生成头（`gen/fuzzer_decode.v1.cuh`、`gen/fuzzer_invoke.v1.cuh`），并复用稳定接口 `fuzzer_decode_v1/fuzzer_invoke_v1`；
  3. fuzzer 按 `KernelManifest` 对输入进行编码/归一化，确保提交样本可 decode；
  4. 编译并链接 per-kernel `origin/rapid2` fuzz targets；
  5. 输出可直接启动 fuzz 的目标清单与构建日志。
- 该里程碑之后的增强项包括 feedback 自动改写、VCONFIG 参数化与
  device-side sanitizer；当前 feedback 与 VCONFIG 的已落地状态和剩余边界
  分别见 [`docs/feedback.md`](docs/feedback.md)、[`docs/vconfig.md`](docs/vconfig.md)、[`docs/sanitizer.md`](docs/sanitizer.md)。

---

## Phase 4：反馈自动改写（Feedback Rewrite）

自动 CFG 与 payload-memory 插桩、task-local map、backend transport 和 Rust
novelty 已经落地。当前剩余工作是用 `rapid-simt-memcov-v1` 直接替换 coarse
`rapid-memory-bitset-v1` 的设备端特征计算，并重新运行 backend equivalence、
性能 gate 和正式 RQ2；详细公式、实施边界与验收标准见
[`docs/feedback.md`](docs/feedback.md)。

---

## Phase 5：VCONFIG metadata 参数化与 Phase 2 virtual-dimension rewrite

固定物理 persistent launch，并通过 runtime metadata 为每个 task 注入逻辑
grid/block；VCONFIG 不进入 raw payload。当前实现已经完成：

- `RapidTaskEnvelopeHeader` 携带 `RapidVConfig + payload_size`，payload 仍是
  原始 `arg-pack-v1`。
- fuzzer 侧 envelope-aware normalize/mutate/clamp 已接入 manifest
  `launch_policy`。
- Phase 2 默认请求 VConfig transform，并通过
  `tools/rapid-vconfig-instrument` 在 `kernel.device.bc` 上把
  `gridDim`/`blockDim` 读改写为 `RapidKernelContext::vconfig`。
- Phase 2 在 `build_spec.json`、`metadata.phase2.json`、
  `rewrite_summary.json` 中记录 `vconfig_requested`、`vconfig_enabled` 和可选
  `vconfig_disabled_reason`。
- 固定 main-path `barrier0` 通过 inactive path virtual sync 支持；固定
  trip loop barrier 支持 pass 的 PHI self-loop 形态与 Clang `optnone`
  header/body/latch 形态，并通过 inactive sync-only loop 保持 barrier
  轮次一致；NVVM warp primitive 通过 warp-aligned logical slicing 和 32
  倍数 physical block candidates 支持。data-dependent/divergent/helper
  barrier、inline asm 仍保守禁用 VConfig，但保留基础 decode/invoke backend
  可构建。

剩余工作主要是 multi-block physical persistent launch、同步语义安全改写、
以及更大规模 third-party campaign 刷新。详细设计与阶段边界见
[`docs/vconfig.md`](docs/vconfig.md)。

### 附：RQ1 运行时评估基建（已落地）

- `benchmark/rq1/`：端到端 RQ1 性能基准，含 6 个 third-party workload
  （来源可溯、provenance 校验）、7 种后端变体 / 9 个 campaign 配置
  （cufuzz、origin、rapid、rapid2 及 no-feedback 变体，rapid 系按窗口
  拆分 w1/w2）与 campaign 执行器；perf workload 通过
  `vconfig_reserved: false` 构建无 VConfig 重写的 base bitcode。
- `cuda-kernel/cufuzz/`：upstream 风格 baseline 后端。
- 运行时约束：fuzz 完成窗口上限 32，RAPID2 任务队列 64（窗口不得超过队列
  容量，详见 `docs/feedback.md`）。

---

## Phase 6：Device-side Sanitizer

Sanitizer 插桩进入 RAPID device 执行路径，并通过 per-task runtime record region 返回 OOB/canary/uninit/sync 等诊断；详细设计与非目标见 [`docs/sanitizer.md`](docs/sanitizer.md)。

---

## Phase 7：调度、隔离、工程化

### 目标

- 支持大规模 kernel 库长期稳定运行。

### 实现

- `.so` 打包策略采用 shard（建议每 16~64 kernels/so）。
- 调度策略：先选 kernel（或 shard），再做变异。
- 失败隔离：worker 进程化，坏样本不拖垮全局 campaign。
- 统一产物：crash 元数据包含 manifest hash、backend、GPU/driver、seed。

### 验收门槛

- 24h 稳定性通过。
- 自动恢复与回放链路可用。

## 4. 执行顺序（建议）

建议按以下顺序推进，减少返工：

1. `Phase 0` -> 冻结 schema/envelope/interface mapping
2. `Phase 1` -> Kernel 发现 + manifest/rewrite plan 产出
3. `Phase 2` -> 以 `kernel.bc` 为输入真源执行 LLVM IR rewrite，并接入 `fuzzer_decode_v1/fuzzer_invoke_v1` + 生成头
4. `Phase 3` -> fuzzer 侧按 manifest 进行输入编码/归一化（arg-pack-v1）
5. `Phase 4` -> feedback 改写稳定化
6. `Phase 5` -> VCONFIG metadata 参数化
7. `Phase 6` -> device-side sanitizer 插桩与 per-task 诊断
8. `Phase 7` -> 大规模工程化

## 5. 当前仓库对应关系（便于开工）

- 异步执行主路径：`cuda-fuzzer/src/async_fuzzer.rs`
- GPU 执行提交器：`cuda-fuzzer/src/gpu_executor.rs`
- 动态加载与共享 coverage map：`cuda-fuzzer/src/cuda_backend/`(`sync.rs`、`ordered.rs`、`async.rs`)
- rapid2 C 接口：`cuda-kernel/rapid2/libafl_interface_shared.cuh`（由 Phase2 backend harness 导出）
- virtual dim pass：`rapid-llvm/llvm/lib/Transforms/Scalar/VirtualDim.cpp`
- Phase 2 VConfig tool：`tools/rapid-vconfig-instrument/`
- vconfig pass 单元/E2E 参考：`tests/vconfig/build_vdim_end2end.sh`、
  `tests/vconfig/test_context_virtual_dim.py`
- backend VConfig feedback E2E：`tests/feedback_e2e/test_vconfig_e2e.py`

## 6. 第一轮落地任务（Next）

- Task A：冻结 `KernelManifest` 最小字段（args 类型、顺序、符号信息、`align_bytes`）。
- Task B：冻结 `Input Envelope v1`（raw bytes）与 decode 插入点。
- Task C：落地 `fuzzer_decode_v1/fuzzer_invoke_v1` 稳定壳层（先以单 kernel 打通）。
- Task D：实现 `KernelManifest -> gen/fuzzer_decode.v1.cuh + gen/fuzzer_invoke.v1.cuh` 的最小 codegen 闭环，并接入 rapid2/origin。

完成 A/B/C/D 后，再开始批量化（Phase 1/2）。
