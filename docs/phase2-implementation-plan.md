# Phase 2 实施计划（LLVM IR 改写 + 稳定 decode/invoke 接口接入）

本文档用于收敛 Phase 2 的执行口径。

- Phase 2 消费 Phase 1 的稳定产物（`kernel.bc` + `kernel.ptx` + `manifest.json` + `metadata.json`）；
- 以 `kernel.bc` 为输入真源，在 LLVM IR 层执行 `__global__ -> __device__` 改写，并补齐最小 rewrite 输入；必要时可以经过 `llvm-dis/llvm-as` 的 `.ll` roundtrip，但 rewrite 语义与校验应以 IR 结构为准，而不是依赖脆弱的文本替换；
- rewrite 目标是产出一个可直接被 `fuzzer_invoke_v1` 调用的 rewritten `entry_symbol`；
- 生成 `gen/fuzzer_decode.v1.cuh` / `gen/fuzzer_invoke.v1.cuh`，将可变逻辑收敛到生成文件；
- `origin/rapid/rapid2` 仅保留稳定壳层调用：`fuzzer_decode_v1` + `fuzzer_invoke_v1`。

本阶段计划目标：

- 提供独立 Phase 2 入口（例如 `scripts/kernel-rewrite/cli.py`），以 Phase 1 run 目录为输入。
- 建立 plan-driven rewrite 流程：读取 `kernel.bc`，构建内部 rewrite plan，执行 `__global__ -> __device__` rewrite，产出单一 rewritten `entry_symbol`。
- 将 `.ll` 仅作为可逆中间形式、debug 输出或回归对比材料；rewrite 的语义定义与校验以 LLVM IR 结构为准。
- 建立 `rewrite/`、`codegen/`、`common.py`、`contracts/` 分层，避免把参数展开、decode、invoke 逻辑散落到手写 backend 文件中。
- 建立统一 contract / support gate：明确支持哪些 manifest arg 形态，未支持输入统一 fail fast，并写入 per-kernel `metadata.phase2.json`。
- 生成可编译的 `phase2/gen/fuzzer_decode.v1.cuh` 和 `phase2/gen/fuzzer_invoke.v1.cuh`，由 `origin/rapid/rapid2` 共享。

## 0. 体验目标（从 Phase 1 产物直达可调用产物）

Phase 2 以“复用现有 kernel-smoke 流程、最小新增步骤”为目标：

- 使用 `tests/kernel_pipeline_smoke/fixtures/` 作为本地基线输入，先跑通 Phase 1 提取；
- 以 `build/kernel-smoke/<run_id>/kernels/<kernel_id>/` 为单位执行改写与代码生成；
- 不要求业务项目改源码；仅在 pipeline 后处理阶段增加 Phase 2；
- 保证同一输入在 `origin/rapid/rapid2` 下 decode/invoke 语义一致。

## 1. 边界

### 1.1 In Scope

- 读取并校验 Phase 1 产物目录（每 kernel）：
  - `kernel.bc`
  - `kernel.ptx`
  - `manifest.json`
  - `metadata.json`
- 构建最小 rewrite 输入（Phase 2 负责解析出 rewritten `entry_symbol` 与 payload 参数布局，Phase 1 已提供稳定的 `symbol_name` / `display_name` / `args` / `type_info` / `output_hashes` 输入契约）。
- 在 LLVM IR 层执行 `__global__ -> __device__` 改写，产出 rewritten `entry_symbol` 与校验报告。
- 生成并接入：
  - `gen/fuzzer_decode.v1.cuh`
  - `gen/fuzzer_invoke.v1.cuh`
- 将 `origin/rapid/rapid2` 的可变参数展开逻辑迁入生成头；手写文件仅保留稳定壳层。
- 对 decode 增加最小安全约束：边界/溢出/`align_bytes` 检查；debug 诊断模式下可开启断言检查，尽早暴露 decode / contract 问题。

### 1.2 Out of Scope

- 不做 fuzzer 侧 manifest 动态消费与 arg-pack 归一化（Phase 3）。
- 不做覆盖率自动改写的正式落地（Phase 4），但 Phase 2 设计必须预留对 rewritten `entry_symbol` 的插桩位点。
- 不做 vdim metadata 参数化的正式落地（Phase 5），但 Phase 2 设计必须预留 metadata 参数接入 rewritten `entry_symbol` 的位点。
- 不做 sanitizer 深化 triage 自动化（Phase 6）。

### 1.3 与现有 kernel-smoke 阶段关系

当前 `scripts/kernel-smoke/` 以 variant 为单位流式执行 `discover / emit_bc / manifest`，并在 run 级汇总 `discover.json` / `index.json` / `summary.json`（可选 `profile.json`，内部恢复态 `variant_progress.json`）。

Phase 2 计划在其后新增后处理链路（先以独立脚本 `scripts/kernel-rewrite/cli.py` 实现）：

- `rewrite_plan`：生成/校验最小 rewrite 输入；
- `rewrite_exec`：执行 LLVM IR 改写并验证符号/签名；
- `codegen`：生成 `gen/fuzzer_decode.v1.cuh` 与 `gen/fuzzer_invoke.v1.cuh`；
- `rewrite_summary`：输出阶段统计与失败原因。

## 2. 阶段衔接

### 2.1 Phase 1 -> Phase 2 输入契约

Phase 2 以 Phase 1 的 run 目录为输入真源：

- run-level：`build/kernel-smoke/<run_id>/index.json`、`discover.json`、`summary.json`
- optional / internal：`profile.json`（可选）、`variant_progress.json`（恢复态）
- per-kernel：`kernels/<kernel_id>/kernel.bc`、`kernel.ptx`、`manifest.json`、`metadata.json`

其中 `manifest.json` 的 `args[].type_info` 可为非内建参数类型（例如 struct/class/enum、CUDA 特殊类型、第三方库类型）提供：

- `type_info.kind` / `definition.loc`：真实声明形态与定义位置；
- Phase 1 还会为每个 kernel 生成一个 `type_shim.v1.cuh`，供 Phase 2 的 decode/invoke 直接消费；`type_info/definition.loc` 仍作为定义区域和调试定位真源保留。

该信息可作为 Phase 2 codegen 的头文件依赖真源，避免仅靠 `type` 字符串猜测 include。

最小可处理条件：

- `metadata.build_status == "built"`
- `manifest` 可通过当前 `docs/kernel-manifest.schema.json`（现行 schema version = `1`）
- `kernel.bc` 非空且可被 LLVM 工具链读取

### 2.2 Phase 2 输出契约（建议）

建议在每个 kernel 目录下新增 `phase2/` 子目录，避免污染 Phase 1 合同文件：

```text
kernels/<kernel_id>/
├── kernel.bc
├── kernel.ptx
├── manifest.json
├── metadata.json
└── phase2/
    ├── kernel.device.bc
    ├── build_spec.json
    ├── gen/
    │   ├── fuzzer_decode.v1.cuh
    │   └── fuzzer_invoke.v1.cuh
    └── metadata.phase2.json
```

其中：

- `kernel.device.bc`：改写后的 device 模块
- `build_spec.json`：`schema_version`、`kernel_id`、`entry_symbol`、`device_bc`、`decode_header`、`invoke_header`、`shim_header`
- `metadata.phase2.json`：`schema_version`、`kernel_id`、`input_symbol`、`entry_symbol`、`phase2_status`、`failure_reason`、`failure_detail`；当失败原因有更具体来源时可附加 `failure_context`

## 3. 数据契约

### 3.1 rewrite 输入最小字段（内部）

- `schema_version`
- `kernel_id`
- `input_symbol`（原始 kernel 符号）
- `entry_symbol`（`__global__ -> __device__` 后的目标 device 符号）
- `payload_args`（来自 manifest args 的索引/类型/align）

上述字段是 Phase 2 内部 rewrite 输入，不再单独落盘成 `rewrite_plan.json`；执行期校验也由 Phase 2 内部逻辑负责。

### 3.2 生成头契约

- `gen/fuzzer_decode.v1.cuh`：
  - 提供 `DecodedKernelArgs` 结构；
  - 提供 raw bytes -> 参数视图/参数值 的解码逻辑；
  - 按 `docs/input-envelope-v1.md` 的 materialization 规则生成代码：
    - `scalar`：按 `size_bytes` 读取 little-endian scalar；
    - `opaque_val`：按 `size_bytes` 读取固定长度对象字节，并 materialize 成本地 by-value 值；
    - `opaque_with_ptr`：不能整体当黑盒 copy；必须按 `type_layout` 下钻，递归 materialize 其 scalar、payload pointer、`opaque_val`、`opaque_with_ptr` child；
    - `pointer_role=payload_buffer`：跳过 canonical padding 后读取 `[len:u64][payload+filler]`，其中 `len` 只覆盖后续 `[payload+filler]` segment，不包含 8-byte length field；decoder materialize 出指向 payload segment 起始地址的 pointer；manifest 必须同时提供 `pointee_layout`，但 Phase 2 当前不递归 materialize pointee；
    - `pointer_role=derived_pointer`：读取 `[offset:u64]`，结合已经 materialized 的 `base_arg` 计算 `base + offset`，并在 generated decoder 中执行 bounds/alignment 检查；当前未实现时必须 fail fast，不能落入 `payload_buffer` 路径；
    - `pointer_role=external_device_pointer`：由 runtime/harness source 提供；当前主路径未实现时必须 fail fast。
  - 对所有参数统一 include `type_shim.v1.cuh`；对非内建顶层 by-value 参数，按 `align_bytes + sizeof(T)` 读对象字节；
  - 执行最小边界与对齐防御检查。
- `gen/fuzzer_invoke.v1.cuh`：
  - 提供稳定入口所需的参数展开逻辑；
  - 将 `DecodedKernelArgs` 中已经 materialized 的参数绑定到 rewritten `entry_symbol` 调用表达式；
  - 不重新 parse raw bytes，不重新解释 manifest，不执行 `base_arg + offset` pointer materialization；
  - 提供编译期约束（参数个数/类型不匹配时编译失败）。

头文件依赖建议：

- `type_shim.v1.cuh` 作为 decode/invoke 的统一类型依赖入口；
- 基础 C/C++ 内建类型可由空 shim 或最小 shim 覆盖；
- 非内建参数类型不再要求 Phase 2 直接 include 原项目头文件。

约束：`fuzzer_decode_v1` / `fuzzer_invoke_v1` 由 Phase2 生成头提供，不承载仓库内手写可变参数细节；可变细节收敛到 rewritten `entry_symbol` 与生成头。

### 3.3 符号与参数通道

- `input_symbol`：Phase 1 发现出的原始 `__global__` kernel。
- `entry_symbol`：Phase 2 执行 `__global__ -> __device__` 后得到的 rewritten device 符号；`gen/fuzzer_invoke.v1.cuh` 只调用它。
- `payload_args`：仅由 `fuzzer_decode_v1` 从 raw bytes 解码。
- 未来若需要 metadata 参数，可在不改变当前最小 rewrite 输入集合的前提下再扩展。

### 3.3.1 Phase 2 处理规则

- Phase 2 必须读取并校验 Phase 1 `manifest.json`。
- 每个 per-kernel manifest 必须只包含一个 kernel；如果 `kernels` 数组不是 exactly one kernel，Phase 2 必须 fail fast。
- `kind=pointer` 参数必须携带 `pointer_role` 和 `pointee_layout`；缺失任一字段都必须 fail fast。
- Phase 2 必须把 `kind` 视为 pointer/scalar/aggregate 分类的唯一入口，
  并用 `pointee_layout.kind` 判断 pointer 背后一层类型是否需要
  `type_shim.v1.cuh`。不得用 `args[].type` 的字符串形态推断语义：
  不得用 `type.endswith("*")` 判断 pointer，不得维护 builtin spelling
  表判断 scalar/size_t/typedef，也不得用 typedef alias 名称猜测 pointee。
  `type` 只用于 generated C/C++ codegen spelling；其可见性由 Phase 1
  产出的 `type_shim.v1.cuh` 和 include dirs 负责。
- `pointer_role=payload_buffer` 必须按 input-envelope-v1 生成 decode。
- `pointer_role=derived_pointer` 必须按 `base_arg + offset` 语义生成 decode；在该路径实现前，Phase 2 必须 fail fast。
- `pointer_role=external_device_pointer` 必须由 runtime/harness source 绑定；在该路径实现前，Phase 2 必须 fail fast。
- `kind=opaque_val` 表示 pointer-free inline aggregate，Phase 2 可按 `read_object_bytes<T>` 处理；若节点是 fixed array，则按 element 展开，避免生成 C++ array assignment。
- `kind=opaque_with_ptr` 表示必须下钻 materialize 的 inline aggregate；Phase 2 contract 必须要求完整可用的 `type_layout`，并对每个 child 递归生成 decode。若遇到 incomplete layout、unsupported pointer role、union 等当前能力外节点，Phase 2 必须 fail fast，并在诊断中保留具体 `index`。
- manifest `constraints` 必须进入 Phase 2 contract，并由
  `fuzzer_decode.v1.cuh` 生成对应检查；当前 v1 支持
  `scalar_le_buffer_len(unit=bytes|elements)`、`scalar_compare_const`、
  `scalar_compare_scalar`、以及 `count_fits_buffer`。
- `fuzzer_decode.v1.cuh` 负责 input envelope materialization 和 decode-time validation。
- `fuzzer_invoke.v1.cuh` 只负责调用 rewritten `entry_symbol`，不得重新 parse raw bytes，不得重新解释 manifest，不得执行 pointer materialization。

### 3.3.1.1 opaque_with_ptr 下钻计划

`opaque_with_ptr` 是 Phase 2 从“按顶层黑盒值 decode”进入“按字段 materialize”的分界。

处理计划：

1. contract 层把 manifest `type_layout` 解析为递归 `KernelLayoutNode`，root path 来自顶层 arg `name`，并保留每个嵌套节点的 `index`、kind、size/align、type_info、pointer_role。`index` 形如 `params.inner.buf`；array element 模板保留 `"name": "$element"`，并使用 `params.inner[].buf` 这类 `[]` 路径。
2. `opaque_val` child 使用 copy-safe materialization；fixed array child 按 element 展开。
3. `pointer && pointer_role=payload_buffer` child 使用标准 input envelope：canonical padding 加 `[len:u64][payload+filler]`，与顶层 payload pointer 完全同一套规则；`pointee_layout` 仅作为 pointee 事实保留，不会让 Phase 2 对 pointee 对象做额外字段 decode。
4. `opaque_with_ptr` child 继续递归，直到 leaf。
5. 当前不支持的 child role 或 incomplete layout 不写进 manifest 的 support 字段，而是在 Phase 2 contract/generator 层 fail fast。
6. invoke 层不关心字段过程，只接收已 materialized 的原始 top-level arg 值并调用 rewritten entry。

### 3.3.2 Constraint 处理计划

Phase 2 必须把 KernelManifest 中的 `constraints` 转换为 decode-time validation predicate。该设计的目标是让每一种 manifest constraint 只负责解析自己的语义，而 decode codegen 只面对统一的 predicate IR。

处理流程：

1. `KernelManifest.constraints[]` 中每个 entry 必须带 `kind`。
2. Phase 2 contract 层按 `kind` 分发到对应 constraint parser。
3. 每个 parser 返回一个或多个 `KernelConstraintPredicate`。
4. `KernelContract.constraint: list[KernelConstraintPredicate]` 保存所有解析后的 predicate。
5. `fuzzer_decode.v1.cuh` codegen 遍历 `KernelContract.constraint`，每个 predicate 生成一个 `RAPID_DECODE_ASSERT(<positive predicate>)`。

当前 predicate IR：

- `KernelConstraintPredicate(lhs, op, rhs)` 表示一个必须成立的约束；
- `op` 支持 `<`、`<=`、`==`、`!=`、`>=`、`>`；
- `ConstraintExpr` 当前包含：
  - `ArgValueExpr(arg_index, field_path=())`：已 materialized 的 scalar/by-value 参数值；`field_path` 是相对 JSON segment array，缺省时引用顶层 arg 自身；
  - `PayloadLenExpr(arg_index, field_path=())`：某个 `pointer_role=payload_buffer` 参数或字段的 decoded payload segment 长度；
  - `ConstExpr(value)`：整数常量；
  - `BinaryExpr(op, lhs, rhs)`：表达式组合，当前用于 element/count 到 byte 长度的 lowering。

当前 Phase 2 已实现的 constraint parser：

- `scalar_le_buffer_len(unit=bytes|elements)`：
  - manifest 语义：`scalar_arg[.scalar_path] <= buffer_arg[.buffer_path].payload_len`
  - `unit=bytes` parser 输出：`ArgValueExpr(scalar_arg, scalar_path) <= PayloadLenExpr(buffer_arg, buffer_path)`
  - `unit=elements` parser 输出：`ArgValueExpr(scalar_arg, scalar_path) * pointee_size_bytes <= PayloadLenExpr(buffer_arg, buffer_path)`
  - decode 生成：`RAPID_DECODE_ASSERT(decoded.<scalar> <= <buffer>_len_u64);`
- `scalar_compare_const`：
  - manifest 语义：`scalar_arg[.scalar_path] op value`
  - parser 输出：`ArgValueExpr(scalar_arg, scalar_path) op ConstExpr(value)`
  - decode 生成示例：`RAPID_DECODE_ASSERT(decoded.params.nested.inner_count >= 0);`
- `scalar_compare_scalar`：
  - manifest 语义：`lhs_arg[.lhs_path] op rhs_arg[.rhs_path]`
  - parser 输出：`ArgValueExpr(lhs_arg, lhs_path) op ArgValueExpr(rhs_arg, rhs_path)`
  - decode 生成示例：`RAPID_DECODE_ASSERT(decoded.params.nested.choice.tag <= decoded.params.count);`
- `count_fits_buffer`：
  - manifest 语义：`count_arg[.count_path] * elem_size_bytes <= buffer_arg[.buffer_path].payload_len_bytes`
  - parser 输出：`BinaryExpr("*", ArgValueExpr(count_arg, count_path), ConstExpr(elem_size_bytes)) <= PayloadLenExpr(buffer_arg, buffer_path)`
  - decode 生成示例：`RAPID_DECODE_ASSERT((decoded.params.n * 4) <= params_data_len_u64);`

详细 constraint 规则见 `docs/constraint.md`。

### 3.4 `.bc` / `.ll` 处理口径

- `kernel.bc` 是 Phase 2 的输入真源；
- `.ll` 可以作为 debug 输出、回归对比材料，或经 `llvm-dis/llvm-as` 的可逆中间表示；
- 关键不在“是否经过 `.ll`”，而在“rewrite 是否按 LLVM IR 结构语义定义和验证”；
- 允许 `bc -> ll -> bc` 的 roundtrip，但不建议长期依赖 raw 文本 regex/search-replace 作为唯一 rewrite 机制，尤其是在需要 body instrumentation、签名扩展、metadata 参数注入时。

### 3.5 decode 对齐与失败语义

按 `docs/input-envelope-v1.md`：

- payload 仍是 raw bytes，不包含 `kernel_id` / `vdim` 头；
- 按 manifest `args` 顺序解码，遵守 `align_bytes`；
- 任一检查失败都不应继续按当前布局解释 payload；
- generated decode 默认提供一个薄宏：如果调用方没有提前定义 `RAPID_DECODE_ASSERT(cond)`，则仅在 `!NDEBUG && RAPID_DECODE_DEBUG_ASSERT` 时映射为 `assert(cond)`，否则为 `((void)0)`；
- 如果 release fuzz/runtime 仍需要保留 decode 检查，调用方必须在 include generated decode header 前自定义 `RAPID_DECODE_ASSERT(cond)`；
- 稳定接口保持 `fuzzer_decode_v1` / `fuzzer_invoke_v1` 不变。

### 3.6 失败原因（建议扩展）

在 Phase 2 统计中统一失败原因编码：

- `phase2_input_invalid`
- `rewrite_plan_invalid`
- `rewrite_exec_fail`
- `rewrite_verify_fail`
- `codegen_fail`
- `harness_compile_fail`

`phase2_input_invalid` 是阶段分类，不应掩盖具体 unsupported 真因。
`failure_detail` 必须保留直接 reason code 或 path-qualified reason，例如
`params.nested.choice:union_not_supported`。对
`scoped_type_shim_unsupported`，`failure_context` 应携带
`type_shim_missing_dependencies` 和相关 header/include 信息，方便区分
“当前不支持 scoped/namespace type shim” 与“layout/codegen 走错路径”。

## 4. 实施步骤

1. 基线准备：用 `tests/kernel_pipeline_smoke/fixtures/` 跑一轮 Phase 1，确认每 kernel 目录产物齐全。
2. M1（rewrite 输入实体化）：为每个 `build_status=built` kernel 构建最小 rewrite 输入。
3. M2（改写执行）：基于 `kernel.bc + rewrite 输入` 执行 IR 级改写，产出 `kernel.device.bc`。
4. M3（改写校验）：将改写结果汇总进 `metadata.phase2.json`，确认 `entry_symbol` 可解析、已完成 `__global__ -> __device__` 转换，且签名与 codegen 合同一致。
5. M4（decode 代码生成）：生成 `gen/fuzzer_decode.v1.cuh`，包含 `DecodedKernelArgs` 与最小防御检查。
6. M5（invoke 代码生成）：生成 `gen/fuzzer_invoke.v1.cuh`，完成 `DecodedKernelArgs` + metadata/context 到 rewritten `entry_symbol` 的绑定表达式。
7. M6（三后端接入）：`origin/rapid/rapid2` 仅保留稳定壳层调用，统一 include 对应 `gen/`。
8. M7（阶段总结）：输出 run-level `rewrite_summary.json` 与 per-kernel `metadata.phase2.json`。

## 4.1 本地 smoke 执行建议

建议先在 fixture 上建立固定回归：

- 输入：`tests/kernel_pipeline_smoke/fixtures/*.cu`
- 先跑 Phase 1：得到 `build/kernel-smoke/<run_id>/kernels/<kernel_id>/...`
- 再跑 Phase 2：在每个 kernel 目录新增 `phase2/` 产物
- 最后挑选至少一个 kernel 编译 `origin/rapid2`，验证 decode + invoke 链路

## 5. 验收标准

- AT1：每个 `build_status=built` 的 kernel 均产出 `phase2/metadata.phase2.json`，并记录 `input_symbol` / `entry_symbol`。
- AT2：每个目标 kernel 均产出 `phase2/kernel.device.bc`，并可通过 `entry_symbol` 符号、签名与 `__global__ -> __device__` 状态校验。
- AT3：每个目标 kernel 均生成 `phase2/gen/fuzzer_decode.v1.cuh` 和 `phase2/gen/fuzzer_invoke.v1.cuh`，且 invoke 头只依赖 rewritten `entry_symbol`。
- AT4：`origin/rapid/rapid2` 三后端对同一输入的 decode/invoke 语义一致。
- AT5：非法输入/contract 不一致可通过 `RAPID_DECODE_ASSERT` 暴露；默认宏只有在 `!NDEBUG && RAPID_DECODE_DEBUG_ASSERT` 下启用，release 或未显式开启 debug decode assert 时为 no-op。
- AT6：Phase 2 失败可追踪（kernel_id、阶段、failure_reason、日志路径）。

## 6. 文档与口径对齐要求

- `IMPLEMENTATION_PLAN.md` 的 Phase 2 描述与本文一致：
  - Phase 2 负责以 `kernel.bc` 为输入真源的 IR rewrite 执行与 decode/invoke 生成接入。
- `docs/input-envelope-v1.md` 与本文一致：
  - Input Envelope v1 仍为 raw bytes；
  - decode 防御规则由 Phase 2 生成代码实现；
  - metadata 参数未来直接接入 rewritten `entry_symbol`，不进入 payload。
- `docs/kernel-rewrite-pipeline.md` 与本文一致：
  - Phase 1 合同产物作为 Phase 2 输入；
  - Phase 2 为后处理扩展阶段，不回改 Phase 1 合同语义。

## 7. 第一轮落地任务（Next）

- Task A：定义包含 `input_symbol` / `entry_symbol` / `payload_args` 的最小 rewrite 输入与生成入口。
- Task B：实现单 kernel IR rewrite helper（先覆盖当前 fixture lane；允许使用 `.ll` roundtrip，但校验必须按 IR 结构语义落地）。
- Task C：实现 `KernelManifest -> gen/fuzzer_decode.v1.cuh` 最小 codegen。
- Task D：实现 `KernelManifest + rewritten entry_symbol -> gen/fuzzer_invoke.v1.cuh` 最小 codegen。
- Task E：在 `origin/rapid/rapid2` 接入同一套生成头并完成 smoke 回归。
- Task F：将 manifest `constraints` 接入 Phase 2 `KernelContract` 与 `fuzzer_decode.v1.cuh` 生成。现状已覆盖 `scalar_le_buffer_len(unit=bytes|elements)`、`scalar_compare_const`、`scalar_compare_scalar`、`count_fits_buffer`。
- Task G：维护 `docs/constraint.md`，记录 constraint schema kind、Phase 2 parser 支持状态、predicate IR 和 decode lowering 规则。

完成 A/B/C/D/E/F/G 后，再推进批量化与并行化执行。

## 8. TODO（后续扩展）

- 扩展 schema/plan/codegen 以表达 `derived_pointer` 的 `offset_unit`、offset constraints，并实现 decode 侧 `base_arg + offset` materialization。
- 扩展 `external_device_pointer` 的 runtime/harness source 绑定策略。
- 扩展 coverage instrumentation、metadata 参数化、sanitizer triage 与 richer codegen。
