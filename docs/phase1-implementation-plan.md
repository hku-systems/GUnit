# Phase 1 实施计划（发现 + 一核一 BC + Manifest 元数据）

本文档用于收敛 Phase 1 口径：

- Phase 1 做 kernel 发现、每 kernel 独立 `kernel.bc` 产出、`KernelManifest` 元数据提取与 `metadata.json` 产出；
- Phase 1 不做 decode/invoke 生成，不做 fuzzer 动态编码，不做 device/harness 构建；
- `align_bytes` 与 `size_bytes` 在 Phase 1 进入 manifest，供 Phase 2/3 消费。
- 发现/提取采用 `artifact` 单一路径：PTX/IR 作为产物真源，通过 Clang helper 对每个 PTX `.entry` 提取实例化后的元数据，不使用源码正则扫描补充。

## 0. 体验目标（无感接入）

Phase 1 以"无感接入现有构建流程"为目标：

- 在用户按原流程编译项目（含大型项目）时，自动识别 `__global__` kernel 并产出 Phase 1 元数据。
- 优先通过替换编译器入口（`clang/clang++` wrapper）实现接入，不要求业务方改源码与构建脚本语义。
- 除编译器路径/环境变量注入外，尽量不引入额外操作步骤。
- 工具与测试隔离：pipeline 工具不放在 `tests/`，`tests/` 仅放 fixture 与测试驱动。

## 1. 边界

### 1.1 In Scope

- 自动发现目标库全部 `__global__` kernel。
- 为每个发现的 `__global__` kernel 产出独立 `kernel.bc`（一核一 BC）。
- 提取每个 kernel 的参数签名（`index/name/type`）与 ABI 布局元数据（`size_bytes/align_bytes`）。
- 支持 `artifact` 发现模式：PTX/IR 专用，不使用源码正则补充 kernel。
- 基于 LLVM IR 进行 kernel 产物提取与存在性校验（exact symbol extraction）。
- 生成并校验 `KernelManifest`（符合 `docs/kernel-manifest.schema.json`）。
- 生成并校验 `metadata.json`（符合 `docs/artifact-metadata.schema.json`）。
- 生成批次索引（发现数量、失败原因、跳过原因）。
- 提供编译期拦截能力：通过 clang/clang++ wrapper 捕获编译命令并完成 kernel 发现与元数据产出。

### 1.2 Out of Scope

- 不做 `fuzzer_decode_v1` / `fuzzer_invoke_v1` 生成接入（Phase 2）。
- 不做 fuzzer 读取 manifest 并动态编码输入（Phase 3）。
- 不改 `tests/vconfig/`（该目录用于后续阶段链路）。
- 不改 `cuda-kernel/` 主执行链路语义（Phase 1 先独立验证）。

## 1.3 集成方式（推荐）

- 方案：提供 `rapid-clang` / `rapid-clang++`（或等价 wrapper），对外保持与原 `clang/clang++` 兼容的参数语义。
- 接入：通过 `CC/CXX`、`CMAKE_C_COMPILER/CMAKE_CXX_COMPILER`、`PATH` 前置等方式替换编译器入口。
- 运行：wrapper 在转发真实编译动作的同时，记录/提取与 CUDA kernel 发现相关信息，并输出 Phase 1 产物。
- 实现建议：wrapper 使用 Python（`scripts/kernel-smoke/`），优先复用现有 clang/llvm 工具链，不要求修改 LLVM。
- 兼容要求：
  - 不改变目标项目构建结果语义；
  - 对非 CUDA 编译单元开销最小；
  - 失败时给出可追踪日志（命令、文件、失败原因）。

## 1.4 目录与结构（本阶段新增）

- 工具目录（非 tests）：
  - `scripts/kernel-smoke/cli.py`
  - `scripts/kernel-smoke/print_type_defs.py`
  - `scripts/kernel-smoke/pipeline/emit_bc.py`
  - `scripts/kernel-smoke/pipeline/manifest.py`
  - `scripts/kernel-smoke/pipeline/capture_db.py`
  - `scripts/kernel-smoke/adapters/replay.py`
  - `scripts/kernel-smoke/adapters/toolchain.py`
  - `scripts/kernel-smoke/utils/compile_args.py`（为 Clang helper 复用编译参数规范化）
  - `scripts/kernel-smoke/utils/kernel_id.py`
- 文档目录：
  - `docs/kernel-smoke-pipeline.md`
- 测试目录（仅 fixture + driver）：
  - `tests/kernel_pipeline_smoke/fixtures/complex_kernels.cu`
  - `tests/kernel_pipeline_smoke/fixtures/expected/`
  - `tests/kernel_pipeline_smoke/drivers/test_emit_bc.py`
  - `tests/kernel_pipeline_smoke/drivers/test_manifest.py`
- 运行产物目录（构建输出）：
  - `build/kernel-smoke/<run_id>/variant_progress.json`
  - `build/kernel-smoke/<run_id>/discover.json`
  - `build/kernel-smoke/<run_id>/index.json`
  - `build/kernel-smoke/<run_id>/summary.json`
  - `build/kernel-smoke/<run_id>/profile.json`（可选，`--profile` 时生成）
  - `build/kernel-smoke/<run_id>/kernels/<kernel_id>/kernel.bc`
  - `build/kernel-smoke/<run_id>/kernels/<kernel_id>/kernel.ptx`
  - `build/kernel-smoke/<run_id>/kernels/<kernel_id>/manifest.json`
  - `build/kernel-smoke/<run_id>/kernels/<kernel_id>/metadata.json`

## 2. 阶段衔接

- **Phase 1 -> Phase 2**：提供稳定 `KernelManifest`（含 `align_bytes`、`size_bytes`、`type_info`、`definition` 区域）与每 kernel 的 `metadata.json`，并在每 kernel 目录生成 `type_shim.v1.cuh`；由 Phase 2 以 `kernel.bc` 为输入真源执行 `__global__ -> __device__` 的 LLVM IR rewrite，并消费该 shim 生成 decode/invoke 代码接入 harness。
- **Phase 1 -> Phase 3**：提供同一份 `KernelManifest`，由 Phase 3 的 fuzzer 在启动时加载并用于输入编码/归一化。

## 3. 数据契约

### 3.1 Manifest 字段（最小集）

- `manifest.json` 当前 schema 版本为 `1`。

- kernel: `symbol_name`, `display_name`
- arg: `index`, `name`, `type`, `kind`, `size_bytes`, `align_bytes`, `domain`（可选）
- arg 条件必填：
  - `kind=pointer` 时必须有 `pointer_role` 和 `pointee_layout`；
  - `pointer_role=derived_pointer` 时必须有 `base_arg`，并且必须有 `offset_bytes` 或 `offset_arg`；
  - `pointer_role=external_device_pointer` 时必须有 `source`；
  - `kind!=pointer` 时不得携带 `pointer_role` 或 `pointee_layout`。
- arg: `type_info`（可选，artifact/helper 可提供）
  - `kind`, `qualified_name`, `usr`, `decl_loc`, `definition`
  - `type_info.kind` 直接使用 C/C++ declaration form，例如 `struct`、`class`、`union`、`enum`
  - 节点自身的 `type` 是当前 codegen 消费的类型字符串；不再要求重复输出 `spelled_type` / `canonical_type`
- others: `source_file`, `source_line`, `origin_project`（可选）

### 3.2 Artifact Metadata 字段（最小集）

- `metadata.json` 必须满足 `docs/artifact-metadata.schema.json`。
- 必填字段至少包括：`schema_version`、`target_lib`、`kernel_id`、`kernel_symbol`、`build_status`、`output_hashes`。
- 状态约束：
  - `build_status=built` 时 `failure_reason=null`；
  - `build_status in {failed, skipped}` 时必须有 `failure_reason`。

### 3.3 ABI 布局与编码对齐约束

- `size_bytes` 为必填字段，表示顶层 kernel 参数类型的 ABI `sizeof(arg_type)`：指针参数记录指针宽度，by-value struct/aggregate 记录整个聚合类型大小。
- `align_bytes` 为必填字段，表示 `arg-pack-v1` 中该参数起始 offset 必须满足的编码对齐策略；它不是 `size_bytes` 的别名，不能简单按 size 填充。
- 对 by-value 标量/aggregate，默认对齐策略通常可取编译期 ABI `alignof(arg_type)`；对 `pointer_role=payload_buffer` 的 pointer，`align_bytes` 表示 decoder/target 对该 buffer 起始地址要求的对齐，可能来自 pointee 访问宽度、kernel 访问模式或显式配置，而不一定等于指针本身 `alignof(pointer)`。
- artifact/helper 路径必须由 Clang `ASTContext` 和后续策略提供精确 `size_bytes/align_bytes`。
- source-text/AST regex inference helpers have been removed from the production
  tree. 进入 Phase 2/3 的 artifact manifest 必须使用 Clang helper 产出的精确
  `kind/size_bytes/align_bytes/type_layout`。
- `kind` 为必填字段，表示参数或字段的 materialization 分类，而不是 ABI 宽度。Phase 1 使用统一四类 kind：`scalar | pointer | opaque_val | opaque_with_ptr`。对 `kind=pointer`，新增 `pointer_role` 作为 pointer-only 语义字段，并强制 `pointee_layout` 记录一层解引用后的类型布局；非 pointer 参数不应携带 `pointer_role` 或 `pointee_layout`。**最小字段**表述时，对 pointer arg 必须补上条件必填字段，否则读者会误以为 pointer 只要有 `kind` 就足够：
  - `pointer`：顶层 ABI 参数是 pointer；是否按 fuzzer-owned raw byte buffer 编码由 `pointer_role` 决定；`pointee_layout` 描述 `T *` 背后的 `T`，用于后续 element size、mutation policy、以及 pointer-to-struct 场景，不改变 pointer 节点自身的 materialization 分类；
    - `pointer_role=payload_buffer`：fuzzer 负责构造 device allocation，arg-pack-v1 编码为 canonical padding 加 `[len:u64][bytes + filler : len]`；
    - `pointer_role=derived_pointer`：由 `base_arg + offset` 表达 alias/subview，不独立编码 buffer；
    - `pointer_role=external_device_pointer`：由 runtime/harness metadata 提供，不来自 fuzz payload；
  - `scalar`：整数、浮点、enum 等固定宽度 by-value 参数；
  - `opaque_val`：不含 pointer subtree 的 inline aggregate/value，可按整体对象字节 copy；
  - `opaque_with_ptr`：含 pointer subtree、union、或无法证明 pointer-free 的 inline aggregate/value，不能当作黑盒整体 copy，后续 decode/fuzzer 必须按 `type_layout` 下钻 materialize。
- Phase 1 的类型系统通常只能可靠提取“这是 pointer”这个 ABI/type fact，不能自动证明它是 `payload_buffer`。`pointer_role` 应来自 sidecar/policy/annotation/人工配置。若 `kind=pointer` 缺失 `pointer_role` 或 `pointee_layout`，manifest 生成和下游加载都必须 fail fast，避免旧产物被静默当成 `payload_buffer` 或丢失 pointer-to-struct 所需事实。

### 3.3.1 Recursive type_layout 下钻规则

- `type_layout` 是 manifest 的事实层，目标是尽可能记录 C/C++ inline aggregate 的真实递归布局；它不记录当前 Phase2/Phase3 是否支持某个节点。
- 顶层 arg 和 `type_layout` 节点使用同一套 kind：`scalar | pointer | opaque_val | opaque_with_ptr`。
- `type_layout` root 不重复顶层 arg `name`；嵌套字段节点必须带字符串 `index`，用于稳定标识递归位置。规则：
  - 普通字段使用点路径：`params.inner.buf`；
  - array element 模板保留 `"name": "$element"`，`index` 使用 `[]`：`params.inner[].buf`；
  - 如果顶层 arg 本身是 array，root 下钻路径同样使用 `params[]...`。
- 下钻规则：
  - scalar/enum/float leaf -> `kind=scalar`；
  - real pointer leaf -> `kind=pointer`，必须带 `pointer_role` 和 `pointee_layout`；
  - struct/class/constant array 若 subtree 可证明没有 pointer -> `kind=opaque_val`，继续保留 `fields` 或 `element/element_count` 作为事实；
  - struct/class/constant array 若 subtree 含 pointer 或布局不完整 -> `kind=opaque_with_ptr`，继续保留可见的 `fields` 或 `element/element_count`；
  - union 保守标为 `opaque_with_ptr`，通过 `type_info.kind=union` 保留事实，后续支持策略由 Phase2/Phase3 决定；
  - 无法展开的 opaque/partial 节点保守标为 `opaque_with_ptr`，并保留 `layout_status=opaque|partial`。
- 固定数组是 inline aggregate，不因为元素个数固定就转成 `pointer`；只有源码/ABI 上真实 pointer field 才使用 `kind=pointer`。
- Phase 1 需要区分“layout 可下钻”与“对象可安全 materialize”：
  - 成员函数默认忽略，不阻止 public data fields 下钻；
  - public data fields 可以进入 `type_layout.fields`；
  - private/protected data field、virtual method/vptr、non-trivially-copyable、reference field、const assignment blocker 必须写成 `materialization_status="unsafe"`、`materialization_reason_codes[]` 和 `materialization_blockers[]`；
  - 这些字段可以出现在 arg、`type_layout` root、嵌套 layout node，以及 `kernels[].others` summary 上；
  - `scoped_type_shim_unsupported` 只表示 type shim/header 生成边界，不得用于表达 private field、virtual/vptr、reference、const 等对象 materialization 问题。
- manifest 不写泛化的 `supported` / `unsupported_reason`。不支持的原因应在 Phase2/Phase3 contract 或 generator 诊断中给出，并带具体 `index`；Phase1 的 `materialization_*` 只记录事实和风险，不等同于 Phase2 当前是否拒绝。

- TODO：
  - `pointer_role` 现阶段先按 `payload_buffer` 作为当前兼容路径，但后续需要通过分析判断是 `payload_buffer`、`derived_pointer` 还是其他 role。
  - `derived_pointer` 的 `offset_unit`、offset constraints、以及 Phase 2 decode 的 `base_arg + offset` materialization 仍待补齐。

- 若低层诊断工具无法精确获得 `align_bytes`，其输出不得进入 artifact manifest；Phase 1 主路径必须使用 Clang helper 的 `ASTContext` 结果：
  - 标量参数：使用 `alignof(type)`；
  - 指针参数：使用当前 ABI/策略要求的 payload 起始对齐；
  - `void*` / `uint8_t*` 等无法从 pointee 访问宽度推导的情形仍由 helper/policy 明确给出。
- 建议取值为 2 的幂（1/2/4/8/...）。
- 对于可能被内核按宽类型访问的缓冲区（例如 `atomic` 以 `int*` 访问），必须在 manifest 明确给出所需对齐。

### 3.4 一核一产物目录约束

- 每个 kernel 的核心产物必须共置于同一目录：`kernel.bc`、`manifest.json`、`metadata.json`。
- 禁止按类型拆分为跨目录产物（例如把所有 `bc` 与所有 manifest 分离存放）。
- `kernel_id` 需稳定可复现，建议格式：`<symbol_name>__<short_hash(signature+source_loc)>`。

### 3.5 变体标识（已实现完整签名）

- 为支持同一 source 的多配置/多架构编译记录，Phase 1 采用完整变体键：
  - `variant_id = sha256(source_abs | compiler_basename | sorted_relevant_flags | sorted_env | cwd)[:12]`
- `capture_db.py` 按 variant_id 去重：同一变体内取最优记录（exit_code==0 优先），不同变体均保留。
- 变体级持久化状态写入 `variant_progress.json` 与 `discover.json`，不再单独输出 `capture_sources.json`。
- AST/IR/预处理产物均按 variant 维度组织，禁止跨 variant 复用发现结果。
- `kernel_id` 哈希需包含 variant 语义，避免同源多配置冲突。
- `summary.json` 包含按变体统计的 `reconciliation` 信息：
  - `replay_failed_by_variant`：各变体重放失败计数
  - `missing_in_ir_by_variant`：在预处理发现但未出现在 IR 的 kernel 符号
  - `extra_in_ir_by_variant`：IR 中存在但未被发现的符号（尽力而为）
  - `per_variant`：每个变体的详细匹配统计

### 3.6 发现与提取真源约束（新增）

- `artifact` 模式是 Phase 1 唯一运行模式：
  - 发现集仅来自 PTX `.entry` / IR 符号，不使用源码正则扫描补充或回退；
  - 签名细节（`args/source_line/display_name`）由 helper 驱动的 Clang metadata 提取，并按 PTX `.entry` exact symbol identity 落盘。
- IR 真源（Artifact Source of Truth）：
  - kernel 是否真实可提取、最终提取符号（mangled）与 `kernel.bc/kernel.ptx` 以 IR 为准。
- 运行方式约束：
  - capture 模式按 variant 流式执行（discover -> emit -> manifest），并支持 variant 级断点续跑。
- 对账要求：
  - summary 必须提供 AST 与 IR 对账结果（缺失/多出/匹配统计），可按 variant 回溯。

## 4. 实施步骤

1. 先实现复杂 smoke fixture：新增自建用例（同文件多个 `__global__`、宏拼接/宏声明、模板实例化、`__global__` 调 `__device__`）。
2. 接入：启用编译器 wrapper（替换 clang/clang++，工具位于 `scripts/kernel-smoke/`）。
3. Artifact 元数据路径：对每个 variant 先生成 `module.bc -> module.ptx`，以 PTX `.entry` 作为发现真源，再通过 Clang helper 提取实例化后的 `args/source_line/display_name`。
4. IR 精确提取：提取阶段使用 IR exact mangled symbol + recursive extraction，避免同名/模板/重载误提取。
5. 统一对账：summary 固化 discovered vs IR 对账（missing/extra/match），并提供 variant 维度追踪与失败摘要。
6. 归档：为每个 kernel 写入同目录产物（`kernel.bc` + `kernel.ptx` + `manifest.json` + `metadata.json`）。
7. 校验：schema 校验 + 完整性校验（字段齐全、索引连续、路径存在）。
8. 索引：输出 `discover.json` + `index.json` + `summary.json`（discovered/succeeded/failed/skipped + reason + variant 维度统计 + reconciliation）。

## 4.1 验收分层

- 最小验收（本地快速回归）：
  - 使用一个自建复杂 smoke fixture 验证端到端：
    - 同文件多个 `__global__`；
    - 通过宏拼接/宏声明形成 `__global__`；
    - 模板实例化产生 `__global__`；
    - `__global__` 内调用 `__device__` helper；
    - 并完成一核一 BC + 同目录元数据产出（manifest + metadata）+ schema 通过。
- 大型项目验收（阶段门禁）：
  - 在"按原流程构建"的前提下完成 kernel 发现与产物输出，不要求修改业务代码。
  - 验收时至少覆盖 1 个深度学习框架 + 1 个编译/算子栈项目。

建议候选（最终以当期环境可构建性为准）：

- 深度学习框架：`PyTorch`（优先）
- 深度学习框架备选：`TensorFlow`
- 编译/算子栈：`TVM`
- CUDA Kernel DSL/编译工具：`Triton`

## 5. 验收标准

- AT1：发现集完整，且可追踪失败/跳过原因。
- AT1.1：复杂 smoke fixture 中宏/模板/`__device__` 调用场景均被正确发现。
- AT1.2：每个发现 kernel 产出独立 `kernel.bc` 与 `kernel.ptx`。
- AT2：每个 kernel 的 args 至少包含 `index/type`，并补齐 `align_bytes`。
- AT3：manifest 能通过 `docs/kernel-manifest.schema.json` 校验。
- AT4：metadata 能通过 `docs/artifact-metadata.schema.json` 校验。
- AT5：同一输入仓库与配置下，manifest 输出稳定（字段与顺序可复现）。
- AT5.1：同一输入下 `kernel_id` 稳定，且对应目录结构稳定可复现。
- AT6：在 wrapper 接入模式下，目标项目保持"原构建流程可用"，无需改业务源码。
- AT7：至少完成 1 个大型项目试跑并产出可复核的发现/失败统计与日志。
- AT8：Phase 1 只接受 artifact 运行模式；CLI 不再提供 semantic 或 source-regex 运行入口。
- AT9：summary.json 包含 reconciliation 段，可追溯发现集与 IR 符号集的差异。
- AT10：manifest 的 kernel 签名字段（`symbol_name/args/source_loc/display_name`）来自 PTX entry identity 与 Clang helper metadata。
- AT11：kernel 提取使用 IR 精确符号（mangled）并具备依赖提取能力，避免同名/重载误提取。
- AT12：文本正则路径不得作为 Phase 1 artifact 主路径；`fallback_kernels` 在 artifact runs 中保持为 0。

## 6. 文档对齐要求

- `IMPLEMENTATION_PLAN.md` 的 Phase 1/2/3 描述必须与本文档一致。
- `docs/input-envelope-v1.md` 必须明确：
  - Phase 2 负责 decode 代码生成；
  - Phase 3 负责 fuzzer 侧 manifest 动态消费与 arg-pack 编码。
