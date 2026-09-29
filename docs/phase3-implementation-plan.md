# Phase 3 实施计划（Fuzzer 侧 Manifest 驱动输入编码）

本文档用于收敛 Phase 3 的执行口径。

- Phase 3 消费 Phase 1 产出的 `manifest.json` 与 Phase 2 生成的 decode/invoke 合同；
- Phase 3 不改变 `Input Envelope v1`，payload 仍保持 raw bytes；
- Phase 3 负责让 fuzzer 提交的输入稳定满足 `fuzzer_decode_v1` 的解码约束；
- Phase 3 的目标不是“再造一套 decode”，而是让 Rust 侧 encode/normalize 与 GPU 侧 decode 合同完全对齐；
- 当前 `cuda-fuzzer/src/arg_pack_v1.rs` 已经是 manifest-driven encoder：它支持顶层
  `scalar` / `pointer` / `opaque_val` / `opaque_with_ptr`，并能对
  `opaque_with_ptr + complete type_layout` 做递归 field-aware 编码。

## 0. 当前阶段判断（与 Phase 2 的关系）

- Phase 2 主链路已经基本完成：
  - backend `.so` 可从 Phase 2 产物生成；
  - `origin / rapid / rapid2` 已统一走 `fuzzer_decode_v1 + fuzzer_invoke_v1`；
  - 生成的 `.so` 可被 `fuzzer` / `fuzzer_async` 加载和执行。
- 但当前还不能宣称“Phase 2 + Phase 3 全部闭环完成”，因为：
  - 真正完成闭环仍需要在 CUDA-capable GPU 上证明 Phase3 payload 能与 Phase2 生成的 `.so` 长时间运行；
  - normalizer 仍保留 raw bytes fallback，但 fallback 的输出会被重新打包成 canonical arg-pack-v1；
  - 不完整 `opaque_with_ptr`、非 `payload_buffer` pointer role、union/materialization blocker 等 manifest 形状仍应 fail fast。

本阶段的任务就是把这些 Phase 3 缺口补完。

## 1. 边界

### 1.1 In Scope

- fuzzer 启动时加载目标 `KernelManifest`（先按单 kernel 目标实现）。
- 从 manifest 推导 Rust 侧 `ArgPackSpec` / `ArgSpec`。
- 实现按 manifest `args` 顺序的输入编码（`arg-pack-v1`）。
- 在变异链中加入结构归一化/修复，保证提交到 harness 的输入可解析。
- 统一 sync / async 两条提交流水线的 encode 行为。
- 提供默认 seed 生成逻辑。
- 提供 mutation-disabled fuzzing 模式：`--no-mutate` 仍走 LibAFL
  scheduler/stage/feedback，但每次 stage iteration 都提交 manifest canonical seed。
- 提供最小可观测性：manifest 路径、arg-pack normalize/repack/fallback/mutation/seed
  计数，以及 payload domain clamp 计数；decode failure 计数作为后续扩展。

### 1.2 Out of Scope

- 不做多 kernel dispatch（仍先按单 target kernel 假设）。
- 不把 `vdim` 编码进 payload。
- 不做覆盖率自动改写（Phase 4）。
- 不做 metadata 参数注入（Phase 5）。
- 不支持不完整 `opaque_with_ptr` 的 field-aware materialization。
- 不支持 union active-member encoding；union 继续作为当前 v1 fail-fast 边界。
- 不支持非 `payload_buffer` pointer role 的 payload 生成。

## 2. 阶段衔接

### 2.1 Phase 1 -> Phase 3

Phase 1 负责提供：

- `manifest.json`
- `metadata.json`

其中 `manifest.json` 是 Phase 3 的编码真源，特别是：

- `args[*].index`
- `args[*].name`
- `args[*].type`
- `args[*].align_bytes`
- `args[*].size_bytes`
- `args[*].domain`（若存在）

### 2.2 Phase 2 -> Phase 3

Phase 2 负责提供：

- `gen/fuzzer_decode.v1.cuh`
- `gen/fuzzer_invoke.v1.cuh`
- `build_spec.json`

Phase 3 必须保证：

- Rust 侧 encode 出来的字节流，在当前 target 对应的 `fuzzer_decode_v1` 下可稳定解码；
- 不能再依赖仓库根目录的静态 `kernelmanifest.json` 作为所有目标的默认合同；
- 任何 decode 约束都应来源于 manifest / envelope 合同，而不是散落在 fuzzer 侧的手写猜测逻辑。

## 3. 数据契约

### 3.1 Rust 侧内部模型（建议新增）

- `KernelManifest`
- `KernelSpec`
- `ArgSpec`
- `ArgKind`
- `ArgPackSpec`

建议最小字段：

- `index`
- `name`
- `type`
- `align_bytes`
- `size_bytes`
- `domain`
- `kind`（manifest 必填）
- `pointer_role`（pointer-only 语义字段；只允许用于 `kind=pointer`，`kind=pointer` 时缺失必须 fail fast）
- `pointee_layout`（pointer-only pointee 事实字段；`kind=pointer` 时必须提供一层解引用后的布局）
- `type_layout`（可选，aggregate 内部递归布局）

Rust 侧不能用 `size_bytes` 判断编码策略；`size_bytes` 只描述 ABI 布局。Phase 3 加载 manifest 后必须读取 `args[*].kind`，并对 `kind=pointer` 继续读取 `args[*].pointer_role`：

- `pointer, pointer_role=payload_buffer` -> `ArgKind::Pointer` / `PointerRole::PayloadBuffer`：fuzzer-owned buffer，编码为 canonical padding 加 `[len:u64][bytes + filler : len]`；pointee type 不改变基础 wire format，只影响后续 element size、长度单位、对齐和 constraints/domain；
- `pointer, pointer_role=derived_pointer`：`base_arg + offset` alias/subview，不独立编码 buffer；Phase 3 v1 应 fail fast；
- `pointer, pointer_role=external_device_pointer`：runtime/harness metadata 提供的 device pointer，不来自 fuzz payload；Phase 3 v1 应 fail fast；
- `scalar` -> `ArgKind::Scalar`：整数、`float`、`double`、enum 等固定宽度值，按 `size_bytes` 写 little-endian；
- `opaque_val` -> `ArgKind::OpaqueVal`：pointer-free by-value struct/aggregate，按 `size_bytes` 固定长度对象字节处理；fixed array 是 inline aggregate，属于该类或下方的 `opaque_with_ptr`，不是 payload pointer；
- `opaque_with_ptr` -> `ArgKind::OpaqueWithPtr`：by-value struct/aggregate subtree 含 pointer，或无法证明 pointer-free；不能按顶层黑盒 bytes 处理，必须按 `type_layout` 下钻生成 input envelope；

Rust 可以用 `type` 对 `kind` 做一致性校验并输出诊断，但不应在缺失 `kind` 时重新猜测并继续 fuzz loop。
`pointer_role` 和 `pointee_layout` 是 `kind=pointer` 下的 pointer-only 字段；如果 pointer 参数缺失任一字段，或非 pointer 参数携带这些字段，应在 manifest load 阶段报错。

递归 `type_layout` 的字段节点使用字符串 `index` 标识位置，例如 `params.n`、`params.inner[].buf`。`args[*].index` 仍是顶层参数的 numeric ABI index；`type_layout.fields[*].index` 是嵌套 manifest index，二者处在不同层级，不复用类型。

`size_bytes` 为必填字段，语义是顶层 kernel 参数类型的 ABI `sizeof(arg_type)`：

- 标量参数：对应该标量类型的 `sizeof`；
- 指针参数：对应指针本身宽度，不是 pointee buffer 的长度；只有 `pointer_role=payload_buffer` 时，pointee buffer 的实际长度才在 `arg-pack-v1` 中由该指针参数段前置的 `[len:u64]` 表达，并可由后续 `domain.max_len` / kernel-level `constraints` 限制；
- by-value struct/aggregate：对应整个聚合类型的 `sizeof`。若字段布局无法完整提取，Phase 3 第一轮仍可按固定长度 opaque bytes 处理，但不能省略 `size_bytes`。

`domain` 是单参数/单字段约束，当前 schema 规定为以下 kind：

- `int_range`：整数闭区间，`min/max` 用十进制字符串表示，避免 JSON number 精度损失；
- `float_range`：浮点闭区间，可声明 `allow_nan`；
- `enum`：枚举值集合，枚举 discriminant 用字符串表示；ABI 大小仍由承载该 domain 的 arg/field 的 `size_bytes` 表达，不在 `domain` 内重复记录；
- `bytes`：`pointer_role=payload_buffer` 的 buffer `min_len/max_len/elem_size_bytes/nullable`。

`kernels[*].constraints` 是跨参数/跨字段关系，不放进某个参数的 `domain`。第一轮 schema 只定义小集合：

- `scalar_le_buffer_len`：标量值不超过某 `payload_buffer` pointer 长度；
- `scalar_compare_const`：标量 arg 或 nested scalar field 与整数常量比较；
- `scalar_compare_scalar`：两个标量 arg 或 nested scalar field 之间比较；
- `count_fits_buffer`：`count * elem_size_bytes <= buffer.len`。

Constraint path 字段使用相对 JSON segment array，例如
`["nested", "choice", "tag"]` 表示被引用 arg 下的
`.nested.choice.tag`；缺省 path 表示直接引用顶层 arg 本身。

`type_layout` 是可选递归下钻布局，只补充 aggregate 内部结构，不重复顶层 arg 的 `type/kind/size_bytes/align_bytes`。根节点只保留：

```json
"type_layout": {
  "layout_status": "complete | partial | opaque",
  "fields": []
}
```

字段节点的 `kind` 与顶层 arg 统一，取值为 `scalar | pointer | opaque_val | opaque_with_ptr`。字段可继续包含 `fields` 或 `element/element_count`，也可携带自己的 `domain`。Phase 3 对 `opaque_with_ptr` 不能继续按顶层 `size_bytes` 做黑盒 mutation；必须下钻到 leaf，按 leaf 的标准 envelope 规则编码。

### 3.2 当前支持的参数子集

当前实现支持：

- `pointer_role=payload_buffer` 的 pointer leaf raw byte buffer 编码（`kind=pointer`）；
- `scalar`，按 `size_bytes` little-endian 编码；
- `opaque_val`，按 `size_bytes` 固定宽度对象字节编码；
- `opaque_with_ptr + complete type_layout`，按 manifest field/array 顺序递归下钻到 leaf 后编码；
- nested field constraints，包括 `scalar_le_buffer_len(unit=bytes|elements)`、
  `scalar_compare_const`、`scalar_compare_scalar`、`count_fits_buffer`。

对不支持的参数类型：

- 启动时明确报错并拒绝进入 fuzz loop；
- 不允许 silent fallback。

### 3.3 `arg-pack-v1` 编码合同

沿用 `docs/input-envelope-v1.md`：

- 所有 length 字段为 `u64` little-endian，且位于 8 字节对齐 offset；
- `payload_buffer` pointer 的 `align_bytes` 约束作用在 payload 起始地址上；
  encoder 在 length field 前插入 canonical padding，使
  `len_offset + sizeof(uint64_t)` 满足该对齐；
- 只有 `pointer_role=payload_buffer` 参数编码为 canonical padding 加 `[len:u64][bytes + filler : len]`；
- 对 pointer 参数，manifest 的 `size_bytes` 只描述 ABI 指针宽度，不参与决定 buffer 段长度；encoder 只有在 `pointer_role=payload_buffer` 时才按 decode 合同写入 `[len:u64]`，再写入对应长度的 bytes/filler；
- payload 后的 filler bytes 计入该 pointer 的 encoded length；length field
  前的 canonical padding 由 decoder/packer 同步跳过；
- 参数起始偏移按 `align_bytes` 对齐；
- 标量参数按 little-endian 编码；
- `size <= input_len` 且 `size <= output_len` 等最小一致性约束必须被保证。

### 3.4 Canonical 语义

Phase 3 提交到 harness 的字节流必须是 canonical 的：

- 即使原始 bytes 已经“勉强可 parse”，也应在必要时重打包，确保同一 semantic args 只有一种稳定编码布局；
- 不再保留“parse 通过但布局非规范”的原始 payload。

### 3.5 复杂类型、跨参数约束与 domain 的后续契约

第一轮不要试图一次性完整理解所有 C/C++ 类型语义，但 manifest 必须区分“ABI 布局必需信息”和“可选语义提示”：

- `size_bytes/align_bytes` 是 ABI 必需信息，所有参数都必须有；
- struct/aggregate 字段布局若完整可得，可放入 `type_layout`，作为 field-aware mutator 的输入；`opaque_val` 可以按顶层 `size_bytes` 固定长度对象字节编码/变异；`opaque_with_ptr` 不能整体 black-box copy，若布局不可得或 child role 不支持，应启动阶段 fail fast；若连顶层 `sizeof/alignof` 都不可得，也应启动阶段 fail fast；
- 参数间关系不要靠 `input/output/size` 名字猜测，应在 manifest 中使用 kernel-level `constraints`，例如 `scalar_arg <= buffer_arg.len`、`scalar_compare_const`、`scalar_compare_scalar`、`count * elem_size <= buffer_len`；
- `domain` 更适合作为单参数值域/变异提示，例如整数 range、enum values、payload buffer 的 min/max length；payload buffer 的 max length 应在这里或统一配置中表达，而不是用特殊 `size_bytes` 值表达；涉及多个参数的一致性约束应放入 `constraints`，不要塞进单个 arg 的 `domain`；
- 整数 domain 的边界/枚举值建议用字符串承载，避免 JSON number 的 53-bit 精度问题。

当前策略是：mutation 在 arg-pack value 层执行，所有出口都通过 manifest-driven
normalize/repair；最终提交的输入必须 canonical 且满足已知 domain/constraints。

### 3.6 Phase 1 manifest 生成前置改造

Phase 3 依赖 manifest 作为编码真源，因此进入 Rust 侧实现前，Phase 1 的 manifest 生成链路需要先完成以下前置改造。

#### 3.6.1 必须先完成

1. `manifest.py` 规范化并写出 `args[*].type_layout`
   - 位置：`scripts/kernel-smoke/pipeline/manifest.py`
   - 当前 `_manifest_arg()` 已保留 `domain` / `type_info`，并会规范化上游传入的 `type_layout`；
   - 规则：root `type_layout` 不重复顶层 arg 的 `index/type/kind/size_bytes/align_bytes`，仅保留 `layout_status` 以及 `fields` 或 `element/element_count`；嵌套 pointer node 必须带 `pointer_role/pointee_layout`。

2. `manifest.py` 透传 `kernels[*].constraints`
   - 位置：`scripts/kernel-smoke/pipeline/manifest.py`
   - 当前 writer 会保留 upstream `constraints`；
   - 规则：constraints 是 kernel-level list；path 字段是相对 JSON segment array，例如 `["nested", "inner_count"]`。

3. 修正静态/手写 manifest 的 `domain` 格式
   - 旧格式：`{"kind":"range", "min":0}`；
   - 新格式：`{"kind":"int_range", "min":"0"}`；
   - 整数边界和 enum discriminant 必须用字符串承载，避免 JSON number 精度问题。

4. 确保所有 arg 都满足最小必填字段
   - `index/name/type/kind/size_bytes/align_bytes`；
   - pointer arg 还必须有 `pointer_role/pointee_layout`；
   - artifact 主路径应由 Clang helper 生成 `kind/size_bytes/align_bytes`；
   - 低层 source/AST 诊断 helper 可以在单元测试里推断简单字段，但 Phase 1 artifact manifest 不能依赖它作为类型真源。

#### 3.6.2 建议随后完成

1. Artifact helper fail-fast contract
   - production source-text/AST regex inference helper 已删除；
   - `scripts/kernel-smoke/pipeline/manifest.py` 现在要求 helper 显式写出
     `kind/size_bytes/align_bytes`，pointer 还必须写出 `pointer_role/pointee_layout`；
   - 缺少 `kind` 不再 fallback 到 spelling 表推断。

2. Clang helper 生成 `type_layout`
   - 位置：`scripts/kernel-smoke/tooling/clang_entry_metadata.cc`；
   - 建议分阶段实现：
     1. aggregate 参数先输出 `type_layout.layout_status = "opaque"` 或 `"partial"`；
     2. 能看到完整 `RecordDecl` 时输出直接字段；
     3. 再递归下钻 nested struct/array/union；
   - Phase 3 可直接处理 `opaque_val`；遇到 `opaque_with_ptr` 必须依赖完整 `type_layout` 做 field-aware materialization。

3. Clang helper 生成可确定的 `domain`
   - enum 参数/字段可从 `EnumDecl` 生成 `domain.kind = "enum"`；
   - range、pointer `max_len` 等通常不是 C++ 类型系统事实，不应无依据生成；可来自 annotation、sidecar config 或全局 fuzz 配置。

4. `constraints` 先只做透传，不做名字启发式
   - 普通签名无法可靠证明 `size <= input.len`；
   - 第一轮来源建议是人工 sidecar、annotation 或后续分析 pass；
   - 不建议仅根据 `input/output/size/count/n` 等名字自动生成约束。

#### 3.6.3 测试改造

- `tests/kernel_pipeline_smoke/drivers/test_manifest.py`
  - 检查 `kind/size_bytes/align_bytes` 必填；
  - 若存在 `domain`，检查新 `int_range/float_range/enum/bytes` 格式；
  - 若存在 `constraints`，检查为 kernel-level list，并覆盖
    `scalar_compare_const` / `scalar_compare_scalar` 这类 path-based
    scalar constraints；
  - 若存在 `type_layout`，检查至少有 `layout_status`。
- `tests/kernel_pipeline_smoke/drivers/test_type_resolution.py` 与 `test_print_type_defs.py`
  - 手写 manifest 样例补齐 `kind/size_bytes/align_bytes`；pointer 样例还必须补齐 `pointer_role/pointee_layout`。
- 若实现 Clang helper 的 `type_layout` / enum domain，补 helper 输出 golden 或 contract 测试。

## 4. 实施步骤

### M1：Manifest 加载与绑定

目标：

- 让 fuzzer 绑定“当前目标 `.so` 对应的 artifact manifest”。

建议：

- CLI 必须显式传入 manifest 参数：`<path_to_library> --manifest <path_to_manifest>`；
- 不读取 `RAPID_KERNEL_MANIFEST`，也不根据目标 `.so` 路径自动查找 `manifest.json`；
- 不再默认 silent fallback 到 `../cuda-kernel/kernelmanifest.json`。

验收：

- 启动时打印 manifest 路径 / kernel_id / args 概览；
- manifest 缺失或不支持时，启动阶段 fail fast。
- 未传 `--manifest` 时，启动阶段打印 usage 并退出，不进入 fuzz loop。

### M2：通用 ArgPackSpec 推导

目标：

- 从 `manifest.json` 推导 Rust 侧 `ArgPackSpec`。

要求：

- 不再写死 `input_align/output_align/size_align` 三元组；
- 不再假设 arg0/1/2 的语义一定是 input/output/size；
- legacy helper 可以保留“第一个 pointer 作为 input、第二个 pointer 作为 output、第一
  个 scalar 作为 size”的兼容 wrapper，但正常 fuzz loop 必须通过 manifest 推导
  arg-pack 结构，不能在提交路径散落固定索引语义。

### M3：通用编码器

目标：

- 实现真正的 manifest-driven `pack_arg_pack_v1()`。

要求：

- 按 `args` 顺序编码；
- 使用 `align_bytes`；
- length 字段固定为 `u64`；
- 标量类型按 little-endian 写入；
- 对 `size_t` 这类 ABI 相关标量，优先使用 manifest 的 `size_bytes` 而不是 Rust 本机默认宽度猜测。
- `size_bytes` 缺失或与当前 decode 合同不一致时必须 fail fast，不允许回退到 Rust 侧猜测。

### M4：严格 normalize / canonical repack

目标：

- 用“解析语义后重打包”的方式替换当前 heuristic normalize。

建议：

- `normalize_arg_pack_v1(raw)` 分两种情况：
  1. raw 已能按当前 target manifest 解析 -> 重新 pack 成 canonical bytes；
  2. raw 无法解析 -> 使用明确的 fallback 策略生成合法输入（短期可保留对半切，但必须记日志/计数）。

验收：

- 对同一 semantic args，normalize 前后字节流稳定可复现；
- 不再因为“根 manifest 和 artifact manifest 对齐不一致”而触发 `_rapid_invalid`。

### M5：默认 seed 生成

目标：

- 根据 manifest 自动生成合法 seed。

建议：

- `pointer_role=payload_buffer`：给一个小 non-empty buffer；
- `size_t` / 整数：取 domain 最小值、默认值或 `0`；
- 若存在 `input/output/size` 这类当前模式，再保持 `size <= input/output len`。

### M6：mutator 集成

目标：

- 让 mutation 之后的输入在提交前总能回到可 decode 状态。

当前保持：

- `ArgPackStructureMutator`：选择一个 manifest leaf 做结构化 mutation，然后重新应用
  domain/constraints 并 canonical repack；
- `ArgPackHavocMutator`：在 arg-pack value 层堆叠多次 scalar/buffer mutation，包括
  buffer chunk insert/delete，并经同一 domain/constraint repair 出口重打包；
- `VConfigMutator`：变异 manifest 允许的 logical launch shape；
- `ArgPackNormalizeMutator`：把 legacy/raw bytes 迁移为当前 manifest 的 canonical
  arg-pack；
- `RapidInputMutator`：每次从 payload structure、payload havoc、VConfig、structure +
  VConfig、havoc + VConfig 五个 plan 中等概率选择一个；关闭 VConfig mutation 时只在
  两种 payload plan 中选择。这里的 payload 是完整 kernel-argument payload，包括
  scalar、pointer buffer、opaque 和 nested leaves，而不是只指 buffer bytes；
- 每个 plan 最后都调用 `ArgPackNormalizeMutator`，统一修复 domain、跨参数 constraint、
  VConfig launch constraint 和 envelope `payload_size`；
- `RapidInputMutator::fixed()` 用于 `--no-mutate`：每个 stage iteration 都提交 manifest
  canonical full-envelope seed 并返回 `MutationResult::Mutated`，但不执行随机 payload、
  buffer-length 或 VConfig mutation。

但要求：

- `ArgPackNormalizeMutator` 使用当前目标绑定的 `ArgPackSpec`；
- sync / async 提交前的 normalize 必须和 mutator 阶段使用同一套 manifest-driven 逻辑。

### M7：sync / async 执行路径对齐

目标：

- `fuzzer` 和 `fuzzer_async` 提交的 bytes 完全一致地走 manifest-driven encode。

特别注意：

- async 路径必须把“真正提交给 GPU 的 normalized bytes”与 pending task 关联好，避免 crash / coverage attribution 指向 mutation 前的旧 bytes。

### M8：可观测性与失败语义

目标：

- 让 encode/decode failure 可见、可统计。

当前已有：

- loaded manifest path；
- loaded kernel summary；
- `normalize_calls`；
- `normalize_repack_count`；
- `fallback_repair_count`；
- `mutation_calls`；
- `seed_generation_count`；
- `payload_clamp_count`，表示 canonical pack 前根据 pointer bytes domain
  对 payload 长度做了截断或补齐。

后续建议计数：

- manifest_load_fail
- manifest_unsupported
- decode_contract_mismatch_count

fuzz 启动时已经打印 loaded manifest path / kernel summary；固定 payload 路径和
正常退出路径会打印 Phase3 arg-pack counters。

## 5. 风险与缓解

### 5.1 Manifest / artifact 漂移

风险：

- fuzzer 读取的 manifest 与目标 `.so` 实际 decode 合同不一致。

缓解：

- 强制按目标 `.so` 绑定 artifact manifest；
- 测试中覆盖“根 manifest != artifact manifest”场景。

### 5.2 类型系统过于理想化

风险：

- 试图一次支持全部 C/C++ 参数类型，导致 Phase 3 迟迟无法落地。

缓解：

- 当前 Phase 3 v1 支持 `pointer_role=payload_buffer` raw byte buffers、
  常见标量、`opaque_val` 固定宽度对象字节，以及
  `opaque_with_ptr + complete type_layout` 的递归 field-aware 编码；
- pointer 的 `pointee_layout` 是描述性 metadata，不会把 pointer leaf
  变成 inline aggregate；例如 `void *` 的 opaque pointee 不应阻止
  payload-buffer 编码；
- 非 `payload_buffer` pointer role、union/incomplete inline
  `opaque_with_ptr` 等不支持类型必须明确 fail fast，不做猜测。

### 5.3 非法输入处理不平滑

风险：

- 若 decode assert 不是显式 opt-in，默认 fuzz 路径仍会把 CUDA context 打坏。

缓解：

- 将 decode assert 收敛为显式 `RAPID_DECODE_DEBUG_ASSERT` 诊断模式，不再由默认 fuzz 构建隐式开启；
- 保持稳定接口不变。

### 5.4 async attribution 不一致

风险：

- 异步完成时记录的 input 不是最终提交给 GPU 的 canonical bytes。

缓解：

- pending task 中保存 normalized bytes 或其稳定拷贝。

## 6. 验收标准

- AT1：fuzzer 只能通过启动入口显式传入 `--manifest <path>` 加载目标 artifact 对应的 `manifest.json`，不再依赖仓库根 manifest silent fallback。
- AT2：当前 `vulnerable_kernel` 的生成 `.so` 在 sync / async 两条路径下，不再因 manifest 对齐不一致触发 `_rapid_invalid`。
- AT3：对当前支持的参数子集，fuzzer 提交到 harness 的输入 100% 可被 `fuzzer_decode_v1` 解析。
- AT4：默认 seed 由 manifest 自动生成，并可通过 decode。
- AT5：同一 manifest 下，origin / rapid / rapid2 的 decode 语义一致。
- AT6：decode failure 有计数/日志，且不会导致进程级崩溃。
- AT7：legacy corpus 输入会在执行前迁移/规范化到 canonical `arg-pack-v1`。
- AT8：async 路径记录的 pending/completed 输入与实际提交的 normalized bytes 一致。

## 7. 当前落地状态与下一步

已落地：

- Rust 侧 `KernelManifest` / `ArgSpec` / `ArgPackSpec` 模型；
- fuzzer 启动入口显式 manifest 初始化；缺少 `--manifest` 时 fail fast；
- manifest-driven `pack / parse / normalize / mutate`；
- sync / async 提交路径统一到同一 manifest-driven normalizer；
- recursive `opaque_with_ptr + complete type_layout` encoder；
- `--dump-seed`、`--runs N`、`--no-mutate`；其中 `--runs N` 限制当前
  LibAFL fuzzing mode 的 loop 次数，`--no-mutate --runs N` 用于 bounded
  canonical-seed smoke；
- `scripts/phase3_smoke.py` 可对一个 Phase1/2 run directory 中已经
  `phase2_status=built` 且已有 backend artifact 的 kernel 批量执行固定 payload
  `--no-mutate --runs N` smoke，并可用 `--build-fuzzer` 先构建对应的 sync/async fuzzer；
- `scripts/phase3_smoke.py --dump-seeds` 可在无 CUDA driver / 无 backend artifact
  环境中批量执行 `fuzzer --dump-seed`，验证 Phase3 能加载 built kernel 的
  manifest 并生成 canonical seed payload；
- payload bytes domain clamp 与 `payload_clamp_count`；
- Phase3 unit tests 与 no-GPU `.so` load smoke。

继续推进：

- 在 CUDA-capable GPU 上验证 origin / rapid / rapid2 与 Phase2 `.so` 的长时间执行；
- 补充 decode failure 计数；
- 完善 timeout 状态从 RAPID2 `TaskResult` 到 Rust `ExitKind` 的传递；
- richer scalar/domain support；
- 多 kernel / 更复杂参数类型。
