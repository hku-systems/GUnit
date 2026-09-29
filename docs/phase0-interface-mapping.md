# Phase 0 Interface Mapping

本文件把当前 `origin / rapid / rapid2` 的已存在接口做基线对照，供 Phase 0 冻结协议时直接引用。

## 1) Exported C ABI（当前）

- `origin`
  - `extern "C" void libafl_target(uint8_t *input, size_t size)`
  - 参考：`cuda-kernel/origin/harness.cpp`

- `rapid`（有序完成异步接口；旧的 `libafl_target` / `libafl_is_running` / `libafl_queue_size` 同步接口已移除）
  - `extern "C" uint64_t libafl_submit_with_id(const uint8_t *input, size_t size)`
  - `extern "C" void libafl_set_target_timeout_ms(uint64_t timeout_ms)`
  - `extern "C" size_t libafl_poll_results(TaskResult *results, size_t max_count)`
  - `extern "C" size_t libafl_release_tasks(const uint64_t *task_ids, size_t count)`
  - `extern "C" void libafl_get_ordered_queue_counts(OrderedQueueCounts *out_counts)`
  - `extern "C" void libafl_wait()`
  - `extern "C" void libafl_stop()`
  - 参考：`cuda-kernel/rapid/harness.cpp`

- `rapid2` Phase2 backend
  - `extern "C" int rapid2_initialize(int num_kernels = 1, size_t max_input_size = MAX_INPUT_SIZE)`
  - `extern "C" void libafl_set_target_timeout_ms(uint64_t timeout_ms)`
  - `extern "C" uint64_t libafl_submit_with_id(const uint8_t *input, size_t size)`
  - `extern "C" void libafl_wait()`
  - `extern "C" void libafl_wait_for_completion()`
  - `extern "C" void libafl_stop()`
  - `extern "C" size_t libafl_poll_results(TaskResult *results, size_t max_count)`
  - `extern "C" size_t libafl_release_tasks(const uint64_t *task_ids, size_t count)`
  - `extern "C" void libafl_get_queue_counts(LibAflQueueCounts *out_counts)`
  - 参考：`cuda-kernel/rapid2/libafl_interface_shared.cuh`

## 2) `vulnerable_kernel` 调用点（当前）

- `origin`：`cuda-kernel/origin/wrapper.cu`
- `rapid`：`cuda-kernel/rapid/wrapper.cu`
- `rapid2`：`cuda-kernel/rapid2/pipelined_kernel_impl.cuh`

当前待测 kernel 是 device 形式：

- `__device__ void vulnerable_kernel(volatile uint8_t *input, volatile uint8_t *output, size_t size)`
- 参考：`cuda-kernel/kernel.cuh:8`, `cuda-kernel/kernel.cu:5`

## 3) Phase 0 冻结建议

- 统一上层输入仍是 `bytes`（`uint8_t* data, size_t size`）。
- `Input Envelope v1` 采用 raw bytes（不含 `kernel_id/schema_id/vdim` 头部字段）。
- decode 逻辑在 Phase 2 基于 `KernelManifest` 元数据与 rewrite plan 代码生成并编译进 harness（不在运行时动态解释 schema）。
- decode 规则统一，三后端语义一致。
- Phase 1 的 `kernel-smoke` 产物现在把 PTX `.entry` linkage symbol 当作 artifact 身份源头；
  `manifest.symbol_name` 应继续保持 exact symbol identity，供后续 `llvm-extract` / rewrite /
  decode 集成直接引用。
- Phase 1 的 artifact metadata 提取已经按“每个 PTX entry -> 一个实例化后参数列表”的方向收敛；
  Phase 2/Phase 3 不应再退回到基于短源码名的模糊匹配。

## 4) decode hook 推荐插入点（按当前修订）

- `rapid2` 主 decode 点（推荐）：`cuda-kernel/rapid2/pipelined_kernel_impl.cuh` 中 `fuzzer_invoke_v1` 调用前
  - 即在 persistent harness kernel 内、调用 `vulnerable_kernel` 前解析 `current_combined->data` 并形成 `(input, output, size)`
  - 采用 `arg-pack-v1` decode，按 `KernelManifest.args` 顺序解析：
    - scalar / by-value 参数直接按 fixed-size little-endian bytes；
    - 只有 `kind=pointer, pointer_role=payload_buffer` 的 fuzzer-owned buffer pointer 使用 canonical padding 加 `[len:u64][bytes + filler : len]`；pointer manifest 必须同时携带 `pointee_layout`，但 pointee layout 不改变 pointer leaf 自身的 envelope；
    - `pointer_role=derived_pointer` / `external_device_pointer` 不应被当作独立 payload buffer 编码，当前 v1 应 fail fast 或由后续 runtime relation/metadata 提供。
- `origin`/`rapid` 同样采用 `arg-pack-v1` decode：
  - `origin`：`cuda-kernel/origin/wrapper.cu` 中 `fuzzer_invoke_v1` 调用前完成解码
  - `rapid`：`cuda-kernel/rapid/wrapper.cu` 中 `fuzzer_invoke_v1` 调用前完成解码

说明：`vdim` 不从 `uint8_t` payload 解析，应走运行时 metadata 通道。

## 5) 与当前 Phase 1 工具链的对齐说明

- `scripts/kernel-smoke/` 的 artifact 模式现在是 PTX-first：
  - 先从 variant 级 `module.bc -> module.ptx` 提取 `.entry` symbols
  - 再用 Clang helper 为每个 entry 提取实例化后的 `args / line / qualified_name`，以及可选的扁平 `type_info` 类型定义元数据
  - 最后再按 exact symbol identity 生成单-kernel `kernel.bc / kernel.ptx / manifest`
- 因此，Phase 0/2 接口约束里最重要的一点仍然是：
  - linkage symbol 要稳定
  - decode/invoke 生成逻辑最终要绑定到 exact exported symbol，而不是依赖源码短名
