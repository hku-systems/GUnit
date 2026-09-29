# kernel_pipeline_rewrite

`tests/kernel_pipeline_rewrite/` 只保留 Phase 2（kernel-rewrite）端到端测试。

- `drivers/test_phase2_cli_e2e.py`
  - 以一个已经存在的 Phase 1 E2E `run_dir` 作为输入
  - 在测试实现里，这个输入目录通常由 `tests/kernel_pipeline_smoke/drivers/shared_e2e.py` 提供
  - Phase 2 只消费该 `run_dir` 并运行 `scripts/kernel-rewrite/cli.py`
  - 只断言 Phase 2 的最终阶段结果：`rewrite_summary.json` 与 per-kernel `metadata.phase2.json`
- `drivers/test_phase2_runtime_e2e.py`
  - 同样以一个已经存在的 Phase 1 E2E `run_dir` 作为输入，并先执行 Phase 2
  - 遍历 `fixtures/cases/` 下所有 checked-in runtime cases
  - 通过 `tests/kernel_pipeline_rewrite/drivers/shared_runtime.py` 自动完成：global wrapper 编译、rewrite 后 bitcode 链接、PTX 生成、CUDA driver API 运行
  - 验证最终生成链路可真实执行
- `drivers/shared_phase2.py`
  - 输入一个 Phase 1 `run_dir`
  - 运行 `scripts/kernel-rewrite/cli.py`
  - 返回统一的 Phase 2 最终结果视图，供 tests 只对最终状态做断言
- `drivers/shared_runtime.py`
  - 基于已有的 Phase 2 产物目录，执行完整 runtime smoke（wrapper -> bitcode -> PTX -> host driver run）
- `fixtures/`
  - 固定保存 runtime smoke 的 `wrapper.cu`、`host.cpp`、`Makefile`
  - `shared_runtime.py` 只向该夹具传入 Phase 2 产物路径和输出目录，不再临时拼接源码字符串

目录约束：

- 这里不放 Phase 2 的 unit test / 合同分类测试 / 中间态产物细节测试。
- Phase 2 的测试入口统一要求以 Phase 1 E2E 产物目录为输入。

## 如何运行

从仓库根目录执行：

```bash
PATH="$(pwd)/.venv/kernel-smoke/bin:$PATH" \
python3 -m unittest \
  tests.kernel_pipeline_rewrite.drivers.test_phase2_cli_e2e \
  tests.kernel_pipeline_rewrite.drivers.test_phase2_runtime_e2e
```

也可以直接手动跑两步：

```bash
PATH="$(pwd)/.venv/kernel-smoke/bin:$PATH" \
python3 scripts/kernel-rewrite/cli.py \
  --run-dir build/kernel-rewrite-demo-fixed/fixtures-all-e2e

PATH="$(pwd)/.venv/kernel-smoke/bin:$PATH" \
python3 tests/kernel_pipeline_rewrite/drivers/run_runtime_cases.py \
  --run-dir build/kernel-rewrite-demo-fixed/fixtures-all-e2e
```

说明：

- 当前 Phase 2 测试语义上要求“输入已经是一个可用的 Phase 1 E2E `run_dir`”；在测试实现里，这个输入通常由 `kernel_pipeline_smoke` 的共享 E2E helper 负责准备，因此本机仍需具备对应的 Phase 1 运行条件。
- `artifact` 路径依赖 `cxxfilt`，因此运行这组测试时需要把 `.venv/kernel-smoke/bin` 放进 `PATH`，确保共享 E2E 使用可用的 Python 环境。
- 推荐理解是：先有一个可用的 Phase 1 `run_dir`，再由 `shared_phase2.py` / `shared_runtime.py` 基于该目录继续做 Phase 2 E2E 断言；测试代码里如果调用 `shared_e2e.py`，那只是为了准备这个输入夹具。
- `fixtures/cases/` 下每个 case 都带 checked-in 的 `input_payload.bin` / `expected_payload.bin` / `case.json`，当前 runtime smoke 会逐个运行它们。
