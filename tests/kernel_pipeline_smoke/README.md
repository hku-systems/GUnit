# kernel_pipeline_smoke

`tests/kernel_pipeline_smoke/` 是 kernel-smoke 测试入口目录，分为两部分：

- `drivers/`：Python `unittest` 驱动测试（`test_*.py`）
- `fixtures/`：CUDA fixture 源码、Makefile、期望符号数据

## 目录说明

- `drivers/test_emit_bc.py`
  - variant-resume 轻量哨兵测试
  - capture 模式主 E2E 断言（summary + manifest symbols）
  - helper-backed artifact 模式的模板实例化断言
- `drivers/test_manifest.py`
  - manifest/metadata/index 合约断言
- `drivers/test_demangled_filter.py`
  - `utils/demangle.py` 中 demangle/qualified-name 归一化 helper 的单测
- `drivers/test_fixtures_build.py`
  - fixtures Makefile 目标与产物断言（含 `.o/.so/.a`、runner、nm）
- `drivers/test_mini_project.py`
  - mini project 结构与 all-build 产物断言
- `drivers/shared_e2e.py`
  - 共享 E2E helper：一次 `make all` + 一次 `cli --capture-dir`，同进程内缓存复用

Phase 2 (`scripts/kernel-rewrite/`) 的测试已单独放到 `tests/kernel_pipeline_rewrite/`，避免和 Phase 1 `kernel-smoke` 驱动混在同一目录。

- `fixtures/Makefile`
  - `all`: 构建 `fixtures_runner` + `mini_project` + `lib_only`
  - `compile_fixtures`: 仅编译 fixture `.o`
  - `run`: 运行 `fixtures_runner`
  - `clean`: 清理产物
- `fixtures/complex_kernels.cu`
  - 混合场景 kernel（direct/macro/template/非 extern C）
- `fixtures/math_kernels.cu`
  - 算术类 kernel（add/mul/clamp/fma）
- `fixtures/ns_kernels.cu`
  - namespace + CUDA intrinsic 场景（验证发现与排除）
- `fixtures/cublas_host_api_smoke.cu`
  - 项目 kernel + 可选 cuBLAS host API 场景（验证不误识别库符号）
- `fixtures/template_instantiation_kernels.cu`
  - 模板 kernel 显式实例化场景（验证 artifact metadata 能区分实例化后的参数类型）
- `fixtures/lib_only_kernels.cu`
  - 专用于 `.so/.a` 产物路径的 kernel 源文件
- `fixtures/fixture_runner.cu`
  - 汇总调用各 wrapper 并输出 PASS/FAIL
- `fixtures/fixture_api.cuh`
  - 各 fixture wrapper 的 `extern "C"` 声明
- `fixtures/common_helpers.cuh`
  - 跨 fixture 共享的 device helper
- `fixtures/expected/symbols.json`
  - artifact/helper E2E 测试使用的期望 symbol 列表
- `fixtures/project/`
  - mini-project 子工程（`main.cu`/`kernels.cu`/`Makefile`）

## 如何运行

从仓库根目录执行。

### 1) 跑全部 drivers（推荐）

```bash
python3 -m unittest discover -s tests/kernel_pipeline_smoke/drivers -p "test_*.py"
```

说明：不需要你先手工跑一次 E2E。`drivers/shared_e2e.py` 会在首次被调用时自动执行一次
`make all + cli --capture-dir`，并在同一进程内复用结果。

如果当前系统 `python3` 环境缺少 `cxxfilt`，建议这样运行：

```bash
PATH="$(pwd)/.venv/kernel-smoke/bin:$PATH" \
python3 -m unittest discover -s tests/kernel_pipeline_smoke/drivers -p "test_*.py"
```

### 2) 从 E2E 主路径开始（调试用，非推荐）

先跑共享 E2E 主测试：

```bash
python3 -m unittest tests.kernel_pipeline_smoke.drivers.test_emit_bc.EmitAndResumeTest.test_capture_mode_e2e_summary_and_symbols
```

再跑基于共享产物的断言：

```bash
python3 -m unittest tests.kernel_pipeline_smoke.drivers.test_manifest
python3 -m unittest tests.kernel_pipeline_smoke.drivers.test_fixtures_build
python3 -m unittest tests.kernel_pipeline_smoke.drivers.test_mini_project
```

## 共享 E2E 的行为

`drivers/shared_e2e.py` 会在首次调用时执行：

1. 选择 `scripts/kernel-smoke/` 下与真实编译器同名的 wrapper 入口（例如 `nvcc` / `clang++`）
2. `make -C tests/kernel_pipeline_smoke/fixtures all`
3. 运行 `python3 scripts/kernel-smoke/cli.py run --capture-dir ... --mode artifact`

当前测试路径会直接使用 `scripts/kernel-smoke/` 里的同名 wrapper 入口（例如 `nvcc` / `clang++`），
这些入口都是指向同一个 `compiler-dispatch.sh` 的 symlink；dispatcher 会按入口名解析真实编译器，
再前置 `rapid-wrap` 完成 capture。

`shared_e2e.py` 在同一 Python 进程内使用模块级缓存（`_CTX`）：

- 首次调用 `get_capture_e2e_context()` 才会真正执行一次 `make all` 和 `cli --capture-dir`
- 后续测试再次调用时直接复用同一个上下文，不会重复编译
- 进程退出时通过 `atexit` 自动执行 `make clean` 并删除临时目录

注意：如果你分多条 `python3 -m unittest ...` 命令运行，每条命令都是新进程，缓存不会共享，
会各自重新编译一次。这也是本 README 推荐使用单条 `unittest discover` 的原因。

## 依赖

- Python 3
- `make`
- CUDA 编译器（`clang++` 或 `nvcc`，否则相关测试会 `skip`）
- 可选：`llvm-nm`/`nm`（符号检查测试）
- `artifact` E2E 还会按需构建 `scripts/kernel-smoke/tooling/clang_entry_metadata.cc`；helper discovery 优先使用 `KSMOKE_CLANG_HELPER_BIN`，否则从 `PATH` 中查找 `clang++`/`llvm-config` 并回退扫描常见 LLVM 安装目录。
