# Rapid CUDA Fuzzer - Claude 协作指南

> **重要**: 请先阅读 `AGENTS.md` 了解项目架构、开发哲学和技术细节。
> 本文件专注于定义 Claude 与 Codex CLI 的协作流程。

## 工作流程

本项目采用 **Claude 规划 + Codex CLI 执行** 的协作模式：

### 1. 复杂任务处理流程

对于非平凡的实现任务（3步以上或涉及多个文件），遵循以下流程：

1. **Claude 进入 Plan 模式**
   - 使用 `EnterPlanMode` 工具
   - 探索相关代码库和架构
   - 理解现有模式和约定
   - 制定详细的实现计划

2. **用户审批计划**
   - 呈现计划给用户
   - 等待用户确认或调整

3. **用户使用 Codex CLI 执行编码**
   - **直接在终端运行**: `codex "<任务描述>"`
   - **传递计划文件**: `codex "按照 <plan-file.md> 执行实现"`
   - **模型配置**: 默认值取决于本地配置和账户，可通过 `--model` 覆盖
   - Codex CLI 的主机、GPU 和网络访问取决于本地 sandbox 与 approval
     配置；硬件测试前应先确认当前权限。

4. **Claude 验证和审查**
   - 用户告知 Codex 完成后，Claude 检查输出
   - 运行测试验证功能
   - 必要时进行调整或建议下一步

### 2. 何时使用 Codex CLI

主动建议用户使用 Codex CLI 的场景：
- 实现计划中的具体编码步骤（3+ 文件修改）
- 重构大量代码
- 需要深入调试的问题
- **需要 GPU 或特殊硬件访问的测试**
- **需要长时间运行的集成测试**
- 需要"第二意见"的实现方案

### 3. 何时 Claude 自己处理

直接处理的场景：
- 简单的单文件修改
- 明显的 bug 修复
- 配置文件调整
- 文档更新
- 快速原型验证

## 与 AGENTS.md 的关系

- **AGENTS.md**: 项目架构、技术栈、API 细节、开发哲学（必读）
- **CLAUDE.md** (本文件): Claude ↔ Codex CLI 协作流程和工作方式

两个文件互补，都是会话开始时的必读文档。

## Codex CLI 使用指南

### 基本命令

```bash
# 简单任务（前台执行，默认模型）
codex "修复 fuzzer_async.rs 中的编译错误"

# 复杂任务：显式选择模型和推理级别
codex --model gpt-5.6-sol -c 'model_reasoning_effort="high"' \
  "按照 .claude/plans/xxx.md 实现所有任务"

# 恢复会话
codex resume <session-id>
```

如需持久化配置，在 `~/.codex/config.toml` 中设置：

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "high"
```

## 遵循 AGENTS.md 的开发哲学

在规划和执行时，始终遵循 `AGENTS.md` 中的开发哲学：
- 不要猜测 API，仔细阅读文档
- 不要模糊执行，寻求明确和确认
- 不要跳过验证，主动测试
- 不要破坏架构，遵循标准和约定
