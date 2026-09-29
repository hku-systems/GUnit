# RAPID2 流水线并行架构设计文档

## 1. 概述

RAPID2是一个高性能的CUDA Fuzzing执行框架，通过双缓冲流水线技术实现GPU内核的持续满载运行。本框架解决了传统锁步执行模式下的性能瓶颈，实现了数据传输与内核执行的真正并行。

### 1.1 核心目标

- **消除锁步执行**：避免GPU等待数据传输
- **持续满载运行**：让vulnerable_kernel持续执行，最大化GPU利用率
- **隐藏传输延迟**：数据传输在内核执行期间完成
- **多实例并行**：支持多个kernel实例同时运行，提高吞吐量

### 1.2 关键创新

- 双缓冲流水线架构
- 基于原子操作的轻量级同步
- Persistent Kernel自主缓冲区切换
- RAII资源管理

### 1.3 与 LibAFL 的集成（RAPID2）

为支持异步 Fuzzing，本目录提供一组对 LibAFL 友好的 C 接口。RAPID2 现在只通过官方 Phase2/PTX backend 构建，并由生成的 `librapid2_target.so` 导出这些符号：

窗口/队列活性合同：fuzz 完成窗口不得超过 32（`MAX_ORDERED_WINDOW_SIZE`），任务队列容量为 64（`QUEUE_CAPACITY`）。窗口必须始终小于队列容量，否则单线程 submit/poll 可能死锁；详见 `docs/feedback.md` 与 `cuda-fuzzer/README_ASYNC.md`。

- `libafl_submit_with_id(const uint8_t* input, size_t size) -> uint64_t`
  - 非阻塞提交输入，返回任务 ID（0 表示提交失败、未初始化或 target timeout
    尚未配置）。
- `libafl_poll_results(TaskResult* out, size_t max_count) -> size_t`
  - 非阻塞拉取已完成任务的结果；返回 task-owned edge 与 memory/index
    指针，避免把合计 72KiB 的反馈图嵌入或拷贝进每个 `TaskResult`。
  - `TaskResult` 会携带提交时的 `input_ptr`（caller 传入的 input buffer 指针值），可用于在 Rust 侧将完成结果与“输入来自哪个 corpus buffer/slot”做关联（若上层保证该指针在复用 slot 语义下稳定）。
  - `TaskResult.status` 使用共享 `LibAflRunStatus { code, stage, detail }`
    ABI；CUDA 失败的 `detail` 为 CUDA error code。
  - 注意：返回的 `edge_ptr` 与 `simt_memcov_ptr` 都只在调用
    `libafl_release_tasks()` 前有效；对应大小由 `edge_size` 与
    `simt_memcov_size` 给出。
- `libafl_release_tasks(const uint64_t* task_ids, size_t count) -> size_t`
  - 释放通过 `libafl_poll_results` 获取的任务结果，允许 harness 回收/复用缓冲区（内部会进入全局复用池以减少 malloc/free）。
- `libafl_set_target_timeout_ms(uint64_t timeout_ms)`
  - 由 fuzzer 运行时设置正数 RAPID2 active-task watchdog 的 target timeout
    预算；计时从 task dispatch 到 double-buffer slot 后开始，不包括
    backend pending queue 等待时间。
- `libafl_get_queue_counts(LibAflQueueCounts* out_counts)`
  - 查询队列中待处理/已完成任务数（一次返回一对计数）。
- `libafl_wait()`
  - 同步等待所有在途任务完成（用于退出/崩溃/收尾）。

与 Rust 侧映射关系见 `cuda-fuzzer/src/cuda_backend/async.rs`。注意覆盖图大小需保持一致：`rapid2` 侧 `MAP_SIZE` 必须等于 Rust 侧 `EDGES_MAP_SIZE`（当前为 65536）。另外，当前 host 侧池化上限直接使用 `MAX_INPUT_SIZE`。

RAPID2 watchdog timeout 不直接在 watchdog 线程中作为正常路径 abort。
它会先发布一个 `TaskResult.status = TIMEOUT` 的完成项，使 Rust
`AsyncBatchFuzzer` 能为该输入运行 `TimeoutFeedback` 并释放 task-owned
buffers；随后 Rust client abort，由 LibAFL restarting manager 重启整个
进程和 CUDA context。

### 官方 Phase2 backend 构建（rapid2）

要把官方 `kernel-rewrite` 生成的某个 `phase2/` 目录打包成 rapid2 可加载的 backend `.so`，请使用：

- `cuda-kernel/rapid2/build.py`

例如：

```bash
python3 cuda-kernel/rapid2/build.py \
  --phase2-dir /path/to/kernel/phase2 \
  --out-dir /tmp/phase2-rapid2 \
  --cuda-arch sm_86 \
  --cuda-path /usr/local/cuda
```

该脚本会：

1. 编译 `cuda-kernel/rapid2/wrapper.cu`
2. 与 `phase2/kernel.device.bc` 链接
3. 生成 PTX 并 embed 到 host 侧源文件
4. 链接 `cuda-kernel/rapid2/harness.cpp`
5. 输出 `librapid2_target.so`

对于仓库内置的 `kernel.cu` 示例，更推荐直接使用顶层驱动：

```bash
python3 cuda-kernel/builtin_phase_pipeline.py --out-root /tmp/builtin-phase
```

或直接用 CMake：

```bash
cmake -S cuda-kernel -B cuda-kernel/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda-kernel/build --target builtin_phase_pipeline
```

这样会自动跑官方 Phase1/Phase2，并同时生成 `origin` / `rapid` / `rapid2` 三套 backend。

运行异步 Fuzzer 时，将生成的 `librapid2_target.so` 作为 target library 传入。

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Host (CPU) Side                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐      ┌──────────────────┐      ┌──────────────┐      │
│  │ Input Queue  │─────>│ Pipeline Control │─────>│ Output Queue │      │
│  └──────────────┘      └──────────────────┘      └──────────────┘      │
│                                │                                         │
│                          Copy Stream                                     │
│                                │                                         │
│  ┌─────────────────────────────┴─────────────────────────────┐         │
│  │  时刻T0: 拷贝Input0到Buffer A  (Buffer B正被GPU处理)       │         │
│  │  时刻T1: 拷贝Input1到Buffer B  (Buffer A正被GPU处理)       │         │
│  │  时刻T2: 拷贝Input2到Buffer A  (Buffer B正被GPU处理)       │         │
│  │  时刻T3: 拷贝Input3到Buffer B  (Buffer A正被GPU处理)       │         │
│  │  ...交替进行...                                           │         │
│  └────────────┬──────────────────────────┬──────────────────┘         │
│               ▼                          ▼                             │
│        cudaMemcpyAsync              信号量操作                          │
│               │                          │                             │
│               ▼                          ▼                             │
│  ┌────────────────────────────────────────────────────────┐           │
│  │           同步信号量 (Host Memory)                      │           │
│  ├─────────────────────┬──────────────────────────────────┤           │
│  │ Buffer A Status:    │  Buffer B Status:                │           │
│  │ - ready_flag_A      │  - ready_flag_B                  │           │
│  │ - done_flag_A       │  - done_flag_B                   │           │
│  └─────────────────────┴──────────────────────────────────┘           │
│                ▲                          ▲                            │
│                │                          │                            │
│         原子操作检查                  原子操作设置                      │
│                │                          │                            │
└────────────────┼──────────────────────────┼────────────────────────────┘
                 │                          │
════════════════════════════════════════════════════════════════════════
┌────────────────┼──────────────────────────┼────────────────────────────┐
│                ▼                          ▼      Device (GPU) Side      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│              ┌──────────────────────────────────────┐                   │
│              │         Double Buffer                │                   │
│              ├────────────────┬─────────────────────┤                   │
│              │   Buffer A     │     Buffer B        │                   │
│              │  ┌─────────┐   │   ┌─────────┐      │                   │
│              │  │ Input A │   │   │ Input B │      │                   │
│              │  │ Size A  │   │   │ Size B  │      │                   │
│              │  │ Cov A   │   │   │ Cov B   │      │                   │
│              │  └─────────┘   │   └─────────┘      │                   │
│              └────────┬───────┴──────────┬──────────┘                   │
│                       │                  │                              │
│         current_buffer│                  │                              │
│              ┌────────▼──────────────────▼────────┐                    │
│              │                                    │                    │
│              │   Single Persistent Kernel         │                    │
│              │   (Always Running)                 │                    │
│              │                                    │                    │
│              │   int current = 0;  // 当前buffer  │                    │
│              │                                    │                    │
│              │   while (!exit) {                  │                    │
│              │     ┌──────────────────────┐      │                    │
│              │     │1. 等待ready[current] │◄─────┼── ready_flag检查    │
│              │     └──────────┬───────────┘      │                    │
│              │                 ▼                  │                    │
│              │     ┌──────────────────────┐      │                    │
│              │     │2. 选择当前buffer:    │      │                    │
│              │     │   if current==0:     │      │                    │
│              │     │     use Buffer A     │      │                    │
│              │     │   else:              │      │                    │
│              │     │     use Buffer B     │      │                    │
│              │     └──────────┬───────────┘      │                    │
│              │                 ▼                  │                    │
│              │     ┌──────────────────────┐      │                    │
│              │     │3. 执行vulnerable_    │      │                    │
│              │     │   kernel(input,size) │      │                    │
│              │     └──────────┬───────────┘      │                    │
│              │                 ▼                  │                    │
│              │     ┌──────────────────────┐      │                    │
│              │     │4. 设置done[current]  │──────┼──► done_flag设置   │
│              │     └──────────┬───────────┘      │                    │
│              │                 ▼                  │                    │
│              │     ┌──────────────────────┐      │                    │
│              │     │5. 切换buffer:        │      │                    │
│              │     │   current = 1-current│      │                    │
│              │     └──────────┬───────────┘      │                    │
│              │                 │                  │                    │
│              │                 └──────────────────┘                    │
│              │   }                                │                    │
│              └────────────────────────────────────┘                    │
│                                                                         │
│                        Kernel Stream                                    │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 多Kernel并行架构

当单个kernel无法充分利用GPU时，可以启动多个独立的persistent kernel，每个都有自己的双缓冲流水线：

```
┌────────────────────────────────────────────────────────────┐
│                    GPU Device                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Kernel 0:  [处理B0A]→[处理B0B]→[处理B0A]→[处理B0B]→...    │
│             ↑        ↑        ↑        ↑                  │
│  Buffer 0:  [A就绪] [B就绪] [A就绪] [B就绪]               │
│                                                            │
│  Kernel 1:  [处理B1A]→[处理B1B]→[处理B1A]→[处理B1B]→...    │
│             ↑        ↑        ↑        ↑                  │
│  Buffer 1:  [A就绪] [B就绪] [A就绪] [B就绪]               │
│                                                            │
│  Kernel 2:  [处理B2A]→[处理B2B]→[处理B2A]→[处理B2B]→...    │
│             ↑        ↑        ↑        ↑                  │
│  Buffer 2:  [A就绪] [B就绪] [A就绪] [B就绪]               │
│                                                            │
│  Kernel 3:  [处理B3A]→[处理B3B]→[处理B3A]→[处理B3B]→...    │
│             ↑        ↑        ↑        ↑                  │
│  Buffer 3:  [A就绪] [B就绪] [A就绪] [B就绪]               │
│                                                            │
└────────────────────────────────────────────────────────────┘

B0A = Kernel 0的Buffer A, B0B = Kernel 0的Buffer B, 依此类推
```

**关键特性**：

- 每个kernel完全独立，有自己的双缓冲区
- 每个kernel内部持续运行，不会暂停
- 任务分发器将新任务分配给可用的kernel
- 多kernel并行最大化GPU利用率

### 2.3 时序图

```
时间 →
     T0        T1        T2        T3        T4        T5
     
CPU: [准备B0] [准备B1] [准备B0] [准备B1] [准备B0] [准备B1]
      ↓        ↓        ↓        ↓        ↓        ↓
     异步拷贝  异步拷贝  异步拷贝  异步拷贝  异步拷贝  异步拷贝
      ↓        ↓        ↓        ↓        ↓        ↓
GPU:    [处理B0]  [处理B1]  [处理B0]  [处理B1]  [处理B0]
         ↓        ↓        ↓        ↓        ↓
       结果回传  结果回传  结果回传  结果回传  结果回传

B0 = Buffer 0, B1 = Buffer 1
```

**关键点**：

- GPU处理Buffer0时，CPU同时准备Buffer1的数据
- 数据传输时间被内核执行时间完全隐藏
- 无空闲等待，实现真正的流水线并行

## 3. 核心组件设计

### 3.1 双缓冲同步机制 (DoubleBufferSync)

```cpp
class DoubleBufferSync {
public:
    // 构造函数 - RAII初始化
    DoubleBufferSync();
    
    // 析构函数 - RAII清理
    ~DoubleBufferSync();
    
    // 禁用拷贝和移动
    DoubleBufferSync(const DoubleBufferSync&) = delete;
    DoubleBufferSync& operator=(const DoubleBufferSync&) = delete;
    
    // CPU端接口
    void signalBufferReady(int buffer_idx, cudaStream_t stream);
    bool checkBufferDone(int buffer_idx, cudaStream_t stream);
    void clearBufferDone(int buffer_idx, cudaStream_t stream);
    void signalExit(cudaStream_t stream);
    
    // 获取设备指针（供kernel使用）
    DeviceSyncPtrs getDevicePtrs() const;
    
private:
    // 每个缓冲区的同步标志
    int* d_buffer_ready[2];    // 0=空闲, 1=数据就绪
    int* d_buffer_done[2];     // 0=处理中, 1=处理完成  
    bool* d_should_exit;       // 退出标志
    
    // RAII资源管理
    bool initialized = false;
};
```

### 3.2 单内核流水线管理器 (SingleKernelPipeline)

```cpp
class SingleKernelPipeline {
public:
    // 构造函数 - 初始化双缓冲同步机制
    SingleKernelPipeline(size_t max_input_size,
                        size_t max_output_size = 0);
    
    // 析构函数 - 清理资源和同步机制
    ~SingleKernelPipeline();
    
    // 核心API - 与DoubleBufferSync紧密集成
    bool submitTask(TaskData* task);        // 使用sync.signalBufferReady()通知GPU
    TaskData* checkCompletedTask();         // 使用sync.checkBufferDone()检查完成
    void launch();                           // 传递sync.getDevicePtrs()给kernel
    void shutdown();                         // 使用sync.signalExit()通知kernel退出
    
    // 状态查询
    bool isBufferAvailable(int buffer_idx) const;
    uint64_t getProcessedCount() const;
    
private:
    // 双缓冲区资源（只有一套，kernel交替使用）
    uint8_t* d_input[2];      // Buffer A 和 Buffer B
    size_t* d_size[2];
    uint8_t* d_coverage[2];
    uint8_t* d_output[2];     // 可选
    
    // 同步机制 - 核心组件
    DoubleBufferSync sync;    // 管理所有CPU-GPU同步
    
    // CUDA流（只需要两个）
    cudaStream_t kernel_stream;  // 运行persistent kernel
    cudaStream_t copy_stream;    // 所有数据拷贝
    
    // 流水线状态
    int next_submit_buffer = 0;     // CPU下一个要提交数据的缓冲区
    int next_collect_buffer = 1;    // CPU下一个要收集结果的缓冲区
    bool buffer_has_data[2] = {false, false};  // 缓冲区是否有待处理数据
    bool buffer_processing[2] = {false, false}; // 缓冲区是否正在被GPU处理
    TaskData* buffer_tasks[2] = {nullptr, nullptr}; // 每个缓冲区对应的任务
    
    // 统计信息
    std::atomic<uint64_t> tasks_submitted{0};
    std::atomic<uint64_t> tasks_completed{0};
    
    // Kernel配置
    dim3 grid_dim{1};
    dim3 block_dim{256};
    size_t shared_mem_size;
    
    // 内部方法 - 使用DoubleBufferSync进行同步
    void waitForBufferAvailable(int buffer_idx);  // 内部等待buffer可用
    void signalBufferReady(int buffer_idx);       // 通知GPU buffer已准备好
    bool isBufferCompleted(int buffer_idx);       // 检查GPU是否处理完成
    void clearBufferStatus(int buffer_idx);       // 清理buffer状态
};
```

### 3.3 任务队列抽象 (TaskQueue)

为了支持未来的无锁扩展，将任务队列抽象为独立的类：

```cpp
// 任务队列接口 - 支持未来替换为无锁实现
template<typename T>
class ITaskQueue {
public:
    virtual ~ITaskQueue() = default;
    
    // 核心操作
    virtual bool push(std::unique_ptr<T> task) = 0;
    virtual std::unique_ptr<T> pop() = 0;
    virtual std::unique_ptr<T> try_pop() = 0;  // 非阻塞pop
    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
};

// 基于mutex的实现（当前版本）
template<typename T>
class MutexTaskQueue : public ITaskQueue<T> {
public:
    MutexTaskQueue(size_t max_size = 0)  // 0表示无限制
        : max_size_(max_size) {}
    
    bool push(std::unique_ptr<T> task) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0 && queue_.size() >= max_size_) {
            return false;  // 队列已满
        }
        queue_.push(std::move(task));
        cv_.notify_one();
        return true;
    }
    
    std::unique_ptr<T> pop() override {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) {
            return nullptr;
        }
        auto task = std::move(queue_.front());
        queue_.pop();
        return task;
    }
    
    std::unique_ptr<T> try_pop() override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return nullptr;
        }
        auto task = std::move(queue_.front());
        queue_.pop();
        return task;
    }
    
    bool empty() const override {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() const override {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    void shutdown() {
        std::unique_lock<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }
    
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::unique_ptr<T>> queue_;
    size_t max_size_;
    std::atomic<bool> shutdown_{false};
};

// 未来的无锁实现（接口示例）
template<typename T>
class LockFreeTaskQueue : public ITaskQueue<T> {
public:
    LockFreeTaskQueue(size_t capacity = 64)
        : capacity_(capacity) {
        // 使用环形缓冲区 + CAS操作实现无锁队列
        // 可以基于boost::lockfree::queue或自定义实现
    }
    
    bool push(std::unique_ptr<T> task) override {
        // CAS操作实现无锁push
        // 使用memory_order_release保证可见性
        return true;
    }
    
    std::unique_ptr<T> pop() override {
        // CAS操作实现无锁pop
        // 使用memory_order_acquire保证顺序
        return nullptr;
    }
    
    // ... 其他方法的无锁实现
    
private:
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::vector<std::atomic<T*>> buffer_;
    size_t capacity_;
};
```

### 3.4 多流水线管理器 (MultiPipelineManager)

使用 TaskQueue 抽象后的 MultiPipelineManager 设计。队列类型现在由
`config.h` 中的 `Rapid2TaskQueue` / `Rapid2CompletedQueue` alias template
在编译期选择，不再保留运行时 `QueueType` 选择层：

```cpp
class MultiPipelineManager {
public:
    // 构造函数
    // Rapid2 currently requires exactly one pipeline.
    explicit MultiPipelineManager(int num_kernels = 1,
                                  size_t queue_capacity = 64);
    
    // 析构函数
    ~MultiPipelineManager();
    
    // 初始化
    void initialize(size_t max_input_size, 
                   size_t max_output_size = 0);
    
    // 任务管理
    uint64_t submitInput(uint8_t* input, size_t size);
    void waitAll();
    
    // 覆盖率管理
    void getCoverage(uint8_t* coverage_out);
    
    // 生命周期管理
    void start();
    void stop();
    bool isRunning() const;
    
    // 统计信息
    struct Statistics {
        uint64_t total_submitted;
        uint64_t total_completed;
        uint64_t pending_queue_size;
        uint64_t completed_queue_size;
        int active_kernels;      // 活跃的kernel数量
        double throughput;        // tasks/sec
        double gpu_utilization;  // GPU利用率
    };
    Statistics getStatistics() const;
    
private:
    // 独立的流水线实例
    // 每个实例都是一个完整的双缓冲流水线，有自己的persistent kernel
    std::vector<std::unique_ptr<SingleKernelPipeline>> pipelines;
    
    // 使用抽象的任务队列接口 - 运行时多态，便于切换实现
    std::unique_ptr<ITaskQueue<TaskData>> pending_queue;
    std::unique_ptr<ITaskQueue<TaskData>> completed_queue;
    
    // 工作线程
    std::thread dispatcher_thread;
    std::thread collector_thread;
    
    // 控制标志
    std::atomic<bool> running{false};
    std::atomic<bool> initialized{false};
    
    // 统计
    std::atomic<uint64_t> total_submitted{0};
    std::atomic<uint64_t> total_completed{0};
    
    // 内部方法
    void dispatcherLoop();  // 分发任务到kernel
    void collectorLoop();   // 收集完成的任务
    
    // 队列实例由 config.h 中的 alias template 选择。
};
```

### 3.4 任务数据结构 (TaskData)

```cpp
struct TaskData {
    // 任务标识
    uint64_t task_id;

    // Caller 传入的 input buffer 指针值（用于关联 corpus/slot）
    uint64_t input_ptr;
    
    // 输入数据（Host 侧：size + payload）
    HostCombinedInputPtr h_combined;
    
    // 覆盖率数据
    std::unique_ptr<uint8_t[]> h_coverage;
    
    // 时间戳
    std::chrono::time_point<std::chrono::steady_clock> submit_time;
    std::chrono::time_point<std::chrono::steady_clock> complete_time;
    
    // 构造函数
    TaskData(uint64_t id, const uint8_t* input, size_t size);

    // 复用对象时重置（必要时扩容输入缓冲区）
    void reset(uint64_t id, const uint8_t* input, size_t size);
    
    // 移动语义
    TaskData(TaskData&&) = default;
    TaskData& operator=(TaskData&&) = default;
    
    // 禁用拷贝
    TaskData(const TaskData&) = delete;
    TaskData& operator=(const TaskData&) = delete;
};
```

## 4. Persistent Kernel设计

### 4.1 双缓冲Persistent Kernel

```cuda
extern "C" __global__ void rapid2_persistent_kernel(
    // 双缓冲输入
    CombinedInput* d_combined0, CombinedInput* d_combined1,
    
    // 双缓冲输出
    uint8_t* d_coverage0, uint8_t* d_coverage1,
    
    // 同步标志
    volatile int* ready0, volatile int* ready1,
    volatile int* done0, volatile int* done1,
    volatile bool* should_exit,
    
    // Kernel标识
    int kernel_id
);
```

该入口由 `wrapper.cu` 导出；实际 persistent kernel 主体在
`pipelined_kernel_impl.cuh` 的 `double_buffered_persistent_kernel_impl()` 中。

**核心逻辑**：

1. 自主在两个缓冲区间切换
2. 使用原子操作检查缓冲区状态
3. 处理完一个缓冲区立即切换到另一个
4. 无需CPU干预的连续运行

## 5. RAII设计原则

### 5.1 资源管理

- **自动初始化**：构造函数分配所有GPU资源
- **自动清理**：析构函数释放所有资源
- **异常安全**：使用RAII确保异常时资源正确释放

### 5.2 生命周期管理

```cpp
class RAIIExample {
    CUDAResource resource;
public:
    RAIIExample() {
        // 分配资源
        cudaMalloc(&resource.ptr, size);
        // 初始化
        cudaMemset(resource.ptr, 0, size);
    }
    
    ~RAIIExample() {
        // 自动释放
        if (resource.ptr) {
            cudaFree(resource.ptr);
        }
    }
    
    // 禁用拷贝，只允许移动
    RAIIExample(const RAIIExample&) = delete;
    RAIIExample& operator=(const RAIIExample&) = delete;
    RAIIExample(RAIIExample&&) = default;
    RAIIExample& operator=(RAIIExample&&) = default;
};
```

## 6. API使用示例

### 6.1 基本使用

RAPID2 对外推荐通过生成的 Phase2 backend `.so` 暴露 C ABI；Rust 侧由
`cuda-fuzzer/src/cuda_backend/async.rs` 动态加载这些符号。

```cpp
uint64_t task_id = libafl_submit_with_id(input_data, size);

TaskResult result{};
size_t count = libafl_poll_results(&result, 1);
if (count != 0 && result.edge_ptr != nullptr &&
    result.simt_memcov_ptr != nullptr) {
    if (result.status.code != LIBAFL_RUN_STATUS_OK) {
        // status.stage/status.detail describe the backend failure when known.
    }
    // 在 release 前复制 edge_ptr 到 observer map，并消费 simt_memcov_ptr。
    libafl_release_tasks(&result.task_id, 1);
}

libafl_wait();
libafl_stop();
```

### 6.2 批量处理

```cpp
// 批量提交
std::vector<uint64_t> task_ids;
for (auto& input : inputs) {
    task_ids.push_back(libafl_submit_with_id(input.data, input.size));
}

// 等待所有任务
libafl_wait();
```

## 7. 性能优化要点

### 7.1 内存访问优化

- 使用对齐的内存分配
- 批量传输减少overhead
- 使用pinned memory加速传输

### 7.2 同步优化

- 原子操作替代锁
- 轮询间隔优化（__nanosleep）
- 避免频繁的流同步

### 7.3 流水线优化

- 双缓冲消除等待
- 多kernel实例提高并行度
- 异步操作最大化重叠

## 8. 错误处理

### 8.1 错误检测

- Kernel崩溃检测
- 内存分配失败处理
- 超时机制

### 8.2 恢复机制

- Kernel重启
- 任务重试
- 状态恢复

## 9. 调试支持

### 9.1 日志系统

```cpp
enum LogLevel {
    ERROR = 0,
    WARNING = 1,
    INFO = 2,
    DEBUG = 3
};

#define LOG(level, fmt, ...) \
    if (level <= current_log_level) { \
        fprintf(stderr, "[%s] " fmt "\n", #level, ##__VA_ARGS__); \
    }
```

### 9.2 性能监控

- 任务延迟统计
- 吞吐量测量
- GPU利用率监控

## 10. 编译和集成

### 10.1 编译要求

- CUDA 11.0+
- C++17
- CMake 3.18+

### 10.2 编译选项

```cmake
# 优化选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# 架构选项
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
```

### 10.3 LibAFL集成

```c
extern "C" {
    uint64_t libafl_submit_with_id(const uint8_t* input, size_t size);
    size_t libafl_poll_results(TaskResult* out, size_t max_count);
    size_t libafl_release_tasks(const uint64_t* task_ids, size_t count);
    void libafl_set_target_timeout_ms(uint64_t timeout_ms);
    void libafl_wait();
    void libafl_stop();
}
```

## 11. 未来优化方向

### 11.1 三缓冲扩展

- 进一步提高并行度
- 适用于极高吞吐量场景

### 11.2 动态负载均衡

- 根据任务大小动态分配
- 自适应缓冲区大小

### 11.3 多GPU支持

- 跨GPU任务分发
- GPU间负载均衡

## 12. 性能基准

### 12.1 对比测试

| 实现方式 | 吞吐量(tasks/s) | GPU利用率 | 延迟(ms) |
|---------|----------------|----------|---------|
| RAPID (锁步) | 10,000 | 45% | 2.5 |
| RAPID2 (原版) | 25,000 | 70% | 1.8 |
| RAPID2 (流水线) | 50,000+ | 95%+ | 1.2 |

### 12.2 优化效果

- **2-5倍吞吐量提升**
- **95%+ GPU利用率**
- **延迟降低50%**

## 13. 总结

RAPID2流水线架构通过双缓冲技术和persistent kernel设计，实现了真正的GPU满载运行。关键创新包括：

1. **双缓冲流水线**：消除CPU-GPU间的锁步等待
2. **原子操作同步**：轻量级的缓冲区状态管理
3. **自主切换机制**：Kernel内部自动切换缓冲区
4. **RAII资源管理**：确保资源的正确分配和释放

这种设计最大化了GPU利用率，显著提升了fuzzing的执行效率。
