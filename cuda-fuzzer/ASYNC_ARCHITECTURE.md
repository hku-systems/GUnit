# Async Fuzzer Architecture - Redesign

## Overview

We redesigned the async fuzzer around a custom `AsyncBatchFuzzer` instead of modifying the `Executor`.

## Key Design Decisions

### 1. Why AsyncBatchFuzzer instead of AsyncExecutor?

- **Problem**: StdFuzzer's synchronous execution model doesn't fit async GPU execution.
- **Solution**: Create AsyncBatchFuzzer that implements the `Fuzzer` trait with async semantics.
- **Rationale**: The Executor should remain simple (just submit tasks); complexity belongs in the Fuzzer.

### 2. Architecture Components

```
┌────────────────────────────────────────┐
│         AsyncBatchFuzzer               │  <- Custom Fuzzer implementation
│  - Manages batch submission            │
│  - Polls for results asynchronously    │
│  - Evaluates feedback when ready       │
│  - Decouples execution from evaluation │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│          GpuExecutor                   │  <- Simple executor
│  - Only submits tasks to GPU           │
│  - Returns immediately (non-blocking)  │
│  - No coverage collection              │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│       AsyncMutationalStage             │  <- Custom stage
│  - Generates mutations                 │
│  - Submits via fuzzer.submit_input()   │
│  - Works with AsyncBatchFuzzer         │
└────────────────────────────────────────┘
```

## Implementation Details

### AsyncBatchFuzzer (`async_fuzzer.rs`)

```rust
pub struct AsyncBatchFuzzer<CS, F, OF, I> {
    scheduler: CS,
    feedback: F,
    objective: OF,
    cuda_backend: Rc<AsyncCudaBackend>,
    pending_tasks: VecDeque<PendingTask<I>>,
    // ... batch management
}

impl Fuzzer for AsyncBatchFuzzer {
    fn fuzz_one() {
        // 1. Submit batch via stages
        // 2. Poll and process completed tasks
        // 3. Evaluate feedback asynchronously
    }
}

impl Evaluator for AsyncBatchFuzzer {
    // Implements evaluation interface for LibAFL compatibility
}
```

### GpuExecutor (`gpu_executor.rs`)

```rust
pub struct GpuExecutor<I, OT, S> {
    cuda_backend: Rc<AsyncCudaBackend>,
    observers: OT,
}

impl Executor for GpuExecutor {
    fn run_target(&mut self, input: &I) -> Result<ExitKind, Error> {
        // Simply submit and return Ok
        self.cuda_backend.submit_with_id(&input.target_bytes());
        Ok(ExitKind::Ok)
    }
}
```

### AsyncMutationalStage (`async_mutational_stage.rs`)

```rust
pub struct AsyncMutationalStage {
    mutator: M,
    mutations_per_input: usize,
}

impl Stage for AsyncMutationalStage {
    fn perform(&mut self, fuzzer: &mut AsyncBatchFuzzer, ...) {
        // Generate mutations and submit via fuzzer
        for _ in 0..self.mutations_per_input {
            let mutated = self.mutator.mutate(input)?;
            fuzzer.submit_input(mutated)?;
        }
    }
}
```

## Key Differences from Previous Approach

### Before (AsyncExecutor - Wrong Approach)
- Tried to handle async logic in Executor
- Complex background thread management in wrong layer
- Fighting against LibAFL's design

### After (AsyncBatchFuzzer - Correct Approach)
- Fuzzer handles async complexity
- Executor remains simple
- Clean separation of concerns
- Works with LibAFL's architecture

## Usage

```rust
// Create async fuzzer
let mut fuzzer = AsyncBatchFuzzer::new(
    scheduler,
    feedback,
    objective,
    cuda_backend.clone(),
)
.with_batch_size(100)
.with_max_pending(32);

// The ordered window is capped at 32 and must not exceed the RAPID2 queue
// capacity (64).

// Simple GPU executor
let mut executor = GpuExecutor::new(
    cuda_backend.clone(),
    observers,
);

// Async mutation stage
let async_stage = AsyncMutationalStage::new(mutator)
    .with_mutations_per_input(100);

// Run fuzzing
fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut manager);
```

## Benefits

1. **Clean Architecture**: Each component has a single, clear responsibility
2. **LibAFL Compatible**: Works within LibAFL's trait system
3. **True Async**: Batch submission and delayed evaluation
4. **Performance**: GPU can process while CPU generates more inputs
5. **Maintainable**: Easy to understand and modify

## Future Improvements

1. **Adaptive Batching**: Adjust batch size based on GPU load
2. **Priority Queue**: Prioritize interesting inputs for faster processing
3. **Multi-GPU Support**: Distribute across multiple GPUs
4. **Statistics**: Better tracking of async performance metrics

## Conclusion

By moving the async logic from Executor to Fuzzer (as you correctly identified), we achieve a much cleaner and more maintainable architecture that properly leverages LibAFL's design while enabling true asynchronous GPU fuzzing.
