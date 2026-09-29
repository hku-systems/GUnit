use std::time::Duration;

/// Shared fuzzer target-execution timeout budget.
///
/// The synchronous backend passes this to LibAFL's in-process timeout executor.
/// RAPID2 asynchronous backends receive the same budget through the backend
/// timeout config ABI and enforce it from the task dispatch point, so backend
/// queue wait time is not counted as target execution time.
pub const DEFAULT_TARGET_TIMEOUT: Duration = Duration::from_secs(10);
