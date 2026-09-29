// Expose modules for use by binary targets
pub mod arg_pack_v1;
pub mod async_fuzzer;
pub mod async_mutational_stage;
pub mod coverage_telemetry;
pub mod cuda_backend;
pub mod fuzzer_frontend;
pub mod fuzzer_restarting;
pub mod fuzzer_timeout;
pub mod fuzzer_workdir;
pub mod gpu_client_guard;
pub mod gpu_executor;
pub mod mutators;
pub mod ordered_fuzzer;
pub mod ordered_window;
#[cfg(feature = "profiling")]
pub mod profiling;
pub mod retirement;
pub mod simt_memcov_feedback;
mod submission_context;
pub mod supply_pool;
pub mod task_time_feedback;
