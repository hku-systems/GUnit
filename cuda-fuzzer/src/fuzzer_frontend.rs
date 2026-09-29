use std::{num::NonZeroUsize, path::PathBuf};

use libafl::{feedbacks::MapFeedbackMetadata, Error, HasMetadata, HasNamedMetadata};
use serde::{Deserialize, Serialize};

use crate::arg_pack_v1::{
    arg_pack_stats_v1, default_seed_rapid_input_v1, normalize_rapid_input_v1,
};
use crate::simt_memcov_feedback::BitsetNoveltyStateMetadata;

#[cfg(feature = "profiling")]
pub fn profile_feedback<F>(feedback: F) -> crate::profiling::ProfiledFeedback<F> {
    crate::profiling::ProfiledFeedback::new(feedback)
}

#[cfg(not(feature = "profiling"))]
pub fn profile_feedback<F>(feedback: F) -> F {
    feedback
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RunMode {
    Fuzz {
        mutate: bool,
        runs: Option<u64>,
        duration_secs: Option<u64>,
    },
    Benchmark {
        warmup_runs: u64,
        min_duration_secs: u64,
    },
    RawThroughput {
        duration_secs: u64,
    },
    CoverageCampaign {
        duration_secs: u64,
        coverage_log: PathBuf,
        vconfig_mutation: bool,
    },
    DumpSeed,
}

impl RunMode {
    pub fn is_bounded_fuzz(&self) -> bool {
        matches!(
            self,
            Self::Fuzz { runs: Some(_), .. }
                | Self::Fuzz {
                    duration_secs: Some(_),
                    ..
                }
        )
    }

    pub fn mutating_seconds(&self) -> Option<u64> {
        match self {
            Self::Fuzz {
                mutate: true,
                duration_secs,
                ..
            } => *duration_secs,
            _ => None,
        }
    }
}

pub const BENCHMARK_RESULT_PREFIX: &str = "RAPID_BENCHMARK_RESULT ";
pub const MUTATING_RESULT_PREFIX: &str = "RAPID_MUTATING_RESULT ";
pub const RAW_THROUGHPUT_RESULT_PREFIX: &str = "RAPID_RAW_RESULT ";
pub const BENCHMARK_RNG_SEED: u64 = 0;
pub const BENCHMARK_CHUNK_ITERATIONS: u64 = 100;
pub const MAX_ORDERED_WINDOW_SIZE: usize = 32;
pub const DEFAULT_ASYNC_SUPPLY_THREADS: usize = 2;
pub const DEFAULT_ASYNC_RETIRE_WORKERS: usize = 2;
const CFG_FEEDBACK_NAME: &str = "cfg_sites";
const PROFILE_REEXEC_ENV: &str = "RAPID_PROFILE_REEXECED";

pub fn gpu_client_guard_required() -> bool {
    std::env::var_os(PROFILE_REEXEC_ENV).is_none()
}

pub fn reexec_profiled_client_after_fork(profile_enabled: bool) -> Result<(), String> {
    #[cfg(unix)]
    if profile_enabled && gpu_client_guard_required() {
        use std::os::unix::process::CommandExt;

        let executable = std::env::current_exe()
            .map_err(|error| format!("failed to resolve profiling client executable: {error}"))?;
        let error = std::process::Command::new(executable)
            .args(std::env::args_os().skip(1))
            .env(PROFILE_REEXEC_ENV, "1")
            .exec();
        return Err(format!("failed to re-exec profiling client: {error}"));
    }
    Ok(())
}

pub fn async_benchmark_chunk_iterations(profile_enabled: bool) -> u64 {
    if profile_enabled {
        1
    } else {
        BENCHMARK_CHUNK_ITERATIONS
    }
}

pub fn async_benchmark_stage_max_iterations(profile_enabled: bool) -> NonZeroUsize {
    let iterations = if profile_enabled {
        1
    } else {
        crate::async_mutational_stage::DEFAULT_MUTATIONAL_MAX_ITERATIONS
    };
    NonZeroUsize::new(iterations).expect("async stage iteration count must be positive")
}

pub fn campaign_stage_max_iterations() -> NonZeroUsize {
    NonZeroUsize::new(1).expect("coverage campaign stage iteration count must be positive")
}

pub fn fixed_rng_seed_from_env() -> Result<u64, String> {
    let raw = std::env::var("RAPID_FIXED_SEED")
        .map_err(|_| "coverage campaign requires RAPID_FIXED_SEED".to_string())?;
    raw.parse::<u64>()
        .map_err(|_| format!("invalid RAPID_FIXED_SEED: {raw}"))
}

pub fn effective_window_size(configured: Option<NonZeroUsize>, default: usize) -> NonZeroUsize {
    configured.unwrap_or_else(|| {
        NonZeroUsize::new(default).expect("default window size must be positive")
    })
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub requested_min_duration_ns: u64,
    pub measured_iterations: u64,
    pub completed: u64,
    pub elapsed_ns: u64,
    pub corpus_size: u64,
    pub solutions: u64,
    pub pending: u64,
    pub mutation_enabled: bool,
}

impl BenchmarkResult {
    pub fn new(
        requested_min_duration_ns: u64,
        measured_iterations: u64,
        completed: u64,
        elapsed_ns: u64,
        corpus_size: u64,
        solutions: u64,
        pending: u64,
    ) -> Result<Self, String> {
        if completed == 0 {
            return Err("benchmark completed no executions".to_string());
        }
        if pending != 0 {
            return Err(format!("benchmark exited with {pending} pending execution"));
        }
        if elapsed_ns == 0 {
            return Err("benchmark elapsed time is zero".to_string());
        }
        if elapsed_ns < requested_min_duration_ns {
            return Err("benchmark ended before its minimum duration".to_string());
        }
        Ok(Self {
            requested_min_duration_ns,
            measured_iterations,
            completed,
            elapsed_ns,
            corpus_size,
            solutions,
            pending,
            mutation_enabled: false,
        })
    }

    pub fn machine_line(&self) -> Result<String, serde_json::Error> {
        Ok(format!(
            "{BENCHMARK_RESULT_PREFIX}{}",
            serde_json::to_string(self)?
        ))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MutatingResult {
    pub requested_seconds: u64,
    pub executions: u64,
    pub corpus_size: u64,
    pub solutions: u64,
    pub mutation_calls: u64,
    pub coverage_nonzero_bytes: u64,
    pub simt_memcov_nonzero_bits: u64,
    pub pending: u64,
    pub completed: u64,
    pub outstanding: u64,
    pub in_flight: u64,
    pub queued_submissions: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FeedbackActivity {
    pub coverage_nonzero_bytes: u64,
    pub simt_memcov_nonzero_bits: u64,
}

pub fn feedback_activity<S>(state: &S) -> Result<FeedbackActivity, Error>
where
    S: HasMetadata + HasNamedMetadata,
{
    let cfg = state.named_metadata::<MapFeedbackMetadata<u8>>(CFG_FEEDBACK_NAME)?;
    let simt = state.metadata::<BitsetNoveltyStateMetadata>()?;
    Ok(FeedbackActivity {
        coverage_nonzero_bytes: cfg.history_map.iter().filter(|byte| **byte != 0).count() as u64,
        simt_memcov_nonzero_bits: simt.seen.iter().map(|byte| byte.count_ones() as u64).sum(),
    })
}

impl MutatingResult {
    pub fn machine_line(&self) -> Result<String, serde_json::Error> {
        Ok(format!(
            "{MUTATING_RESULT_PREFIX}{}",
            serde_json::to_string(self)?
        ))
    }
}

pub fn print_mutating_result(result: &MutatingResult) -> Result<(), Error> {
    println!(
        "{}",
        result
            .machine_line()
            .map_err(|error| Error::illegal_state(error.to_string()))?
    );
    Ok(())
}

pub fn mutating_result(
    requested_seconds: u64,
    executions: u64,
    corpus_size: u64,
    solutions: u64,
    feedback_activity: FeedbackActivity,
    pending: u64,
    completed: u64,
    outstanding: u64,
    in_flight: u64,
    queued_submissions: u64,
) -> MutatingResult {
    MutatingResult {
        requested_seconds,
        executions,
        corpus_size,
        solutions,
        mutation_calls: arg_pack_stats_v1().mutation_calls,
        coverage_nonzero_bytes: feedback_activity.coverage_nonzero_bytes,
        simt_memcov_nonzero_bits: feedback_activity.simt_memcov_nonzero_bits,
        pending,
        completed,
        outstanding,
        in_flight,
        queued_submissions,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawThroughputResult {
    pub requested_duration_ns: u64,
    pub warmup_tasks: u64,
    pub total_tasks: u64,
    pub submitted_tasks: u64,
    pub released_tasks: u64,
    pub elapsed_ns: u64,
    pub execs_per_second: f64,
    pub non_ok_statuses: u64,
    pub canary_statuses: u64,
    pub queue_samples: u64,
    pub max_pending_queue: u64,
    pub max_completed_queue: u64,
    pub final_pending_queue: u64,
    pub final_completed_queue: u64,
    pub max_host_in_flight: u64,
    pub skip_cov_read: bool,
    pub skip_coverage_copy: bool,
}

impl RawThroughputResult {
    pub fn machine_line(&self) -> Result<String, serde_json::Error> {
        Ok(format!(
            "{RAW_THROUGHPUT_RESULT_PREFIX}{}",
            serde_json::to_string(self)?
        ))
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct FuzzerArgs {
    pub lib_path: String,
    pub manifest_path: PathBuf,
    pub mode: RunMode,
    pub window_size: Option<NonZeroUsize>,
    pub supply_threads: usize,
    pub retire_workers: usize,
    pub retire_batch: NonZeroUsize,
}

pub fn parse_args(args: &[String]) -> Result<FuzzerArgs, String> {
    parse_args_impl(args, false)
}

pub fn parse_async_args(args: &[String]) -> Result<FuzzerArgs, String> {
    parse_args_impl(args, true)
}

fn parse_args_impl(args: &[String], allow_async_options: bool) -> Result<FuzzerArgs, String> {
    if args.len() < 4 {
        return Err("invalid argument count".to_string());
    }
    if args[2] != "--manifest" {
        return Err("missing --manifest".to_string());
    }

    let mut mutate = true;
    let mut runs = None;
    let mut mutate_seconds = None;
    let mut dump_seed = false;
    let mut benchmark = false;
    let mut warmup_runs = None;
    let mut benchmark_seconds = None;
    let mut raw_throughput_seconds = None;
    let mut window_size = None;
    let mut supply_threads = None;
    let mut retire_workers = None;
    let mut retire_batch = None;
    let mut coverage_seconds = None;
    let mut coverage_log = None;
    let mut vconfig_mutation = None;
    let mut idx = 4usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--dump-seed" => {
                dump_seed = true;
                idx += 1;
            }
            "--no-mutate" => {
                mutate = false;
                idx += 1;
            }
            "--runs" => {
                if runs.is_some() {
                    return Err("duplicate --runs".to_string());
                }
                let Some(raw_runs) = args.get(idx + 1) else {
                    return Err("missing --runs value".to_string());
                };
                runs = Some(parse_positive_u64(raw_runs)?);
                idx += 2;
            }
            "--mutate-seconds" => {
                if mutate_seconds.is_some() {
                    return Err("duplicate --mutate-seconds".to_string());
                }
                let Some(raw_seconds) = args.get(idx + 1) else {
                    return Err("missing --mutate-seconds value".to_string());
                };
                mutate_seconds = Some(parse_positive_u64(raw_seconds)?);
                idx += 2;
            }
            "--benchmark" => {
                if benchmark {
                    return Err("duplicate --benchmark".to_string());
                }
                benchmark = true;
                idx += 1;
            }
            "--warmup-runs" => {
                if warmup_runs.is_some() {
                    return Err("duplicate --warmup-runs".to_string());
                }
                let Some(raw_runs) = args.get(idx + 1) else {
                    return Err("missing --warmup-runs value".to_string());
                };
                warmup_runs = Some(parse_positive_u64(raw_runs)?);
                idx += 2;
            }
            "--benchmark-seconds" => {
                if benchmark_seconds.is_some() {
                    return Err("duplicate --benchmark-seconds".to_string());
                }
                let Some(raw_seconds) = args.get(idx + 1) else {
                    return Err("missing --benchmark-seconds value".to_string());
                };
                benchmark_seconds = Some(parse_positive_u64(raw_seconds)?);
                idx += 2;
            }
            "--raw-throughput-seconds" => {
                if !allow_async_options {
                    return Err(
                        "--raw-throughput-seconds is only available in fuzzer_async".to_string()
                    );
                }
                if raw_throughput_seconds.is_some() {
                    return Err("duplicate --raw-throughput-seconds".to_string());
                }
                let Some(raw_seconds) = args.get(idx + 1) else {
                    return Err("missing --raw-throughput-seconds value".to_string());
                };
                raw_throughput_seconds = Some(parse_positive_u64(raw_seconds)?);
                idx += 2;
            }
            "--window-size" => {
                if window_size.is_some() {
                    return Err("duplicate --window-size".to_string());
                }
                let Some(raw_window_size) = args.get(idx + 1) else {
                    return Err("missing --window-size value".to_string());
                };
                let parsed = raw_window_size
                    .parse::<usize>()
                    .map_err(|_| format!("invalid window size: {raw_window_size}"))?;
                let Some(nonzero) = NonZeroUsize::new(parsed) else {
                    return Err("window size must be positive".to_string());
                };
                if nonzero.get() > MAX_ORDERED_WINDOW_SIZE {
                    return Err(format!(
                        "window size exceeds maximum {MAX_ORDERED_WINDOW_SIZE}"
                    ));
                }
                window_size = Some(nonzero);
                idx += 2;
            }
            "--supply-threads" => {
                if !allow_async_options {
                    return Err("--supply-threads is only available in fuzzer_async".to_string());
                }
                if supply_threads.is_some() {
                    return Err("duplicate --supply-threads".to_string());
                }
                let Some(raw_supply_threads) = args.get(idx + 1) else {
                    return Err("missing --supply-threads value".to_string());
                };
                let parsed = raw_supply_threads
                    .parse::<usize>()
                    .map_err(|_| format!("invalid supply thread count: {raw_supply_threads}"))?;
                if parsed > MAX_ORDERED_WINDOW_SIZE {
                    return Err(format!(
                        "supply thread count exceeds maximum {MAX_ORDERED_WINDOW_SIZE}"
                    ));
                }
                supply_threads = Some(parsed);
                idx += 2;
            }
            "--retire-worker" => {
                if !allow_async_options {
                    return Err("--retire-worker is only available in fuzzer_async".to_string());
                }
                if retire_workers.is_some() {
                    return Err("duplicate --retire-worker".to_string());
                }
                let Some(raw_retire_workers) = args.get(idx + 1) else {
                    return Err("missing --retire-worker value".to_string());
                };
                let parsed = raw_retire_workers
                    .parse::<usize>()
                    .map_err(|_| format!("invalid retire worker count: {raw_retire_workers}"))?;
                if parsed > MAX_ORDERED_WINDOW_SIZE {
                    return Err(format!(
                        "retire worker count exceeds maximum {MAX_ORDERED_WINDOW_SIZE}"
                    ));
                }
                retire_workers = Some(parsed);
                idx += 2;
            }
            "--retire-workers" => {
                return Err("--retire-workers has been removed; use --retire-worker N".to_string());
            }
            "--retire-batch" => {
                if !allow_async_options {
                    return Err("--retire-batch is only available in fuzzer_async".to_string());
                }
                if retire_batch.is_some() {
                    return Err("duplicate --retire-batch".to_string());
                }
                let Some(raw_retire_batch) = args.get(idx + 1) else {
                    return Err("missing --retire-batch value".to_string());
                };
                let parsed = raw_retire_batch
                    .parse::<usize>()
                    .map_err(|_| format!("invalid retire batch: {raw_retire_batch}"))?;
                retire_batch = Some(
                    NonZeroUsize::new(parsed)
                        .ok_or_else(|| "retire batch must be positive".to_string())?,
                );
                idx += 2;
            }
            "--coverage-seconds" => {
                if coverage_seconds.is_some() {
                    return Err("duplicate --coverage-seconds".to_string());
                }
                let Some(raw_seconds) = args.get(idx + 1) else {
                    return Err("missing --coverage-seconds value".to_string());
                };
                coverage_seconds = Some(parse_positive_u64(raw_seconds)?);
                idx += 2;
            }
            "--coverage-log" => {
                if coverage_log.is_some() {
                    return Err("duplicate --coverage-log".to_string());
                }
                let Some(raw_path) = args.get(idx + 1) else {
                    return Err("missing --coverage-log value".to_string());
                };
                coverage_log = Some(PathBuf::from(raw_path));
                idx += 2;
            }
            "--vconfig" => {
                if vconfig_mutation.is_some() {
                    return Err("duplicate --vconfig".to_string());
                }
                let Some(raw_vconfig) = args.get(idx + 1) else {
                    return Err("missing --vconfig value".to_string());
                };
                vconfig_mutation = Some(match raw_vconfig.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err("--vconfig must be on or off".to_string()),
                });
                idx += 2;
            }
            _ => return Err("invalid run mode".to_string()),
        }
    }

    let coverage_requested =
        coverage_seconds.is_some() || coverage_log.is_some() || vconfig_mutation.is_some();
    let mode = if coverage_requested {
        if benchmark
            || warmup_runs.is_some()
            || benchmark_seconds.is_some()
            || raw_throughput_seconds.is_some()
            || dump_seed
            || !mutate
            || runs.is_some()
            || mutate_seconds.is_some()
            || supply_threads.is_some_and(|count| count > 0)
        {
            return Err("coverage campaign cannot be combined with other run modes".to_string());
        }
        RunMode::CoverageCampaign {
            duration_secs: coverage_seconds
                .ok_or_else(|| "missing --coverage-seconds".to_string())?,
            coverage_log: coverage_log.ok_or_else(|| "missing --coverage-log".to_string())?,
            vconfig_mutation: vconfig_mutation.ok_or_else(|| "missing --vconfig".to_string())?,
        }
    } else if let Some(duration_secs) = raw_throughput_seconds {
        if benchmark
            || warmup_runs.is_some()
            || benchmark_seconds.is_some()
            || dump_seed
            || !mutate
            || runs.is_some()
            || mutate_seconds.is_some()
            || supply_threads.is_some_and(|count| count > 0)
        {
            return Err("raw throughput cannot be combined with other run modes".to_string());
        }
        RunMode::RawThroughput { duration_secs }
    } else if benchmark || warmup_runs.is_some() || benchmark_seconds.is_some() {
        if !benchmark || dump_seed || !mutate || runs.is_some() || mutate_seconds.is_some() {
            return Err("benchmark cannot be combined with fuzzing options".to_string());
        }
        RunMode::Benchmark {
            warmup_runs: warmup_runs.ok_or_else(|| "missing --warmup-runs".to_string())?,
            min_duration_secs: benchmark_seconds
                .ok_or_else(|| "missing --benchmark-seconds".to_string())?,
        }
    } else if dump_seed {
        if window_size.is_some() {
            return Err("--dump-seed cannot be combined with --window-size".to_string());
        }
        if !mutate || runs.is_some() || mutate_seconds.is_some() {
            return Err("--dump-seed cannot be combined with fuzzing options".to_string());
        }
        RunMode::DumpSeed
    } else {
        if mutate_seconds.is_some() && (!mutate || runs.is_some()) {
            return Err("--mutate-seconds requires mutating fuzz mode without --runs".to_string());
        }
        RunMode::Fuzz {
            mutate,
            runs,
            duration_secs: mutate_seconds,
        }
    };

    let mutating_fuzz = matches!(mode, RunMode::Fuzz { mutate: true, .. });
    let default_workers = allow_async_options && mutating_fuzz;
    let supply_threads = supply_threads.unwrap_or(if default_workers {
        DEFAULT_ASYNC_SUPPLY_THREADS
    } else {
        0
    });
    if supply_threads > 0 && !mutating_fuzz {
        return Err("--supply-threads requires mutating fuzz mode".to_string());
    }
    let retire_workers = retire_workers.unwrap_or(if default_workers {
        DEFAULT_ASYNC_RETIRE_WORKERS
    } else {
        0
    });
    if retire_batch.is_some() && retire_workers == 0 {
        return Err("--retire-batch requires a positive retire worker count".to_string());
    }
    if retire_workers > 0 && !matches!(mode, RunMode::Fuzz { .. }) {
        return Err("--retire-worker requires fuzz mode".to_string());
    }
    let retire_batch = retire_batch
        .unwrap_or_else(|| NonZeroUsize::new(crate::retirement::DEFAULT_RETIRE_BATCH).unwrap());
    if retire_workers > 0
        && retire_batch.get().div_ceil(retire_workers) > crate::retirement::MAX_RETIREMENT_IN_FLIGHT
    {
        return Err(format!(
            "retire batch requires more than {} retirement credits per worker",
            crate::retirement::MAX_RETIREMENT_IN_FLIGHT
        ));
    }

    Ok(FuzzerArgs {
        lib_path: args[1].clone(),
        manifest_path: PathBuf::from(&args[3]),
        mode,
        window_size,
        supply_threads,
        retire_workers,
        retire_batch,
    })
}

pub fn print_seed_hex() {
    let seed = normalize_rapid_input_v1(&default_seed_rapid_input_v1());
    for byte in seed {
        print!("{byte:02x}");
    }
    println!();
}

pub fn print_arg_pack_stats() {
    let stats = arg_pack_stats_v1();
    println!(
        "Arg-pack stats: normalize_calls={}, normalize_repack_count={}, invalid_repair_count={}, mutation_calls={}, seed_generation_count={}, payload_clamp_count={}",
        stats.normalize_calls,
        stats.normalize_repack_count,
        stats.invalid_repair_count,
        stats.mutation_calls,
        stats.seed_generation_count,
        stats.payload_clamp_count
    );
}

pub fn print_final_statistics(corpus_size: usize, solutions: usize, executions: u64) {
    println!("Final statistics:");
    println!("  Corpus size: {corpus_size}");
    println!("  Crashes found: {solutions}");
    println!("  Total executions: {executions}");
}

fn parse_positive_u64(raw: &str) -> Result<u64, String> {
    let value = raw
        .parse::<u64>()
        .map_err(|_| format!("invalid positive integer: {raw}"))?;
    if value == 0 {
        return Err("value must be positive".to_string());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libafl_bolts::serdeany::{NamedSerdeAnyMap, SerdeAnyMap};

    #[derive(Default)]
    struct FeedbackState {
        metadata: SerdeAnyMap,
        named_metadata: NamedSerdeAnyMap,
    }

    impl HasMetadata for FeedbackState {
        fn metadata_map(&self) -> &SerdeAnyMap {
            &self.metadata
        }

        fn metadata_map_mut(&mut self) -> &mut SerdeAnyMap {
            &mut self.metadata
        }
    }

    impl HasNamedMetadata for FeedbackState {
        fn named_metadata_map(&self) -> &NamedSerdeAnyMap {
            &self.named_metadata
        }

        fn named_metadata_map_mut(&mut self) -> &mut NamedSerdeAnyMap {
            &mut self.named_metadata
        }
    }

    fn valid_args(extra: &[&str]) -> Vec<String> {
        ["fuzzer", "libtarget.so", "--manifest", "manifest.json"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn async_profile_benchmark_checks_the_deadline_after_each_fuzz_iteration() {
        assert_eq!(async_benchmark_chunk_iterations(true), 1);
        assert_eq!(async_benchmark_stage_max_iterations(true).get(), 1);
        assert_eq!(
            async_benchmark_chunk_iterations(false),
            BENCHMARK_CHUNK_ITERATIONS
        );
        assert_eq!(
            async_benchmark_stage_max_iterations(false).get(),
            crate::async_mutational_stage::DEFAULT_MUTATIONAL_MAX_ITERATIONS
        );
    }

    #[test]
    fn parse_args_accepts_default_fuzz_mode() {
        let parsed = parse_args(&valid_args(&[])).unwrap();

        assert_eq!(parsed.lib_path, "libtarget.so");
        assert_eq!(parsed.manifest_path, PathBuf::from("manifest.json"));
        assert_eq!(parsed.window_size, None);
        assert_eq!(parsed.supply_threads, 0);
        assert_eq!(
            parsed.mode,
            RunMode::Fuzz {
                mutate: true,
                runs: None,
                duration_secs: None,
            }
        );
    }

    #[test]
    fn parse_args_enables_and_disables_supply_threads_explicitly() {
        let default = parse_async_args(&valid_args(&[])).unwrap();
        assert_eq!(default.supply_threads, 2);

        let enabled = parse_async_args(&valid_args(&["--supply-threads", "3"])).unwrap();
        assert_eq!(enabled.supply_threads, 3);

        let disabled = parse_async_args(&valid_args(&["--supply-threads", "0"])).unwrap();
        assert_eq!(disabled.supply_threads, 0);
        assert_eq!(
            disabled.mode,
            RunMode::Fuzz {
                mutate: true,
                runs: None,
                duration_secs: None,
            }
        );

        assert_eq!(
            parse_async_args(&valid_args(&["--supply-threads"])).unwrap_err(),
            "missing --supply-threads value"
        );
        assert_eq!(
            parse_async_args(&valid_args(&["--supply-threads", "33"])).unwrap_err(),
            "supply thread count exceeds maximum 32"
        );
    }

    #[test]
    fn async_retirement_defaults_to_two_workers_and_accepts_zero() {
        let default = parse_async_args(&valid_args(&[])).unwrap();
        assert_eq!(default.retire_workers, 2);
        assert_eq!(default.retire_batch.get(), 8);

        let parallel = parse_async_args(&valid_args(&["--retire-worker", "4"])).unwrap();
        assert_eq!(parallel.retire_workers, 4);

        let disabled = parse_async_args(&valid_args(&["--retire-worker", "0"])).unwrap();
        assert_eq!(disabled.retire_workers, 0);

        assert_eq!(
            parse_args(&valid_args(&["--retire-worker", "2"])).unwrap_err(),
            "--retire-worker is only available in fuzzer_async"
        );
        assert_eq!(
            parse_async_args(&valid_args(&["--retire-worker"])).unwrap_err(),
            "missing --retire-worker value"
        );
        assert_eq!(
            parse_async_args(&valid_args(&["--retire-batch", "16"]))
                .unwrap()
                .retire_batch
                .get(),
            16
        );
        assert_eq!(
            parse_async_args(&valid_args(&[
                "--retire-worker",
                "2",
                "--retire-batch",
                "129",
            ]))
            .unwrap_err(),
            "retire batch requires more than 64 retirement credits per worker"
        );
        assert_eq!(
            parse_async_args(&valid_args(&[
                "--retire-worker",
                "0",
                "--retire-batch",
                "2",
            ]))
            .unwrap_err(),
            "--retire-batch requires a positive retire worker count"
        );
        assert_eq!(
            parse_async_args(&valid_args(&["--retire-worker", "33"])).unwrap_err(),
            "retire worker count exceeds maximum 32"
        );
        assert_eq!(
            parse_async_args(&valid_args(&["--retire-workers", "2"])).unwrap_err(),
            "--retire-workers has been removed; use --retire-worker N"
        );
    }

    #[test]
    fn synchronous_parser_rejects_async_supply_option() {
        assert_eq!(
            parse_args(&valid_args(&["--supply-threads", "2"])).unwrap_err(),
            "--supply-threads is only available in fuzzer_async"
        );
    }

    #[test]
    fn supply_threads_only_apply_to_mutating_fuzz_mode() {
        assert_eq!(
            parse_async_args(&valid_args(&["--no-mutate", "--supply-threads", "2"])).unwrap_err(),
            "--supply-threads requires mutating fuzz mode"
        );
        assert_eq!(
            parse_async_args(&valid_args(&[
                "--benchmark",
                "--warmup-runs",
                "2",
                "--benchmark-seconds",
                "30",
                "--supply-threads",
                "2",
            ]))
            .unwrap_err(),
            "--supply-threads requires mutating fuzz mode"
        );
    }

    #[test]
    fn async_parser_accepts_raw_throughput_mode() {
        let parsed = parse_async_args(&valid_args(&[
            "--raw-throughput-seconds",
            "10",
            "--window-size",
            "32",
        ]))
        .unwrap();

        assert_eq!(parsed.window_size.unwrap().get(), 32);
        assert_eq!(parsed.mode, RunMode::RawThroughput { duration_secs: 10 });
        assert_eq!(parsed.supply_threads, 0);
        assert_eq!(parsed.retire_workers, 0);
    }

    #[test]
    fn raw_throughput_is_async_only_and_exclusive() {
        assert_eq!(
            parse_args(&valid_args(&["--raw-throughput-seconds", "10"])).unwrap_err(),
            "--raw-throughput-seconds is only available in fuzzer_async"
        );
        for incompatible in [
            vec![
                "--benchmark",
                "--warmup-runs",
                "2",
                "--benchmark-seconds",
                "5",
            ],
            vec!["--dump-seed"],
            vec!["--no-mutate"],
            vec!["--runs", "2"],
            vec!["--supply-threads", "2"],
        ] {
            let args = ["--raw-throughput-seconds", "10"]
                .into_iter()
                .chain(incompatible)
                .collect::<Vec<_>>();
            assert_eq!(
                parse_async_args(&valid_args(&args)).unwrap_err(),
                "raw throughput cannot be combined with other run modes"
            );
        }
    }

    #[test]
    fn raw_throughput_result_is_machine_readable_json() {
        let result = RawThroughputResult {
            requested_duration_ns: 1_000_000_000,
            warmup_tasks: 32,
            total_tasks: 20_000,
            submitted_tasks: 20_000,
            released_tasks: 20_000,
            elapsed_ns: 1_000_000_000,
            execs_per_second: 20_000.0,
            non_ok_statuses: 0,
            canary_statuses: 0,
            queue_samples: 100,
            max_pending_queue: 30,
            max_completed_queue: 2,
            final_pending_queue: 0,
            final_completed_queue: 0,
            max_host_in_flight: 32,
            skip_cov_read: true,
            skip_coverage_copy: false,
        };

        let line = result.machine_line().unwrap();
        assert!(line.starts_with(RAW_THROUGHPUT_RESULT_PREFIX));
        let json = &line[RAW_THROUGHPUT_RESULT_PREFIX.len()..];
        let decoded: RawThroughputResult = serde_json::from_str(json).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn parse_args_accepts_complete_coverage_campaign() {
        let parsed = parse_args(&valid_args(&[
            "--coverage-seconds",
            "60",
            "--coverage-log",
            "coverage.jsonl",
            "--vconfig",
            "on",
            "--window-size",
            "4",
        ]))
        .unwrap();

        assert_eq!(parsed.window_size.unwrap().get(), 4);
        assert_eq!(
            parsed.mode,
            RunMode::CoverageCampaign {
                duration_secs: 60,
                coverage_log: PathBuf::from("coverage.jsonl"),
                vconfig_mutation: true,
            }
        );

        let parsed = parse_async_args(&valid_args(&[
            "--vconfig",
            "off",
            "--coverage-log",
            "coverage.jsonl",
            "--coverage-seconds",
            "60",
        ]))
        .unwrap();
        assert_eq!(parsed.supply_threads, 0);
        assert_eq!(parsed.retire_workers, 0);
        assert_eq!(
            parsed.mode,
            RunMode::CoverageCampaign {
                duration_secs: 60,
                coverage_log: PathBuf::from("coverage.jsonl"),
                vconfig_mutation: false,
            }
        );
    }

    #[test]
    fn coverage_campaign_rejects_retired_sampling_options() {
        for option in ["--sample-interval-ms", "--sample-interval-seconds"] {
            assert_eq!(
                parse_args(&valid_args(&[
                    "--coverage-seconds",
                    "20",
                    "--coverage-log",
                    "coverage.jsonl",
                    "--vconfig",
                    "on",
                    option,
                    "1",
                ]))
                .unwrap_err(),
                "invalid run mode"
            );
        }
    }

    #[test]
    fn coverage_campaign_requires_every_argument_and_validates_vconfig() {
        for incomplete in [
            vec!["--coverage-log", "coverage.jsonl", "--vconfig", "on"],
            vec!["--coverage-seconds", "60", "--vconfig", "on"],
            vec![
                "--coverage-seconds",
                "60",
                "--coverage-log",
                "coverage.jsonl",
            ],
        ] {
            assert!(parse_args(&valid_args(&incomplete)).is_err());
        }

        assert_eq!(
            parse_args(&valid_args(&[
                "--coverage-seconds",
                "60",
                "--coverage-log",
                "coverage.jsonl",
                "--vconfig",
                "unsupported",
            ]))
            .unwrap_err(),
            "--vconfig must be on or off"
        );
    }

    #[test]
    fn coverage_campaign_rejects_legacy_mode_options() {
        let coverage = [
            "--coverage-seconds",
            "60",
            "--coverage-log",
            "coverage.jsonl",
            "--vconfig",
            "on",
        ];
        for incompatible in [
            vec!["--dump-seed"],
            vec!["--no-mutate"],
            vec!["--runs", "3"],
            vec![
                "--benchmark",
                "--warmup-runs",
                "2",
                "--benchmark-seconds",
                "30",
            ],
        ] {
            let args = coverage
                .iter()
                .copied()
                .chain(incompatible)
                .collect::<Vec<_>>();
            assert_eq!(
                parse_args(&valid_args(&args)).unwrap_err(),
                "coverage campaign cannot be combined with other run modes"
            );
        }
    }

    #[test]
    fn legacy_rq1_command_shapes_remain_exact() {
        let cases = [
            (
                vec![],
                RunMode::Fuzz {
                    mutate: true,
                    runs: None,
                    duration_secs: None,
                },
                None,
            ),
            (vec!["--dump-seed"], RunMode::DumpSeed, None),
            (
                vec!["--no-mutate", "--runs", "7"],
                RunMode::Fuzz {
                    mutate: false,
                    runs: Some(7),
                    duration_secs: None,
                },
                None,
            ),
            (
                vec![
                    "--benchmark",
                    "--warmup-runs",
                    "100",
                    "--benchmark-seconds",
                    "30",
                ],
                RunMode::Benchmark {
                    warmup_runs: 100,
                    min_duration_secs: 30,
                },
                None,
            ),
            (
                vec![
                    "--benchmark",
                    "--warmup-runs",
                    "100",
                    "--benchmark-seconds",
                    "30",
                    "--window-size",
                    "4",
                ],
                RunMode::Benchmark {
                    warmup_runs: 100,
                    min_duration_secs: 30,
                },
                Some(4),
            ),
        ];

        for (extra, expected_mode, expected_window) in cases {
            let parsed = parse_args(&valid_args(&extra)).unwrap();
            assert_eq!(parsed.mode, expected_mode);
            assert_eq!(parsed.window_size.map(NonZeroUsize::get), expected_window);
        }
    }

    #[test]
    fn parse_args_accepts_finite_runs_mode() {
        let parsed = parse_args(&valid_args(&["--runs", "3"])).unwrap();

        assert_eq!(parsed.lib_path, "libtarget.so");
        assert_eq!(parsed.manifest_path, PathBuf::from("manifest.json"));
        assert_eq!(
            parsed.mode,
            RunMode::Fuzz {
                mutate: true,
                runs: Some(3),
                duration_secs: None,
            }
        );
    }

    #[test]
    fn parse_args_accepts_timed_mutating_mode() {
        let parsed = parse_args(&valid_args(&["--mutate-seconds", "30"])).unwrap();

        assert_eq!(
            parsed.mode,
            RunMode::Fuzz {
                mutate: true,
                runs: None,
                duration_secs: Some(30),
            }
        );
        assert!(parse_args(&valid_args(&["--mutate-seconds", "30", "--runs", "2",])).is_err());
        assert!(parse_args(&valid_args(&["--mutate-seconds", "30", "--no-mutate",])).is_err());
    }

    #[test]
    fn parse_args_accepts_ordered_window_size() {
        let parsed = parse_args(&valid_args(&["--window-size", "32", "--runs", "3"])).unwrap();

        assert_eq!(parsed.window_size.unwrap().get(), 32);
        assert_eq!(
            parsed.mode,
            RunMode::Fuzz {
                mutate: true,
                runs: Some(3),
                duration_secs: None,
            }
        );
    }

    #[test]
    fn configured_window_size_overrides_async_default() {
        let configured = NonZeroUsize::new(4);
        let default = crate::async_fuzzer::DEFAULT_ASYNC_MAX_PENDING;

        assert_eq!(default, 2);
        assert_eq!(effective_window_size(configured, default).get(), 4);
        assert_eq!(effective_window_size(None, default).get(), 2);
    }

    #[test]
    fn parse_args_rejects_invalid_ordered_window_size() {
        assert_eq!(
            parse_args(&valid_args(&["--window-size", "0"])).unwrap_err(),
            "window size must be positive"
        );
        assert_eq!(
            parse_args(&valid_args(&["--window-size", "33"])).unwrap_err(),
            "window size exceeds maximum 32"
        );
        assert_eq!(
            parse_args(&valid_args(&["--window-size"])).unwrap_err(),
            "missing --window-size value"
        );
        assert_eq!(
            parse_args(&valid_args(&["--window-size", "2", "--window-size", "3"])).unwrap_err(),
            "duplicate --window-size"
        );
        assert_eq!(
            parse_args(&valid_args(&["--window-size", "2", "--dump-seed"])).unwrap_err(),
            "--dump-seed cannot be combined with --window-size"
        );
    }

    #[test]
    fn parse_args_accepts_dump_seed_mode() {
        let parsed = parse_args(&valid_args(&["--dump-seed"])).unwrap();

        assert_eq!(parsed.lib_path, "libtarget.so");
        assert_eq!(parsed.manifest_path, PathBuf::from("manifest.json"));
        assert_eq!(parsed.mode, RunMode::DumpSeed);
    }

    #[test]
    fn parse_args_accepts_no_mutate_fuzz_mode() {
        let parsed = parse_args(&valid_args(&["--no-mutate"])).unwrap();

        assert_eq!(parsed.lib_path, "libtarget.so");
        assert_eq!(parsed.manifest_path, PathBuf::from("manifest.json"));
        assert_eq!(
            parsed.mode,
            RunMode::Fuzz {
                mutate: false,
                runs: None,
                duration_secs: None,
            }
        );
    }

    #[test]
    fn parse_args_accepts_bounded_no_mutate_fuzz_mode() {
        let parsed = parse_async_args(&valid_args(&["--no-mutate", "--runs", "3"])).unwrap();

        assert_eq!(parsed.lib_path, "libtarget.so");
        assert_eq!(parsed.manifest_path, PathBuf::from("manifest.json"));
        assert_eq!(parsed.supply_threads, 0);
        assert_eq!(parsed.retire_workers, 0);
        assert_eq!(
            parsed.mode,
            RunMode::Fuzz {
                mutate: false,
                runs: Some(3),
                duration_secs: None,
            }
        );
    }

    #[test]
    fn parse_args_accepts_runs_before_no_mutate() {
        let parsed = parse_args(&valid_args(&["--runs", "3", "--no-mutate"])).unwrap();

        assert_eq!(parsed.lib_path, "libtarget.so");
        assert_eq!(parsed.manifest_path, PathBuf::from("manifest.json"));
        assert_eq!(
            parsed.mode,
            RunMode::Fuzz {
                mutate: false,
                runs: Some(3),
                duration_secs: None,
            }
        );
    }

    #[test]
    fn parse_args_rejects_zero_runs() {
        let err = parse_args(&valid_args(&["--runs", "0"])).unwrap_err();
        assert_eq!(err, "value must be positive");
    }

    #[test]
    fn parse_args_rejects_missing_runs_value() {
        let err = parse_args(&valid_args(&["--runs"])).unwrap_err();
        assert_eq!(err, "missing --runs value");
    }

    #[test]
    fn parse_args_rejects_removed_fixed_modes() {
        assert_eq!(
            parse_args(&valid_args(&["--fixed"])).unwrap_err(),
            "invalid run mode"
        );
        assert_eq!(
            parse_args(&valid_args(&["--no-mutate-runs", "3"])).unwrap_err(),
            "invalid run mode"
        );
    }

    #[test]
    fn parse_args_accepts_fixed_input_benchmark() {
        let parsed = parse_async_args(&valid_args(&[
            "--benchmark",
            "--warmup-runs",
            "2",
            "--benchmark-seconds",
            "30",
        ]))
        .unwrap();

        assert_eq!(parsed.supply_threads, 0);
        assert_eq!(parsed.retire_workers, 0);
        assert_eq!(
            parsed.mode,
            RunMode::Benchmark {
                warmup_runs: 2,
                min_duration_secs: 30,
            }
        );
        assert_eq!(parsed.window_size, None);
    }

    #[test]
    fn parse_args_accepts_ordered_fixed_input_benchmark() {
        let parsed = parse_args(&valid_args(&[
            "--window-size",
            "4",
            "--benchmark",
            "--warmup-runs",
            "100",
            "--benchmark-seconds",
            "30",
        ]))
        .unwrap();

        assert_eq!(parsed.window_size.unwrap().get(), 4);
        assert_eq!(
            parsed.mode,
            RunMode::Benchmark {
                warmup_runs: 100,
                min_duration_secs: 30,
            }
        );
    }

    #[test]
    fn benchmark_requires_warmup_and_minimum_duration() {
        assert!(parse_args(&valid_args(&["--benchmark", "--benchmark-seconds", "30"])).is_err());
        assert!(parse_args(&valid_args(&["--benchmark", "--warmup-runs", "2"])).is_err());
        assert!(parse_args(&valid_args(&[
            "--benchmark",
            "--warmup-runs",
            "2",
            "--benchmark-seconds",
            "30",
            "--no-mutate",
        ]))
        .is_err());
        assert!(parse_args(&valid_args(&[
            "--benchmark",
            "--warmup-runs",
            "2",
            "--benchmark-seconds",
            "30",
            "--runs",
            "5",
        ]))
        .is_err());
    }

    #[test]
    fn benchmark_result_is_minimal_machine_readable_json() {
        let result = BenchmarkResult::new(1_000, 5, 17, 1_000, 1, 0, 0).unwrap();
        let line = result.machine_line().unwrap();
        let json = line.strip_prefix(BENCHMARK_RESULT_PREFIX).unwrap();
        let decoded: BenchmarkResult = serde_json::from_str(json).unwrap();

        assert_eq!(decoded, result);
        assert!(!decoded.mutation_enabled);
        assert_eq!(
            json,
            r#"{"requested_min_duration_ns":1000,"measured_iterations":5,"completed":17,"elapsed_ns":1000,"corpus_size":1,"solutions":0,"pending":0,"mutation_enabled":false}"#
        );
        assert_eq!(
            BenchmarkResult::new(1_000, 5, 0, 1_000, 1, 0, 0).unwrap_err(),
            "benchmark completed no executions"
        );
        assert_eq!(
            BenchmarkResult::new(1_000, 5, 5, 1_000, 1, 0, 1).unwrap_err(),
            "benchmark exited with 1 pending execution"
        );
        assert_eq!(
            BenchmarkResult::new(1_001, 5, 5, 1_000, 1, 0, 0).unwrap_err(),
            "benchmark ended before its minimum duration"
        );
    }

    #[test]
    fn mutating_result_reports_growth_execution_and_drain_provenance() {
        let result = MutatingResult {
            requested_seconds: 30,
            executions: 128,
            corpus_size: 3,
            solutions: 0,
            mutation_calls: 100,
            coverage_nonzero_bytes: 3,
            simt_memcov_nonzero_bits: 7,
            pending: 0,
            completed: 0,
            outstanding: 0,
            in_flight: 0,
            queued_submissions: 0,
        };
        let line = result.machine_line().unwrap();
        let json = line.strip_prefix(MUTATING_RESULT_PREFIX).unwrap();

        assert_eq!(
            serde_json::from_str::<MutatingResult>(json).unwrap(),
            result
        );
        assert_eq!(
            json,
            r#"{"requested_seconds":30,"executions":128,"corpus_size":3,"solutions":0,"mutation_calls":100,"coverage_nonzero_bytes":3,"simt_memcov_nonzero_bits":7,"pending":0,"completed":0,"outstanding":0,"in_flight":0,"queued_submissions":0}"#
        );
    }

    #[test]
    fn feedback_activity_counts_persisted_cfg_and_simt_coverage() {
        let mut state = FeedbackState::default();
        state.add_named_metadata(
            CFG_FEEDBACK_NAME,
            MapFeedbackMetadata::with_history_map(vec![0_u8, 1, 0, 3], 0),
        );
        let mut simt = BitsetNoveltyStateMetadata::default();
        simt.seen[0] = 0b101;
        state.add_metadata(simt);

        assert_eq!(
            feedback_activity(&state).unwrap(),
            FeedbackActivity {
                coverage_nonzero_bytes: 2,
                simt_memcov_nonzero_bits: 2,
            }
        );
    }

    #[test]
    fn benchmark_uses_a_fixed_rng_seed() {
        assert_eq!(BENCHMARK_RNG_SEED, 0);
    }

    #[test]
    fn coverage_campaign_stage_runs_exactly_one_evaluation() {
        assert_eq!(campaign_stage_max_iterations().get(), 1);
    }

    #[test]
    fn reexeced_profiled_client_keeps_inherited_gpu_guard() {
        let original = std::env::var_os(PROFILE_REEXEC_ENV);
        std::env::remove_var(PROFILE_REEXEC_ENV);
        assert!(gpu_client_guard_required());
        std::env::set_var(PROFILE_REEXEC_ENV, "1");
        assert!(!gpu_client_guard_required());
        if let Some(value) = original {
            std::env::set_var(PROFILE_REEXEC_ENV, value);
        } else {
            std::env::remove_var(PROFILE_REEXEC_ENV);
        }
    }
}
