use std::env::args;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

use libafl::{
    corpus::{Corpus, InMemoryOnDiskCorpus, OnDiskCorpus},
    events::{EventConfig, EventRestarter, ProgressReporter, SendExiting, SimpleEventManager},
    executors::inprocess::InProcessExecutor,
    feedback_or, feedback_or_fast,
    feedbacks::{CrashFeedback, MaxMapFeedback, TimeFeedback, TimeoutFeedback},
    fuzzer::StdFuzzer,
    inputs::{BytesInput, HasTargetBytes},
    monitors::{MultiMonitor, SimpleMonitor},
    observers::{CanTrack, HitcountsMapObserver, StdMapObserver, TimeObserver},
    schedulers::QueueScheduler,
    stages::StdMutationalStage,
    state::{HasCorpus, HasExecutions, HasSolutions, StdState},
    Error, Fuzzer,
};
use libafl_bolts::{current_nanos, rands::StdRand, tuples::tuple_list, AsSlice};
use libloading::Library;

use cuda_fuzzer::arg_pack_v1::{
    arg_pack_manifest_summary_v1, default_seed_rapid_input_v1, init_arg_pack_manifest,
    normalize_rapid_input_v1,
};
use cuda_fuzzer::async_fuzzer::DEFAULT_ASYNC_MAX_PENDING;
use cuda_fuzzer::async_mutational_stage::AsyncMutationalStage;
use cuda_fuzzer::coverage_telemetry::{CoverageLogSession, CoverageTelemetry};
use cuda_fuzzer::cuda_backend::{
    run_status_requires_fresh_process, KernelTimingStats, OrderedCudaBackend, SyncCudaBackend,
    SyncRunResult, EDGES_MAP_SIZE,
};
use cuda_fuzzer::fuzzer_frontend::{self, RunMode};
use cuda_fuzzer::fuzzer_restarting::setup_oom_safe_restarting_mgr;
use cuda_fuzzer::fuzzer_timeout::DEFAULT_TARGET_TIMEOUT;
use cuda_fuzzer::fuzzer_workdir::FuzzerWorkDir;
use cuda_fuzzer::mutators::RapidInputMutator;
use cuda_fuzzer::ordered_fuzzer::{OrderedBatchFuzzer, OrderedGpuExecutor};
#[cfg(feature = "profiling")]
use cuda_fuzzer::profiling::ProfileSession;
use cuda_fuzzer::simt_memcov_feedback::SimtMemCovFeedback;
use cuda_fuzzer::task_time_feedback::TaskTimeFeedback;

fn usage(program: &str) -> ! {
    eprintln!(
        "Usage: {program} <path_to_library> --manifest <path_to_manifest> [--window-size W] [--dump-seed] [--no-mutate] [--runs N | --mutate-seconds S] [--benchmark --warmup-runs M --benchmark-seconds S] [--coverage-seconds S --coverage-log PATH --vconfig on|off]"
    );
    eprintln!("This is the ORIGINAL (synchronous) version of the fuzzer");
    std::process::exit(1);
}

fn telemetry_error(error: anyhow::Error) -> Error {
    Error::illegal_state(error.to_string())
}

#[cfg(feature = "profiling")]
type ActiveProfile = Option<ProfileSession>;
#[cfg(not(feature = "profiling"))]
type ActiveProfile = ();

#[cfg(feature = "profiling")]
fn profiling_enabled() -> Result<bool, Error> {
    ProfileSession::enabled_from_env().map_err(Error::illegal_state)
}

#[cfg(not(feature = "profiling"))]
fn profiling_enabled() -> Result<bool, Error> {
    Ok(false)
}

#[cfg(feature = "profiling")]
fn start_profiling() -> Result<ActiveProfile, Error> {
    ProfileSession::start_from_env().map_err(Error::illegal_state)
}

#[cfg(not(feature = "profiling"))]
fn start_profiling() -> Result<ActiveProfile, Error> {
    Ok(())
}

#[cfg(feature = "profiling")]
fn finish_profiling<F>(
    profile: ActiveProfile,
    started: Instant,
    snapshot_device: F,
) -> Result<u64, Error>
where
    F: FnOnce() -> Option<KernelTimingStats>,
{
    match profile {
        Some(profile) => profile
            .stop_after_measuring(started, snapshot_device)
            .map_err(Error::illegal_state),
        None => Ok(started.elapsed().as_nanos().min(u64::MAX as u128) as u64),
    }
}

#[cfg(not(feature = "profiling"))]
fn finish_profiling<F>(
    _profile: ActiveProfile,
    started: Instant,
    _snapshot_device: F,
) -> Result<u64, Error>
where
    F: FnOnce() -> Option<KernelTimingStats>,
{
    Ok(started.elapsed().as_nanos().min(u64::MAX as u128) as u64)
}

fn canonicalize_loaded_corpus<C>(corpus: &C) -> Result<(), Error>
where
    C: Corpus<BytesInput>,
{
    let ids = corpus.ids().collect::<Vec<_>>();
    for id in ids {
        let mut testcase = corpus.get(id)?.borrow_mut();
        corpus.load_input_into(&mut testcase)?;
        let input = testcase
            .input()
            .as_ref()
            .ok_or_else(|| Error::illegal_state("loaded corpus entry has no input".to_string()))?;
        let target = input.target_bytes();
        let canonical = normalize_rapid_input_v1(target.as_slice());
        if canonical != target.as_slice() {
            testcase.set_input(BytesInput::new(canonical));
            corpus.store_input_from(&testcase)?;
        }
    }
    Ok(())
}

fn run_sync_coverage_campaign(
    cuda_backend: &mut SyncCudaBackend,
    duration_secs: u64,
    coverage_log: &std::path::Path,
    vconfig_mutation: bool,
) -> Result<(), Error> {
    let workdir = FuzzerWorkDir::new("sync-coverage").map_err(|error| {
        Error::illegal_state(format!(
            "failed to create coverage campaign workdir: {error}"
        ))
    })?;
    let monitor = SimpleMonitor::new(|message| println!("{message}"));
    let mut manager = SimpleEventManager::new(monitor);
    let edges_observer = unsafe {
        HitcountsMapObserver::new(StdMapObserver::from_mut_ptr(
            "edges",
            cuda_backend.cov_map_ptr(),
            EDGES_MAP_SIZE,
        ))
        .track_indices()
    };
    let time_observer = TimeObserver::new("time");
    let map_feedback = MaxMapFeedback::with_name("cfg_sites", &edges_observer);
    let simt_memcov_feedback = SimtMemCovFeedback::new();
    let simt_memcov_handle = simt_memcov_feedback.clone();
    let mut feedback = fuzzer_frontend::profile_feedback(feedback_or!(
        map_feedback,
        simt_memcov_feedback,
        TimeFeedback::new(&time_observer)
    ));
    let mut objective = feedback_or_fast!(CrashFeedback::new(), TimeoutFeedback::new());
    let mut state = StdState::new(
        StdRand::with_seed(
            fuzzer_frontend::fixed_rng_seed_from_env().map_err(Error::illegal_argument)?,
        ),
        InMemoryOnDiskCorpus::new(workdir.corpus_dir()).unwrap(),
        OnDiskCorpus::new(workdir.crashes_dir()).unwrap(),
        &mut feedback,
        &mut objective,
    )?;
    let scheduler = QueueScheduler::new();
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);
    let telemetry = CoverageTelemetry::new();
    let telemetry_handle = telemetry.clone();
    let mut harness = |input: &BytesInput| {
        let target = input.target_bytes();
        let normalized = normalize_rapid_input_v1(target.as_slice());
        simt_memcov_handle
            .record_submission()
            .expect("failed to record CUDA feedback submission");
        telemetry_handle
            .record_submission()
            .expect("failed to record coverage telemetry submission");
        let result = cuda_backend.run(&normalized);
        let edge_map =
            unsafe { std::slice::from_raw_parts(cuda_backend.cov_map_ptr(), EDGES_MAP_SIZE) };
        let memory_map = cuda_backend.simt_memcov_bits();
        telemetry_handle
            .record_completion(edge_map, memory_map)
            .expect("failed to record coverage telemetry completion");
        simt_memcov_handle
            .observe(memory_map)
            .expect("failed to observe synchronous CUDA feedback bitset");
        finish_sync_execution(result)
    };
    let mut executor = InProcessExecutor::with_timeout(
        &mut harness,
        tuple_list!(edges_observer, time_observer),
        &mut fuzzer,
        &mut state,
        &mut manager,
        DEFAULT_TARGET_TIMEOUT,
    )?;
    state
        .corpus_mut()
        .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    let mutator = RapidInputMutator::mutating(vconfig_mutation);
    let mut stages = tuple_list!(StdMutationalStage::with_max_iterations(
        mutator,
        fuzzer_frontend::campaign_stage_max_iterations(),
    ));

    manager.report_progress(&mut state)?;
    let mut session =
        CoverageLogSession::start(coverage_log, telemetry).map_err(telemetry_error)?;
    while session.elapsed() < Duration::from_secs(duration_secs) {
        let _ = fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut manager)?;
    }
    session.finish().map_err(telemetry_error)?;
    manager.on_restart(&mut state)?;
    fuzzer_frontend::print_arg_pack_stats();
    Ok(())
}

fn finish_sync_execution(result: SyncRunResult) -> libafl::executors::ExitKind {
    if run_status_requires_fresh_process(result.status) {
        eprintln!(
            "ORIGIN NON-RECOVERABLE STATUS: status_code={} stage={} detail={:#x}. \
             Aborting client process so the restarting manager can create a fresh CUDA context.",
            result.status.code, result.status.stage, result.status.detail,
        );
        std::process::abort();
    }
    result.exit_kind
}

fn run_bounded_no_mutate_fuzz_loop(
    cuda_backend: &mut SyncCudaBackend,
    warmup_runs: u64,
    runs: u64,
    benchmark_min_duration_secs: Option<u64>,
) -> Result<(), Error> {
    let profile_timing_start = cuda_backend.profile_timing_start_handle();
    let benchmark = benchmark_min_duration_secs.is_some();
    let workdir = FuzzerWorkDir::new("sync-no-mutate").map_err(|err| {
        Error::illegal_state(format!("failed to create bounded fuzzer workdir: {err}"))
    })?;
    let monitor = SimpleMonitor::new(move |s| {
        if !benchmark {
            println!("{s}");
        }
    });
    let mut mgr = SimpleEventManager::new(monitor);

    let edges_observer = unsafe {
        HitcountsMapObserver::new(StdMapObserver::from_mut_ptr(
            "edges",
            cuda_backend.cov_map_ptr(),
            EDGES_MAP_SIZE,
        ))
        .track_indices()
    };
    let time_observer = TimeObserver::new("time");

    let map_feedback = MaxMapFeedback::with_name("cfg_sites", &edges_observer);
    let simt_memcov_feedback = SimtMemCovFeedback::new();
    let simt_memcov_handle = simt_memcov_feedback.clone();
    let mut feedback = fuzzer_frontend::profile_feedback(feedback_or!(
        map_feedback,
        simt_memcov_feedback,
        TimeFeedback::new(&time_observer)
    ));
    let mut objective = feedback_or_fast!(CrashFeedback::new(), TimeoutFeedback::new());

    let mut state = StdState::new(
        StdRand::with_seed(if benchmark {
            fuzzer_frontend::BENCHMARK_RNG_SEED
        } else {
            current_nanos()
        }),
        InMemoryOnDiskCorpus::new(workdir.corpus_dir()).unwrap(),
        OnDiskCorpus::new(workdir.crashes_dir()).unwrap(),
        &mut feedback,
        &mut objective,
    )?;

    let scheduler = QueueScheduler::new();
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    let mut harness = |input: &BytesInput| {
        let target = input.target_bytes();
        let normalized = normalize_rapid_input_v1(target.as_slice());
        simt_memcov_handle
            .record_submission()
            .expect("failed to record CUDA feedback submission");
        let result = cuda_backend.run(&normalized);
        simt_memcov_handle
            .observe(cuda_backend.simt_memcov_bits())
            .expect("failed to observe synchronous CUDA feedback bitset");
        finish_sync_execution(result)
    };
    let mut executor = InProcessExecutor::with_timeout(
        &mut harness,
        tuple_list!(edges_observer, time_observer),
        &mut fuzzer,
        &mut state,
        &mut mgr,
        DEFAULT_TARGET_TIMEOUT,
    )?;

    state
        .corpus_mut()
        .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    let mutator = RapidInputMutator::fixed();
    let mut stages = tuple_list!(StdMutationalStage::new(mutator));

    mgr.report_progress(&mut state)?;
    if warmup_runs > 0 {
        let _ = fuzzer.fuzz_loop_for(
            &mut stages,
            &mut executor,
            &mut state,
            &mut mgr,
            warmup_runs,
        )?;
    }
    if let Some(seconds) = benchmark_min_duration_secs {
        println!(
            "Running fixed-input LibAFL benchmark: warmup_runs={warmup_runs}, minimum_seconds={seconds}, workdir={}",
            workdir.root().display()
        );
    } else {
        println!(
            "Running bounded fixed-input LibAFL loop: runs={runs}, workdir={}",
            workdir.root().display()
        );
    }
    let completed_before = *state.executions();
    let profile_enabled = profiling_enabled()?;
    if profile_enabled {
        profile_timing_start
            .start()
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    let profile = start_profiling()?;
    let started = Instant::now();
    let measured_iterations = if let Some(seconds) = benchmark_min_duration_secs {
        let min_duration = Duration::from_secs(seconds);
        let mut iterations = 0u64;
        while started.elapsed() < min_duration {
            let _ = fuzzer.fuzz_loop_for(
                &mut stages,
                &mut executor,
                &mut state,
                &mut mgr,
                fuzzer_frontend::BENCHMARK_CHUNK_ITERATIONS,
            )?;
            iterations = iterations.saturating_add(fuzzer_frontend::BENCHMARK_CHUNK_ITERATIONS);
        }
        iterations
    } else {
        let _ = fuzzer.fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut mgr, runs)?;
        runs
    };
    drop(executor);
    drop(harness);
    let elapsed_ns = finish_profiling(profile, started, || {
        let device = profile_enabled
            .then(|| profile_timing_start.snapshot())
            .flatten();
        cuda_backend.stop();
        device
    })?;
    let completed = state
        .executions()
        .checked_sub(completed_before)
        .ok_or_else(|| Error::illegal_state("execution counter regressed".to_string()))?;
    if let Some(seconds) = benchmark_min_duration_secs {
        let requested_min_duration_ns = Duration::from_secs(seconds)
            .as_nanos()
            .min(u64::MAX as u128) as u64;
        let result = fuzzer_frontend::BenchmarkResult::new(
            requested_min_duration_ns,
            measured_iterations,
            completed,
            elapsed_ns,
            state.corpus().count() as u64,
            state.solutions().count() as u64,
            0,
        )
        .map_err(Error::illegal_state)?;
        println!(
            "{}",
            result
                .machine_line()
                .map_err(|err| Error::illegal_state(err.to_string()))?
        );
    }
    mgr.on_restart(&mut state)?;
    fuzzer_frontend::print_arg_pack_stats();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use libafl::corpus::InMemoryCorpus;
    use std::{
        fs,
        path::{Path, PathBuf},
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn c_compiler() -> Option<&'static str> {
        ["cc", "clang", "gcc"]
            .into_iter()
            .find(|compiler| Command::new(compiler).arg("--version").output().is_ok())
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{}-{nanos}", std::process::id()))
    }

    fn compile_shared_library(compiler: &str, temp_dir: &Path, source: &str) -> PathBuf {
        fs::create_dir_all(temp_dir).expect("create temp test directory");
        let source_path = temp_dir.join("backend.c");
        let library_path = temp_dir.join("libbackend.so");
        fs::write(&source_path, source).expect("write backend source");
        let output = Command::new(compiler)
            .args(["-shared", "-fPIC"])
            .arg(&source_path)
            .arg("-o")
            .arg(&library_path)
            .output()
            .expect("compile backend source");
        assert!(
            output.status.success(),
            "failed to compile backend stub: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        library_path
    }

    #[test]
    fn sync_imports_are_canonical_before_mutation() {
        let temp_dir = unique_temp_dir("rapid-sync-import-canonical");
        fs::create_dir_all(&temp_dir).unwrap();
        let manifest_path = temp_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            r#"{
              "schema_version": 1,
              "kernels": [{
                "symbol_name": "sync_import_kernel",
                "display_name": "sync_import_kernel",
                "args": [],
                "constraints": []
              }]
            }"#,
        )
        .unwrap();
        init_arg_pack_manifest(&manifest_path).unwrap();
        let mut corpus = InMemoryCorpus::<BytesInput>::new();
        let id = corpus
            .add(BytesInput::new(vec![0xff, 0xee, 0xdd]).into())
            .unwrap();

        canonicalize_loaded_corpus(&corpus).unwrap();

        let input = corpus.cloned_input_for_id(id).unwrap();
        let bytes = input.target_bytes();
        assert_eq!(bytes.as_slice(), normalize_rapid_input_v1(bytes.as_slice()));
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn library_without_sync_target_selects_default_ordered_backend() {
        let Some(compiler) = c_compiler() else {
            eprintln!("skipping backend selection test: no C compiler found");
            return;
        };
        let temp_dir = unique_temp_dir("rapid-backend-selection-ordered");
        let library_path = compile_shared_library(
            compiler,
            &temp_dir,
            r#"
#include <stddef.h>
#include <stdint.h>

uint64_t libafl_submit_with_id(const uint8_t *input, size_t size) {
  (void)input;
  (void)size;
  return 1;
}
"#,
        );

        let selected = select_backend_mode(library_path.to_str().unwrap(), None);

        assert_eq!(
            selected,
            BackendMode::Ordered(NonZeroUsize::new(DEFAULT_ASYNC_MAX_PENDING).unwrap())
        );
    }

    #[test]
    fn explicit_window_selects_ordered_backend_even_with_sync_target() {
        let Some(compiler) = c_compiler() else {
            eprintln!("skipping backend selection test: no C compiler found");
            return;
        };
        let temp_dir = unique_temp_dir("rapid-backend-selection-explicit");
        let library_path = compile_shared_library(
            compiler,
            &temp_dir,
            r#"
#include <stddef.h>
#include <stdint.h>

void libafl_target(const uint8_t *input, size_t size) {
  (void)input;
  (void)size;
}
"#,
        );

        let selected = select_backend_mode(library_path.to_str().unwrap(), NonZeroUsize::new(4));

        assert_eq!(
            selected,
            BackendMode::Ordered(NonZeroUsize::new(4).unwrap())
        );
    }

    #[test]
    fn sync_target_selects_sync_backend_without_explicit_window() {
        let Some(compiler) = c_compiler() else {
            eprintln!("skipping backend selection test: no C compiler found");
            return;
        };
        let temp_dir = unique_temp_dir("rapid-backend-selection-sync");
        let library_path = compile_shared_library(
            compiler,
            &temp_dir,
            r#"
#include <stddef.h>
#include <stdint.h>

void libafl_target(const uint8_t *input, size_t size) {
  (void)input;
  (void)size;
}
"#,
        );

        let selected = select_backend_mode(library_path.to_str().unwrap(), None);

        assert_eq!(selected, BackendMode::Sync);
    }
}

fn load_cuda_backend(lib_path: &str) -> SyncCudaBackend {
    let cuda_backend =
        SyncCudaBackend::new(lib_path).expect("Failed to load synchronous CUDA backend");
    eprintln!(
        "RAPID FUZZER CLIENT LOADED CUDA BACKEND pid={} lib={}",
        std::process::id(),
        lib_path
    );
    cuda_backend
}

fn load_ordered_cuda_backend(lib_path: &str) -> Rc<OrderedCudaBackend> {
    let backend = Rc::new(
        OrderedCudaBackend::new(lib_path)
            .expect("Failed to load rapid ordered-window CUDA backend"),
    );
    backend.set_target_timeout(DEFAULT_TARGET_TIMEOUT);
    eprintln!(
        "RAPID FUZZER CLIENT LOADED ORDERED CUDA BACKEND pid={} lib={}",
        std::process::id(),
        lib_path
    );
    backend
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendMode {
    Sync,
    Ordered(NonZeroUsize),
}

fn cuda_backend_exports_sync_target(lib_path: &str) -> bool {
    let library = unsafe { Library::new(lib_path) }
        .unwrap_or_else(|error| panic!("Failed to inspect CUDA backend ABI: {error}"));
    unsafe {
        library
            .get::<unsafe extern "C" fn(*const u8, usize)>(b"libafl_target")
            .is_ok()
    }
}

fn select_backend_mode(
    lib_path: &str,
    configured_window_size: Option<NonZeroUsize>,
) -> BackendMode {
    if let Some(window_size) = configured_window_size {
        return BackendMode::Ordered(window_size);
    }
    if cuda_backend_exports_sync_target(lib_path) {
        BackendMode::Sync
    } else {
        BackendMode::Ordered(fuzzer_frontend::effective_window_size(
            None,
            DEFAULT_ASYNC_MAX_PENDING,
        ))
    }
}

fn run_bounded_ordered_no_mutate_fuzz_loop(
    cuda_backend: Rc<OrderedCudaBackend>,
    window_size: NonZeroUsize,
    warmup_runs: u64,
    runs: u64,
    benchmark_min_duration_secs: Option<u64>,
) -> Result<(), Error> {
    let benchmark = benchmark_min_duration_secs.is_some();
    let workdir = FuzzerWorkDir::new("ordered-no-mutate").map_err(|error| {
        Error::illegal_state(format!("failed to create ordered fuzzer workdir: {error}"))
    })?;
    let monitor = SimpleMonitor::new(move |message| {
        if !benchmark {
            println!("{message}");
        }
    });
    let mut manager = SimpleEventManager::new(monitor);
    let edges_observer = unsafe {
        HitcountsMapObserver::new(StdMapObserver::from_mut_ptr(
            "edges",
            cuda_backend.cov_map_ptr(),
            EDGES_MAP_SIZE,
        ))
        .track_indices()
    };
    let map_feedback = MaxMapFeedback::with_name("cfg_sites", &edges_observer);
    let simt_memcov_feedback = SimtMemCovFeedback::new();
    let simt_memcov_handle = simt_memcov_feedback.clone();
    let task_time_feedback = TaskTimeFeedback::new();
    let task_time_handle = task_time_feedback.clone();
    let mut feedback = fuzzer_frontend::profile_feedback(feedback_or!(
        map_feedback,
        simt_memcov_feedback,
        task_time_feedback
    ));
    let mut objective = feedback_or_fast!(CrashFeedback::new(), TimeoutFeedback::new());
    let mut state = StdState::new(
        StdRand::with_seed(if benchmark {
            fuzzer_frontend::BENCHMARK_RNG_SEED
        } else {
            current_nanos()
        }),
        InMemoryOnDiskCorpus::<BytesInput>::new(workdir.corpus_dir()).unwrap(),
        OnDiskCorpus::new(workdir.crashes_dir()).unwrap(),
        &mut feedback,
        &mut objective,
    )?;
    let scheduler = QueueScheduler::new();
    let mut fuzzer = OrderedBatchFuzzer::new(
        scheduler,
        feedback,
        objective,
        Rc::clone(&cuda_backend),
        simt_memcov_handle,
        task_time_handle,
    );
    let mut executor = OrderedGpuExecutor::new(
        Rc::clone(&cuda_backend),
        tuple_list!(edges_observer),
        window_size,
    );
    state
        .corpus_mut()
        .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    let mutator = RapidInputMutator::fixed();
    let mut stages = tuple_list!(AsyncMutationalStage::new(mutator));

    manager.report_progress(&mut state)?;
    if warmup_runs > 0 {
        let _ = fuzzer.fuzz_loop_for(
            &mut stages,
            &mut executor,
            &mut state,
            &mut manager,
            warmup_runs,
        )?;
    }
    if let Some(seconds) = benchmark_min_duration_secs {
        println!(
            "Running ordered fixed-input benchmark: window_size={}, warmup_runs={warmup_runs}, minimum_seconds={seconds}, workdir={}",
            window_size.get(),
            workdir.root().display()
        );
    } else {
        println!(
            "Running bounded ordered fixed-input loop: window_size={}, runs={runs}, workdir={}",
            window_size.get(),
            workdir.root().display()
        );
    }
    let completed_before = *state.executions();
    let profile_enabled = profiling_enabled()?;
    if profile_enabled {
        cuda_backend
            .profile_timing_start()
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    let profile = start_profiling()?;
    let started = Instant::now();
    let measured_iterations = if let Some(seconds) = benchmark_min_duration_secs {
        let minimum = Duration::from_secs(seconds);
        let mut iterations = 0u64;
        while started.elapsed() < minimum {
            let _ = fuzzer.fuzz_loop_for(
                &mut stages,
                &mut executor,
                &mut state,
                &mut manager,
                fuzzer_frontend::BENCHMARK_CHUNK_ITERATIONS,
            )?;
            iterations = iterations.saturating_add(fuzzer_frontend::BENCHMARK_CHUNK_ITERATIONS);
        }
        iterations
    } else {
        let _ = fuzzer.fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut manager, runs)?;
        runs
    };
    fuzzer.force_evaluation(&mut executor, &mut state, &mut manager)?;
    let elapsed_ns = finish_profiling(profile, started, || {
        let device = profile_enabled
            .then(|| cuda_backend.profile_timing_snapshot())
            .flatten();
        cuda_backend.stop();
        device
    })?;
    let completed = state
        .executions()
        .checked_sub(completed_before)
        .ok_or_else(|| Error::illegal_state("execution counter regressed".to_string()))?;
    let counts = cuda_backend.get_queue_counts();
    let pending = counts
        .pending
        .saturating_add(counts.completed)
        .saturating_add(counts.outstanding) as u64;
    if let Some(seconds) = benchmark_min_duration_secs {
        let result = fuzzer_frontend::BenchmarkResult::new(
            Duration::from_secs(seconds)
                .as_nanos()
                .min(u64::MAX as u128) as u64,
            measured_iterations,
            completed,
            elapsed_ns,
            state.corpus().count() as u64,
            state.solutions().count() as u64,
            pending,
        )
        .map_err(Error::illegal_state)?;
        println!(
            "{}",
            result
                .machine_line()
                .map_err(|error| Error::illegal_state(error.to_string()))?
        );
    }
    println!(
        "Ordered final statistics: window_size={} submitted={} completed={} pending={} outstanding={} corpus={} solutions={}",
        window_size.get(),
        executor.total_submitted(),
        completed,
        counts.pending.saturating_add(counts.completed),
        counts.outstanding,
        state.corpus().count(),
        state.solutions().count(),
    );
    fuzzer_frontend::print_arg_pack_stats();
    manager.on_restart(&mut state)?;
    Ok(())
}

fn run_ordered_coverage_campaign(
    cuda_backend: Rc<OrderedCudaBackend>,
    window_size: NonZeroUsize,
    duration_secs: u64,
    coverage_log: &std::path::Path,
    vconfig_mutation: bool,
) -> Result<(), Error> {
    let workdir = FuzzerWorkDir::new("ordered-coverage").map_err(|error| {
        Error::illegal_state(format!(
            "failed to create ordered coverage workdir: {error}"
        ))
    })?;
    let monitor = SimpleMonitor::new(|message| println!("{message}"));
    let mut manager = SimpleEventManager::new(monitor);
    let edges_observer = unsafe {
        HitcountsMapObserver::new(StdMapObserver::from_mut_ptr(
            "edges",
            cuda_backend.cov_map_ptr(),
            EDGES_MAP_SIZE,
        ))
        .track_indices()
    };
    let map_feedback = MaxMapFeedback::with_name("cfg_sites", &edges_observer);
    let simt_memcov_feedback = SimtMemCovFeedback::new();
    let simt_memcov_handle = simt_memcov_feedback.clone();
    let task_time_feedback = TaskTimeFeedback::new();
    let task_time_handle = task_time_feedback.clone();
    let mut feedback = fuzzer_frontend::profile_feedback(feedback_or!(
        map_feedback,
        simt_memcov_feedback,
        task_time_feedback
    ));
    let mut objective = feedback_or_fast!(CrashFeedback::new(), TimeoutFeedback::new());
    let mut state = StdState::new(
        StdRand::with_seed(
            fuzzer_frontend::fixed_rng_seed_from_env().map_err(Error::illegal_argument)?,
        ),
        InMemoryOnDiskCorpus::<BytesInput>::new(workdir.corpus_dir()).unwrap(),
        OnDiskCorpus::new(workdir.crashes_dir()).unwrap(),
        &mut feedback,
        &mut objective,
    )?;
    let telemetry = CoverageTelemetry::new();
    let scheduler = QueueScheduler::new();
    let mut fuzzer = OrderedBatchFuzzer::new(
        scheduler,
        feedback,
        objective,
        Rc::clone(&cuda_backend),
        simt_memcov_handle,
        task_time_handle,
    )
    .with_coverage_telemetry(telemetry.clone());
    let mut executor = OrderedGpuExecutor::new(
        Rc::clone(&cuda_backend),
        tuple_list!(edges_observer),
        window_size,
    );
    state
        .corpus_mut()
        .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    let mutator = RapidInputMutator::mutating(vconfig_mutation);
    let mut stages = tuple_list!(AsyncMutationalStage::with_max_iterations(
        mutator,
        fuzzer_frontend::campaign_stage_max_iterations(),
    ));

    manager.report_progress(&mut state)?;
    let mut session =
        CoverageLogSession::start(coverage_log, telemetry).map_err(telemetry_error)?;
    while session.elapsed() < Duration::from_secs(duration_secs) {
        let _ = fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut manager)?;
    }
    fuzzer.force_evaluation(&mut executor, &mut state, &mut manager)?;
    session.finish().map_err(telemetry_error)?;
    let counts = cuda_backend.get_queue_counts();
    println!(
        "Ordered coverage campaign complete: window_size={} submitted={} completed={} pending={} outstanding={}",
        window_size.get(),
        executor.total_submitted(),
        *state.executions(),
        counts.pending.saturating_add(counts.completed),
        counts.outstanding,
    );
    manager.on_restart(&mut state)?;
    fuzzer_frontend::print_arg_pack_stats();
    Ok(())
}

fn run_ordered_fuzzer(
    lib_path: &str,
    mode: RunMode,
    window_size: NonZeroUsize,
) -> Result<(), Error> {
    if let RunMode::CoverageCampaign {
        duration_secs,
        coverage_log,
        vconfig_mutation,
    } = &mode
    {
        let cuda_backend = load_ordered_cuda_backend(lib_path);
        return run_ordered_coverage_campaign(
            cuda_backend,
            window_size,
            *duration_secs,
            coverage_log,
            *vconfig_mutation,
        );
    }
    if let RunMode::Fuzz {
        mutate: false,
        runs: Some(runs),
        ..
    } = mode
    {
        let cuda_backend = load_ordered_cuda_backend(lib_path);
        return run_bounded_ordered_no_mutate_fuzz_loop(cuda_backend, window_size, 0, runs, None);
    }
    if let RunMode::Benchmark {
        warmup_runs,
        min_duration_secs,
    } = mode
    {
        let cuda_backend = load_ordered_cuda_backend(lib_path);
        return run_bounded_ordered_no_mutate_fuzz_loop(
            cuda_backend,
            window_size,
            warmup_runs,
            0,
            Some(min_duration_secs),
        );
    }
    let bounded_fuzz = mode.is_bounded_fuzz();
    let mutating_seconds = mode.mutating_seconds();

    let monitor = MultiMonitor::new(|message| println!("{message}"));
    let (saved_state, mut manager) =
        match setup_oom_safe_restarting_mgr(monitor, 1337, EventConfig::AlwaysUnique) {
            Ok(result) => result,
            Err(Error::ShuttingDown) => return Ok(()),
            Err(error) => panic!("Failed to setup the ordered restarter: {error}"),
        };
    let profile_enabled = profiling_enabled()?;
    fuzzer_frontend::reexec_profiled_client_after_fork(profile_enabled)
        .map_err(Error::illegal_state)?;
    let cuda_backend = load_ordered_cuda_backend(lib_path);
    let edges_observer = unsafe {
        HitcountsMapObserver::new(StdMapObserver::from_mut_ptr(
            "edges",
            cuda_backend.cov_map_ptr(),
            EDGES_MAP_SIZE,
        ))
        .track_indices()
    };
    let map_feedback = MaxMapFeedback::with_name("cfg_sites", &edges_observer);
    let simt_memcov_feedback = SimtMemCovFeedback::new();
    let simt_memcov_handle = simt_memcov_feedback.clone();
    let task_time_feedback = TaskTimeFeedback::new();
    let task_time_handle = task_time_feedback.clone();
    let mut feedback = fuzzer_frontend::profile_feedback(feedback_or!(
        map_feedback,
        simt_memcov_feedback,
        task_time_feedback
    ));
    let mut objective = feedback_or_fast!(CrashFeedback::new(), TimeoutFeedback::new());
    let mut state = saved_state.unwrap_or_else(|| {
        let seed = std::env::var("RAPID_FIXED_SEED")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or_else(current_nanos);
        StdState::new(
            StdRand::with_seed(seed),
            InMemoryOnDiskCorpus::<BytesInput>::new(PathBuf::from("./corpus")).unwrap(),
            OnDiskCorpus::new(PathBuf::from("./crashes")).unwrap(),
            &mut feedback,
            &mut objective,
        )
        .unwrap()
    });
    let scheduler = QueueScheduler::new();
    let mut fuzzer = OrderedBatchFuzzer::new(
        scheduler,
        feedback,
        objective,
        Rc::clone(&cuda_backend),
        simt_memcov_handle,
        task_time_handle,
    );
    let mut executor = OrderedGpuExecutor::new(
        Rc::clone(&cuda_backend),
        tuple_list!(edges_observer),
        window_size,
    );
    let mutator = match mode {
        RunMode::Fuzz { mutate: false, .. } => RapidInputMutator::fixed(),
        _ => RapidInputMutator::default(),
    };
    let mut stages = tuple_list!(if mutating_seconds.is_some() {
        AsyncMutationalStage::with_max_iterations(
            mutator,
            fuzzer_frontend::campaign_stage_max_iterations(),
        )
    } else {
        AsyncMutationalStage::new(mutator)
    });

    if state.must_load_initial_inputs() {
        state
            .load_initial_inputs(
                &mut fuzzer,
                &mut executor,
                &mut manager,
                &[PathBuf::from("./corpus")],
            )
            .unwrap_or_else(|error| {
                eprintln!("Failed to load ordered initial corpus: {error}");
                std::process::exit(1);
            });
    }
    if state.corpus().count() == 0 {
        state
            .corpus_mut()
            .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    }
    println!(
        "=== RAPID Ordered Fuzzer: window_size={} FIFO execution / FIFO feedback retirement ===",
        window_size.get()
    );
    if profile_enabled {
        cuda_backend
            .profile_timing_start()
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    let profile = start_profiling()?;
    let profile_started = Instant::now();
    manager.report_progress(&mut state)?;
    match mode {
        RunMode::Fuzz {
            runs: Some(runs), ..
        } => {
            let _ =
                fuzzer.fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut manager, runs)?;
        }
        RunMode::Fuzz {
            duration_secs: Some(seconds),
            ..
        } => {
            while profile_started.elapsed() < Duration::from_secs(seconds) {
                let _ = fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut manager)?;
            }
        }
        _ => fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut manager)?,
    }
    fuzzer.force_evaluation(&mut executor, &mut state, &mut manager)?;
    let counts = cuda_backend.get_queue_counts();
    if counts.pending != 0 || counts.completed != 0 || counts.outstanding != 0 {
        return Err(Error::illegal_state(format!(
            "ordered drain left pending={} completed={} outstanding={}",
            counts.pending, counts.completed, counts.outstanding
        )));
    }
    let feedback_activity = fuzzer_frontend::feedback_activity(&state)?;
    let _elapsed_ns = finish_profiling(profile, profile_started, || {
        let device = profile_enabled
            .then(|| cuda_backend.profile_timing_snapshot())
            .flatten();
        cuda_backend.stop();
        device
    })?;
    fuzzer_frontend::print_final_statistics(
        state.corpus().count(),
        state.solutions().count(),
        *state.executions(),
    );
    if let Some(requested_seconds) = mutating_seconds {
        fuzzer_frontend::print_mutating_result(&fuzzer_frontend::mutating_result(
            requested_seconds,
            *state.executions(),
            state.corpus().count() as u64,
            state.solutions().count() as u64,
            feedback_activity,
            counts.pending as u64,
            counts.completed as u64,
            counts.outstanding as u64,
            0,
            0,
        ))?;
    }
    if bounded_fuzz {
        manager.on_shutdown()?;
    } else {
        manager.on_restart(&mut state)?;
    }
    fuzzer_frontend::print_arg_pack_stats();
    Ok(())
}

fn main() -> Result<(), Error> {
    // --- Setup ---
    #[cfg(debug_assertions)]
    env_logger::init();

    let args: Vec<String> = args().collect();
    let parsed = fuzzer_frontend::parse_args(&args).unwrap_or_else(|_| usage(&args[0]));

    init_arg_pack_manifest(&parsed.manifest_path).unwrap_or_else(|err| {
        eprintln!(
            "Failed to initialize manifest-driven arg-pack spec from {}: {err}",
            parsed.manifest_path.display()
        );
        std::process::exit(1);
    });
    if parsed.mode == RunMode::DumpSeed {
        fuzzer_frontend::print_seed_hex();
        return Ok(());
    }
    println!(
        "Loaded manifest: {} ({})",
        parsed.manifest_path.display(),
        arg_pack_manifest_summary_v1()
    );

    if let BackendMode::Ordered(window_size) =
        select_backend_mode(&parsed.lib_path, parsed.window_size)
    {
        return run_ordered_fuzzer(&parsed.lib_path, parsed.mode, window_size);
    }

    if let RunMode::CoverageCampaign {
        duration_secs,
        coverage_log,
        vconfig_mutation,
    } = &parsed.mode
    {
        let mut cuda_backend = load_cuda_backend(&parsed.lib_path);
        return run_sync_coverage_campaign(
            &mut cuda_backend,
            *duration_secs,
            coverage_log,
            *vconfig_mutation,
        );
    }

    if let RunMode::Fuzz {
        mutate: false,
        runs: Some(runs),
        ..
    } = parsed.mode
    {
        let mut cuda_backend = load_cuda_backend(&parsed.lib_path);
        return run_bounded_no_mutate_fuzz_loop(&mut cuda_backend, 0, runs, None);
    }
    if let RunMode::Benchmark {
        warmup_runs,
        min_duration_secs,
    } = parsed.mode
    {
        let mut cuda_backend = load_cuda_backend(&parsed.lib_path);
        return run_bounded_no_mutate_fuzz_loop(
            &mut cuda_backend,
            warmup_runs,
            0,
            Some(min_duration_secs),
        );
    }
    let bounded_fuzz = parsed.mode.is_bounded_fuzz();
    let mutating_seconds = parsed.mode.mutating_seconds();

    // --- Monitor ---
    let monitor = MultiMonitor::new(|s| println!("{}", s));

    // --- Event Manager ---
    let (state, mut restarting_mgr) =
        match setup_oom_safe_restarting_mgr(monitor, 1337, EventConfig::AlwaysUnique) {
            Ok(res) => res,
            Err(err) => match err {
                Error::ShuttingDown => {
                    return Ok(());
                }
                _ => {
                    panic!("Failed to setup the restarter: {err}");
                }
            },
        };
    let profile_enabled = profiling_enabled()?;
    fuzzer_frontend::reexec_profiled_client_after_fork(profile_enabled)
        .map_err(Error::illegal_state)?;

    let mut cuda_backend = load_cuda_backend(&parsed.lib_path);
    let profile_timing_start = cuda_backend.profile_timing_start_handle();

    // --- Observers ---
    let edges_observer = unsafe {
        HitcountsMapObserver::new(StdMapObserver::from_mut_ptr(
            "edges",
            cuda_backend.cov_map_ptr(),
            EDGES_MAP_SIZE,
        ))
        .track_indices()
    };
    let time_observer = TimeObserver::new("time");

    // --- Feedbacks & Objectives ---
    let map_feedback = MaxMapFeedback::with_name("cfg_sites", &edges_observer);
    let simt_memcov_feedback = SimtMemCovFeedback::new();
    let simt_memcov_handle = simt_memcov_feedback.clone();
    let mut feedback = fuzzer_frontend::profile_feedback(feedback_or!(
        map_feedback,
        simt_memcov_feedback,
        TimeFeedback::new(&time_observer)
    ));
    let mut objective = feedback_or_fast!(CrashFeedback::new(), TimeoutFeedback::new());

    // --- State ---
    let mut state = state.unwrap_or_else(|| {
        // Use fixed seed for deterministic testing when RAPID_FIXED_SEED env var is set
        let seed = std::env::var("RAPID_FIXED_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(current_nanos);
        StdState::new(
            StdRand::with_seed(seed),
            InMemoryOnDiskCorpus::new(PathBuf::from("./corpus")).unwrap(),
            OnDiskCorpus::new(PathBuf::from("./crashes")).unwrap(),
            &mut feedback,
            &mut objective,
        )
        .unwrap()
    });

    // --- Scheduler ---
    let scheduler = QueueScheduler::new();

    // --- Fuzzer ---
    let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

    // --- Harness & Executor ---
    println!("=== Starting ORIGINAL Fuzzer ===");
    println!("Using synchronous execution (baseline)");

    let mut harness = |input: &BytesInput| {
        let target = input.target_bytes();
        let normalized = normalize_rapid_input_v1(target.as_slice());
        simt_memcov_handle
            .record_submission()
            .expect("failed to record CUDA feedback submission");
        let result = cuda_backend.run(&normalized);
        simt_memcov_handle
            .observe(cuda_backend.simt_memcov_bits())
            .expect("failed to observe synchronous CUDA feedback bitset");
        finish_sync_execution(result)
    };
    let mut executor = InProcessExecutor::with_timeout(
        &mut harness,
        tuple_list!(edges_observer, time_observer),
        &mut fuzzer,
        &mut state,
        &mut restarting_mgr,
        DEFAULT_TARGET_TIMEOUT,
    )?;

    // --- Corpus Loading ---
    if state.must_load_initial_inputs() {
        state
            .load_initial_inputs(
                &mut fuzzer,
                &mut executor,
                &mut restarting_mgr,
                &[PathBuf::from("./corpus")],
            )
            .unwrap_or_else(|err| {
                eprintln!("Failed to load initial corpus: {err}");
                std::process::exit(1);
            });
        println!("Loaded {} initial inputs.", state.corpus().count());
    }

    // If corpus is empty, add a default seed
    if state.corpus().count() == 0 {
        println!("Corpus is empty, adding default seed...");
        let seed = BytesInput::new(default_seed_rapid_input_v1());
        state.corpus_mut().add(seed.into())?;
    }
    // The sync harness repairs only its submission copy. Persist canonical
    // bytes for imported and resumed corpus entries before mutation begins.
    canonicalize_loaded_corpus(state.corpus())?;

    // --- Mutators & Stages ---
    let mutator = match parsed.mode {
        RunMode::Fuzz { mutate: false, .. } => RapidInputMutator::fixed(),
        _ => RapidInputMutator::default(),
    };
    let mut stages = tuple_list!(if mutating_seconds.is_some() {
        StdMutationalStage::with_max_iterations(
            mutator,
            fuzzer_frontend::campaign_stage_max_iterations(),
        )
    } else {
        StdMutationalStage::new(mutator)
    });

    // Trigger an initial progress report so that, with `introspection` enabled,
    // the monitor receives the first `UpdatePerfMonitor` event immediately
    // (otherwise it will show NaN until the first timed report is due).
    if profile_enabled {
        profile_timing_start
            .start()
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    let profile = start_profiling()?;
    let profile_started = Instant::now();
    restarting_mgr.report_progress(&mut state)?;

    // --- Fuzzing Loop ---
    match parsed.mode {
        RunMode::Fuzz {
            runs: Some(runs), ..
        } => {
            let _ = fuzzer.fuzz_loop_for(
                &mut stages,
                &mut executor,
                &mut state,
                &mut restarting_mgr,
                runs,
            )?;
        }
        RunMode::Fuzz {
            duration_secs: Some(seconds),
            ..
        } => {
            while profile_started.elapsed() < Duration::from_secs(seconds) {
                let _ =
                    fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut restarting_mgr)?;
            }
        }
        _ => fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut restarting_mgr)?,
    }

    let feedback_activity = fuzzer_frontend::feedback_activity(&state)?;
    drop(executor);
    drop(harness);
    let _elapsed_ns = finish_profiling(profile, profile_started, || {
        let device = profile_enabled
            .then(|| profile_timing_start.snapshot())
            .flatten();
        cuda_backend.stop();
        device
    })?;
    fuzzer_frontend::print_final_statistics(
        state.corpus().count(),
        state.solutions().count(),
        *state.executions(),
    );
    if let Some(requested_seconds) = mutating_seconds {
        fuzzer_frontend::print_mutating_result(&fuzzer_frontend::mutating_result(
            requested_seconds,
            *state.executions(),
            state.corpus().count() as u64,
            state.solutions().count() as u64,
            feedback_activity,
            0,
            0,
            0,
            0,
            0,
        ))?;
    }

    // --- Graceful Exit ---
    if bounded_fuzz {
        restarting_mgr.on_shutdown()?;
    } else {
        restarting_mgr.on_restart(&mut state)?;
    }
    fuzzer_frontend::print_arg_pack_stats();
    Ok(())
}
