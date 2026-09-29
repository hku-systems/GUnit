use std::collections::VecDeque;
use std::env::args;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use libafl::{
    corpus::{Corpus, InMemoryOnDiskCorpus, OnDiskCorpus},
    events::{EventConfig, EventRestarter, ProgressReporter, SendExiting, SimpleEventManager},
    feedback_or, feedback_or_fast,
    feedbacks::{CrashFeedback, MaxMapFeedback, TimeoutFeedback},
    inputs::{BytesInput, HasTargetBytes},
    monitors::{MultiMonitor, SimpleMonitor},
    observers::{CanTrack, StdMapObserver},
    schedulers::QueueScheduler,
    state::{HasCorpus, HasExecutions, HasSolutions, StdState},
    Error, Fuzzer,
};
use libafl_bolts::{current_nanos, rands::StdRand, tuples::tuple_list, AsSlice};

use cuda_fuzzer::arg_pack_v1::{
    arg_pack_manifest_summary_v1, default_seed_rapid_input_v1, init_arg_pack_manifest,
    normalize_rapid_input_v1,
};
use cuda_fuzzer::async_fuzzer::{AsyncBatchFuzzer, DEFAULT_ASYNC_MAX_PENDING};
use cuda_fuzzer::async_mutational_stage::AsyncMutationalStage;
use cuda_fuzzer::async_mutational_stage::DEFAULT_MUTATIONAL_MAX_ITERATIONS;
use cuda_fuzzer::coverage_telemetry::{CoverageLogSession, CoverageTelemetry};
use cuda_fuzzer::cuda_backend::{
    run_status_requires_fresh_process, AsyncCudaBackend, KernelTimingStats, LibAflQueueCounts,
    EDGES_MAP_SIZE, LIBAFL_RUN_STATUS_OK,
};
use cuda_fuzzer::fuzzer_frontend::{
    self, RunMode, DEFAULT_ASYNC_RETIRE_WORKERS, DEFAULT_ASYNC_SUPPLY_THREADS,
};
use cuda_fuzzer::fuzzer_restarting::setup_oom_safe_restarting_mgr;
use cuda_fuzzer::fuzzer_timeout::DEFAULT_TARGET_TIMEOUT;
use cuda_fuzzer::fuzzer_workdir::FuzzerWorkDir;
use cuda_fuzzer::gpu_client_guard::try_acquire_for_existing_broker;
use cuda_fuzzer::gpu_executor::{
    GpuExecutor, HasPendingTasks, PendingTask, RetirementSubmission, RetirementSubmitExecutor,
};
use cuda_fuzzer::mutators::RapidInputMutator;
#[cfg(feature = "profiling")]
use cuda_fuzzer::profiling::ProfileSession;
use cuda_fuzzer::retirement::{
    discard_queued_submissions, materialize_completions, retirement_channel, AcquiredCompletion,
    OrderedPreparedInbox, RetirementCredits, RetirementHandoff, RetirementStats,
    MAX_RETIREMENT_IN_FLIGHT,
};
use cuda_fuzzer::simt_memcov_feedback::SimtMemCovFeedback;
use cuda_fuzzer::supply_pool::{ProducerPool, SupplyStage};
use cuda_fuzzer::task_time_feedback::TaskTimeFeedback;

fn usage(program: &str) -> ! {
    eprintln!(
        "Usage: {program} <path_to_library> --manifest <path_to_manifest> [--window-size W] [--supply-threads N] [--retire-worker N] [--retire-batch N] [--dump-seed] [--no-mutate] [--runs N | --mutate-seconds S] [--benchmark --warmup-runs M --benchmark-seconds S] [--raw-throughput-seconds S] [--coverage-seconds S --coverage-log PATH --vconfig on|off]"
    );
    eprintln!();
    eprintln!("This is the ASYNC version using:");
    eprintln!("  - AsyncBatchFuzzer for batch processing");
    eprintln!("  - GpuExecutor for simple GPU submission");
    eprintln!("  - Delayed feedback evaluation");
    eprintln!();
    eprintln!("Performance controls (N must be in 0..=32):");
    eprintln!(
        "  --supply-threads N: host mutation producers (default {DEFAULT_ASYNC_SUPPLY_THREADS}; 0 disables)"
    );
    eprintln!(
        "  --retire-worker N: coverage materializers (default {DEFAULT_ASYNC_RETIRE_WORKERS}; 0 disables)"
    );
    eprintln!("  --retire-batch N: maximum owner retirement batch (default 8)");
    eprintln!(
        "  worker defaults apply to mutating fuzz mode; specialized modes select compatible single-threaded paths automatically"
    );
    eprintln!("  --raw-throughput-seconds S: isolated RAPID2 submit/poll measurement");
    eprintln!("  profiling: build with --features profiling and set RAPID_PROFILE=1");
    eprintln!("  raw coverage skips: RAPID_RAW_SKIP_COV_READ=1, then RAPID2_SKIP_COVERAGE_COPY=1");
    eprintln!("  guard: one restarting-manager client per visible GPU; the broker holds no lock");
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

fn submit_retirement_input(
    cuda_backend: &AsyncCudaBackend,
    credits: &RetirementCredits,
    pending: &mut VecDeque<PendingTask<BytesInput>>,
    submission: RetirementSubmission<BytesInput>,
    max_gpu_pending: &AtomicUsize,
) -> Result<(), Error> {
    credits.submission_dequeued();
    let task_id = match cuda_backend.submit(submission.input.target_bytes().as_slice()) {
        Ok(task_id) => task_id.get(),
        Err(error) => {
            credits.release(1);
            return Err(Error::illegal_state(error.to_string()));
        }
    };
    pending.push_back(PendingTask {
        task_id,
        input: submission.input,
        corpus_id: submission.corpus_id,
        from_supply: submission.from_supply,
    });
    max_gpu_pending.fetch_max(pending.len(), Ordering::Relaxed);
    Ok(())
}

fn run_retirement_io_loop(
    cuda_backend: &AsyncCudaBackend,
    window_size: NonZeroUsize,
    submissions: mpsc::Receiver<RetirementSubmission<BytesInput>>,
    retirement: RetirementHandoff<AcquiredCompletion<BytesInput>>,
    credits: &RetirementCredits,
    cancelled: &AtomicBool,
    max_gpu_pending: &AtomicUsize,
) -> Result<(), Error> {
    let mut pending = VecDeque::with_capacity(window_size.get());
    let mut submissions_closed = false;
    let mut next_sequence = 0_u64;

    loop {
        if cancelled.load(Ordering::Acquire) {
            discard_queued_submissions(&submissions, credits);
            return Ok(());
        }

        // Do not touch the backend before this thread owns a submitted task.
        // In particular, an empty-queue poll must not become a startup gate
        // that prevents this same thread from receiving the first submission.
        if pending.is_empty() && !submissions_closed {
            if credits.try_acquire() {
                match submissions.recv() {
                    Ok(submission) => submit_retirement_input(
                        cuda_backend,
                        credits,
                        &mut pending,
                        submission,
                        max_gpu_pending,
                    )?,
                    Err(_) => {
                        credits.release(1);
                        submissions_closed = true;
                    }
                }
            } else {
                thread::yield_now();
                continue;
            }
        }

        while pending.len() < window_size.get() && !submissions_closed && credits.try_acquire() {
            match submissions.try_recv() {
                Ok(submission) => submit_retirement_input(
                    cuda_backend,
                    credits,
                    &mut pending,
                    submission,
                    max_gpu_pending,
                )?,
                Err(mpsc::TryRecvError::Empty) => {
                    credits.release(1);
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    credits.release(1);
                    submissions_closed = true;
                }
            }
        }

        if pending.is_empty() {
            if submissions_closed {
                break;
            }
            continue;
        }

        let completed = cuda_backend.poll_completed_tasks(pending.len());
        let completed_ids = completed
            .iter()
            .map(|result| result.task_id)
            .collect::<Vec<_>>();
        if completed.len() > pending.len() {
            release_acquired_ids(cuda_backend, credits, &completed_ids)?;
            return Err(Error::illegal_state(format!(
                "RAPID2 returned {} retirement completions with only {} submitted tasks",
                completed.len(),
                pending.len()
            )));
        }
        if let Some((expected, actual)) = pending
            .iter()
            .zip(&completed)
            .find(|(pending_task, result)| pending_task.task_id != result.task_id)
        {
            let error = Error::illegal_state(format!(
                "RAPID2 retirement order violation: expected task_id={}, got task_id={}",
                expected.task_id, actual.task_id
            ));
            release_acquired_ids(cuda_backend, credits, &completed_ids)?;
            return Err(error);
        }

        let mut acquired = VecDeque::with_capacity(completed.len());
        for result in completed {
            let pending_task = pending
                .pop_front()
                .expect("completion batch was validated against pending tasks");
            acquired.push_back(AcquiredCompletion::with_sequence(
                next_sequence,
                result,
                pending_task,
            ));
            next_sequence = next_sequence.saturating_add(1);
        }
        while let Some(completion) = acquired.pop_front() {
            if let Err(error) = retirement.handoff(completion) {
                let mut unsent_ids = vec![error.0.result.task_id];
                unsent_ids.extend(acquired.iter().map(|item| item.result.task_id));
                release_acquired_ids(cuda_backend, credits, &unsent_ids)?;
                return Err(Error::illegal_state(
                    "retirement execution context stopped before handoff".to_string(),
                ));
            }
        }

        if submissions_closed && pending.is_empty() {
            break;
        }
        if pending.len() == window_size.get() {
            cuda_backend.wait_for_completion();
        } else {
            thread::yield_now();
        }
    }
    Ok(())
}

fn release_acquired_ids(
    cuda_backend: &AsyncCudaBackend,
    credits: &RetirementCredits,
    task_ids: &[u64],
) -> Result<(), Error> {
    let released = cuda_backend.release_completed_tasks(task_ids);
    credits.release(released);
    if released != task_ids.len() {
        return Err(Error::illegal_state(format!(
            "released {released} of {} acquired retirement tasks",
            task_ids.len()
        )));
    }
    Ok(())
}

fn print_retirement_statistics(
    stats: RetirementStats,
    admission_window: usize,
    max_gpu_pending: usize,
    max_retirement_credits_held: usize,
) {
    println!(
        "Retirement statistics: processed/sec={:.2}, processed={}, batches={}, batch[min/avg/max]={}/{:.2}/{}, max_queue_depth={}, max_in_flight={}, admission_window={}, max_gpu_pending={}, max_retirement_credits_held={}",
        stats.processed_per_second(),
        stats.processed,
        stats.batches,
        stats.min_batch,
        stats.average_batch(),
        stats.max_batch,
        stats.max_queue_depth,
        stats.max_in_flight,
        admission_window,
        max_gpu_pending,
        max_retirement_credits_held,
    );
}

fn run_restarting_retirement_fuzzer(
    lib_path: String,
    window_size: NonZeroUsize,
    supply_threads: usize,
    mutate: bool,
    configured_runs: Option<u64>,
    configured_duration_secs: Option<u64>,
    retire_workers: usize,
    retire_batch: NonZeroUsize,
) -> Result<(), Error> {
    let _gpu_client_guard = if fuzzer_frontend::gpu_client_guard_required() {
        try_acquire_for_existing_broker(1337).map_err(Error::illegal_state)?
    } else {
        None
    };
    let monitor = MultiMonitor::new(|message| println!("{message}"));
    let (restored_state, mut manager) =
        match setup_oom_safe_restarting_mgr(monitor, 1337, EventConfig::AlwaysUnique) {
            Ok(result) => result,
            Err(Error::ShuttingDown) => return Ok(()),
            Err(error) => return Err(error),
        };
    let profile_enabled = profiling_enabled()?;
    fuzzer_frontend::reexec_profiled_client_after_fork(profile_enabled)
        .map_err(Error::illegal_state)?;

    let cuda_backend = load_cuda_backend(&lib_path);
    let loader_observer = unsafe {
        StdMapObserver::from_mut_ptr("edges", cuda_backend.cov_map_ptr(), EDGES_MAP_SIZE)
            .track_indices()
    };
    let map_feedback = MaxMapFeedback::with_name("cfg_sites", &loader_observer);
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
    let mut state = restored_state.unwrap_or_else(|| {
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

    let mut fuzzer = AsyncBatchFuzzer::new(
        QueueScheduler::new(),
        feedback,
        objective,
        Arc::clone(&cuda_backend),
        simt_memcov_handle,
        task_time_handle,
    )
    .with_max_pending(window_size.get());

    if state.must_load_initial_inputs() {
        println!("Loading initial corpus...");
        let mut loader = GpuExecutor::new(Arc::clone(&cuda_backend), tuple_list!(loader_observer));
        state
            .load_initial_inputs(
                &mut fuzzer,
                &mut loader,
                &mut manager,
                &[PathBuf::from("./corpus")],
            )
            .unwrap_or_else(|error| {
                eprintln!("Failed to load initial corpus: {error}");
                std::process::exit(1);
            });
        fuzzer.force_evaluation(&mut loader, &mut state, &mut manager)?;
        println!("Loaded {} initial inputs.", state.corpus().count());
    }
    if state.corpus().count() == 0 {
        println!("Corpus is empty, adding default seed...");
        state
            .corpus_mut()
            .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    }

    const SUBMISSION_QUEUE_CAPACITY: usize = DEFAULT_MUTATIONAL_MAX_ITERATIONS;
    const MAX_SUBMISSIONS_PER_FUZZ_ONE: usize = DEFAULT_MUTATIONAL_MAX_ITERATIONS;
    let retirement_capacity = NonZeroUsize::new(MAX_RETIREMENT_IN_FLIGHT).unwrap();
    assert!(
        retirement_capacity
            .get()
            .saturating_sub(1)
            .saturating_add(MAX_SUBMISSIONS_PER_FUZZ_ONE)
            <= retirement_capacity
                .get()
                .saturating_add(SUBMISSION_QUEUE_CAPACITY),
        "external retirement admission can exceed downstream capacity"
    );
    let credits = Arc::new(RetirementCredits::new(retirement_capacity));
    let (submission_sender, submission_receiver) = mpsc::sync_channel(SUBMISSION_QUEUE_CAPACITY);
    let (retirement_handoff, materializer_inbox) = retirement_channel(retirement_capacity);
    let (prepared_handoff, prepared_inbox) = retirement_channel(retirement_capacity);
    let edges_observer = StdMapObserver::owned("edges", vec![0; EDGES_MAP_SIZE]).track_indices();
    let mut fuzzer =
        fuzzer.with_external_retirement(NonZeroUsize::new(MAX_SUBMISSIONS_PER_FUZZ_ONE).unwrap());
    let mut executor = RetirementSubmitExecutor::new(
        Arc::clone(&cuda_backend),
        submission_sender,
        Arc::clone(&credits),
        tuple_list!(edges_observer),
    );

    let supply_pool = NonZeroUsize::new(supply_threads).map(ProducerPool::new);
    if let Some(pool) = supply_pool.as_ref() {
        let supply_stats = pool.stats();
        fuzzer.set_supply_stats(Arc::clone(&supply_stats));
        executor.set_supply_stats(supply_stats);
        println!("Host supply pool enabled: threads={supply_threads}");
    }
    let mutator = if mutate {
        RapidInputMutator::default()
    } else {
        RapidInputMutator::fixed()
    };
    let stage_iterations = configured_duration_secs.map_or_else(
        || NonZeroUsize::new(DEFAULT_MUTATIONAL_MAX_ITERATIONS).unwrap(),
        |_| fuzzer_frontend::campaign_stage_max_iterations(),
    );
    let mut stages = tuple_list!(SupplyStage::new(
        AsyncMutationalStage::with_max_iterations(mutator, stage_iterations)
            .with_canonical_output(),
        supply_pool,
        stage_iterations,
    ));

    // RestartingMgr may fork, so create the submit/poll thread only after setup
    // has returned in the fuzzing client process.
    let io_backend = Arc::clone(&cuda_backend);
    let io_credits = Arc::clone(&credits);
    let retirement_cancelled = Arc::new(AtomicBool::new(false));
    let io_cancelled = Arc::clone(&retirement_cancelled);
    let max_gpu_pending = Arc::new(AtomicUsize::new(0));
    let io_max_gpu_pending = Arc::clone(&max_gpu_pending);
    let io_worker = thread::Builder::new()
        .name("rapid-submit-poll".to_string())
        .spawn(move || {
            run_retirement_io_loop(
                io_backend.as_ref(),
                window_size,
                submission_receiver,
                retirement_handoff,
                io_credits.as_ref(),
                io_cancelled.as_ref(),
                io_max_gpu_pending.as_ref(),
            )
        })
        .map_err(|error| {
            Error::illegal_state(format!("failed to spawn submit/poll thread: {error}"))
        })?;

    let mut materializers = Vec::with_capacity(retire_workers);
    let per_worker_batch_size =
        NonZeroUsize::new(retire_batch.get().div_ceil(retire_workers)).unwrap();
    // Materializers only copy task-local maps and build sparse words. They
    // avoid shared libafl_cov_map races; the owner swaps its observer backing.
    for worker_index in 0..retire_workers {
        let worker_inbox = materializer_inbox.clone();
        let worker_output = prepared_handoff.clone();
        let worker_backend = Arc::clone(&cuda_backend);
        let worker_credits = Arc::clone(&credits);
        let worker = thread::Builder::new()
            .name(format!("rapid-materialize-{worker_index}"))
            .spawn(move || {
                let started = Instant::now();
                let mut stats = RetirementStats::default();
                'materialize: while let Some(batch) =
                    worker_inbox.recv_full_batch(per_worker_batch_size)
                {
                    let batch_len = batch.len();
                    let prepared = materialize_completions(
                        batch,
                        worker_backend.as_ref(),
                        worker_credits.as_ref(),
                    );
                    stats.record_batch(batch_len);
                    for completion in prepared {
                        if worker_output.handoff(completion).is_err() {
                            break 'materialize;
                        }
                    }
                }
                stats.finish(started, worker_inbox.max_queue_depth());
                stats
            })
            .map_err(|error| {
                Error::illegal_state(format!("failed to spawn materialize worker: {error}"))
            })?;
        materializers.push(worker);
    }
    drop(materializer_inbox);
    drop(prepared_handoff);
    let mut retirement_inbox = OrderedPreparedInbox::new(prepared_inbox, 0);

    manager.report_progress(&mut state)?;
    if profile_enabled {
        cuda_backend
            .profile_timing_start()
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    let profile = start_profiling()?;
    let profile_started = Instant::now();

    println!(
        "RAPID2 retirement enabled: materializers={}, retire_batch={}, window_size={}",
        retire_workers, retire_batch, window_size
    );
    println!("Starting async batch fuzzing loop...");
    println!("Press Ctrl+C to stop and synchronize remaining tasks.");
    let started = Instant::now();
    let mut stats = RetirementStats::default();
    let mut iterations = 0u64;
    let mut stop_generation = false;
    let mut failure = None;
    'generation: while !stop_generation
        && configured_runs.is_none_or(|limit| iterations < limit)
        && configured_duration_secs
            .is_none_or(|seconds| profile_started.elapsed() < Duration::from_secs(seconds))
    {
        if let Err(error) = manager.maybe_report_progress(&mut state, Duration::from_secs(15)) {
            if matches!(error, Error::ShuttingDown) {
                stop_generation = true;
            } else {
                failure = Some(error);
            }
            break;
        }
        let submitted_before = executor.total_submitted();
        match fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut manager) {
            Ok(_) => iterations = iterations.saturating_add(1),
            Err(Error::ShuttingDown) => stop_generation = true,
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
        stats.observe_in_flight(executor.total_submitted());

        if executor.total_submitted() == submitted_before {
            continue;
        }
        // Admit another fuzz_one only below this watermark. One fuzz_one can
        // still add MAX_SUBMISSIONS_PER_FUZZ_ONE entries; the startup assertion
        // proves the submission queue plus retirement credits can absorb it.
        while stats.backlog_is_full(executor.total_submitted(), retirement_capacity) {
            let batch = match retirement_inbox.try_recv_batch(retire_batch) {
                Ok(Some(batch)) => batch,
                Ok(None) => match retirement_inbox.recv_batch(retire_batch) {
                    Ok(Some(batch)) => batch,
                    Ok(None) => {
                        failure = Some(Error::illegal_state(
                            "submit/poll thread stopped before retirement drain".to_string(),
                        ));
                        break 'generation;
                    }
                    Err(error) => {
                        failure = Some(Error::illegal_state(error));
                        break 'generation;
                    }
                },
                Err(error) => {
                    failure = Some(Error::illegal_state(error));
                    break 'generation;
                }
            };
            let batch_len = batch.len();
            if let Err(error) =
                fuzzer.retire_prepared_batch(&mut executor, &mut state, &mut manager, batch, true)
            {
                failure = Some(error);
                break 'generation;
            }
            stats.record_batch(batch_len);
        }
        if iterations.is_multiple_of(100) {
            println!(
                "[Async] Iteration: {}, Corpus: {}, Execs: {}, Pending: {}, Submitted: {}, Evaluated: {}",
                iterations,
                state.corpus().count(),
                state.executions(),
                credits.in_flight(),
                executor.total_submitted(),
                state.executions(),
            );
        }
    }

    if failure.is_some() {
        retirement_cancelled.store(true, Ordering::Release);
    }
    executor.close_submissions();
    if failure.is_none() {
        loop {
            match retirement_inbox.recv_batch(retire_batch) {
                Ok(Some(batch)) => {
                    let batch_len = batch.len();
                    match fuzzer.retire_prepared_batch(
                        &mut executor,
                        &mut state,
                        &mut manager,
                        batch,
                        true,
                    ) {
                        Ok(()) => stats.record_batch(batch_len),
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    failure = Some(Error::illegal_state(error));
                    break;
                }
            }
        }
    }
    if failure.is_some() {
        retirement_cancelled.store(true, Ordering::Release);
        retirement_inbox.drain_unordered(retire_batch);
    }
    match io_worker.join() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => failure = Some(error),
        Err(_) => {
            failure = Some(Error::illegal_state(
                "submit/poll thread panicked".to_string(),
            ));
        }
    }
    let mut materializer_stats = RetirementStats::default();
    for worker in materializers {
        match worker.join() {
            Ok(worker_stats) => {
                materializer_stats.processed = materializer_stats
                    .processed
                    .saturating_add(worker_stats.processed);
                materializer_stats.batches = materializer_stats
                    .batches
                    .saturating_add(worker_stats.batches);
                materializer_stats.max_queue_depth = materializer_stats
                    .max_queue_depth
                    .max(worker_stats.max_queue_depth);
                if worker_stats.processed != 0 {
                    materializer_stats.min_batch = if materializer_stats.min_batch == 0 {
                        worker_stats.min_batch
                    } else {
                        materializer_stats.min_batch.min(worker_stats.min_batch)
                    };
                    materializer_stats.max_batch =
                        materializer_stats.max_batch.max(worker_stats.max_batch);
                }
            }
            Err(_) => {
                failure = Some(Error::illegal_state(
                    "materialize worker panicked".to_string(),
                ));
            }
        }
    }
    if credits.in_flight() != 0 || credits.queued_submissions() != 0 {
        failure = Some(Error::illegal_state(format!(
            "retirement drain left in_flight={} queued_submissions={}",
            credits.in_flight(),
            credits.queued_submissions()
        )));
    }
    let queue_counts = cuda_backend.get_queue_counts();
    if queue_counts.pending != 0 || queue_counts.completed != 0 {
        failure = Some(Error::illegal_state(format!(
            "retirement drain left backend_pending={} backend_completed={}",
            queue_counts.pending, queue_counts.completed
        )));
    }
    let feedback_activity = fuzzer_frontend::feedback_activity(&state)?;
    stages.0.shutdown();
    let _elapsed_ns = finish_profiling(profile, profile_started, || {
        let device = profile_enabled
            .then(|| cuda_backend.profile_timing_snapshot())
            .flatten();
        cuda_backend.stop();
        device
    })?;
    stats.finish(started, retirement_inbox.max_queue_depth());
    fuzzer_frontend::print_final_statistics(
        state.corpus().count(),
        state.solutions().count(),
        *state.executions(),
    );
    fuzzer_frontend::print_arg_pack_stats();
    let finite_run_campaign_complete =
        failure.is_none() && configured_runs.is_some_and(|limit| iterations >= limit);
    let timed_campaign_complete = failure.is_none()
        && configured_duration_secs
            .is_some_and(|seconds| profile_started.elapsed() >= Duration::from_secs(seconds));
    if timed_campaign_complete && mutate {
        if let Some(requested_seconds) = configured_duration_secs {
            fuzzer_frontend::print_mutating_result(&fuzzer_frontend::mutating_result(
                requested_seconds,
                *state.executions(),
                state.corpus().count() as u64,
                state.solutions().count() as u64,
                feedback_activity,
                queue_counts.pending as u64,
                queue_counts.completed as u64,
                0,
                credits.in_flight() as u64,
                credits.queued_submissions() as u64,
            ))?;
        }
    }
    if stop_generation || finite_run_campaign_complete || timed_campaign_complete {
        manager.on_shutdown()?;
    } else {
        manager.on_restart(&mut state)?;
    }
    print_retirement_statistics(
        stats,
        retirement_capacity.get(),
        max_gpu_pending.load(Ordering::Acquire),
        credits.maximum(),
    );
    println!(
        "Materializer statistics: workers={}, processed={}, batches={}, batch[min/avg/max]={}/{:.2}/{}, max_input_queue_depth={}",
        retire_workers,
        materializer_stats.processed,
        materializer_stats.batches,
        materializer_stats.min_batch,
        materializer_stats.average_batch(),
        materializer_stats.max_batch,
        materializer_stats.max_queue_depth,
    );
    cuda_backend.stop();
    if let Some(error) = failure {
        return Err(error);
    }
    println!("Async fuzzer exited successfully.");
    Ok(())
}

fn load_cuda_backend(lib_path: &str) -> Arc<AsyncCudaBackend> {
    let cuda_backend = Arc::new(
        AsyncCudaBackend::new(lib_path).expect("Failed to load asynchronous CUDA backend"),
    );
    cuda_backend.set_target_timeout(DEFAULT_TARGET_TIMEOUT);
    eprintln!(
        "RAPID FUZZER CLIENT LOADED CUDA BACKEND pid={} lib={}",
        std::process::id(),
        lib_path
    );
    cuda_backend
}

const CANARY_STATUS_DETAIL: u32 = 0x4341_4e59;

#[derive(Clone, Copy)]
struct RawThroughputConfig {
    requested_duration: Duration,
    skip_cov_read: bool,
    skip_coverage_copy: bool,
}

impl RawThroughputConfig {
    fn from_env(duration_secs: u64) -> Result<Self, Error> {
        let skip_cov_read = std::env::var("RAPID_RAW_SKIP_COV_READ").as_deref() == Ok("1");
        let skip_coverage_copy = std::env::var("RAPID2_SKIP_COVERAGE_COPY").as_deref() == Ok("1");
        if skip_coverage_copy && !skip_cov_read {
            return Err(Error::illegal_argument(
                "RAPID2_SKIP_COVERAGE_COPY=1 requires RAPID_RAW_SKIP_COV_READ=1".to_string(),
            ));
        }
        Ok(Self {
            requested_duration: Duration::from_secs(duration_secs),
            skip_cov_read,
            skip_coverage_copy,
        })
    }
}

struct RawThroughputMeasurement {
    submitted: u64,
    released: u64,
    elapsed_ns: u64,
    execs_per_second: f64,
    non_ok_statuses: u64,
    canary_statuses: u64,
    queue_samples: u64,
    max_pending_queue: usize,
    max_completed_queue: usize,
    final_counts: LibAflQueueCounts,
    max_host_in_flight: usize,
    fatal_status_seen: bool,
    coverage_read_error: Option<String>,
}

fn run_raw_warmup(
    cuda_backend: &AsyncCudaBackend,
    seed: &[u8],
    warmup_tasks: usize,
) -> Result<(), Error> {
    for _ in 0..warmup_tasks {
        cuda_backend
            .submit(seed)
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    cuda_backend.wait();

    let mut remaining = warmup_tasks;
    let mut non_ok = 0usize;
    let mut fatal = false;
    while remaining > 0 {
        let results = cuda_backend.poll_completed_tasks(remaining);
        if results.is_empty() {
            std::hint::spin_loop();
            continue;
        }
        let task_ids = results
            .iter()
            .map(|result| {
                if result.status.code != LIBAFL_RUN_STATUS_OK {
                    non_ok += 1;
                    fatal |= run_status_requires_fresh_process(result.status);
                    eprintln!(
                        "RAPID_RAW_STATUS phase=warmup task_id={} code={} stage={} detail={} canary={}",
                        result.task_id,
                        result.status.code,
                        result.status.stage,
                        result.status.detail,
                        result.status.detail == CANARY_STATUS_DETAIL
                    );
                }
                result.task_id
            })
            .collect::<Vec<_>>();
        let released = cuda_backend.release_completed_tasks(&task_ids);
        if released != task_ids.len() {
            return Err(Error::illegal_state(format!(
                "raw warmup released {released} of {} completed tasks",
                task_ids.len()
            )));
        }
        remaining = remaining.checked_sub(released).ok_or_else(|| {
            Error::illegal_state("raw warmup in-flight counter underflow".to_string())
        })?;
        if non_ok != 0 {
            break;
        }
    }

    if fatal {
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        std::process::abort();
    }
    if non_ok != 0 {
        cuda_backend.stop();
        return Err(Error::illegal_state(format!(
            "raw warmup observed {non_ok} non-OK task statuses"
        )));
    }
    let counts = cuda_backend.get_queue_counts();
    if remaining != 0 || counts.pending != 0 || counts.completed != 0 {
        cuda_backend.stop();
        return Err(Error::illegal_state(format!(
            "raw warmup did not drain: host_in_flight={remaining}, backend_pending={}, backend_completed={}",
            counts.pending, counts.completed
        )));
    }
    println!("Raw warmup complete: tasks={warmup_tasks}");
    Ok(())
}

fn measure_raw_throughput(
    cuda_backend: &AsyncCudaBackend,
    seed: &[u8],
    window_size: usize,
    config: RawThroughputConfig,
) -> Result<RawThroughputMeasurement, Error> {
    let started = Instant::now();
    let mut submitted = 0u64;
    let mut released = 0u64;
    let mut host_in_flight = 0usize;
    let mut max_host_in_flight = 0usize;
    let mut non_ok_statuses = 0u64;
    let mut canary_statuses = 0u64;
    let mut fatal_status_seen = false;
    let mut coverage_checksum = 0u64;
    let mut completed_batches = 0u64;
    let mut next_queue_sample_batch = 0u64;
    let mut queue_samples = 0u64;
    let mut max_pending_queue = 0usize;
    let mut max_completed_queue = 0usize;
    let mut draining = false;
    let mut coverage_read_error = None;

    loop {
        if !draining {
            while host_in_flight < window_size && started.elapsed() < config.requested_duration {
                cuda_backend
                    .submit(seed)
                    .map_err(|error| Error::illegal_state(error.to_string()))?;
                submitted = submitted.saturating_add(1);
                host_in_flight += 1;
                max_host_in_flight = max_host_in_flight.max(host_in_flight);
            }
            if started.elapsed() >= config.requested_duration {
                cuda_backend.wait();
                draining = true;
            }
        }

        if completed_batches >= next_queue_sample_batch {
            let counts = cuda_backend.get_queue_counts();
            queue_samples = queue_samples.saturating_add(1);
            max_pending_queue = max_pending_queue.max(counts.pending);
            max_completed_queue = max_completed_queue.max(counts.completed);
            next_queue_sample_batch = completed_batches.saturating_add(1024);
        }

        let results = cuda_backend.poll_completed_tasks(window_size);
        if results.is_empty() {
            if draining && host_in_flight == 0 {
                break;
            }
            std::hint::spin_loop();
            continue;
        }
        completed_batches = completed_batches.saturating_add(1);

        let mut task_ids = Vec::with_capacity(results.len());
        for result in &results {
            if result.status.code != LIBAFL_RUN_STATUS_OK {
                non_ok_statuses = non_ok_statuses.saturating_add(1);
                let is_canary = result.status.detail == CANARY_STATUS_DETAIL;
                canary_statuses = canary_statuses.saturating_add(u64::from(is_canary));
                fatal_status_seen |= run_status_requires_fresh_process(result.status);
                eprintln!(
                    "RAPID_RAW_STATUS task_id={} code={} stage={} detail={} canary={}",
                    result.task_id,
                    result.status.code,
                    result.status.stage,
                    result.status.detail,
                    is_canary
                );
            }
            if !config.skip_cov_read {
                let edge = unsafe { result.edge_slice() };
                let simt = unsafe { result.simt_memcov_slice() };
                match (edge, simt) {
                    (Ok(edge), Ok(simt)) => {
                        for byte in edge.iter().chain(simt) {
                            coverage_checksum = coverage_checksum.wrapping_add(u64::from(*byte));
                        }
                    }
                    (Err(error), _) | (_, Err(error)) => {
                        coverage_read_error = Some(error.to_string());
                    }
                }
            }
            task_ids.push(result.task_id);
        }

        let released_now = cuda_backend.release_completed_tasks(&task_ids);
        if released_now != task_ids.len() {
            return Err(Error::illegal_state(format!(
                "raw throughput released {released_now} of {} completed tasks",
                task_ids.len()
            )));
        }
        released = released.saturating_add(released_now as u64);
        host_in_flight = host_in_flight.checked_sub(released_now).ok_or_else(|| {
            Error::illegal_state("raw throughput in-flight counter underflow".to_string())
        })?;

        if non_ok_statuses != 0 || coverage_read_error.is_some() {
            break;
        }
        if draining && host_in_flight == 0 {
            break;
        }
    }

    std::hint::black_box(coverage_checksum);
    let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let execs_per_second = if elapsed_ns == 0 {
        0.0
    } else {
        released as f64 * 1_000_000_000.0 / elapsed_ns as f64
    };

    Ok(RawThroughputMeasurement {
        submitted,
        released,
        elapsed_ns,
        execs_per_second,
        non_ok_statuses,
        canary_statuses,
        queue_samples,
        max_pending_queue,
        max_completed_queue,
        final_counts: cuda_backend.get_queue_counts(),
        max_host_in_flight,
        fatal_status_seen,
        coverage_read_error,
    })
}

fn run_raw_throughput(
    cuda_backend: Arc<AsyncCudaBackend>,
    window_size: NonZeroUsize,
    duration_secs: u64,
) -> Result<(), Error> {
    let config = RawThroughputConfig::from_env(duration_secs)?;
    let seed = normalize_rapid_input_v1(&default_seed_rapid_input_v1());

    println!(
        "Running raw RAPID2 throughput loop: window_size={}, seconds={}, skip_cov_read={}, skip_coverage_copy={}",
        window_size.get(),
        duration_secs,
        config.skip_cov_read,
        config.skip_coverage_copy
    );

    let warmup_tasks = window_size.get();
    run_raw_warmup(cuda_backend.as_ref(), &seed, warmup_tasks)?;
    let measurement =
        measure_raw_throughput(cuda_backend.as_ref(), &seed, window_size.get(), config)?;

    println!(
        "Raw throughput: total_tasks={}, submitted={}, released={}, elapsed_ns={}, exec/s={:.3}",
        measurement.released,
        measurement.submitted,
        measurement.released,
        measurement.elapsed_ns,
        measurement.execs_per_second
    );
    println!(
        "Raw queue counts: samples={}, max_pending={}, max_completed={}, final_pending={}, final_completed={}, max_host_in_flight={}",
        measurement.queue_samples,
        measurement.max_pending_queue,
        measurement.max_completed_queue,
        measurement.final_counts.pending,
        measurement.final_counts.completed,
        measurement.max_host_in_flight
    );
    println!(
        "Raw status counts: non_ok={}, canary={}",
        measurement.non_ok_statuses, measurement.canary_statuses
    );

    let result = fuzzer_frontend::RawThroughputResult {
        requested_duration_ns: config.requested_duration.as_nanos().min(u64::MAX as u128) as u64,
        warmup_tasks: warmup_tasks as u64,
        total_tasks: measurement.released,
        submitted_tasks: measurement.submitted,
        released_tasks: measurement.released,
        elapsed_ns: measurement.elapsed_ns,
        execs_per_second: measurement.execs_per_second,
        non_ok_statuses: measurement.non_ok_statuses,
        canary_statuses: measurement.canary_statuses,
        queue_samples: measurement.queue_samples,
        max_pending_queue: measurement.max_pending_queue as u64,
        max_completed_queue: measurement.max_completed_queue as u64,
        final_pending_queue: measurement.final_counts.pending as u64,
        final_completed_queue: measurement.final_counts.completed as u64,
        max_host_in_flight: measurement.max_host_in_flight as u64,
        skip_cov_read: config.skip_cov_read,
        skip_coverage_copy: config.skip_coverage_copy,
    };
    let machine_line = result
        .machine_line()
        .map_err(|error| Error::illegal_state(error.to_string()))?;

    if measurement.fatal_status_seen {
        println!("{machine_line}");
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        std::process::abort();
    }

    cuda_backend.stop();
    println!("{machine_line}");
    if measurement.non_ok_statuses != 0 {
        return Err(Error::illegal_state(format!(
            "raw throughput observed {} non-OK task statuses",
            measurement.non_ok_statuses
        )));
    }
    if let Some(error) = measurement.coverage_read_error {
        return Err(Error::illegal_state(error));
    }
    if measurement.submitted != measurement.released {
        return Err(Error::illegal_state(format!(
            "raw throughput submitted {} tasks but released {}",
            measurement.submitted, measurement.released
        )));
    }
    Ok(())
}

fn run_bounded_no_mutate_fuzz_loop(
    cuda_backend: Arc<AsyncCudaBackend>,
    window_size: NonZeroUsize,
    warmup_runs: u64,
    runs: u64,
    benchmark_min_duration_secs: Option<u64>,
) -> Result<(), Error> {
    let benchmark = benchmark_min_duration_secs.is_some();
    let profile_enabled = profiling_enabled()?;
    let workdir = FuzzerWorkDir::new("async-no-mutate").map_err(|err| {
        Error::illegal_state(format!("failed to create bounded fuzzer workdir: {err}"))
    })?;
    let monitor = SimpleMonitor::new(move |s| {
        if !benchmark {
            println!("{s}");
        }
    });
    let mut mgr = SimpleEventManager::new(monitor);

    // RAPID records boolean CFG-presence bytes with atomicOr(1), so hitcount
    // bucketing would only rescan an already classified 0/1 map.
    let edges_observer = unsafe {
        StdMapObserver::from_mut_ptr("edges", cuda_backend.cov_map_ptr(), EDGES_MAP_SIZE)
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
    let mut fuzzer = AsyncBatchFuzzer::new(
        scheduler,
        feedback,
        objective,
        Arc::clone(&cuda_backend),
        simt_memcov_handle,
        task_time_handle,
    )
    .with_max_pending(window_size.get());
    let mut executor = GpuExecutor::new(Arc::clone(&cuda_backend), tuple_list!(edges_observer));

    state
        .corpus_mut()
        .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    let mutator = RapidInputMutator::fixed();
    let mut stages = tuple_list!(AsyncMutationalStage::with_max_iterations(
        mutator,
        fuzzer_frontend::async_benchmark_stage_max_iterations(profile_enabled),
    )
    .with_canonical_output());

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
            "Running async fixed-input LibAFL benchmark: window_size={}, warmup_runs={warmup_runs}, minimum_seconds={seconds}, workdir={}",
            window_size.get(),
            workdir.root().display()
        );
    } else {
        println!(
            "Running bounded async fixed-input LibAFL loop: window_size={}, runs={runs}, workdir={}",
            window_size.get(),
            workdir.root().display()
        );
    }
    let completed_before = *state.executions();
    if profile_enabled {
        cuda_backend
            .profile_timing_start()
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    let profile = start_profiling()?;
    let started = Instant::now();
    let measured_iterations = if let Some(seconds) = benchmark_min_duration_secs {
        let min_duration = Duration::from_secs(seconds);
        let chunk_iterations = fuzzer_frontend::async_benchmark_chunk_iterations(profile_enabled);
        let mut iterations = 0u64;
        while started.elapsed() < min_duration {
            for _ in 0..chunk_iterations {
                let _ = fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut mgr)?;
            }
            iterations = iterations.saturating_add(chunk_iterations);
        }
        fuzzer.force_evaluation(&mut executor, &mut state, &mut mgr)?;
        iterations
    } else {
        let _ = fuzzer.fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut mgr, runs)?;
        runs
    };
    cuda_backend.wait();
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
    let queue_counts = cuda_backend.get_queue_counts();
    let pending = queue_counts.pending.saturating_add(queue_counts.completed) as u64;
    cuda_backend.stop();
    if profile_enabled {
        drop(stages);
        drop(fuzzer);
        drop(executor);
        drop(cuda_backend);
    }
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
            pending,
        )
        .map_err(Error::illegal_state)?;
        println!(
            "{}",
            result
                .machine_line()
                .map_err(|err| Error::illegal_state(err.to_string()))?
        );
    }
    fuzzer_frontend::print_final_statistics(
        state.corpus().count(),
        state.solutions().count(),
        *state.executions(),
    );
    fuzzer_frontend::print_arg_pack_stats();
    mgr.on_restart(&mut state)?;
    Ok(())
}

fn run_coverage_campaign(
    cuda_backend: Arc<AsyncCudaBackend>,
    window_size: NonZeroUsize,
    duration_secs: u64,
    coverage_log: &std::path::Path,
    vconfig_mutation: bool,
) -> Result<(), Error> {
    let workdir = FuzzerWorkDir::new("async-coverage").map_err(|error| {
        Error::illegal_state(format!("failed to create async coverage workdir: {error}"))
    })?;
    let monitor = SimpleMonitor::new(|message| println!("{message}"));
    let mut manager = SimpleEventManager::new(monitor);
    let edges_observer = unsafe {
        StdMapObserver::from_mut_ptr("edges", cuda_backend.cov_map_ptr(), EDGES_MAP_SIZE)
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
    let mut fuzzer = AsyncBatchFuzzer::new(
        scheduler,
        feedback,
        objective,
        Arc::clone(&cuda_backend),
        simt_memcov_handle,
        task_time_handle,
    )
    .with_max_pending(window_size.get())
    .with_coverage_telemetry(telemetry.clone());
    let mut executor = GpuExecutor::new(Arc::clone(&cuda_backend), tuple_list!(edges_observer));
    state
        .corpus_mut()
        .add(BytesInput::new(default_seed_rapid_input_v1()).into())?;
    let mutator = RapidInputMutator::mutating(vconfig_mutation);
    let mut stages = tuple_list!(AsyncMutationalStage::with_max_iterations(
        mutator,
        fuzzer_frontend::campaign_stage_max_iterations(),
    )
    .with_canonical_output());

    let _profiling = start_profiling()?;

    manager.report_progress(&mut state)?;
    let mut session =
        CoverageLogSession::start(coverage_log, telemetry).map_err(telemetry_error)?;
    while session.elapsed() < Duration::from_secs(duration_secs) {
        let _ = fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut manager)?;
    }
    fuzzer.force_evaluation(&mut executor, &mut state, &mut manager)?;
    session.finish().map_err(telemetry_error)?;
    let queue_counts = cuda_backend.get_queue_counts();
    fuzzer_frontend::print_final_statistics(
        state.corpus().count(),
        state.solutions().count(),
        *state.executions(),
    );
    println!(
        "Coverage campaign drain: executor_pending={} backend_pending={} backend_completed={}",
        executor.pending_count(),
        queue_counts.pending,
        queue_counts.completed,
    );
    fuzzer_frontend::print_arg_pack_stats();
    manager.on_restart(&mut state)?;
    Ok(())
}

fn main() -> Result<(), Error> {
    // --- Setup ---
    #[cfg(debug_assertions)]
    env_logger::init();

    let args: Vec<String> = args().collect();
    let parsed = fuzzer_frontend::parse_async_args(&args).unwrap_or_else(|_| usage(&args[0]));
    let window_size =
        fuzzer_frontend::effective_window_size(parsed.window_size, DEFAULT_ASYNC_MAX_PENDING);

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

    if let RunMode::RawThroughput { duration_secs } = parsed.mode {
        let cuda_backend = load_cuda_backend(&parsed.lib_path);
        return run_raw_throughput(cuda_backend, window_size, duration_secs);
    }

    println!("=== Async Batch Fuzzer ===");
    println!("Architecture: AsyncBatchFuzzer + GpuExecutor");
    println!("Features:");
    println!("  - Batch submission of inputs");
    println!("  - Asynchronous GPU execution");
    println!("  - Delayed coverage evaluation");
    println!("  - Decoupled execution and feedback");

    if let RunMode::CoverageCampaign {
        duration_secs,
        coverage_log,
        vconfig_mutation,
    } = &parsed.mode
    {
        let cuda_backend = load_cuda_backend(&parsed.lib_path);
        return run_coverage_campaign(
            cuda_backend,
            window_size,
            *duration_secs,
            coverage_log,
            *vconfig_mutation,
        );
    }

    if parsed.retire_workers == 0 {
        if let RunMode::Fuzz {
            mutate: false,
            runs: Some(runs),
            ..
        } = parsed.mode
        {
            let cuda_backend = load_cuda_backend(&parsed.lib_path);
            return run_bounded_no_mutate_fuzz_loop(cuda_backend, window_size, 0, runs, None);
        }
    }
    if let RunMode::Benchmark {
        warmup_runs,
        min_duration_secs,
    } = parsed.mode
    {
        let cuda_backend = load_cuda_backend(&parsed.lib_path);
        return run_bounded_no_mutate_fuzz_loop(
            cuda_backend,
            window_size,
            warmup_runs,
            0,
            Some(min_duration_secs),
        );
    }

    if parsed.retire_workers > 0 {
        let (mutate, configured_runs, configured_duration_secs) = match parsed.mode {
            RunMode::Fuzz {
                mutate,
                runs,
                duration_secs,
            } => (mutate, runs, duration_secs),
            _ => unreachable!("retirement worker is restricted to fuzz mode"),
        };
        return run_restarting_retirement_fuzzer(
            parsed.lib_path,
            window_size,
            parsed.supply_threads,
            mutate,
            configured_runs,
            configured_duration_secs,
            parsed.retire_workers,
            parsed.retire_batch,
        );
    }

    // --- Monitor ---
    let monitor = MultiMonitor::new(|s| println!("{}", s));

    // --- Event Manager ---
    // The first invocation owns only the broker. A later invocation is a GPU
    // client; keep one such client per visible device because RAPID2's
    // persistent kernel can starve a second CUDA context indefinitely.
    let _gpu_client_guard = if fuzzer_frontend::gpu_client_guard_required() {
        try_acquire_for_existing_broker(1337).map_err(Error::illegal_state)?
    } else {
        None
    };
    let (state, mut mgr) =
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

    let cuda_backend = load_cuda_backend(&parsed.lib_path);

    // --- Observers ---
    // AsyncBatchFuzzer updates this map before evaluating each completion.
    let edges_observer = unsafe {
        StdMapObserver::from_mut_ptr("edges", cuda_backend.cov_map_ptr(), EDGES_MAP_SIZE)
            .track_indices()
    };

    // --- Feedbacks & Objectives ---
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

    // --- State ---
    let mut state = state.unwrap_or_else(|| {
        // Use fixed seed for deterministic testing when RAPID_FIXED_SEED env var is set
        let seed = std::env::var("RAPID_FIXED_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
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

    // --- Scheduler ---
    let scheduler = QueueScheduler::new();

    // --- Async Batch Fuzzer ---
    let mut fuzzer = AsyncBatchFuzzer::new(
        scheduler,
        feedback,
        objective,
        Arc::clone(&cuda_backend),
        simt_memcov_handle,
        task_time_handle,
    )
    .with_max_pending(window_size.get());

    // --- GPU Executor ---
    // This executor only submits tasks, doesn't wait for results
    let mut executor = GpuExecutor::new(Arc::clone(&cuda_backend), tuple_list!(edges_observer));

    // --- Corpus Loading ---
    if state.must_load_initial_inputs() {
        println!("Loading initial corpus...");
        state
            .load_initial_inputs(
                &mut fuzzer,
                &mut executor,
                &mut mgr,
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

    // --- Stages ---
    // Producers only see immutable snapshots published by this submit thread.
    let supply_pool = NonZeroUsize::new(parsed.supply_threads).map(ProducerPool::new);
    if let Some(pool) = supply_pool.as_ref() {
        let stats = pool.stats();
        fuzzer.set_supply_stats(Arc::clone(&stats));
        executor.set_supply_stats(stats);
        println!(
            "Host supply pool enabled: threads={}",
            parsed.supply_threads
        );
    }
    let mutator = match parsed.mode {
        RunMode::Fuzz { mutate: false, .. } => RapidInputMutator::fixed(),
        _ => RapidInputMutator::default(),
    };
    let stage_iterations = parsed.mode.mutating_seconds().map_or_else(
        || NonZeroUsize::new(DEFAULT_MUTATIONAL_MAX_ITERATIONS).unwrap(),
        |_| fuzzer_frontend::campaign_stage_max_iterations(),
    );
    let mut stages = tuple_list!(SupplyStage::new(
        AsyncMutationalStage::with_max_iterations(mutator, stage_iterations)
            .with_canonical_output(),
        supply_pool,
        stage_iterations,
    ));

    if profile_enabled {
        cuda_backend
            .profile_timing_start()
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    let profile = start_profiling()?;
    let profile_started = Instant::now();

    // Trigger an initial progress report so that, with `introspection` enabled,
    // the monitor receives the first `UpdatePerfMonitor` event immediately
    // (otherwise it will show NaN until the first timed report is due).
    mgr.report_progress(&mut state)?;

    // --- Fuzzing Loop ---
    println!("Starting async batch fuzzing loop...");
    println!("Press Ctrl+C to stop and synchronize remaining tasks.");

    let bounded_fuzz = parsed.mode.is_bounded_fuzz();
    let mutating_seconds = parsed.mode.mutating_seconds();
    match parsed.mode {
        RunMode::Fuzz {
            runs: Some(runs), ..
        } => {
            let _ = fuzzer.fuzz_loop_for(&mut stages, &mut executor, &mut state, &mut mgr, runs)?;
        }
        RunMode::Fuzz {
            duration_secs: Some(seconds),
            ..
        } => {
            while profile_started.elapsed() < Duration::from_secs(seconds) {
                let _ = fuzzer.fuzz_one(&mut stages, &mut executor, &mut state, &mut mgr)?;
            }
        }
        _ => fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?,
    }

    // --- Cleanup ---
    println!("\nFuzzing stopped. Waiting for remaining tasks...");

    fuzzer.force_evaluation(&mut executor, &mut state, &mut mgr)?;
    let queue_counts = cuda_backend.get_queue_counts();
    if executor.pending_count() != 0 || queue_counts.pending != 0 || queue_counts.completed != 0 {
        return Err(Error::illegal_state(format!(
            "async drain left executor_pending={} backend_pending={} backend_completed={}",
            executor.pending_count(),
            queue_counts.pending,
            queue_counts.completed,
        )));
    }
    let feedback_activity = fuzzer_frontend::feedback_activity(&state)?;
    stages.0.shutdown();
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
    fuzzer_frontend::print_arg_pack_stats();
    if let Some(requested_seconds) = mutating_seconds {
        fuzzer_frontend::print_mutating_result(&fuzzer_frontend::mutating_result(
            requested_seconds,
            *state.executions(),
            state.corpus().count() as u64,
            state.solutions().count() as u64,
            feedback_activity,
            queue_counts.pending as u64,
            queue_counts.completed as u64,
            0,
            0,
            0,
        ))?;
    }

    if bounded_fuzz {
        mgr.on_shutdown()?;
    } else {
        mgr.on_restart(&mut state)?;
    }

    println!("Async fuzzer exited successfully.");
    Ok(())
}
