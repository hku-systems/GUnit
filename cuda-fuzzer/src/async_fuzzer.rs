use core::{fmt::Debug, marker::PhantomData, num::NonZeroUsize, time::Duration};
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;

use libafl::{
    corpus::{Corpus, CorpusId, HasCurrentCorpusId, HasTestcase, Testcase},
    events::{
        Event, EventConfig, EventFirer, EventReceiver, EventWithStats, ProgressReporter,
        SendExiting,
    },
    executors::{Executor, ExitKind, HasObservers},
    feedbacks::{Feedback, MapFeedbackMetadata},
    fuzzer::{Evaluator, EvaluatorObservers, EventProcessor, ExecuteInputResult, Fuzzer},
    inputs::{HasTargetBytes, Input},
    mark_feature_time,
    observers::ObserversTuple,
    schedulers::Scheduler,
    start_timer,
    state::{
        HasCorpus, HasCurrentStageId, HasCurrentTestcase, HasExecutions, HasImported,
        HasLastFoundTime, HasLastReportTime, HasSolutions, MaybeHasClientPerfMonitor, Stoppable,
    },
    Error, HasMetadata, HasNamedMetadata,
};
use libafl_bolts::current_time;
use log;

#[cfg(feature = "introspection")]
use libafl::monitors::stats::PerfFeature;

use crate::coverage_telemetry::CoverageTelemetry;
use crate::cuda_backend::{
    exit_kind_from_run_status, run_status_requires_fresh_process, AsyncCudaBackend,
    LibAflRunStatus, EDGES_MAP_SIZE, LIBAFL_RUN_STATUS_OK, SIMT_MEMCOV_STORAGE_SIZE,
};
use crate::gpu_executor::{HasPendingTasks, RetirementObservers, RetirementSubmitExecutor};
#[cfg(feature = "profiling")]
use crate::profiling::{scope, Segment};
use crate::retirement::{
    copy_coverage_map, has_novel_bits, or_coverage_map, sparse_has_novel_bits, AcquiredCompletion,
    PreparedCompletion,
};
use crate::simt_memcov_feedback::{BitsetNoveltyStateMetadata, SimtMemCovFeedback};
use crate::supply_pool::SupplyStats;
use crate::task_time_feedback::TaskTimeFeedback;

fn record_task_completion(
    telemetry: Option<&CoverageTelemetry>,
    edge_map: &[u8],
    memory_map: &[u8],
) -> Result<(), Error> {
    if let Some(telemetry) = telemetry {
        telemetry
            .record_completion(edge_map, memory_map)
            .map_err(|error| Error::illegal_state(error.to_string()))?;
    }
    Ok(())
}

fn ensure_campaign_drained(
    executor_pending: usize,
    backend_pending: usize,
    backend_completed: usize,
) -> Result<(), Error> {
    if executor_pending != 0 || backend_pending != 0 || backend_completed != 0 {
        return Err(Error::illegal_state(format!(
            "rapid2 drain left executor_pending={executor_pending} backend_pending={backend_pending} backend_completed={backend_completed}"
        )));
    }
    Ok(())
}

fn with_parent_attribution<I, S, T, F>(
    state: &mut S,
    parent_id: Option<CorpusId>,
    evaluate: F,
) -> Result<T, Error>
where
    S: HasCurrentCorpusId + HasTestcase<I>,
    F: FnOnce(&mut S) -> Result<T, Error>,
{
    let saved_corpus_id = state.current_corpus_id()?;
    if let Some(parent_id) = parent_id {
        state.set_corpus_id(parent_id)?;
    }

    let evaluation = evaluate(state);
    if evaluation.is_ok() {
        if let Some(parent_id) = parent_id {
            if let Ok(mut testcase) = state.testcase_mut(parent_id) {
                let scheduled_count = testcase.scheduled_count();
                testcase.set_scheduled_count(scheduled_count + 1);
            }
        }
    }
    let restore = match saved_corpus_id {
        Some(id) => state.set_corpus_id(id),
        None => state.clear_corpus_id(),
    };

    match (evaluation, restore) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(evaluation_error), Err(restore_error)) => Err(Error::illegal_state(format!(
            "completion evaluation failed: {evaluation_error}; restoring parent attribution also failed: {restore_error}"
        ))),
    }
}

fn remaining_submission_budget(capacity: usize, before: u64, after: u64) -> usize {
    let submitted = after.saturating_sub(before).min(usize::MAX as u64) as usize;
    capacity.saturating_sub(submitted)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompletionRestartRequest {
    pub task_id: u64,
    pub status: LibAflRunStatus,
}

fn completion_restart_request(
    task_id: u64,
    status: LibAflRunStatus,
) -> Option<CompletionRestartRequest> {
    run_status_requires_fresh_process(status)
        .then_some(CompletionRestartRequest { task_id, status })
}

trait CompletedTaskReleaser {
    fn release_completed_tasks(&self, task_ids: &[u64]) -> usize;
}

impl CompletedTaskReleaser for AsyncCudaBackend {
    fn release_completed_tasks(&self, task_ids: &[u64]) -> usize {
        AsyncCudaBackend::release_completed_tasks(self, task_ids)
    }
}

fn with_task_release<R, T, F, A>(
    releaser: &R,
    task_ids: &[u64],
    process: F,
    after_release: A,
) -> Result<T, Error>
where
    R: CompletedTaskReleaser,
    F: FnOnce() -> Result<T, Error>,
    A: FnOnce(usize),
{
    let process_result = process();
    let released = {
        #[cfg(feature = "profiling")]
        let _timing = scope(Segment::Release);
        releaser.release_completed_tasks(task_ids)
    };
    after_release(released);
    let release_result = if released == task_ids.len() {
        Ok(())
    } else {
        Err(Error::illegal_state(format!(
            "released {released} of {} completed tasks",
            task_ids.len()
        )))
    };

    match (process_result, release_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(process_error), Ok(())) => Err(process_error),
        (Ok(_), Err(release_error)) => Err(release_error),
        (Err(process_error), Err(release_error)) => Err(Error::illegal_state(format!(
            "completion processing failed: {process_error}; task release also failed: {release_error}"
        ))),
    }
}

// Local copies of LibAFL trait definitions we need for clearer structure.
// We intentionally keep them minimal and independent from LibAFL's implementations.

/// Send a monitor update all 15 (or more) seconds
pub(crate) const STATS_TIMEOUT_DEFAULT: Duration = Duration::from_secs(15);
const DEFAULT_ASYNC_COMPLETION_BATCH_SIZE: usize = 8;
pub const DEFAULT_ASYNC_MAX_PENDING: usize = 2;
const CFG_FEEDBACK_NAME: &str = "cfg_sites";

#[derive(Debug)]
struct BatchNoveltyScratch {
    edge_map: Vec<u8>,
    simt_memcov_map: Vec<u8>,
}

impl BatchNoveltyScratch {
    fn new() -> Self {
        Self {
            edge_map: vec![0; EDGES_MAP_SIZE],
            simt_memcov_map: vec![0; SIMT_MEMCOV_STORAGE_SIZE],
        }
    }

    fn needs_detailed_evaluation<I, S>(
        &mut self,
        state: &S,
        completions: &[AcquiredCompletion<I>],
    ) -> Result<bool, Error>
    where
        S: HasMetadata + HasNamedMetadata,
    {
        if completions
            .iter()
            .any(|completion| completion.result.status.code != LIBAFL_RUN_STATUS_OK)
        {
            return Ok(true);
        }

        self.edge_map.fill(0);
        self.simt_memcov_map.fill(0);
        for completion in completions {
            let edge_map = unsafe { completion.result.edge_slice() }
                .map_err(|error| Error::illegal_state(error.to_string()))?;
            let simt_memcov_map = unsafe { completion.result.simt_memcov_slice() }
                .map_err(|error| Error::illegal_state(error.to_string()))?;
            or_coverage_map(edge_map, &mut self.edge_map).map_err(Error::illegal_state)?;
            or_coverage_map(simt_memcov_map, &mut self.simt_memcov_map)
                .map_err(Error::illegal_state)?;
        }

        let Ok(cfg_metadata) = state.named_metadata::<MapFeedbackMetadata<u8>>(CFG_FEEDBACK_NAME)
        else {
            return Ok(true);
        };
        if cfg_metadata.history_map.len() != EDGES_MAP_SIZE
            // RAPID edge entries are binary presence bytes, so bit novelty is
            // equivalent to MaxMapFeedback's numeric max comparison.
            || has_novel_bits(&self.edge_map, &cfg_metadata.history_map)
                .map_err(Error::illegal_state)?
        {
            return Ok(true);
        }

        let Ok(simt_metadata) = state.metadata::<BitsetNoveltyStateMetadata>() else {
            return Ok(true);
        };
        if simt_metadata.seen.len() != SIMT_MEMCOV_STORAGE_SIZE {
            return Ok(true);
        }
        has_novel_bits(&self.simt_memcov_map, &simt_metadata.seen).map_err(Error::illegal_state)
    }
}

/// Structs with this trait will execute an input
pub trait ExecutesInput<E, EM, I, S> {
    /// Runs the input and triggers observers and feedback
    fn execute_input(
        &mut self,
        state: &mut S,
        executor: &mut E,
        event_mgr: &mut EM,
        input: &I,
    ) -> Result<ExitKind, Error>;
}

/// Evaluates if an input is interesting using the feedback
pub trait ExecutionProcessor<EM, I, OT, S> {
    /// Check the outcome of the execution, find if it is worth for corpus or objectives
    fn check_results(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        observers: &OT,
        exit_kind: &ExitKind,
    ) -> Result<ExecuteInputResult, Error>;

    /// Process `ExecuteInputResult`. Add to corpus, solution or ignore
    fn process_execution(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        exec_res: &ExecuteInputResult,
        exit_kind: &ExitKind,
        observers: &OT,
    ) -> Result<Option<CorpusId>, Error>;

    /// serialize and send event via manager
    fn serialize_and_dispatch(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        exec_res: &ExecuteInputResult,
        observers: &OT,
        exit_kind: &ExitKind,
    ) -> Result<(), Error>;

    /// send event via manager
    fn dispatch_event(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        exec_res: &ExecuteInputResult,
        obs_buf: Option<Vec<u8>>,
        exit_kind: &ExitKind,
    ) -> Result<(), Error>;

    /// Evaluate if a set of observation channels has an interesting state
    fn evaluate_execution(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        observers: &OT,
        exit_kind: &ExitKind,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error>;
}

/// An asynchronous batch fuzzer that decouples execution from evaluation
#[derive(Clone, Copy, Debug)]
struct CompletionBatchSize(NonZeroUsize);

impl CompletionBatchSize {
    fn new(value: usize) -> Self {
        Self(NonZeroUsize::new(value).expect("batch size must be non-zero"))
    }

    fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Debug)]
pub struct AsyncBatchFuzzer<CS, F, OF, I> {
    /// The scheduler
    scheduler: CS,
    /// The feedback
    feedback: F,
    /// The objective feedback
    objective: OF,
    /// Strict CUDA backend for asynchronous operations.
    cuda_backend: Arc<AsyncCudaBackend>,
    /// Shared novelty state populated from task-owned completion bitsets
    simt_memcov_feedback: SimtMemCovFeedback,
    /// Shared task-time feedback populated from task-owned completion results
    task_time_feedback: TaskTimeFeedback,
    /// Optional observation-only cumulative coverage telemetry.
    coverage_telemetry: Option<CoverageTelemetry>,
    /// Maximum number of completions to poll while draining
    batch_size: CompletionBatchSize,
    /// Maximum pending tasks before forcing evaluation
    max_pending: usize,
    /// Statistics
    total_evaluated: u64,
    supply_stats: Option<Arc<SupplyStats>>,
    /// Whether to share objectives with other nodes
    share_objectives: bool,
    /// Completion polling/release is owned by the external retirement pipeline.
    external_retirement: bool,
    /// Maximum submissions allowed before returning to external retirement.
    external_submission_capacity: Option<usize>,
    /// Per-call event budget derived from the remaining submission capacity.
    external_event_budget: Option<usize>,
    batch_novelty: BatchNoveltyScratch,
    /// PhantomData for type parameter I
    phantom: PhantomData<I>,
}

impl<CS, F, OF, I> AsyncBatchFuzzer<CS, F, OF, I> {
    /// Create a new AsyncBatchFuzzer
    pub fn new(
        scheduler: CS,
        feedback: F,
        objective: OF,
        cuda_backend: Arc<AsyncCudaBackend>,
        simt_memcov_feedback: SimtMemCovFeedback,
        task_time_feedback: TaskTimeFeedback,
    ) -> Self {
        Self {
            scheduler,
            feedback,
            objective,
            cuda_backend,
            simt_memcov_feedback,
            task_time_feedback,
            coverage_telemetry: None,
            batch_size: CompletionBatchSize::new(DEFAULT_ASYNC_COMPLETION_BATCH_SIZE),
            max_pending: DEFAULT_ASYNC_MAX_PENDING,
            total_evaluated: 0,
            supply_stats: None,
            share_objectives: false,
            external_retirement: false,
            external_submission_capacity: None,
            external_event_budget: None,
            batch_novelty: BatchNoveltyScratch::new(),
            phantom: PhantomData,
        }
    }

    pub fn set_supply_stats(&mut self, stats: Arc<SupplyStats>) {
        self.supply_stats = Some(stats);
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = CompletionBatchSize::new(batch_size);
        self
    }

    /// Set max pending tasks
    pub fn with_max_pending(mut self, max_pending: usize) -> Self {
        assert!(max_pending != 0, "async window capacity must be non-zero");
        self.max_pending = max_pending;
        self
    }

    /// Enable observation-only cumulative coverage telemetry.
    pub fn with_coverage_telemetry(mut self, telemetry: CoverageTelemetry) -> Self {
        self.coverage_telemetry = Some(telemetry);
        self
    }

    pub fn with_external_retirement(mut self, submission_capacity: NonZeroUsize) -> Self {
        self.external_retirement = true;
        self.external_submission_capacity = Some(submission_capacity.get());
        self
    }
}

// Inherent methods mirroring LibAFL semantics (async-friendly implementations)
// Implement ExecutesInput for AsyncBatchFuzzer
impl<CS, E, EM, F, I, OF, S> ExecutesInput<E, EM, I, S> for AsyncBatchFuzzer<CS, F, OF, I>
where
    E: Executor<EM, I, S, Self> + HasObservers,
    E::Observers: ObserversTuple<I, S>,
    S: HasCurrentCorpusId + MaybeHasClientPerfMonitor,
    I: Input + Clone + HasTargetBytes,
{
    fn execute_input(
        &mut self,
        state: &mut S,
        executor: &mut E,
        event_mgr: &mut EM,
        input: &I,
    ) -> Result<ExitKind, Error> {
        // Pre-execution hooks
        start_timer!(state);
        executor.observers_mut().pre_exec_all(state, input)?;
        mark_feature_time!(state, PerfFeature::PreExecObservers);

        // Submit via executor's run_target (which handles pending tasks)
        start_timer!(state);
        let exit_kind = executor.run_target(self, state, event_mgr, input)?;
        mark_feature_time!(state, PerfFeature::TargetExecution);

        if matches!(exit_kind, ExitKind::Ok) {
            self.simt_memcov_feedback
                .record_submission()
                .map_err(|err| Error::illegal_state(err.to_string()))?;
            if let Some(telemetry) = &self.coverage_telemetry {
                telemetry
                    .record_submission()
                    .map_err(|error| Error::illegal_state(error.to_string()))?;
            }
        }

        // Note: post_exec_all is called in process_completed_tasks after the task completes
        // In async execution, we can't call post_exec here as the task hasn't completed yet

        Ok(exit_kind)
    }
}

// Implement ExecutionProcessor for AsyncBatchFuzzer
impl<CS, EM, F, I, OF, OT, S> ExecutionProcessor<EM, I, OT, S> for AsyncBatchFuzzer<CS, F, OF, I>
where
    CS: Scheduler<I, S>,
    EM: EventFirer<I, S>,
    F: Feedback<EM, I, OT, S>,
    OF: Feedback<EM, I, OT, S>,
    I: Input,
    OT: Serialize,
    S: HasCorpus<I>
        + HasExecutions
        + HasSolutions<I>
        + HasLastFoundTime
        + HasCurrentCorpusId
        + MaybeHasClientPerfMonitor,
{
    fn check_results(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        observers: &OT,
        exit_kind: &ExitKind,
    ) -> Result<ExecuteInputResult, Error> {
        let mut res = ExecuteInputResult::default();

        // Check if it's a solution (objective)
        #[cfg(not(feature = "introspection"))]
        let is_solution = self
            .objective_mut()
            .is_interesting(state, manager, input, observers, exit_kind)?;

        #[cfg(feature = "introspection")]
        let is_solution = self
            .objective_mut()
            .is_interesting_introspection(state, manager, input, observers, exit_kind)?;

        if is_solution {
            res.set_is_solution(true);
        }

        // Check if it's corpus-worthy (feedback)
        #[cfg(not(feature = "introspection"))]
        let corpus_worthy = self
            .feedback_mut()
            .is_interesting(state, manager, input, observers, exit_kind)?;

        #[cfg(feature = "introspection")]
        let corpus_worthy = self
            .feedback_mut()
            .is_interesting_introspection(state, manager, input, observers, exit_kind)?;

        if corpus_worthy {
            res.set_is_corpus(true);
        }

        Ok(res)
    }

    fn process_execution(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        exec_res: &ExecuteInputResult,
        exit_kind: &ExitKind,
        observers: &OT,
    ) -> Result<Option<CorpusId>, Error> {
        let mut corpus_id: Option<CorpusId> = None;

        // Process corpus addition
        if exec_res.is_corpus() {
            let mut testcase = Testcase::from(input.clone());
            testcase.set_executions(*state.executions());
            #[cfg(feature = "track_hit_feedbacks")]
            self.feedback_mut()
                .append_hit_feedbacks(testcase.hit_feedbacks_mut())?;
            self.feedback_mut()
                .append_metadata(state, manager, observers, &mut testcase)?;
            let id = state.corpus_mut().add(testcase)?;
            self.scheduler_mut().on_add(state, id)?;
            corpus_id = Some(id);
        }

        // Process solution addition
        if exec_res.is_solution() {
            let mut testcase = Testcase::from(input.clone());
            testcase.set_executions(*state.executions());
            testcase.add_metadata(*exit_kind);
            // Use state.current_corpus_id() for async safety - corpus.current() may have changed
            testcase.set_parent_id_optional(state.current_corpus_id()?);
            if let Ok(mut tc) = state.current_testcase_mut() {
                tc.found_objective();
            }
            #[cfg(feature = "track_hit_feedbacks")]
            self.objective_mut()
                .append_hit_feedbacks(testcase.hit_objectives_mut())?;
            self.objective_mut()
                .append_metadata(state, manager, observers, &mut testcase)?;
            state.solutions_mut().add(testcase)?;
        }

        Ok(corpus_id)
    }

    fn serialize_and_dispatch(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        exec_res: &ExecuteInputResult,
        observers: &OT,
        exit_kind: &ExitKind,
    ) -> Result<(), Error> {
        // Serialize observers if needed for corpus events
        let observers_buf = if exec_res.is_corpus()
            && manager.should_send()
            && manager.configuration() != EventConfig::AlwaysUnique
        {
            // Serialize observers for network transmission
            Some(postcard::to_allocvec(observers)?)
        } else {
            None
        };

        self.dispatch_event(state, manager, input, exec_res, observers_buf, exit_kind)?;
        Ok(())
    }

    fn dispatch_event(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        exec_res: &ExecuteInputResult,
        observers_buf: Option<Vec<u8>>,
        exit_kind: &ExitKind,
    ) -> Result<(), Error> {
        // Send events to broker if needed
        if manager.should_send() {
            if exec_res.is_corpus() {
                // Send new testcase event
                manager.fire(
                    state,
                    EventWithStats::with_current_time(
                        Event::NewTestcase {
                            input: input.clone(),
                            observers_buf,
                            exit_kind: *exit_kind,
                            corpus_size: state.corpus().count(),
                            client_config: manager.configuration(),
                            forward_id: None,
                        },
                        *state.executions(),
                    ),
                )?;
            }

            if exec_res.is_solution() {
                // Send objective/crash event
                manager.fire(
                    state,
                    EventWithStats::with_current_time(
                        Event::Objective {
                            input: self.share_objectives.then_some(input.clone()),
                            objective_size: state.solutions().count(),
                        },
                        *state.executions(),
                    ),
                )?;
            }
        }

        Ok(())
    }

    fn evaluate_execution(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        observers: &OT,
        exit_kind: &ExitKind,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        let exec_res = self.check_results(state, manager, input, observers, exit_kind)?;
        let corpus_id =
            self.process_execution(state, manager, input, &exec_res, exit_kind, observers)?;
        if send_events {
            self.serialize_and_dispatch(state, manager, input, &exec_res, observers, exit_kind)?;
        }
        if exec_res.is_corpus() || exec_res.is_solution() {
            *state.last_found_time_mut() = current_time();
        }
        self.total_evaluated += 1;
        Ok((exec_res, corpus_id))
    }
}

impl<CS, E, EM, F, I, OF, S, ST> Fuzzer<E, EM, I, S, ST> for AsyncBatchFuzzer<CS, F, OF, I>
where
    CS: Scheduler<I, S>,
    E: HasObservers + Executor<EM, I, S, Self> + HasPendingTasks<I>,
    EM: EventFirer<I, S> + EventReceiver<I, S> + ProgressReporter<S> + SendExiting,
    F: Feedback<EM, I, E::Observers, S>,
    I: Input + Clone + HasTargetBytes,
    OF: Feedback<EM, I, E::Observers, S>,
    S: HasExecutions
        + HasMetadata
        + HasCorpus<I>
        + HasSolutions<I>
        + HasLastReportTime
        + HasLastFoundTime
        + HasImported
        + HasTestcase<I>
        + HasCurrentCorpusId
        + HasCurrentStageId
        + HasNamedMetadata
        + Stoppable
        + MaybeHasClientPerfMonitor,
    E::Observers: DeserializeOwned + Serialize + ObserversTuple<I, S>,
    ST: libafl::stages::StagesTuple<E, EM, S, Self>,
{
    fn fuzz_one(
        &mut self,
        stages: &mut ST,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<CorpusId, Error> {
        #[cfg(feature = "profiling")]
        let scheduler_stage_timing = scope(Segment::SchedulerStage);

        // Init timer for scheduler
        #[cfg(feature = "introspection")]
        state.introspection_stats_mut().start_timer();

        // Get next corpus entry to work on (scheduler)
        let id = if let Some(id) = state.current_corpus_id()? {
            id // we are resuming
        } else {
            let id = self.scheduler.next(state)?;
            state.set_corpus_id(id)?; // set up for resume
            id
        };

        // Mark the elapsed time for the scheduler
        #[cfg(feature = "introspection")]
        state.introspection_stats_mut().mark_scheduler_time();

        // Reset stage index so the first (and only) stage is always "Stage 0" in PerfMonitor.
        #[cfg(feature = "introspection")]
        state.introspection_stats_mut().reset_stage_index();

        let submitted_before = executor.total_submitted();

        // Execute all stages
        stages.perform_all(self, executor, state, manager)?;

        #[cfg(feature = "profiling")]
        drop(scheduler_stage_timing);

        // Init timer for manager
        #[cfg(feature = "introspection")]
        state.introspection_stats_mut().start_timer();

        // Imported events without observer buffers may execute another input.
        // Bound those re-executions so this call cannot fill the submission
        // channel and wait on retirement that only its caller can perform.
        if let Some(capacity) = self.external_submission_capacity {
            self.external_event_budget = Some(remaining_submission_budget(
                capacity,
                submitted_before,
                executor.total_submitted(),
            ));
        }

        // Process events from other nodes and completed GPU tasks
        self.process_events(state, executor, manager)?;

        // Mark the elapsed time for the manager
        #[cfg(feature = "introspection")]
        state.introspection_stats_mut().mark_manager_time();

        if executor.pending_count() > self.max_pending {
            return Err(Error::illegal_state(format!(
                "async window overflow: pending={} capacity={}",
                executor.pending_count(),
                self.max_pending
            )));
        }

        // Return the current corpus id (may not be the latest if async)
        state.clear_corpus_id()?;

        if state.stop_requested() {
            if !self.external_retirement {
                self.force_evaluation(executor, state, manager)?;
            }

            state.discard_stop_request();
            if !self.external_retirement {
                manager.on_shutdown()?;
            }
            return Err(Error::shutting_down());
        }

        Ok(id)
    }

    fn fuzz_loop(
        &mut self,
        stages: &mut ST,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), Error> {
        let mut iteration = 0u64;
        let monitor_timeout = STATS_TIMEOUT_DEFAULT;
        loop {
            manager.maybe_report_progress(state, monitor_timeout)?;
            self.fuzz_one(stages, executor, state, manager)?;

            iteration += 1;
            // #[cfg(feature = "introspection")]
            {
                if iteration.is_multiple_of(100) {
                    println!("[Async] Iteration: {}, Corpus: {}, Execs: {}, Pending: {}, Submitted: {}, Evaluated: {}",
                    iteration,
                    state.corpus().count(),
                    state.executions(),
                    executor.pending_count(),
                    executor.total_submitted(),
                    self.total_evaluated
                );
                }
            }
        }
    }

    fn fuzz_loop_for(
        &mut self,
        stages: &mut ST,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
        iters: u64,
    ) -> Result<CorpusId, Error> {
        if iters == 0 {
            return Err(Error::illegal_argument(
                "Cannot fuzz for 0 iterations!".to_string(),
            ));
        }

        let mut ret = None;
        let monitor_timeout = STATS_TIMEOUT_DEFAULT;

        for _ in 0..iters {
            manager.maybe_report_progress(state, monitor_timeout)?;
            ret = Some(self.fuzz_one(stages, executor, state, manager)?);
        }
        // Final synchronization
        self.force_evaluation(executor, state, manager)?;

        manager.report_progress(state)?;

        Ok(ret.unwrap())
    }
}

// Private helper methods
impl<CS, F, OF, I> AsyncBatchFuzzer<CS, F, OF, I>
where
    I: HasTargetBytes,
{
    /// Write coverage data to the shared map that observers read from
    /// This ensures the coverage data is available for feedback evaluation
    #[inline]
    fn update_coverage_map(
        &self,
        coverage_data: &[u8],
        coverage_map_ptr: *mut u8,
    ) -> Result<(), Error> {
        // Safety: The backend map is the same memory that the observers reference.
        // Both the StdMapObserver in fuzzer_async.rs and this pointer refer to the
        // same shared coverage map. This design allows async GPU results to be
        // written to a location where observers can read them during evaluation.
        let coverage_map =
            unsafe { std::slice::from_raw_parts_mut(coverage_map_ptr, EDGES_MAP_SIZE) };
        copy_coverage_map(coverage_data, coverage_map).map_err(Error::illegal_state)
    }

    fn retire_one<E, EM, S>(
        &mut self,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
        result: crate::cuda_backend::TaskResult,
        pending: crate::gpu_executor::PendingTask<I>,
        detailed_evaluation: bool,
        send_events: bool,
    ) -> Result<
        (
            ExecuteInputResult,
            Option<CorpusId>,
            Option<CompletionRestartRequest>,
        ),
        Error,
    >
    where
        CS: Scheduler<I, S>,
        E: HasObservers,
        EM: EventFirer<I, S>,
        F: Feedback<EM, I, E::Observers, S>,
        I: Input + Clone,
        OF: Feedback<EM, I, E::Observers, S>,
        S: HasCorpus<I>
            + HasMetadata
            + HasExecutions
            + HasSolutions<I>
            + HasLastFoundTime
            + HasCurrentTestcase<I>
            + HasCurrentCorpusId
            + HasTestcase<I>
            + HasNamedMetadata
            + MaybeHasClientPerfMonitor,
        E::Observers: ObserversTuple<I, S> + Serialize,
    {
        // Polling transfers the TaskData into the backend's outstanding map.
        // These task-owned pointers remain valid until the caller releases it.
        let edge_map =
            unsafe { result.edge_slice() }.map_err(|err| Error::illegal_state(err.to_string()))?;
        let simt_memcov_bits = unsafe { result.simt_memcov_slice() }
            .map_err(|err| Error::illegal_state(err.to_string()))?;
        self.commit_materialized_one(
            executor,
            state,
            manager,
            result.task_id,
            result.status,
            result.exec_time_ns,
            pending,
            edge_map,
            simt_memcov_bits,
            true,
            detailed_evaluation,
            send_events,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_materialized_one<E, EM, S>(
        &mut self,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
        task_id: u64,
        status: LibAflRunStatus,
        exec_time_ns: u64,
        pending: crate::gpu_executor::PendingTask<I>,
        edge_map: &[u8],
        simt_memcov_bits: &[u8],
        install_backend_map: bool,
        detailed_evaluation: bool,
        send_events: bool,
    ) -> Result<
        (
            ExecuteInputResult,
            Option<CorpusId>,
            Option<CompletionRestartRequest>,
        ),
        Error,
    >
    where
        CS: Scheduler<I, S>,
        E: HasObservers,
        EM: EventFirer<I, S>,
        F: Feedback<EM, I, E::Observers, S>,
        I: Input + Clone,
        OF: Feedback<EM, I, E::Observers, S>,
        S: HasCorpus<I>
            + HasMetadata
            + HasExecutions
            + HasSolutions<I>
            + HasLastFoundTime
            + HasCurrentTestcase<I>
            + HasCurrentCorpusId
            + HasTestcase<I>
            + HasNamedMetadata
            + MaybeHasClientPerfMonitor,
        E::Observers: ObserversTuple<I, S> + Serialize,
    {
        {
            #[cfg(feature = "profiling")]
            let _timing = scope(Segment::Coverage);
            if install_backend_map {
                record_task_completion(
                    self.coverage_telemetry.as_ref(),
                    edge_map,
                    simt_memcov_bits,
                )?;
                self.update_coverage_map(edge_map, self.cuda_backend.cov_map_ptr())?;
            }
            if detailed_evaluation {
                self.simt_memcov_feedback
                    .prepare_observation(state, simt_memcov_bits)?;
            }
            self.task_time_feedback
                .observe_ns(exec_time_ns)
                .map_err(|err| Error::illegal_state(err.to_string()))?;
        }

        #[cfg(feature = "profiling")]
        let evaluate_timing = scope(Segment::Evaluate);

        *state.executions_mut() += 1;
        if pending.from_supply {
            let stats = self
                .supply_stats
                .as_ref()
                .expect("supply completion requires supply statistics");
            stats.record_completed();
        }

        let res_exit_kind = exit_kind_from_run_status(status);
        if !matches!(res_exit_kind, ExitKind::Ok) {
            log::error!(
                "RAPID2 non-OK completion: task_id={} status_code={} stage={} detail={:#x} exit_kind={:?}",
                task_id,
                status.code,
                status.stage,
                status.detail,
                res_exit_kind
            );
        }
        let restart_request = completion_restart_request(task_id, status);
        let (exec_res, added_id) = with_parent_attribution(state, pending.corpus_id, |state| {
            start_timer!(state);
            executor
                .observers_mut()
                .post_exec_all(state, &pending.input, &res_exit_kind)?;
            mark_feature_time!(state, PerfFeature::PostExecObservers);
            let observers = executor.observers();
            self.scheduler
                .on_evaluation(state, &pending.input, &*observers)?;
            if detailed_evaluation {
                self.evaluate_execution(
                    state,
                    manager,
                    &pending.input,
                    &*observers,
                    &res_exit_kind,
                    send_events,
                )
            } else {
                self.total_evaluated += 1;
                Ok((ExecuteInputResult::default(), None))
            }
        })?;

        #[cfg(feature = "profiling")]
        drop(evaluate_timing);

        Ok((exec_res, added_id, restart_request))
    }

    fn retire_batch<E, EM, S>(
        &mut self,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
        completions: Vec<AcquiredCompletion<I>>,
        send_events: bool,
    ) -> Result<
        (
            ExecuteInputResult,
            Option<CorpusId>,
            Option<CompletionRestartRequest>,
        ),
        Error,
    >
    where
        CS: Scheduler<I, S>,
        E: HasObservers,
        EM: EventFirer<I, S>,
        F: Feedback<EM, I, E::Observers, S>,
        I: Input + Clone,
        OF: Feedback<EM, I, E::Observers, S>,
        S: HasCorpus<I>
            + HasMetadata
            + HasExecutions
            + HasSolutions<I>
            + HasLastFoundTime
            + HasCurrentTestcase<I>
            + HasCurrentCorpusId
            + HasTestcase<I>
            + HasNamedMetadata
            + MaybeHasClientPerfMonitor,
        E::Observers: ObserversTuple<I, S> + Serialize,
    {
        if completions.is_empty() {
            return Ok((ExecuteInputResult::default(), None, None));
        }
        let detailed_evaluation = {
            #[cfg(feature = "profiling")]
            let _timing = scope(Segment::Coverage);
            self.batch_novelty
                .needs_detailed_evaluation(state, &completions)?
        };
        let completion_count = completions.len();
        let mut last_exec_res = ExecuteInputResult::default();
        let mut last_added_id = None;
        let mut restart_after_batch = None;
        for completion in completions {
            let (exec_res, added_id, restart) = self.retire_one(
                executor,
                state,
                manager,
                completion.result,
                completion.pending,
                detailed_evaluation,
                send_events,
            )?;
            last_exec_res = exec_res;
            last_added_id = added_id;
            if restart_after_batch.is_none() {
                restart_after_batch = restart;
            }
        }
        if !detailed_evaluation {
            let stats = self
                .simt_memcov_feedback
                .record_known_uninteresting_batch(state, completion_count)?;
            SimtMemCovFeedback::publish_stats::<EM, I, S>(state, manager, stats)?;
        }
        Ok((last_exec_res, last_added_id, restart_after_batch))
    }

    pub fn retire_prepared_batch<EM, S>(
        &mut self,
        executor: &mut RetirementSubmitExecutor<I, RetirementObservers, S>,
        state: &mut S,
        manager: &mut EM,
        completions: Vec<PreparedCompletion<I>>,
        send_events: bool,
    ) -> Result<(), Error>
    where
        CS: Scheduler<I, S>,
        EM: EventFirer<I, S>,
        F: Feedback<EM, I, RetirementObservers, S>,
        I: Input + Clone,
        OF: Feedback<EM, I, RetirementObservers, S>,
        S: HasCorpus<I>
            + HasMetadata
            + HasExecutions
            + HasSolutions<I>
            + HasLastFoundTime
            + HasCurrentTestcase<I>
            + HasCurrentCorpusId
            + HasTestcase<I>
            + HasNamedMetadata
            + MaybeHasClientPerfMonitor,
        RetirementObservers: ObserversTuple<I, S> + Serialize,
    {
        let mut restart_after_batch = None;
        let mut known_stats = None;
        for mut completion in completions {
            let detailed_evaluation = if completion.status.code != LIBAFL_RUN_STATUS_OK {
                true
            } else {
                self.prepared_needs_detailed_evaluation(state, &completion)
            };
            record_task_completion(
                self.coverage_telemetry.as_ref(),
                &completion.edge_map,
                &completion.simt_memcov_map,
            )?;
            let installed = std::mem::take(&mut completion.edge_map);
            let old_backing = executor.replace_coverage_map(installed);
            let evaluated = self.commit_materialized_one(
                executor,
                state,
                manager,
                completion.task_id,
                completion.status,
                completion.exec_time_ns,
                completion.pending,
                &[],
                &completion.simt_memcov_map,
                false,
                detailed_evaluation,
                send_events,
            );
            let _evaluated_map = executor.replace_coverage_map(old_backing);
            let (_, _, restart) = evaluated?;
            if !detailed_evaluation {
                known_stats = Some(
                    self.simt_memcov_feedback
                        .record_known_uninteresting_batch(state, 1)?,
                );
            }
            if restart_after_batch.is_none() {
                restart_after_batch = restart;
            }
        }
        if let Some(stats) = known_stats {
            SimtMemCovFeedback::publish_stats::<EM, I, S>(state, manager, stats)?;
        }
        if let Some(request) = restart_after_batch {
            abort_after_non_recoverable_completion(request);
        }
        Ok(())
    }

    fn prepared_needs_detailed_evaluation<S>(
        &self,
        state: &S,
        completion: &PreparedCompletion<I>,
    ) -> bool
    where
        S: HasMetadata + HasNamedMetadata,
    {
        let Ok(cfg_metadata) = state.named_metadata::<MapFeedbackMetadata<u8>>(CFG_FEEDBACK_NAME)
        else {
            return true;
        };
        if cfg_metadata.history_map.len() != EDGES_MAP_SIZE
            || sparse_has_novel_bits(&completion.edge_words, &cfg_metadata.history_map)
        {
            return true;
        }
        let Ok(simt_metadata) = state.metadata::<BitsetNoveltyStateMetadata>() else {
            return true;
        };
        simt_metadata.seen.len() != SIMT_MEMCOV_STORAGE_SIZE
            || sparse_has_novel_bits(&completion.simt_words, &simt_metadata.seen)
    }
    /// Process completed tasks from GPU
    #[inline]
    fn process_completed_tasks<E, EM, S>(
        &mut self,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
        send_events: bool,
        refill_input: Option<&I>,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error>
    where
        CS: Scheduler<I, S>,
        E: HasObservers + Executor<EM, I, S, Self> + HasPendingTasks<I>,
        EM: EventFirer<I, S>,
        F: Feedback<EM, I, E::Observers, S>,
        I: Input + Clone,
        OF: Feedback<EM, I, E::Observers, S>,
        S: HasCorpus<I>
            + HasMetadata
            + HasExecutions
            + HasSolutions<I>
            + HasLastFoundTime
            + HasCurrentTestcase<I>
            + HasCurrentCorpusId
            + HasTestcase<I>
            + HasNamedMetadata
            + MaybeHasClientPerfMonitor,
        E::Observers: ObserversTuple<I, S> + Serialize,
    {
        // In async mode, completion processing (post_exec/feedback) happens outside the stage
        // execution path. To keep PerfMonitor output compact and intuitive, we attribute all
        // completion-related work to Stage 0.
        // FIXME: This may not be accurate if multiple stages are used.
        #[cfg(feature = "introspection")]
        state.introspection_stats_mut().reset_stage_index();

        // Poll for completed tasks (batch some results per call)
        let max_results = if refill_input.is_some() {
            1
        } else {
            self.batch_size.get()
        };
        start_timer!(state);
        let results = {
            #[cfg(feature = "profiling")]
            let _timing = scope(Segment::Poll);
            self.cuda_backend.poll_completed_tasks(max_results)
        };
        mark_feature_time!(state, PerfFeature::PollCompletedTasks);
        // Every result returned by poll is acquired. Record all IDs before any
        // fallible validation/evaluation so the release path cannot miss one.
        let to_release: Vec<u64> = results.iter().map(|result| result.task_id).collect();
        let release_backend = Arc::clone(&self.cuda_backend);
        let mut restart_after_batch: Option<CompletionRestartRequest> = None;
        let mut refill_submit_failed = false;

        if refill_input.is_some() && results.is_empty() {
            return Err(Error::illegal_state(
                "completion notification returned without a pollable task".to_string(),
            ));
        }

        let processed = with_task_release(
            release_backend.as_ref(),
            &to_release,
            || {
                let mut completions = Vec::with_capacity(results.len());
                for result in results {
                    start_timer!(state);
                    let pending = executor.take_pending_task(result.task_id).ok_or_else(|| {
                        Error::illegal_state(format!(
                            "completed unknown rapid2 task_id={}",
                            result.task_id
                        ))
                    })?;
                    mark_feature_time!(state, PerfFeature::FindPendingTask);

                    let restart_request = completion_restart_request(result.task_id, result.status);
                    if restart_request.is_none() {
                        if let Some(input) = refill_input {
                            let refill_exit_kind =
                                self.execute_input(state, executor, manager, input)?;
                            refill_submit_failed = !matches!(refill_exit_kind, ExitKind::Ok);
                        }
                    }

                    if restart_after_batch.is_none() {
                        restart_after_batch = restart_request;
                    }
                    completions.push(AcquiredCompletion::new(result, pending));
                }

                let (exec_res, added_id, restart) =
                    self.retire_batch(executor, state, manager, completions, send_events)?;
                if restart_after_batch.is_none() {
                    restart_after_batch = restart;
                }
                Ok((exec_res, added_id))
            },
            |_| {},
        );

        if let Some(request) = restart_after_batch {
            abort_after_non_recoverable_completion(request);
        }
        if refill_submit_failed {
            eprintln!(
                "RAPID2 REFILL FAILURE: backend returned task_id=0. Aborting client process so the restarting manager can reload the backend."
            );
            std::process::abort();
        }

        processed
    }
    /// Force evaluation of all pending tasks
    #[inline]
    pub fn force_evaluation<E, EM, S>(
        &mut self,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), Error>
    where
        CS: Scheduler<I, S>,
        E: HasObservers + Executor<EM, I, S, Self> + HasPendingTasks<I>,
        EM: EventFirer<I, S>,
        F: Feedback<EM, I, E::Observers, S>,
        I: Input + Clone,
        OF: Feedback<EM, I, E::Observers, S>,
        S: HasCorpus<I>
            + HasMetadata
            + HasExecutions
            + HasSolutions<I>
            + HasLastFoundTime
            + HasCurrentTestcase<I>
            + HasCurrentCorpusId
            + HasTestcase<I>
            + HasNamedMetadata
            + MaybeHasClientPerfMonitor,
        E::Observers: ObserversTuple<I, S> + Serialize,
    {
        if self.external_retirement {
            return Ok(());
        }
        // Let the backend's completion notification perform the blocking wait,
        // then drain task-owned results without a fixed host-side sleep.
        {
            #[cfg(feature = "profiling")]
            let _timing = scope(Segment::OtherIdle);
            self.cuda_backend.wait();
        }
        while executor.pending_count() != 0 {
            let pending_before = executor.pending_count();
            let _ = self.process_completed_tasks(executor, state, manager, false, None)?;
            if executor.pending_count() == pending_before {
                let counts = self.cuda_backend.get_queue_counts();
                return Err(Error::illegal_state(format!(
                    "rapid2 drain stalled with {} pending tasks: backend_pending={} backend_completed={}",
                    pending_before, counts.pending, counts.completed
                )));
            }
        }

        let counts = self.cuda_backend.get_queue_counts();
        ensure_campaign_drained(executor.pending_count(), counts.pending, counts.completed)?;

        Ok(())
    }
}

fn abort_after_non_recoverable_completion(request: CompletionRestartRequest) -> ! {
    eprintln!(
        "RAPID2 NON-RECOVERABLE COMPLETION: task_id={} status_code={} stage={} detail={:#x}. Aborting client process so the restarting manager can create a fresh CUDA context.",
        request.task_id,
        request.status.code,
        request.status.stage,
        request.status.detail,
    );
    std::process::abort();
}

// Implement scheduler access
impl<CS, F, OF, I> AsyncBatchFuzzer<CS, F, OF, I> {
    pub fn scheduler(&self) -> &CS {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut CS {
        &mut self.scheduler
    }

    pub fn feedback(&self) -> &F {
        &self.feedback
    }

    pub fn feedback_mut(&mut self) -> &mut F {
        &mut self.feedback
    }

    pub fn objective(&self) -> &OF {
        &self.objective
    }

    pub fn objective_mut(&mut self) -> &mut OF {
        &mut self.objective
    }
}

// Implement EventProcessor trait for AsyncBatchFuzzer
impl<CS, E, EM, F, I, OF, S> EventProcessor<E, EM, I, S> for AsyncBatchFuzzer<CS, F, OF, I>
where
    CS: Scheduler<I, S>,
    E: HasObservers + Executor<EM, I, S, Self> + HasPendingTasks<I>,
    E::Observers: ObserversTuple<I, S> + Serialize + DeserializeOwned,
    EM: EventReceiver<I, S> + EventFirer<I, S>,
    F: Feedback<EM, I, E::Observers, S>,
    I: Input + Clone + HasTargetBytes,
    OF: Feedback<EM, I, E::Observers, S>,
    S: HasCorpus<I>
        + HasSolutions<I>
        + HasExecutions
        + HasMetadata
        + HasLastFoundTime
        + MaybeHasClientPerfMonitor
        + HasCurrentCorpusId
        + HasCurrentTestcase<I>
        + HasTestcase<I>
        + HasNamedMetadata
        + HasImported,
{
    fn process_events(
        &mut self,
        state: &mut S,
        executor: &mut E,
        manager: &mut EM,
    ) -> Result<(), Error> {
        // Process incoming events from other nodes (broker/client mode)
        // This is separate from GPU task processing which happens in evaluate_filtered
        let mut event_budget = self.external_event_budget.take();
        while event_budget.is_none_or(|remaining| remaining > 0) {
            let Some((event, with_observers)) = manager.try_receive(state)? else {
                break;
            };
            if let Some(remaining) = event_budget.as_mut() {
                *remaining -= 1;
            }
            // Handle events from other nodes
            let res = if with_observers {
                match event.event() {
                    libafl::events::Event::NewTestcase {
                        input,
                        observers_buf,
                        exit_kind,
                        ..
                    } => {
                        // Deserialize observers and evaluate
                        let observers: E::Observers =
                            postcard::from_bytes(observers_buf.as_ref().unwrap())?;
                        let res = self.evaluate_execution(
                            state, manager, input, &observers, exit_kind, false,
                        )?;
                        res.1
                    }
                    _ => None,
                }
            } else {
                match event.event() {
                    libafl::events::Event::NewTestcase { input, .. } => {
                        // Re-execute and evaluate the input
                        let res = self.evaluate_input_with_observers(
                            state, executor, manager, input, false,
                        )?;
                        res.1
                    }
                    libafl::events::Event::Objective {
                        input: Some(unwrapped_input),
                        ..
                    } => {
                        let res = self.evaluate_input_with_observers(
                            state,
                            executor,
                            manager,
                            unwrapped_input,
                            false,
                        )?;
                        res.1
                    }
                    _ => None,
                }
            };

            if let Some(item) = res {
                *state.imported_mut() += 1;
                log::debug!("Added received input as item #{item}");

                // for centralize
                manager.on_interesting(state, event)?;
            } else {
                log::debug!("Received input was discarded");
            }
        }

        Ok(())
    }
}

// Implement Evaluator trait for AsyncBatchFuzzer
impl<CS, E, EM, F, I, OF, S> Evaluator<E, EM, I, S> for AsyncBatchFuzzer<CS, F, OF, I>
where
    CS: Scheduler<I, S>,
    E: HasObservers + Executor<EM, I, S, Self> + HasPendingTasks<I>,
    EM: EventFirer<I, S>,
    F: Feedback<EM, I, E::Observers, S>,
    I: Input + Clone + HasTargetBytes,
    OF: Feedback<EM, I, E::Observers, S>,
    S: HasCorpus<I>
        + HasExecutions
        + HasSolutions<I>
        + HasMetadata
        + HasLastFoundTime
        + HasCurrentCorpusId
        + HasTestcase<I>
        + HasNamedMetadata
        + MaybeHasClientPerfMonitor,
    E::Observers: ObserversTuple<I, S> + Serialize,
{
    fn evaluate_filtered(
        &mut self,
        state: &mut S,
        executor: &mut E,
        manager: &mut EM,
        input: &I,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        self.evaluate_input(state, executor, manager, input)
    }

    #[inline]
    fn evaluate_input(
        &mut self,
        state: &mut S,
        executor: &mut E,
        manager: &mut EM,
        input: &I,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        self.evaluate_input_with_observers(state, executor, manager, input, true)
    }

    fn add_input(
        &mut self,
        state: &mut S,
        _executor: &mut E,
        _manager: &mut EM,
        input: I,
    ) -> Result<(CorpusId, ExecuteInputResult), Error> {
        // Add directly to corpus without execution
        let mut testcase = Testcase::from(input);
        testcase.set_executions(*state.executions());
        let id = state.corpus_mut().add(testcase)?;
        Ok((id, ExecuteInputResult::default()))
    }

    fn add_disabled_input(&mut self, state: &mut S, input: I) -> Result<CorpusId, Error> {
        let mut testcase = Testcase::from(input);
        testcase.set_executions(*state.executions());
        testcase.set_disabled(true);
        let id = state.corpus_mut().add_disabled(testcase)?;
        Ok(id)
    }
}

// Implement EvaluatorObservers trait for AsyncBatchFuzzer
impl<CS, E, EM, F, I, OF, S> EvaluatorObservers<E, EM, I, S> for AsyncBatchFuzzer<CS, F, OF, I>
where
    CS: Scheduler<I, S>,
    E: HasObservers + Executor<EM, I, S, Self> + HasPendingTasks<I>,
    E::Observers: ObserversTuple<I, S> + Serialize,
    EM: EventFirer<I, S>,
    F: Feedback<EM, I, E::Observers, S>,
    I: Input + Clone + HasTargetBytes,
    OF: Feedback<EM, I, E::Observers, S>,
    S: HasCorpus<I>
        + HasSolutions<I>
        + HasExecutions
        + HasMetadata
        + HasLastFoundTime
        + HasCurrentCorpusId
        + HasCurrentTestcase<I>
        + HasTestcase<I>
        + HasNamedMetadata
        + MaybeHasClientPerfMonitor,
{
    #[inline]
    fn evaluate_input_with_observers(
        &mut self,
        state: &mut S,
        executor: &mut E,
        manager: &mut EM,
        input: &I,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        if executor.pending_count() >= self.max_pending {
            {
                #[cfg(feature = "profiling")]
                let _timing = scope(Segment::OtherIdle);
                self.cuda_backend.wait_for_completion();
            }
            return self.process_completed_tasks(
                executor,
                state,
                manager,
                send_events,
                Some(input),
            );
        }

        // Fill the initial window without retiring feedback. Once full, each
        // completion is replaced before its heavier feedback path runs.
        let exit_kind = self.execute_input(state, executor, manager, input)?;
        if !matches!(exit_kind, ExitKind::Ok) {
            *state.executions_mut() += 1;
            executor
                .observers_mut()
                .post_exec_all(state, input, &exit_kind)?;
            let observers = executor.observers();
            let _ = self.evaluate_execution(
                state,
                manager,
                input,
                &*observers,
                &exit_kind,
                send_events,
            )?;
            eprintln!(
                "RAPID2 SUBMIT FAILURE: backend returned task_id=0. \
                 Aborting client process so the restarting manager can reload the backend."
            );
            std::process::abort();
        }

        Ok((ExecuteInputResult::default(), None))
    }
}

#[cfg(test)]
mod tests {
    use libafl::{
        corpus::{Corpus, InMemoryCorpus},
        inputs::BytesInput,
        state::StdState,
    };
    use libafl_bolts::rands::StdRand;

    use crate::cuda_backend::{
        LIBAFL_RUN_STATUS_CUDA_ERROR, LIBAFL_RUN_STATUS_INTERNAL_ERROR,
        LIBAFL_RUN_STATUS_INVALID_INPUT, LIBAFL_RUN_STATUS_OK, LIBAFL_RUN_STATUS_TIMEOUT,
    };
    use std::sync::Mutex;

    use super::*;
    use crate::{
        coverage_telemetry::CoverageTelemetry, cuda_backend::SIMT_MEMCOV_STORAGE_SIZE,
        retirement::RetirementCredits,
    };

    #[derive(Debug)]
    struct RecordingReleaser {
        released: Mutex<Vec<u64>>,
        release_count: usize,
    }

    impl RecordingReleaser {
        fn new(release_count: usize) -> Self {
            Self {
                released: Mutex::new(Vec::new()),
                release_count,
            }
        }
    }

    impl CompletedTaskReleaser for RecordingReleaser {
        fn release_completed_tasks(&self, task_ids: &[u64]) -> usize {
            self.released.lock().unwrap().extend_from_slice(task_ids);
            self.release_count
        }
    }

    #[test]
    fn completion_batch_size_preserves_configured_poll_limit() {
        assert_eq!(CompletionBatchSize::new(17).get(), 17);
    }

    #[test]
    #[should_panic(expected = "batch size must be non-zero")]
    fn completion_batch_size_rejects_zero() {
        let _ = CompletionBatchSize::new(0);
    }

    #[test]
    fn releases_acquired_tasks_after_successful_evaluation() {
        let releaser = RecordingReleaser::new(2);
        let credits = RetirementCredits::new(NonZeroUsize::new(2).unwrap());
        assert!(credits.try_acquire());
        assert!(credits.try_acquire());

        let value = with_task_release(
            &releaser,
            &[10, 11],
            || {
                assert!(releaser.released.lock().unwrap().is_empty());
                Ok(17)
            },
            |released| credits.release(released),
        )
        .unwrap();

        assert_eq!(value, 17);
        assert_eq!(*releaser.released.lock().unwrap(), vec![10, 11]);
        assert_eq!(credits.in_flight(), 0);
    }

    #[test]
    fn releases_acquired_tasks_after_evaluation_error() {
        let releaser = RecordingReleaser::new(2);

        let error = with_task_release::<_, (), _, _>(
            &releaser,
            &[20, 21],
            || Err(Error::illegal_state("evaluation failed")),
            |_| {},
        )
        .unwrap_err();

        assert!(error.to_string().contains("evaluation failed"));
        assert_eq!(*releaser.released.lock().unwrap(), vec![20, 21]);
    }

    #[test]
    fn rejects_partial_task_release() {
        let releaser = RecordingReleaser::new(1);

        let error = with_task_release(&releaser, &[30, 31], || Ok(()), |_| {}).unwrap_err();

        assert!(error
            .to_string()
            .contains("released 1 of 2 completed tasks"));
    }

    #[test]
    fn completion_restart_request_only_marks_context_poisoning_statuses() {
        for code in [
            LIBAFL_RUN_STATUS_CUDA_ERROR,
            LIBAFL_RUN_STATUS_TIMEOUT,
            LIBAFL_RUN_STATUS_INTERNAL_ERROR,
        ] {
            let status = LibAflRunStatus {
                code,
                ..LibAflRunStatus::default()
            };
            assert_eq!(
                completion_restart_request(42, status),
                Some(CompletionRestartRequest {
                    task_id: 42,
                    status,
                })
            );
        }

        for code in [LIBAFL_RUN_STATUS_OK, LIBAFL_RUN_STATUS_INVALID_INPUT] {
            assert_eq!(
                completion_restart_request(
                    42,
                    LibAflRunStatus {
                        code,
                        ..LibAflRunStatus::default()
                    }
                ),
                None
            );
        }
    }

    #[test]
    fn async_completion_attribution_unions_distinct_task_maps() {
        let telemetry = CoverageTelemetry::new();
        let mut first_edges = vec![0; EDGES_MAP_SIZE];
        let mut second_edges = vec![0; EDGES_MAP_SIZE];
        let mut first_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        let mut second_memory = vec![0; SIMT_MEMCOV_STORAGE_SIZE];
        first_edges[11] = 0b0000_0001;
        second_edges[22] = 0b0000_0010;
        first_memory[33] = 0b0000_0100;
        second_memory[44] = 0b0000_1000;

        record_task_completion(Some(&telemetry), &first_edges, &first_memory).unwrap();
        record_task_completion(Some(&telemetry), &second_edges, &second_memory).unwrap();

        let snapshot = telemetry.snapshot().unwrap();
        assert_eq!(snapshot.executions_completed, 2);
        assert_eq!(snapshot.cfg_sites, 2);
        assert_eq!(snapshot.memory_features, 2);
    }

    #[test]
    fn batch_novelty_gate_skips_only_fully_known_feedback() {
        let mut state = StdState::new(
            StdRand::with_seed(7),
            InMemoryCorpus::<BytesInput>::new(),
            InMemoryCorpus::<BytesInput>::new(),
            &mut (),
            &mut (),
        )
        .unwrap();
        let mut known_edges = vec![0_u8; EDGES_MAP_SIZE];
        let mut known_simt = vec![0_u8; SIMT_MEMCOV_STORAGE_SIZE];
        known_edges[17] = 1;
        known_simt[23] = 0b0000_0100;
        state.add_named_metadata(
            CFG_FEEDBACK_NAME,
            MapFeedbackMetadata::with_history_map(known_edges.clone(), 0),
        );
        state.add_metadata(BitsetNoveltyStateMetadata {
            seen: known_simt.clone(),
            stats: Default::default(),
        });

        let completion = |task_id, edges: &[u8], simt: &[u8]| {
            AcquiredCompletion::new(
                crate::cuda_backend::TaskResult {
                    task_id,
                    edge_ptr: edges.as_ptr(),
                    edge_size: EDGES_MAP_SIZE as u32,
                    simt_memcov_ptr: simt.as_ptr(),
                    simt_memcov_size: SIMT_MEMCOV_STORAGE_SIZE as u32,
                    status: LibAflRunStatus::default(),
                    ..Default::default()
                },
                crate::gpu_executor::PendingTask {
                    task_id,
                    input: BytesInput::new(vec![task_id as u8]),
                    corpus_id: None,
                    from_supply: false,
                },
            )
        };
        let mut scratch = BatchNoveltyScratch::new();

        assert!(!scratch
            .needs_detailed_evaluation(&state, &[completion(1, &known_edges, &known_simt)],)
            .unwrap());

        let mut novel_edges = known_edges.clone();
        novel_edges[31] = 1;
        assert!(scratch
            .needs_detailed_evaluation(&state, &[completion(2, &novel_edges, &known_simt)],)
            .unwrap());

        let mut novel_simt = known_simt.clone();
        novel_simt[47] = 0b1000_0000;
        assert!(scratch
            .needs_detailed_evaluation(&state, &[completion(3, &known_edges, &novel_simt)],)
            .unwrap());
    }

    #[test]
    fn async_final_drain_rejects_any_pending_work() {
        assert!(ensure_campaign_drained(0, 0, 0).is_ok());
        for (executor_pending, backend_pending, backend_completed) in
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        {
            assert!(
                ensure_campaign_drained(executor_pending, backend_pending, backend_completed)
                    .is_err()
            );
        }
    }

    #[test]
    fn retirement_uses_captured_parent_and_restores_current_id() {
        let mut state = StdState::new(
            StdRand::with_seed(7),
            InMemoryCorpus::<BytesInput>::new(),
            InMemoryCorpus::<BytesInput>::new(),
            &mut (),
            &mut (),
        )
        .unwrap();
        let parent_id = state
            .corpus_mut()
            .add(BytesInput::new(vec![1]).into())
            .unwrap();
        let saved_id = state
            .corpus_mut()
            .add(BytesInput::new(vec![2]).into())
            .unwrap();
        state.set_corpus_id(saved_id).unwrap();

        with_parent_attribution(&mut state, Some(parent_id), |state| {
            assert_eq!(state.current_corpus_id()?, Some(parent_id));
            Ok(())
        })
        .unwrap();

        assert_eq!(state.current_corpus_id().unwrap(), Some(saved_id));
        assert_eq!(state.testcase(parent_id).unwrap().scheduled_count(), 1);
        assert_eq!(state.testcase(saved_id).unwrap().scheduled_count(), 0);
    }

    #[test]
    fn external_event_budget_cannot_overfill_submission_channel() {
        assert_eq!(remaining_submission_budget(128, 10, 10), 128);
        assert_eq!(remaining_submission_budget(128, 10, 110), 28);
        assert_eq!(remaining_submission_budget(128, 10, 138), 0);
        assert_eq!(remaining_submission_budget(128, 10, 200), 0);
    }
}
