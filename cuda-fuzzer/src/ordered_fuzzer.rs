use core::{fmt::Debug, marker::PhantomData, num::NonZeroUsize, time::Duration};
use serde::{de::DeserializeOwned, Serialize};
use std::rc::Rc;

use libafl::{
    corpus::{Corpus, CorpusId, HasCurrentCorpusId, HasTestcase, Testcase},
    events::{Event, EventFirer, EventReceiver, ProgressReporter, SendExiting},
    executors::{Executor, ExitKind, HasObservers},
    feedbacks::Feedback,
    fuzzer::{
        Evaluator, EvaluatorObservers, EventProcessor, ExecuteInputResult, ExecutesInput,
        ExecutionProcessor, Fuzzer, HasScheduler, NopInputFilter, StdFuzzer,
    },
    inputs::{BytesInput, HasTargetBytes, NopToTargetBytes},
    observers::ObserversTuple,
    schedulers::Scheduler,
    state::{
        HasCorpus, HasCurrentStageId, HasCurrentTestcase, HasExecutions, HasImported,
        HasLastFoundTime, HasLastReportTime, HasSolutions, MaybeHasClientPerfMonitor, Stoppable,
    },
    Error, HasMetadata,
};
use libafl_bolts::{tuples::RefIndexable, AsSlice};

use crate::{
    arg_pack_v1::normalize_rapid_input_v1,
    coverage_telemetry::CoverageTelemetry,
    cuda_backend::{
        exit_kind_from_run_status, run_status_requires_fresh_process, OrderedCudaBackend,
        OrderedQueueCounts, TaskResult, EDGES_MAP_SIZE,
    },
    gpu_executor::PendingTask,
    ordered_window::OrderedWindow,
    simt_memcov_feedback::SimtMemCovFeedback,
    task_time_feedback::TaskTimeFeedback,
};

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
    backend_outstanding: usize,
) -> Result<(), Error> {
    if executor_pending != 0
        || backend_pending != 0
        || backend_completed != 0
        || backend_outstanding != 0
    {
        return Err(Error::illegal_state(format!(
            "ordered drain left executor_pending={executor_pending} backend_pending={backend_pending} backend_completed={backend_completed} backend_outstanding={backend_outstanding}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod coverage_campaign_tests {
    use super::{ensure_campaign_drained, record_task_completion};
    use crate::{
        coverage_telemetry::CoverageTelemetry,
        cuda_backend::{EDGES_MAP_SIZE, SIMT_MEMCOV_STORAGE_SIZE},
    };

    #[test]
    fn ordered_completion_attribution_unions_distinct_task_maps() {
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
    fn ordered_completion_attribution_is_disabled_by_default() {
        record_task_completion(
            None,
            &vec![0; EDGES_MAP_SIZE],
            &vec![0; SIMT_MEMCOV_STORAGE_SIZE],
        )
        .unwrap();
    }

    #[test]
    fn ordered_final_drain_rejects_any_pending_work() {
        assert!(ensure_campaign_drained(0, 0, 0, 0).is_ok());
        for (executor_pending, backend_pending, backend_completed, backend_outstanding) in
            [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        {
            assert!(ensure_campaign_drained(
                executor_pending,
                backend_pending,
                backend_completed,
                backend_outstanding,
            )
            .is_err());
        }
    }
}

pub struct OrderedGpuExecutor<OT, S> {
    cuda_backend: Rc<OrderedCudaBackend>,
    window: OrderedWindow<PendingTask<BytesInput>, TaskResult>,
    total_submitted: u64,
    observers: OT,
    phantom: PhantomData<S>,
}

impl<OT, S> OrderedGpuExecutor<OT, S> {
    pub fn new(
        cuda_backend: Rc<OrderedCudaBackend>,
        observers: OT,
        window_size: NonZeroUsize,
    ) -> Self {
        Self {
            cuda_backend,
            window: OrderedWindow::new(window_size),
            total_submitted: 0,
            observers,
            phantom: PhantomData,
        }
    }

    pub fn pending_count(&self) -> usize {
        self.window.pending_len()
    }

    pub fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    pub fn can_submit(&self) -> bool {
        self.window.can_submit()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    pub fn queue_counts(&self) -> OrderedQueueCounts {
        self.cuda_backend.get_queue_counts()
    }

    fn record_completed(&mut self, result: TaskResult) -> Result<(), Error> {
        self.window
            .record_ready(result.task_id, result)
            .map_err(Error::illegal_state)
    }

    fn pop_ready_head(&mut self) -> Option<(u64, PendingTask<BytesInput>, TaskResult)> {
        self.window.pop_ready_head()
    }
}

impl<EM, OT, S, Z> Executor<EM, BytesInput, S, Z> for OrderedGpuExecutor<OT, S>
where
    OT: ObserversTuple<BytesInput, S>,
    S: HasCurrentCorpusId,
{
    fn run_target(
        &mut self,
        _fuzzer: &mut Z,
        state: &mut S,
        _manager: &mut EM,
        input: &BytesInput,
    ) -> Result<ExitKind, Error> {
        if !self.window.can_submit() {
            return Err(Error::illegal_state(
                "ordered executor submitted while its window was full".to_string(),
            ));
        }
        let normalized = normalize_rapid_input_v1(input.target_bytes().as_slice());
        let task_id = match self.cuda_backend.submit(&normalized) {
            Ok(task_id) => task_id.get(),
            Err(_) => return Ok(ExitKind::Crash),
        };
        let pending = PendingTask {
            task_id,
            input: BytesInput::new(normalized),
            corpus_id: state.current_corpus_id()?,
            from_supply: false,
        };
        self.window
            .push_pending(task_id, pending)
            .map_err(Error::illegal_state)?;
        self.total_submitted = self.total_submitted.saturating_add(1);
        Ok(ExitKind::Ok)
    }
}

impl<OT, S> HasObservers for OrderedGpuExecutor<OT, S>
where
    OT: ObserversTuple<BytesInput, S>,
{
    type Observers = OT;

    fn observers(&self) -> RefIndexable<&Self::Observers, Self::Observers> {
        RefIndexable::from(&self.observers)
    }

    fn observers_mut(&mut self) -> RefIndexable<&mut Self::Observers, Self::Observers> {
        RefIndexable::from(&mut self.observers)
    }
}

#[derive(Debug)]
pub struct OrderedBatchFuzzer<CS, F, OF> {
    inner: StdFuzzer<CS, F, NopToTargetBytes, NopInputFilter, OF>,
    cuda_backend: Rc<OrderedCudaBackend>,
    simt_memcov_feedback: SimtMemCovFeedback,
    task_time_feedback: TaskTimeFeedback,
    coverage_telemetry: Option<CoverageTelemetry>,
}

impl<CS, F, OF> OrderedBatchFuzzer<CS, F, OF> {
    pub fn new(
        scheduler: CS,
        feedback: F,
        objective: OF,
        cuda_backend: Rc<OrderedCudaBackend>,
        simt_memcov_feedback: SimtMemCovFeedback,
        task_time_feedback: TaskTimeFeedback,
    ) -> Self {
        Self {
            inner: StdFuzzer::new(scheduler, feedback, objective),
            cuda_backend,
            simt_memcov_feedback,
            task_time_feedback,
            coverage_telemetry: None,
        }
    }

    pub fn with_coverage_telemetry(mut self, telemetry: CoverageTelemetry) -> Self {
        self.coverage_telemetry = Some(telemetry);
        self
    }

    fn update_coverage_map(&self, coverage: &[u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                coverage.as_ptr(),
                self.cuda_backend.cov_map_ptr(),
                EDGES_MAP_SIZE.min(coverage.len()),
            );
        }
    }

    fn record_available_results<OT, S>(
        &self,
        executor: &mut OrderedGpuExecutor<OT, S>,
    ) -> Result<usize, Error>
    where
        OT: ObserversTuple<BytesInput, S>,
    {
        if executor.is_empty() {
            return Ok(0);
        }
        let results = self
            .cuda_backend
            .poll_completed_tasks(executor.pending_count());
        let count = results.len();
        for result in results {
            if let Err(error) = executor.record_completed(result) {
                let released = self.cuda_backend.release_completed_tasks(&[result.task_id]);
                return Err(Error::illegal_state(format!(
                    "failed to record ordered completion task_id={}: {error}; released={released}",
                    result.task_id
                )));
            }
        }
        Ok(count)
    }

    fn process_one_completion<EM, OT, S>(
        &mut self,
        executor: &mut OrderedGpuExecutor<OT, S>,
        state: &mut S,
        manager: &mut EM,
        pending: &PendingTask<BytesInput>,
        result: &TaskResult,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error>
    where
        CS: Scheduler<BytesInput, S>,
        EM: EventFirer<BytesInput, S>,
        F: Feedback<EM, BytesInput, OT, S>,
        OF: Feedback<EM, BytesInput, OT, S>,
        OT: ObserversTuple<BytesInput, S> + Serialize,
        S: HasCorpus<BytesInput>
            + HasExecutions
            + HasSolutions<BytesInput>
            + HasLastFoundTime
            + HasCurrentTestcase<BytesInput>
            + HasCurrentCorpusId
            + HasTestcase<BytesInput>
            + MaybeHasClientPerfMonitor,
    {
        let edge_map = unsafe { result.edge_slice() }
            .map_err(|error| Error::illegal_state(error.to_string()))?;
        let simt_memcov = unsafe { result.simt_memcov_slice() }
            .map_err(|error| Error::illegal_state(error.to_string()))?;
        record_task_completion(self.coverage_telemetry.as_ref(), edge_map, simt_memcov)?;
        self.update_coverage_map(edge_map);
        self.simt_memcov_feedback
            .observe(simt_memcov)
            .map_err(|error| Error::illegal_state(error.to_string()))?;
        self.task_time_feedback
            .observe_ns(result.exec_time_ns)
            .map_err(|error| Error::illegal_state(error.to_string()))?;

        *state.executions_mut() += 1;
        let exit_kind = exit_kind_from_run_status(result.status);
        let saved_corpus_id = state.current_corpus_id()?;
        if let Some(parent_id) = pending.corpus_id {
            state.set_corpus_id(parent_id)?;
        }

        let process_result = (|| {
            executor
                .observers_mut()
                .post_exec_all(state, &pending.input, &exit_kind)?;
            let observers = executor.observers();
            self.inner
                .scheduler_mut()
                .on_evaluation(state, &pending.input, &*observers)?;
            let evaluated = self.inner.evaluate_execution(
                state,
                manager,
                &pending.input,
                &*observers,
                &exit_kind,
                send_events,
            )?;
            if let Some(parent_id) = pending.corpus_id {
                if let Ok(mut testcase) = state.testcase_mut(parent_id) {
                    let scheduled_count = testcase.scheduled_count();
                    testcase.set_scheduled_count(scheduled_count + 1);
                }
            }
            Ok(evaluated)
        })();

        let restore_result = if let Some(id) = saved_corpus_id {
            state.set_corpus_id(id)
        } else {
            state.clear_corpus_id()
        };
        match (process_result, restore_result) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(process_error), Err(restore_error)) => Err(Error::illegal_state(format!(
                "ordered evaluation failed: {process_error}; corpus restoration failed: {restore_error}"
            ))),
        }
    }

    fn process_ready_heads<EM, OT, S>(
        &mut self,
        executor: &mut OrderedGpuExecutor<OT, S>,
        state: &mut S,
        manager: &mut EM,
        send_events: bool,
    ) -> Result<(usize, ExecuteInputResult, Option<CorpusId>), Error>
    where
        CS: Scheduler<BytesInput, S>,
        EM: EventFirer<BytesInput, S>,
        F: Feedback<EM, BytesInput, OT, S>,
        OF: Feedback<EM, BytesInput, OT, S>,
        OT: ObserversTuple<BytesInput, S> + Serialize,
        S: HasCorpus<BytesInput>
            + HasExecutions
            + HasSolutions<BytesInput>
            + HasLastFoundTime
            + HasCurrentTestcase<BytesInput>
            + HasCurrentCorpusId
            + HasTestcase<BytesInput>
            + MaybeHasClientPerfMonitor,
    {
        self.record_available_results(executor)?;
        let mut processed = 0usize;
        let mut last_result = ExecuteInputResult::default();
        let mut last_corpus_id = None;
        while let Some((task_id, pending, result)) = executor.pop_ready_head() {
            let restart_required = run_status_requires_fresh_process(result.status);
            let process_result = self.process_one_completion(
                executor,
                state,
                manager,
                &pending,
                &result,
                send_events,
            );
            let released = self.cuda_backend.release_completed_tasks(&[task_id]);
            let evaluated = match (process_result, released) {
                (Ok(value), 1) => value,
                (Err(error), 1) => return Err(error),
                (Ok(_), count) => {
                    return Err(Error::illegal_state(format!(
                        "released {count} of 1 ordered completion task_id={task_id}"
                    )));
                }
                (Err(error), count) => {
                    return Err(Error::illegal_state(format!(
                        "ordered completion task_id={task_id} failed: {error}; released {count} of 1"
                    )));
                }
            };
            processed += 1;
            last_result = evaluated.0;
            last_corpus_id = evaluated.1;
            if restart_required {
                abort_after_non_recoverable_completion(task_id, result.status);
            }
        }
        Ok((processed, last_result, last_corpus_id))
    }

    fn ensure_submission_slot<EM, OT, S>(
        &mut self,
        executor: &mut OrderedGpuExecutor<OT, S>,
        state: &mut S,
        manager: &mut EM,
        send_events: bool,
    ) -> Result<(), Error>
    where
        CS: Scheduler<BytesInput, S>,
        EM: EventFirer<BytesInput, S>,
        F: Feedback<EM, BytesInput, OT, S>,
        OF: Feedback<EM, BytesInput, OT, S>,
        OT: ObserversTuple<BytesInput, S> + Serialize,
        S: HasCorpus<BytesInput>
            + HasExecutions
            + HasSolutions<BytesInput>
            + HasLastFoundTime
            + HasCurrentTestcase<BytesInput>
            + HasCurrentCorpusId
            + HasTestcase<BytesInput>
            + MaybeHasClientPerfMonitor,
    {
        self.process_ready_heads(executor, state, manager, send_events)?;
        while !executor.can_submit() {
            let (processed, _, _) =
                self.process_ready_heads(executor, state, manager, send_events)?;
            if processed == 0 {
                let counts = executor.queue_counts();
                if counts.pending == 0 && counts.completed == 0 {
                    return Err(Error::illegal_state(format!(
                        "ordered window has {} unretired tasks but backend has no pending or completed task",
                        executor.pending_count()
                    )));
                }
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    pub fn force_evaluation<EM, OT, S>(
        &mut self,
        executor: &mut OrderedGpuExecutor<OT, S>,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), Error>
    where
        CS: Scheduler<BytesInput, S>,
        EM: EventFirer<BytesInput, S>,
        F: Feedback<EM, BytesInput, OT, S>,
        OF: Feedback<EM, BytesInput, OT, S>,
        OT: ObserversTuple<BytesInput, S> + Serialize,
        S: HasCorpus<BytesInput>
            + HasExecutions
            + HasSolutions<BytesInput>
            + HasLastFoundTime
            + HasCurrentTestcase<BytesInput>
            + HasCurrentCorpusId
            + HasTestcase<BytesInput>
            + MaybeHasClientPerfMonitor,
    {
        self.cuda_backend.wait();
        while !executor.is_empty() {
            let (processed, _, _) = self.process_ready_heads(executor, state, manager, false)?;
            if processed == 0 {
                let counts = executor.queue_counts();
                if counts.completed == 0 {
                    return Err(Error::illegal_state(format!(
                        "ordered drain stalled with {} unretired tasks: pending={} completed={} outstanding={}",
                        executor.pending_count(), counts.pending, counts.completed, counts.outstanding
                    )));
                }
                std::thread::yield_now();
            }
        }
        let counts = executor.queue_counts();
        ensure_campaign_drained(
            executor.pending_count(),
            counts.pending,
            counts.completed,
            counts.outstanding,
        )?;
        Ok(())
    }
}

fn abort_after_non_recoverable_completion(
    task_id: u64,
    status: crate::cuda_backend::LibAflRunStatus,
) -> ! {
    eprintln!(
        "RAPID NON-RECOVERABLE COMPLETION: task_id={} status_code={} stage={} detail={:#x}. Aborting client process so the restarting manager can create a fresh CUDA context.",
        task_id, status.code, status.stage, status.detail
    );
    std::process::abort();
}

impl<CS, EM, F, OF, OT, S> ExecutesInput<OrderedGpuExecutor<OT, S>, EM, BytesInput, S>
    for OrderedBatchFuzzer<CS, F, OF>
where
    CS: Scheduler<BytesInput, S>,
    EM: EventFirer<BytesInput, S>,
    F: Feedback<EM, BytesInput, OT, S>,
    OF: Feedback<EM, BytesInput, OT, S>,
    OT: ObserversTuple<BytesInput, S> + Serialize,
    S: HasCorpus<BytesInput>
        + HasExecutions
        + HasSolutions<BytesInput>
        + HasLastFoundTime
        + HasCurrentTestcase<BytesInput>
        + HasCurrentCorpusId
        + HasTestcase<BytesInput>
        + MaybeHasClientPerfMonitor,
{
    fn execute_input(
        &mut self,
        state: &mut S,
        executor: &mut OrderedGpuExecutor<OT, S>,
        manager: &mut EM,
        input: &BytesInput,
    ) -> Result<ExitKind, Error> {
        self.ensure_submission_slot(executor, state, manager, true)?;
        executor.observers_mut().pre_exec_all(state, input)?;
        let exit_kind = executor.run_target(self, state, manager, input)?;
        if matches!(exit_kind, ExitKind::Ok) {
            self.simt_memcov_feedback
                .record_submission()
                .map_err(|error| Error::illegal_state(error.to_string()))?;
            if let Some(telemetry) = &self.coverage_telemetry {
                telemetry
                    .record_submission()
                    .map_err(|error| Error::illegal_state(error.to_string()))?;
            }
        }
        Ok(exit_kind)
    }
}

impl<CS, EM, F, OF, OT, S> EvaluatorObservers<OrderedGpuExecutor<OT, S>, EM, BytesInput, S>
    for OrderedBatchFuzzer<CS, F, OF>
where
    CS: Scheduler<BytesInput, S>,
    EM: EventFirer<BytesInput, S>,
    F: Feedback<EM, BytesInput, OT, S>,
    OF: Feedback<EM, BytesInput, OT, S>,
    OT: ObserversTuple<BytesInput, S> + Serialize,
    S: HasCorpus<BytesInput>
        + HasExecutions
        + HasSolutions<BytesInput>
        + HasLastFoundTime
        + HasCurrentTestcase<BytesInput>
        + HasCurrentCorpusId
        + HasTestcase<BytesInput>
        + MaybeHasClientPerfMonitor,
{
    fn evaluate_input_with_observers(
        &mut self,
        state: &mut S,
        executor: &mut OrderedGpuExecutor<OT, S>,
        manager: &mut EM,
        input: &BytesInput,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        let exit_kind = self.execute_input(state, executor, manager, input)?;
        if !matches!(exit_kind, ExitKind::Ok) {
            *state.executions_mut() += 1;
            executor
                .observers_mut()
                .post_exec_all(state, input, &exit_kind)?;
            let observers = executor.observers();
            let _ = self.inner.evaluate_execution(
                state,
                manager,
                input,
                &*observers,
                &exit_kind,
                send_events,
            )?;
            return Err(Error::illegal_state(
                "ordered CUDA backend rejected submission with task_id=0".to_string(),
            ));
        }
        let (_, result, corpus_id) =
            self.process_ready_heads(executor, state, manager, send_events)?;
        Ok((result, corpus_id))
    }
}

impl<CS, EM, F, OF, OT, S> Evaluator<OrderedGpuExecutor<OT, S>, EM, BytesInput, S>
    for OrderedBatchFuzzer<CS, F, OF>
where
    CS: Scheduler<BytesInput, S>,
    EM: EventFirer<BytesInput, S>,
    F: Feedback<EM, BytesInput, OT, S>,
    OF: Feedback<EM, BytesInput, OT, S>,
    OT: ObserversTuple<BytesInput, S> + Serialize,
    S: HasCorpus<BytesInput>
        + HasExecutions
        + HasSolutions<BytesInput>
        + HasMetadata
        + HasLastFoundTime
        + HasCurrentTestcase<BytesInput>
        + HasCurrentCorpusId
        + HasTestcase<BytesInput>
        + MaybeHasClientPerfMonitor,
{
    fn evaluate_filtered(
        &mut self,
        state: &mut S,
        executor: &mut OrderedGpuExecutor<OT, S>,
        manager: &mut EM,
        input: &BytesInput,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        self.evaluate_input(state, executor, manager, input)
    }

    fn evaluate_input(
        &mut self,
        state: &mut S,
        executor: &mut OrderedGpuExecutor<OT, S>,
        manager: &mut EM,
        input: &BytesInput,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        self.evaluate_input_with_observers(state, executor, manager, input, true)
    }

    fn add_input(
        &mut self,
        state: &mut S,
        _executor: &mut OrderedGpuExecutor<OT, S>,
        _manager: &mut EM,
        input: BytesInput,
    ) -> Result<(CorpusId, ExecuteInputResult), Error> {
        let mut testcase = Testcase::from(input);
        testcase.set_executions(*state.executions());
        let id = state.corpus_mut().add(testcase)?;
        Ok((id, ExecuteInputResult::default()))
    }

    fn add_disabled_input(&mut self, state: &mut S, input: BytesInput) -> Result<CorpusId, Error> {
        let mut testcase = Testcase::from(input);
        testcase.set_executions(*state.executions());
        testcase.set_disabled(true);
        Ok(state.corpus_mut().add_disabled(testcase)?)
    }
}

impl<CS, EM, F, OF, OT, S> EventProcessor<OrderedGpuExecutor<OT, S>, EM, BytesInput, S>
    for OrderedBatchFuzzer<CS, F, OF>
where
    CS: Scheduler<BytesInput, S>,
    EM: EventReceiver<BytesInput, S> + EventFirer<BytesInput, S>,
    F: Feedback<EM, BytesInput, OT, S>,
    OF: Feedback<EM, BytesInput, OT, S>,
    OT: ObserversTuple<BytesInput, S> + Serialize + DeserializeOwned,
    S: HasCorpus<BytesInput>
        + HasSolutions<BytesInput>
        + HasExecutions
        + HasLastFoundTime
        + HasMetadata
        + MaybeHasClientPerfMonitor
        + HasCurrentCorpusId
        + HasCurrentTestcase<BytesInput>
        + HasTestcase<BytesInput>
        + HasImported,
{
    fn process_events(
        &mut self,
        state: &mut S,
        executor: &mut OrderedGpuExecutor<OT, S>,
        manager: &mut EM,
    ) -> Result<(), Error> {
        while let Some((event, with_observers)) = manager.try_receive(state)? {
            let added = if with_observers {
                match event.event() {
                    Event::NewTestcase {
                        input,
                        observers_buf,
                        exit_kind,
                        ..
                    } => {
                        let observers: OT = postcard::from_bytes(observers_buf.as_ref().unwrap())?;
                        self.inner
                            .evaluate_execution(
                                state, manager, input, &observers, exit_kind, false,
                            )?
                            .1
                    }
                    _ => None,
                }
            } else {
                match event.event() {
                    Event::NewTestcase { input, .. } => {
                        self.evaluate_input_with_observers(state, executor, manager, input, false)?
                            .1
                    }
                    Event::Objective {
                        input: Some(input), ..
                    } => {
                        self.evaluate_input_with_observers(state, executor, manager, input, false)?
                            .1
                    }
                    _ => None,
                }
            };
            if let Some(id) = added {
                *state.imported_mut() += 1;
                log::debug!("Added received ordered input as item #{id}");
                manager.on_interesting(state, event)?;
            }
        }
        Ok(())
    }
}

impl<CS, EM, F, OF, OT, S, ST> Fuzzer<OrderedGpuExecutor<OT, S>, EM, BytesInput, S, ST>
    for OrderedBatchFuzzer<CS, F, OF>
where
    CS: Scheduler<BytesInput, S>,
    EM: EventFirer<BytesInput, S>
        + EventReceiver<BytesInput, S>
        + ProgressReporter<S>
        + SendExiting,
    F: Feedback<EM, BytesInput, OT, S>,
    OF: Feedback<EM, BytesInput, OT, S>,
    OT: ObserversTuple<BytesInput, S> + Serialize + DeserializeOwned,
    S: HasExecutions
        + HasMetadata
        + HasCorpus<BytesInput>
        + HasSolutions<BytesInput>
        + HasLastReportTime
        + HasLastFoundTime
        + HasImported
        + HasTestcase<BytesInput>
        + HasCurrentCorpusId
        + HasCurrentStageId
        + Stoppable
        + MaybeHasClientPerfMonitor,
    ST: libafl::stages::StagesTuple<OrderedGpuExecutor<OT, S>, EM, S, Self>,
{
    fn fuzz_one(
        &mut self,
        stages: &mut ST,
        executor: &mut OrderedGpuExecutor<OT, S>,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<CorpusId, Error> {
        let id = if let Some(id) = state.current_corpus_id()? {
            id
        } else {
            let id = self.inner.scheduler_mut().next(state)?;
            state.set_corpus_id(id)?;
            id
        };
        stages.perform_all(self, executor, state, manager)?;
        self.process_events(state, executor, manager)?;
        if let Ok(mut testcase) = state.testcase_mut(id) {
            let scheduled_count = testcase.scheduled_count();
            testcase.set_scheduled_count(scheduled_count + 1);
        }
        state.clear_corpus_id()?;
        if state.stop_requested() {
            state.discard_stop_request();
            manager.on_shutdown()?;
            return Err(Error::shutting_down());
        }
        Ok(id)
    }

    fn fuzz_loop(
        &mut self,
        stages: &mut ST,
        executor: &mut OrderedGpuExecutor<OT, S>,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), Error> {
        loop {
            manager.maybe_report_progress(state, Duration::from_secs(15))?;
            self.fuzz_one(stages, executor, state, manager)?;
        }
    }

    fn fuzz_loop_for(
        &mut self,
        stages: &mut ST,
        executor: &mut OrderedGpuExecutor<OT, S>,
        state: &mut S,
        manager: &mut EM,
        iters: u64,
    ) -> Result<CorpusId, Error> {
        if iters == 0 {
            return Err(Error::illegal_argument(
                "Cannot fuzz for 0 iterations!".to_string(),
            ));
        }
        let mut last = None;
        for _ in 0..iters {
            manager.maybe_report_progress(state, Duration::from_secs(15))?;
            last = Some(self.fuzz_one(stages, executor, state, manager)?);
        }
        self.force_evaluation(executor, state, manager)?;
        manager.report_progress(state)?;
        Ok(last.unwrap())
    }
}
