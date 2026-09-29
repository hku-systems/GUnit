use std::collections::VecDeque;
use std::marker::PhantomData;
use std::{
    num::NonZeroU64,
    sync::{mpsc::SyncSender, Arc},
};

use libafl::{
    corpus::{CorpusId, HasCurrentCorpusId},
    executors::{Executor, ExitKind, HasObservers},
    inputs::{BytesInput, HasTargetBytes},
    observers::{ExplicitTracking, ObserversTuple, StdMapObserver},
    Error,
};
use libafl_bolts::{ownedref::OwnedMutSlice, tuples::RefIndexable, AsSlice};

use crate::arg_pack_v1::normalize_rapid_input_v1;
use crate::cuda_backend::{AsyncCudaBackend, AsyncSubmitError};
#[cfg(feature = "profiling")]
use crate::profiling::{scope, Segment};
use crate::retirement::RetirementCredits;
use crate::submission_context::{current as submission_context, SubmissionContext};
use crate::supply_pool::SupplyStats;

/// Trait for executors that manage pending tasks
pub trait HasPendingTasks<I> {
    /// Remove and return the FIFO head matching the completion ID.
    fn take_pending_task(&mut self, task_id: u64) -> Option<PendingTask<I>>;

    /// Get the number of pending tasks
    fn pending_count(&self) -> usize;

    /// Get the total number of submitted tasks
    fn total_submitted(&self) -> u64;

    /// Get a reference to the asynchronous CUDA backend.
    fn cuda_backend(&self) -> &Arc<AsyncCudaBackend>;
}

/// A pending task that hasn't been evaluated yet
#[derive(Debug, Clone)]
pub struct PendingTask<I> {
    pub task_id: u64,
    pub input: I,
    pub corpus_id: Option<CorpusId>,
    pub from_supply: bool,
}

#[derive(Debug)]
pub struct RetirementSubmission<I> {
    pub input: I,
    pub corpus_id: Option<CorpusId>,
    pub from_supply: bool,
}

#[derive(Debug)]
struct PendingTaskStore<I> {
    fifo: VecDeque<PendingTask<I>>,
}

impl<I> PendingTaskStore<I> {
    fn new() -> Self {
        Self {
            fifo: VecDeque::new(),
        }
    }

    fn insert(&mut self, task: PendingTask<I>) -> Result<(), String> {
        let task_id = task.task_id;
        if task_id == 0 {
            return Err("pending task ID must be nonzero".to_string());
        }
        if self
            .fifo
            .back()
            .is_some_and(|pending| pending.task_id == task_id)
        {
            return Err(format!("duplicate pending task ID {task_id}"));
        }
        self.fifo.push_back(task);
        Ok(())
    }

    fn remove(&mut self, task_id: u64) -> Option<PendingTask<I>> {
        let expected = self.fifo.front()?.task_id;
        assert_eq!(
            task_id, expected,
            "RAPID2 completion order violation: expected task_id={expected}, got task_id={task_id}"
        );
        self.fifo.pop_front()
    }

    fn len(&self) -> usize {
        self.fifo.len()
    }
}

fn submit_bytes_input_and_build_pending_task<F>(
    input: &BytesInput,
    corpus_id: Option<CorpusId>,
    context: SubmissionContext,
    submit: F,
) -> Result<PendingTask<BytesInput>, AsyncSubmitError>
where
    F: FnOnce(&[u8]) -> Result<NonZeroU64, AsyncSubmitError>,
{
    let target_bytes = input.target_bytes();
    let submitted = if context.is_canonical() {
        target_bytes.as_slice().to_vec()
    } else {
        normalize_rapid_input_v1(target_bytes.as_slice())
    };
    let task_id = submit(&submitted)?.get();

    Ok(PendingTask {
        task_id,
        input: BytesInput::new(submitted),
        corpus_id,
        from_supply: context.is_supply(),
    })
}

/// A GPU executor that submits tasks asynchronously without waiting
/// The actual execution happens asynchronously on the GPU
/// This executor manages the queue of pending tasks
pub struct GpuExecutor<I, OT, S> {
    /// The strict asynchronous CUDA backend.
    cuda_backend: Arc<AsyncCudaBackend>,
    /// Queue of pending tasks awaiting evaluation
    pending_tasks: PendingTaskStore<I>,
    /// Total number of submitted tasks
    total_submitted: u64,
    supply_stats: Option<Arc<SupplyStats>>,
    /// The observers
    observers: OT,
    /// Phantom data
    phantom: PhantomData<(I, S)>,
}

impl<I, OT, S> GpuExecutor<I, OT, S> {
    /// Create a new GPU executor
    pub fn new(cuda_backend: Arc<AsyncCudaBackend>, observers: OT) -> Self {
        Self {
            cuda_backend,
            pending_tasks: PendingTaskStore::new(),
            total_submitted: 0,
            supply_stats: None,
            observers,
            phantom: PhantomData,
        }
    }

    pub fn set_supply_stats(&mut self, stats: Arc<SupplyStats>) {
        self.supply_stats = Some(stats);
    }
}

impl<I, OT, S> HasPendingTasks<I> for GpuExecutor<I, OT, S> {
    fn take_pending_task(&mut self, task_id: u64) -> Option<PendingTask<I>> {
        self.pending_tasks.remove(task_id)
    }

    fn pending_count(&self) -> usize {
        self.pending_tasks.len()
    }

    fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    fn cuda_backend(&self) -> &Arc<AsyncCudaBackend> {
        &self.cuda_backend
    }
}

impl<EM, OT, S, Z> Executor<EM, BytesInput, S, Z> for GpuExecutor<BytesInput, OT, S>
where
    OT: ObserversTuple<BytesInput, S>,
    S: HasCurrentCorpusId,
{
    fn run_target(
        &mut self,
        _fuzzer: &mut Z,
        state: &mut S,
        _mgr: &mut EM,
        input: &BytesInput,
    ) -> Result<ExitKind, Error> {
        // Note: We do NOT increment executions here!
        // In async execution, this only submits the task.
        // The execution count will be incremented when the task completes
        // and is processed in AsyncBatchFuzzer::process_completed_tasks

        // Get corpus ID if available (for tracking parent corpus)
        let corpus_id = state.current_corpus_id()?;

        let pending_task = {
            #[cfg(feature = "profiling")]
            let _timing = scope(Segment::Submit);
            submit_bytes_input_and_build_pending_task(
                input,
                corpus_id,
                submission_context(),
                |bytes| self.cuda_backend.submit(bytes),
            )
        };

        if let Ok(pending_task) = pending_task {
            let task_id = pending_task.task_id;
            let from_supply = pending_task.from_supply;
            self.pending_tasks
                .insert(pending_task)
                .map_err(Error::illegal_state)?;
            self.total_submitted += 1;
            if from_supply {
                let stats = self
                    .supply_stats
                    .as_ref()
                    .expect("supply submission requires supply statistics");
                stats.record_submitted();
            }

            // Debug output for tracking
            if self.total_submitted.is_multiple_of(100) {
                log::debug!(
                    "[GpuExecutor] Submitted input #{} with task_id={}, corpus_id={:?}, pending={}",
                    self.total_submitted,
                    task_id,
                    corpus_id,
                    self.pending_tasks.len()
                );
            }
            Ok(ExitKind::Ok)
        } else {
            log::error!("[GpuExecutor] RAPID2 backend rejected submission with task_id=0");
            Ok(ExitKind::Crash)
        }
    }
}

impl<I, OT, S> HasObservers for GpuExecutor<I, OT, S>
where
    OT: ObserversTuple<I, S>,
{
    type Observers = OT;

    fn observers(&self) -> RefIndexable<&Self::Observers, Self::Observers> {
        RefIndexable::from(&self.observers)
    }

    fn observers_mut(&mut self) -> RefIndexable<&mut Self::Observers, Self::Observers> {
        RefIndexable::from(&mut self.observers)
    }
}

pub struct RetirementSubmitExecutor<I, OT, S> {
    cuda_backend: Arc<AsyncCudaBackend>,
    sender: Option<SyncSender<RetirementSubmission<I>>>,
    credits: Arc<RetirementCredits>,
    total_submitted: u64,
    supply_stats: Option<Arc<SupplyStats>>,
    observers: OT,
    phantom: PhantomData<S>,
}

pub type RetirementMapObserver = ExplicitTracking<StdMapObserver<'static, u8, false>, true, false>;
pub type RetirementObservers = (RetirementMapObserver, ());

pub fn replace_observer_coverage_map(
    observer: &mut RetirementMapObserver,
    map: Vec<u8>,
) -> Vec<u8> {
    let old = std::mem::replace(observer.as_mut().map_mut(), OwnedMutSlice::from(map));
    Vec::from(old)
}

impl<I, OT, S> RetirementSubmitExecutor<I, OT, S> {
    pub fn new(
        cuda_backend: Arc<AsyncCudaBackend>,
        sender: SyncSender<RetirementSubmission<I>>,
        credits: Arc<RetirementCredits>,
        observers: OT,
    ) -> Self {
        Self {
            cuda_backend,
            sender: Some(sender),
            credits,
            total_submitted: 0,
            supply_stats: None,
            observers,
            phantom: PhantomData,
        }
    }

    pub fn set_supply_stats(&mut self, stats: Arc<SupplyStats>) {
        self.supply_stats = Some(stats);
    }

    pub fn close_submissions(&mut self) {
        self.sender.take();
    }
}

impl<I, S> RetirementSubmitExecutor<I, RetirementObservers, S> {
    pub fn replace_coverage_map(&mut self, map: Vec<u8>) -> Vec<u8> {
        replace_observer_coverage_map(&mut self.observers.0, map)
    }
}

impl<I, OT, S> HasPendingTasks<I> for RetirementSubmitExecutor<I, OT, S> {
    fn take_pending_task(&mut self, _task_id: u64) -> Option<PendingTask<I>> {
        None
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn total_submitted(&self) -> u64 {
        self.total_submitted
    }

    fn cuda_backend(&self) -> &Arc<AsyncCudaBackend> {
        &self.cuda_backend
    }
}

impl<EM, OT, S, Z> Executor<EM, BytesInput, S, Z> for RetirementSubmitExecutor<BytesInput, OT, S>
where
    OT: ObserversTuple<BytesInput, S>,
    S: HasCurrentCorpusId,
{
    fn run_target(
        &mut self,
        _fuzzer: &mut Z,
        state: &mut S,
        _mgr: &mut EM,
        input: &BytesInput,
    ) -> Result<ExitKind, Error> {
        let context = submission_context();
        let target_bytes = input.target_bytes();
        let submitted = if context.is_canonical() {
            target_bytes.as_slice().to_vec()
        } else {
            normalize_rapid_input_v1(target_bytes.as_slice())
        };
        let corpus_id = state.current_corpus_id()?;
        let sender = self.sender.as_ref().ok_or_else(Error::shutting_down)?;
        self.credits.submission_enqueued();
        if sender
            .send(RetirementSubmission {
                input: BytesInput::new(submitted),
                corpus_id,
                from_supply: context.is_supply(),
            })
            .is_err()
        {
            self.credits.submission_dequeued();
            return Err(Error::shutting_down());
        }
        self.total_submitted = self.total_submitted.saturating_add(1);
        if context.is_supply() {
            let stats = self
                .supply_stats
                .as_ref()
                .expect("supply submission requires supply statistics");
            stats.record_submitted();
        }
        Ok(ExitKind::Ok)
    }
}

impl<I, OT, S> HasObservers for RetirementSubmitExecutor<I, OT, S>
where
    OT: ObserversTuple<I, S>,
{
    type Observers = OT;

    fn observers(&self) -> RefIndexable<&Self::Observers, Self::Observers> {
        RefIndexable::from(&self.observers)
    }

    fn observers_mut(&mut self) -> RefIndexable<&mut Self::Observers, Self::Observers> {
        RefIndexable::from(&mut self.observers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg_pack_v1::{
        default_seed_rapid_input_v1, init_arg_pack_manifest, test_normalize_calls_v1,
    };
    use libafl::observers::CanTrack;

    const TEST_MANIFEST: &str = r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "submit_tracking_kernel",
      "display_name": "submit_tracking_kernel",
      "args": [
        {
          "index": 0,
          "name": "input",
          "type": "uint8_t *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "pointee_layout": {
            "index": "input.*",
            "name": "$pointee",
            "type": "uint8_t",
            "kind": "scalar",
            "size_bytes": 1,
            "align_bytes": 1
          },
          "size_bytes": 8,
          "align_bytes": 8
        }
      ],
      "constraints": []
    }
  ]
}
"#;

    fn init_test_manifest() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let manifest_path = std::env::temp_dir().join(format!(
            "rapid-gpu-executor-test-{}-{nanos}.json",
            std::process::id(),
        ));
        std::fs::write(&manifest_path, TEST_MANIFEST).unwrap();
        let result = init_arg_pack_manifest(&manifest_path);
        let _ = std::fs::remove_file(manifest_path);
        if let Err(err) = result {
            assert_eq!(err, "manifest-driven arg-pack spec already initialized");
        }
    }

    #[test]
    fn retirement_observer_swaps_owned_backing_without_copying() {
        let old_map = vec![0x11; crate::cuda_backend::EDGES_MAP_SIZE];
        let old_ptr = old_map.as_ptr();
        let mut observer = StdMapObserver::owned("edges", old_map).track_indices();
        let new_map = vec![0x22; crate::cuda_backend::EDGES_MAP_SIZE];
        let new_ptr = new_map.as_ptr();

        let returned = replace_observer_coverage_map(&mut observer, new_map);

        assert_eq!(returned.as_ptr(), old_ptr);
        assert_eq!(observer.as_ref().map().as_slice().as_ptr(), new_ptr);
        assert_eq!(observer.as_ref().map().as_slice()[0], 0x22);
    }

    #[test]
    fn async_pending_submit_accept_is_tracked_without_timeout_start() {
        init_test_manifest();
        let input = BytesInput::new(vec![0x41; 40]);
        let expected = normalize_rapid_input_v1(input.target_bytes().as_slice());
        let before = test_normalize_calls_v1();
        let pending = submit_bytes_input_and_build_pending_task(
            &input,
            None,
            SubmissionContext::default(),
            |_| Ok(NonZeroU64::new(7).unwrap()),
        )
        .expect("submit should return a tracked task");

        assert_eq!(test_normalize_calls_v1() - before, 1);
        assert_eq!(pending.task_id, 7);
        assert_eq!(pending.input.target_bytes().as_slice(), expected.as_slice());
        assert_eq!(pending.corpus_id, None);
        assert!(!pending.from_supply);
    }

    #[test]
    fn canonical_submission_bypasses_legacy_normalization() {
        init_test_manifest();
        let canonical = BytesInput::new(default_seed_rapid_input_v1());
        let mut submitted = Vec::new();
        let before = test_normalize_calls_v1();

        let pending = submit_bytes_input_and_build_pending_task(
            &canonical,
            Some(CorpusId::from(9usize)),
            SubmissionContext::Canonical,
            |bytes| {
                submitted.extend_from_slice(bytes);
                Ok(NonZeroU64::new(8).unwrap())
            },
        )
        .unwrap();

        assert_eq!(test_normalize_calls_v1(), before);
        assert_eq!(submitted, canonical.target_bytes().as_slice());
        assert_eq!(pending.input.target_bytes().as_slice(), submitted);
        assert_eq!(pending.corpus_id, Some(CorpusId::from(9usize)));
        assert!(!pending.from_supply);
    }

    #[test]
    fn async_pending_submit_failure_is_explicit() {
        init_test_manifest();
        let input = BytesInput::new(vec![0x41]);

        let error = submit_bytes_input_and_build_pending_task(
            &input,
            None,
            SubmissionContext::default(),
            |_| Err(AsyncSubmitError),
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "RAPID2 backend rejected submission with task_id=0"
        );
    }

    #[test]
    fn pending_task_store_removes_fifo_completion_from_the_head() {
        let mut pending = PendingTaskStore::new();
        pending
            .insert(PendingTask {
                task_id: 10,
                input: "ten",
                corpus_id: None,
                from_supply: false,
            })
            .unwrap();
        pending
            .insert(PendingTask {
                task_id: 11,
                input: "eleven",
                corpus_id: None,
                from_supply: false,
            })
            .unwrap();

        assert_eq!(pending.remove(10).unwrap().input, "ten");
        assert_eq!(pending.remove(11).unwrap().input, "eleven");
        assert_eq!(pending.len(), 0);
    }

    #[test]
    #[should_panic(expected = "RAPID2 completion order violation")]
    fn pending_task_store_rejects_non_head_completion() {
        let mut pending = PendingTaskStore::new();
        pending
            .insert(PendingTask {
                task_id: 10,
                input: "ten",
                corpus_id: None,
                from_supply: false,
            })
            .unwrap();
        pending
            .insert(PendingTask {
                task_id: 11,
                input: "eleven",
                corpus_id: None,
                from_supply: false,
            })
            .unwrap();

        let _ = pending.remove(11);
    }

    #[test]
    fn pending_task_store_rejects_duplicate_task_id() {
        let mut pending = PendingTaskStore::new();
        pending
            .insert(PendingTask {
                task_id: 7,
                input: "first",
                corpus_id: None,
                from_supply: false,
            })
            .unwrap();
        let error = pending
            .insert(PendingTask {
                task_id: 7,
                input: "second",
                corpus_id: None,
                from_supply: false,
            })
            .unwrap_err();

        assert!(error.contains("duplicate pending task ID 7"));
    }
}
