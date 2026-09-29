use std::{
    fmt,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender},
        Arc, RwLock,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use libafl::{
    corpus::{Corpus, CorpusId, HasCurrentCorpusId},
    fuzzer::Evaluator,
    inputs::{BytesInput, HasTargetBytes},
    mutators::{MutationResult, Mutator},
    stages::{Restartable, Stage},
    state::{HasCorpus, HasRand},
    Error,
};
use libafl_bolts::{
    current_nanos,
    rands::{Rand, StdRand},
    AsSlice, Named,
};

use crate::{
    async_mutational_stage::DEFAULT_MUTATIONAL_MAX_ITERATIONS,
    mutators::RapidInputMutator,
    submission_context::{with_submission_context, SubmissionContext},
};

const EMPTY_SNAPSHOT_POLL_INTERVAL: Duration = Duration::from_millis(10);

struct SnapshotEntry {
    parent_id: CorpusId,
    bytes: Arc<[u8]>,
}

#[derive(Default)]
struct CorpusSnapshot {
    generation: u64,
    entries: Vec<SnapshotEntry>,
}

struct PreparedInput {
    parent_id: CorpusId,
    input: BytesInput,
    snapshot_generation: u64,
}

struct WorkerState {
    rand: StdRand,
}

impl WorkerState {
    fn with_seed(seed: u64) -> Self {
        Self {
            rand: StdRand::with_seed(seed),
        }
    }
}

impl HasRand for WorkerState {
    type Rand = StdRand;

    fn rand(&self) -> &Self::Rand {
        &self.rand
    }

    fn rand_mut(&mut self) -> &mut Self::Rand {
        &mut self.rand
    }
}

fn random_batch_size(state: &mut WorkerState) -> usize {
    let limit = NonZeroUsize::new(DEFAULT_MUTATIONAL_MAX_ITERATIONS).unwrap();
    1 + state.rand_mut().below(limit)
}

fn produce_batch(
    entry: &SnapshotEntry,
    snapshot_generation: u64,
    batch_size: usize,
    state: &mut WorkerState,
    mutator: &mut RapidInputMutator,
) -> Result<Vec<PreparedInput>, Error> {
    let mut batch = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let mut input = BytesInput::new(entry.bytes.as_ref().to_vec());
        if mutator.mutate(state, &mut input)? == MutationResult::Skipped {
            continue;
        }
        batch.push(PreparedInput {
            parent_id: entry.parent_id,
            input,
            snapshot_generation,
        });
    }
    Ok(batch)
}

#[derive(Debug, Default)]
pub struct SupplyStats {
    produced: AtomicU64,
    submitted: AtomicU64,
    completed: AtomicU64,
}

impl SupplyStats {
    pub(crate) fn record_produced(&self) {
        self.produced.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_submitted(&self) {
        self.submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_completed(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    fn print_rates(&self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64().max(f64::EPSILON);
        let produced = self.produced.load(Ordering::Relaxed);
        let submitted = self.submitted.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        println!(
            "Supply statistics: produced/sec={:.2}, submitted/sec={:.2}, completed/sec={:.2} (produced={produced}, submitted={submitted}, completed={completed})",
            produced as f64 / seconds,
            submitted as f64 / seconds,
            completed as f64 / seconds,
        );
    }
}

pub struct ProducerPool {
    snapshot: Arc<RwLock<Arc<CorpusSnapshot>>>,
    receiver: Option<Receiver<PreparedInput>>,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
    stats: Arc<SupplyStats>,
    started: Instant,
    published_corpus_count: Option<usize>,
    generation: u64,
    stats_reported: bool,
}

impl fmt::Debug for ProducerPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProducerPool")
            .field("workers", &self.workers.len())
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl ProducerPool {
    pub fn new(thread_count: NonZeroUsize) -> Self {
        let queue_capacity = thread_count
            .get()
            .saturating_mul(DEFAULT_MUTATIONAL_MAX_ITERATIONS);
        let (sender, receiver) = sync_channel(queue_capacity);
        let snapshot = Arc::new(RwLock::new(Arc::new(CorpusSnapshot::default())));
        let shutdown = Arc::new(AtomicBool::new(false));
        let cursor = Arc::new(AtomicU64::new(0));
        let stats = Arc::new(SupplyStats::default());
        let mut workers = Vec::with_capacity(thread_count.get());

        for worker_id in 0..thread_count.get() {
            let worker_snapshot = Arc::clone(&snapshot);
            let worker_shutdown = Arc::clone(&shutdown);
            let worker_cursor = Arc::clone(&cursor);
            let worker_stats = Arc::clone(&stats);
            let worker_sender = sender.clone();
            let name = format!("rapid-supply-{worker_id}");
            let handle = thread::Builder::new()
                .name(name)
                .spawn(move || {
                    producer_loop(
                        worker_id,
                        worker_snapshot,
                        worker_shutdown,
                        worker_cursor,
                        worker_stats,
                        worker_sender,
                    );
                })
                .expect("failed to spawn RAPID supply producer");
            workers.push(handle);
        }
        drop(sender);

        Self {
            snapshot,
            receiver: Some(receiver),
            shutdown,
            workers,
            stats,
            started: Instant::now(),
            published_corpus_count: None,
            generation: 0,
            stats_reported: false,
        }
    }

    pub fn stats(&self) -> Arc<SupplyStats> {
        Arc::clone(&self.stats)
    }

    fn refresh_snapshot<C>(&mut self, corpus: &C) -> Result<(), Error>
    where
        C: Corpus<BytesInput>,
    {
        let count = corpus.count();
        if self.published_corpus_count == Some(count) {
            return Ok(());
        }

        let mut entries = Vec::with_capacity(count);
        for parent_id in corpus.ids() {
            let input = corpus.cloned_input_for_id(parent_id)?;
            entries.push(SnapshotEntry {
                parent_id,
                bytes: Arc::from(input.target_bytes().as_slice()),
            });
        }

        self.generation = self.generation.saturating_add(1);
        let snapshot = Arc::new(CorpusSnapshot {
            generation: self.generation,
            entries,
        });
        *self.snapshot.write().unwrap() = snapshot;
        self.published_corpus_count = Some(count);
        for worker in &self.workers {
            worker.thread().unpark();
        }
        Ok(())
    }

    fn recv_timeout(&self, timeout: Duration) -> Result<PreparedInput, RecvTimeoutError> {
        let receiver = self
            .receiver
            .as_ref()
            .ok_or(RecvTimeoutError::Disconnected)?;
        let deadline = Instant::now() + timeout;
        loop {
            let prepared =
                receiver.recv_timeout(deadline.saturating_duration_since(Instant::now()))?;
            if prepared.snapshot_generation == self.generation {
                return Ok(prepared);
            }
        }
    }

    fn shutdown(&mut self) {
        self.receiver.take();
        self.shutdown.store(true, Ordering::Relaxed);
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                eprintln!("RAPID supply producer panicked during shutdown");
            }
        }
        if !self.stats_reported {
            self.stats.print_rates(self.started.elapsed());
            self.stats_reported = true;
        }
    }
}

impl Drop for ProducerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn producer_loop(
    worker_id: usize,
    snapshot: Arc<RwLock<Arc<CorpusSnapshot>>>,
    shutdown: Arc<AtomicBool>,
    cursor: Arc<AtomicU64>,
    stats: Arc<SupplyStats>,
    sender: SyncSender<PreparedInput>,
) {
    let seed = current_nanos() ^ (worker_id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut state = WorkerState::with_seed(seed);
    let mut mutator = RapidInputMutator::default();

    while !shutdown.load(Ordering::Relaxed) {
        // Clone the published Arc under the lock; this generation stays
        // immutable for the whole batch even if the owner publishes a newer one.
        let current = Arc::clone(&snapshot.read().unwrap());
        if current.entries.is_empty() {
            thread::park_timeout(EMPTY_SNAPSHOT_POLL_INTERVAL);
            continue;
        }

        let batch_size = random_batch_size(&mut state);
        let batch_start = cursor.fetch_add(batch_size as u64, Ordering::Relaxed);
        let entry_index = (batch_start % current.entries.len() as u64) as usize;
        let entry = &current.entries[entry_index];
        let batch = match produce_batch(
            entry,
            current.generation,
            batch_size,
            &mut state,
            &mut mutator,
        ) {
            Ok(batch) => batch,
            Err(error) => {
                log::error!("RAPID supply producer mutation failed: {error}");
                continue;
            }
        };

        for prepared in batch {
            if shutdown.load(Ordering::Relaxed) || sender.send(prepared).is_err() {
                return;
            }
            stats.record_produced();
        }
    }
}

#[derive(Debug)]
pub struct SupplyStage<Inner> {
    inner: Inner,
    pool: Option<ProducerPool>,
    max_iterations: NonZeroUsize,
}

impl<Inner> SupplyStage<Inner> {
    pub fn new(inner: Inner, pool: Option<ProducerPool>, max_iterations: NonZeroUsize) -> Self {
        Self {
            inner,
            pool,
            max_iterations,
        }
    }

    pub fn shutdown(&mut self) {
        if let Some(pool) = self.pool.as_mut() {
            pool.shutdown();
        }
    }
}

impl<Inner> Named for SupplyStage<Inner>
where
    Inner: Named,
{
    fn name(&self) -> &std::borrow::Cow<'static, str> {
        self.inner.name()
    }
}

impl<Inner, S> Restartable<S> for SupplyStage<Inner>
where
    Inner: Restartable<S>,
{
    fn should_restart(&mut self, state: &mut S) -> Result<bool, Error> {
        self.inner.should_restart(state)
    }

    fn clear_progress(&mut self, state: &mut S) -> Result<(), Error> {
        self.inner.clear_progress(state)
    }
}

impl<E, EM, Inner, S, Z> Stage<E, EM, S, Z> for SupplyStage<Inner>
where
    Inner: Stage<E, EM, S, Z>,
    S: HasCorpus<BytesInput> + HasCurrentCorpusId,
    Z: Evaluator<E, EM, BytesInput, S>,
{
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), Error> {
        let Some(pool) = self.pool.as_mut() else {
            return self.inner.perform(fuzzer, executor, state, manager);
        };

        pool.refresh_snapshot(state.corpus())?;
        let Ok(first) = pool.recv_timeout(EMPTY_SNAPSHOT_POLL_INTERVAL) else {
            return self.inner.perform(fuzzer, executor, state, manager);
        };

        let saved_parent = state.current_corpus_id()?;
        let mut prepared = Some(first);
        for _ in 0..self.max_iterations.get() {
            let next = match prepared.take() {
                Some(prepared) => prepared,
                None => match pool.recv_timeout(EMPTY_SNAPSHOT_POLL_INTERVAL) {
                    Ok(prepared) => prepared,
                    Err(_) => break,
                },
            };
            // Carry the snapshot parent through current_corpus_id into the
            // PendingTask; retirement restores it around scheduler/feedback work.
            state.set_corpus_id(next.parent_id)?;
            let evaluation = with_submission_context(SubmissionContext::Supply, || {
                fuzzer.evaluate_filtered(state, executor, manager, &next.input)
            });
            let restore = match saved_parent {
                Some(parent_id) => state.set_corpus_id(parent_id),
                None => state.clear_corpus_id(),
            };
            restore?;
            evaluation?;
            pool.refresh_snapshot(state.corpus())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use libafl::{
        corpus::{Corpus, CorpusId, InMemoryCorpus},
        inputs::{BytesInput, HasTargetBytes},
    };
    use libafl_bolts::AsSlice;

    use super::{produce_batch, random_batch_size, SnapshotEntry, WorkerState};
    use crate::{
        arg_pack_v1::{
            default_seed_rapid_input_v1, init_arg_pack_manifest, normalize_rapid_input_v1,
            parse_rapid_input_envelope, test_normalize_calls_v1,
        },
        mutators::RapidInputMutator,
    };

    const TEST_MANIFEST: &str = r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "supply_pool_kernel",
      "display_name": "supply_pool_kernel",
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
        let path = std::env::temp_dir().join(format!(
            "rapid-supply-pool-test-{}-{nanos}.json",
            std::process::id()
        ));
        std::fs::write(&path, TEST_MANIFEST).unwrap();
        let result = init_arg_pack_manifest(&path);
        let _ = std::fs::remove_file(path);
        if let Err(error) = result {
            assert_eq!(error, "manifest-driven arg-pack spec already initialized");
        }
    }

    #[test]
    fn producer_pool_outputs_canonical_input_with_snapshot_parent() {
        init_test_manifest();
        let mut corpus = InMemoryCorpus::<BytesInput>::new();
        let parent_id = corpus
            .add(BytesInput::new(default_seed_rapid_input_v1()).into())
            .unwrap();
        let mut pool = super::ProducerPool::new(std::num::NonZeroUsize::new(1).unwrap());
        pool.refresh_snapshot(&corpus).unwrap();

        let prepared = pool
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("producer did not supply input");
        pool.shutdown();

        let bytes = prepared.input.target_bytes();
        let bytes = bytes.as_slice();
        assert_eq!(prepared.parent_id, parent_id);
        assert_eq!(prepared.snapshot_generation, 1);
        assert!(parse_rapid_input_envelope(bytes).is_some());
        assert_eq!(bytes, normalize_rapid_input_v1(bytes));
    }

    #[test]
    fn producer_batch_never_exceeds_requested_stage_budget() {
        init_test_manifest();
        let entry = SnapshotEntry {
            parent_id: CorpusId::from(0usize),
            bytes: Arc::from(default_seed_rapid_input_v1()),
        };
        let mut state = WorkerState::with_seed(7);
        let mut mutator = RapidInputMutator::default();

        for _ in 0..256 {
            let batch_size = random_batch_size(&mut state);
            let batch = produce_batch(&entry, 1, batch_size, &mut state, &mut mutator).unwrap();
            assert!((1..=super::DEFAULT_MUTATIONAL_MAX_ITERATIONS).contains(&batch_size));
            assert!(batch.len() <= batch_size);
        }
    }

    #[test]
    fn producer_normalizes_each_prepared_input_once() {
        init_test_manifest();
        let entry = SnapshotEntry {
            parent_id: CorpusId::from(0usize),
            bytes: Arc::from(default_seed_rapid_input_v1()),
        };
        let mut state = WorkerState::with_seed(7);
        let mut mutator = RapidInputMutator::default();
        let before = test_normalize_calls_v1();

        let batch = produce_batch(&entry, 1, 1, &mut state, &mut mutator).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(test_normalize_calls_v1() - before, 1);
    }
}
