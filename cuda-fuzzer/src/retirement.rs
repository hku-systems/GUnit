use std::{
    collections::BTreeMap,
    mem::size_of,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc, Mutex,
    },
    time::Instant,
};

use crate::{
    cuda_backend::{AsyncCudaBackend, LibAflRunStatus, TaskResult},
    gpu_executor::PendingTask,
};

pub const DEFAULT_RETIRE_BATCH: usize = 8;
pub const MAX_RETIREMENT_IN_FLIGHT: usize = 64;

#[derive(Debug)]
pub struct AcquiredCompletion<I> {
    pub sequence: u64,
    pub result: TaskResult,
    pub pending: PendingTask<I>,
}

impl<I> AcquiredCompletion<I> {
    pub fn new(result: TaskResult, pending: PendingTask<I>) -> Self {
        Self {
            sequence: result.task_id,
            result,
            pending,
        }
    }

    pub fn with_sequence(sequence: u64, result: TaskResult, pending: PendingTask<I>) -> Self {
        Self {
            sequence,
            result,
            pending,
        }
    }
}

// Polling acquires the backend slot backing TaskResult's raw pointers. They
// remain valid across this handoff until the retirement worker releases it.
unsafe impl<I: Send> Send for AcquiredCompletion<I> {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseWord {
    pub index: usize,
    pub value: u64,
}

#[derive(Debug)]
pub struct PreparedCompletion<I> {
    pub sequence: u64,
    pub task_id: u64,
    pub pending: PendingTask<I>,
    pub edge_map: Vec<u8>,
    pub simt_memcov_map: Vec<u8>,
    pub edge_words: Vec<SparseWord>,
    pub simt_words: Vec<SparseWord>,
    pub status: LibAflRunStatus,
    pub exec_time_ns: u64,
}

pub fn sparse_words(map: &[u8]) -> Vec<SparseWord> {
    let mut sparse = Vec::new();
    let words = map.len() / size_of::<u64>();
    for index in 0..words {
        let offset = index * size_of::<u64>();
        // Safety: the read is unaligned and contained in the source slice.
        let value = unsafe { map.as_ptr().add(offset).cast::<u64>().read_unaligned() };
        if value != 0 {
            sparse.push(SparseWord { index, value });
        }
    }
    if map.len() % size_of::<u64>() != 0 {
        let index = words;
        let mut value = 0_u64;
        for (shift, byte) in map[words * size_of::<u64>()..].iter().enumerate() {
            value |= u64::from(*byte) << (shift * 8);
        }
        if value != 0 {
            sparse.push(SparseWord { index, value });
        }
    }
    sparse
}

pub fn sparse_has_novel_bits(words: &[SparseWord], seen: &[u8]) -> bool {
    words.iter().any(|word| {
        let offset = word.index * size_of::<u64>();
        if offset >= seen.len() {
            return true;
        }
        let available = (seen.len() - offset).min(size_of::<u64>());
        let mut seen_word = 0_u64;
        for (shift, byte) in seen[offset..offset + available].iter().enumerate() {
            seen_word |= u64::from(*byte) << (shift * 8);
        }
        word.value & !seen_word != 0
    })
}

pub trait CompletedTaskReleaser: Send + Sync {
    fn release_completed_tasks(&self, task_ids: &[u64]) -> usize;
}

impl CompletedTaskReleaser for AsyncCudaBackend {
    fn release_completed_tasks(&self, task_ids: &[u64]) -> usize {
        AsyncCudaBackend::release_completed_tasks(self, task_ids)
    }
}

fn copy_completion<I>(completion: AcquiredCompletion<I>) -> Result<PreparedCompletion<I>, String> {
    // TaskResult points into task-owned buffers: copy both maps completely
    // before release makes that backend slot reusable.
    let edge_map = unsafe { completion.result.edge_slice() }
        .map_err(|error| error.to_string())?
        .to_vec();
    let simt_memcov_map = unsafe { completion.result.simt_memcov_slice() }
        .map_err(|error| error.to_string())?
        .to_vec();
    let edge_words = sparse_words(&edge_map);
    let simt_words = sparse_words(&simt_memcov_map);
    Ok(PreparedCompletion {
        sequence: completion.sequence,
        task_id: completion.result.task_id,
        pending: completion.pending,
        edge_map,
        simt_memcov_map,
        edge_words,
        simt_words,
        status: completion.result.status,
        exec_time_ns: completion.result.exec_time_ns,
    })
}

pub fn materialize_completions<I, R>(
    completions: Vec<AcquiredCompletion<I>>,
    releaser: &R,
    credits: &RetirementCredits,
) -> Vec<Result<PreparedCompletion<I>, String>>
where
    R: CompletedTaskReleaser,
{
    if completions.is_empty() {
        return Vec::new();
    }

    let task_ids = completions
        .iter()
        .map(|completion| completion.result.task_id)
        .collect::<Vec<_>>();
    let mut copied = completions
        .into_iter()
        .map(copy_completion)
        .collect::<Vec<_>>();

    // Release the whole acquired batch only after every raw map pointer has
    // either been copied or rejected by validation.
    let released = releaser.release_completed_tasks(&task_ids);
    credits.release(released);
    if released != task_ids.len() {
        copied[0] = Err(format!(
            "released {released} of {} completed tasks",
            task_ids.len()
        ));
    }
    copied
}

pub fn copy_coverage_map(source: &[u8], destination: &mut [u8]) -> Result<(), String> {
    if source.len() != destination.len() {
        return Err(format!(
            "coverage length mismatch: source={} destination={}",
            source.len(),
            destination.len()
        ));
    }
    destination.copy_from_slice(source);
    Ok(())
}

#[inline]
pub fn merge_coverage_map(source: &[u8], destination: &mut [u8]) -> Result<u64, String> {
    if source.len() != destination.len() {
        return Err(format!(
            "coverage length mismatch: source={} destination={}",
            source.len(),
            destination.len()
        ));
    }

    let words = source.len() / size_of::<u64>();
    let mut added = 0_u64;
    for index in 0..words {
        let offset = index * size_of::<u64>();
        // Safety: u64 is POD, each access stays within the validated slices,
        // and unaligned operations do not require either map to be aligned.
        unsafe {
            let source_word = source.as_ptr().add(offset).cast::<u64>().read_unaligned();
            let destination_ptr = destination.as_mut_ptr().add(offset).cast::<u64>();
            let destination_word = destination_ptr.read_unaligned();
            added += (source_word & !destination_word).count_ones() as u64;
            destination_ptr.write_unaligned(destination_word | source_word);
        }
    }
    for (source_byte, destination_byte) in source[words * size_of::<u64>()..]
        .iter()
        .zip(&mut destination[words * size_of::<u64>()..])
    {
        added += (*source_byte & !*destination_byte).count_ones() as u64;
        *destination_byte |= *source_byte;
    }
    Ok(added)
}

#[inline]
pub fn or_coverage_map(source: &[u8], destination: &mut [u8]) -> Result<(), String> {
    if source.len() != destination.len() {
        return Err(format!(
            "coverage length mismatch: source={} destination={}",
            source.len(),
            destination.len()
        ));
    }

    let words = source.len() / size_of::<u64>();
    for index in 0..words {
        let offset = index * size_of::<u64>();
        // Safety: u64 is POD, each access stays within the validated slices,
        // and unaligned operations do not require either map to be aligned.
        unsafe {
            let source_word = source.as_ptr().add(offset).cast::<u64>().read_unaligned();
            let destination_ptr = destination.as_mut_ptr().add(offset).cast::<u64>();
            let destination_word = destination_ptr.read_unaligned();
            destination_ptr.write_unaligned(destination_word | source_word);
        }
    }
    for (source_byte, destination_byte) in source[words * size_of::<u64>()..]
        .iter()
        .zip(&mut destination[words * size_of::<u64>()..])
    {
        *destination_byte |= *source_byte;
    }
    Ok(())
}

#[inline]
pub fn has_novel_bits(current: &[u8], seen: &[u8]) -> Result<bool, String> {
    if current.len() != seen.len() {
        return Err(format!(
            "coverage length mismatch: current={} seen={}",
            current.len(),
            seen.len()
        ));
    }

    let words = current.len() / size_of::<u64>();
    for index in 0..words {
        let offset = index * size_of::<u64>();
        // Safety: u64 is POD and unaligned reads stay within both slices.
        let (current_word, seen_word) = unsafe {
            (
                current.as_ptr().add(offset).cast::<u64>().read_unaligned(),
                seen.as_ptr().add(offset).cast::<u64>().read_unaligned(),
            )
        };
        if current_word & !seen_word != 0 {
            return Ok(true);
        }
    }
    Ok(current[words * size_of::<u64>()..]
        .iter()
        .zip(&seen[words * size_of::<u64>()..])
        .any(|(current_byte, seen_byte)| current_byte & !seen_byte != 0))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetirementStats {
    pub processed: u64,
    pub batches: u64,
    pub min_batch: usize,
    pub max_batch: usize,
    pub max_queue_depth: usize,
    pub max_in_flight: u64,
    pub elapsed_ns: u64,
}

impl RetirementStats {
    pub fn record_batch(&mut self, size: usize) {
        self.processed = self.processed.saturating_add(size as u64);
        self.batches = self.batches.saturating_add(1);
        self.min_batch = if self.min_batch == 0 {
            size
        } else {
            self.min_batch.min(size)
        };
        self.max_batch = self.max_batch.max(size);
    }

    pub fn finish(&mut self, started: Instant, max_queue_depth: usize) {
        self.max_queue_depth = max_queue_depth;
        self.elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    }

    pub fn processed_per_second(self) -> f64 {
        if self.elapsed_ns == 0 {
            0.0
        } else {
            self.processed as f64 * 1_000_000_000.0 / self.elapsed_ns as f64
        }
    }

    pub fn average_batch(self) -> f64 {
        if self.batches == 0 {
            0.0
        } else {
            self.processed as f64 / self.batches as f64
        }
    }

    pub fn backlog_is_full(&self, submitted: u64, capacity: NonZeroUsize) -> bool {
        submitted.saturating_sub(self.processed) >= capacity.get() as u64
    }

    pub fn observe_in_flight(&mut self, submitted: u64) {
        self.max_in_flight = self
            .max_in_flight
            .max(submitted.saturating_sub(self.processed));
    }
}

#[derive(Default)]
struct QueueDepth {
    current: AtomicUsize,
    maximum: AtomicUsize,
}

pub struct RetirementHandoff<T> {
    sender: SyncSender<T>,
    queue_depth: Arc<QueueDepth>,
}

pub struct RetirementInbox<T> {
    receiver: Arc<Mutex<Receiver<T>>>,
    queue_depth: Arc<QueueDepth>,
}

pub fn retirement_channel<T>(capacity: NonZeroUsize) -> (RetirementHandoff<T>, RetirementInbox<T>) {
    let (sender, receiver) = mpsc::sync_channel(capacity.get());
    let queue_depth = Arc::new(QueueDepth::default());
    (
        RetirementHandoff {
            sender,
            queue_depth: Arc::clone(&queue_depth),
        },
        RetirementInbox {
            receiver: Arc::new(Mutex::new(receiver)),
            queue_depth,
        },
    )
}

impl<T> Clone for RetirementHandoff<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            queue_depth: Arc::clone(&self.queue_depth),
        }
    }
}

impl<T> Clone for RetirementInbox<T> {
    fn clone(&self) -> Self {
        Self {
            receiver: Arc::clone(&self.receiver),
            queue_depth: Arc::clone(&self.queue_depth),
        }
    }
}

impl<T> RetirementHandoff<T> {
    pub fn handoff(&self, job: T) -> Result<(), mpsc::SendError<T>> {
        self.queue_depth.enqueued();
        if let Err(error) = self.sender.send(job) {
            self.queue_depth.dequeued(1);
            return Err(error);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct RetirementCredits {
    limit: usize,
    in_flight: AtomicUsize,
    maximum: AtomicUsize,
    queued_submissions: AtomicUsize,
}

impl RetirementCredits {
    pub fn new(limit: NonZeroUsize) -> Self {
        Self {
            limit: limit.get(),
            in_flight: AtomicUsize::new(0),
            maximum: AtomicUsize::new(0),
            queued_submissions: AtomicUsize::new(0),
        }
    }

    pub fn try_acquire(&self) -> bool {
        let mut current = self.in_flight.load(Ordering::Acquire);
        loop {
            if current >= self.limit {
                return false;
            }
            match self.in_flight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.maximum.fetch_max(current + 1, Ordering::Relaxed);
                    return true;
                }
                Err(actual) => current = actual,
            }
        }
    }

    pub fn release(&self, count: usize) {
        let previous = self.in_flight.fetch_sub(count, Ordering::AcqRel);
        assert!(previous >= count, "retirement credit underflow");
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Acquire)
    }

    pub fn maximum(&self) -> usize {
        self.maximum.load(Ordering::Relaxed)
    }

    pub fn submission_enqueued(&self) {
        self.queued_submissions.fetch_add(1, Ordering::Release);
    }

    pub fn submission_dequeued(&self) {
        let previous = self.queued_submissions.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "retirement submission queue underflow");
    }

    pub fn queued_submissions(&self) -> usize {
        self.queued_submissions.load(Ordering::Acquire)
    }
}

pub fn discard_queued_submissions<T>(receiver: &Receiver<T>, credits: &RetirementCredits) {
    while receiver.recv().is_ok() {
        credits.submission_dequeued();
    }
}

impl<T> RetirementInbox<T> {
    pub fn recv_full_batch(&self, batch_size: NonZeroUsize) -> Option<Vec<T>> {
        let receiver = self.receiver.lock().ok()?;
        let mut batch = Vec::with_capacity(batch_size.get());
        batch.push(receiver.recv().ok()?);
        while batch.len() < batch_size.get() {
            match receiver.recv() {
                Ok(job) => batch.push(job),
                Err(_) => break,
            }
        }
        self.queue_depth.dequeued(batch.len());
        Some(batch)
    }

    pub fn recv_batch(&self, batch_size: NonZeroUsize) -> Option<Vec<T>> {
        let receiver = self.receiver.lock().ok()?;
        let first = receiver.recv().ok()?;
        let batch = Self::collect_available(&receiver, first, batch_size);
        self.queue_depth.dequeued(batch.len());
        Some(batch)
    }

    pub fn try_recv_batch(&self, batch_size: NonZeroUsize) -> Option<Vec<T>> {
        let receiver = self.receiver.lock().ok()?;
        let first = receiver.try_recv().ok()?;
        let batch = Self::collect_available(&receiver, first, batch_size);
        self.queue_depth.dequeued(batch.len());
        Some(batch)
    }

    fn collect_available(receiver: &Receiver<T>, first: T, batch_size: NonZeroUsize) -> Vec<T> {
        let mut batch = Vec::with_capacity(batch_size.get());
        batch.push(first);
        while batch.len() < batch_size.get() {
            match receiver.try_recv() {
                Ok(job) => batch.push(job),
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => break,
            }
        }
        batch
    }

    pub fn max_queue_depth(&self) -> usize {
        self.queue_depth.maximum.load(Ordering::Relaxed)
    }
}

pub struct OrderedPreparedInbox<I> {
    inbox: RetirementInbox<Result<PreparedCompletion<I>, String>>,
    waiting: BTreeMap<u64, PreparedCompletion<I>>,
    next_sequence: u64,
}

impl<I> OrderedPreparedInbox<I> {
    pub fn new(
        inbox: RetirementInbox<Result<PreparedCompletion<I>, String>>,
        first_sequence: u64,
    ) -> Self {
        Self {
            inbox,
            waiting: BTreeMap::new(),
            next_sequence: first_sequence,
        }
    }

    pub fn recv_batch(
        &mut self,
        batch_size: NonZeroUsize,
    ) -> Result<Option<Vec<PreparedCompletion<I>>>, String> {
        while !self.waiting.contains_key(&self.next_sequence) {
            let Some(messages) = self.inbox.recv_batch(batch_size) else {
                if self.waiting.is_empty() {
                    return Ok(None);
                }
                return Err(format!(
                    "materializer output closed before sequence {}",
                    self.next_sequence
                ));
            };
            self.store(messages)?;
        }
        if let Some(messages) = self.inbox.try_recv_batch(batch_size) {
            self.store(messages)?;
        }
        Ok(Some(self.take_ready(batch_size)))
    }

    pub fn try_recv_batch(
        &mut self,
        batch_size: NonZeroUsize,
    ) -> Result<Option<Vec<PreparedCompletion<I>>>, String> {
        if let Some(messages) = self.inbox.try_recv_batch(batch_size) {
            self.store(messages)?;
        }
        if !self.waiting.contains_key(&self.next_sequence) {
            return Ok(None);
        }
        Ok(Some(self.take_ready(batch_size)))
    }

    fn store(
        &mut self,
        messages: Vec<Result<PreparedCompletion<I>, String>>,
    ) -> Result<(), String> {
        for message in messages {
            let prepared = message?;
            let sequence = prepared.sequence;
            if sequence < self.next_sequence || self.waiting.insert(sequence, prepared).is_some() {
                return Err(format!("duplicate prepared sequence {sequence}"));
            }
        }
        Ok(())
    }

    fn take_ready(&mut self, batch_size: NonZeroUsize) -> Vec<PreparedCompletion<I>> {
        let mut ready = Vec::with_capacity(batch_size.get());
        while ready.len() < batch_size.get() {
            let Some(prepared) = self.waiting.remove(&self.next_sequence) else {
                break;
            };
            ready.push(prepared);
            self.next_sequence = self.next_sequence.saturating_add(1);
        }
        ready
    }

    pub fn max_queue_depth(&self) -> usize {
        self.inbox.max_queue_depth()
    }

    pub fn drain_unordered(&mut self, batch_size: NonZeroUsize) {
        self.waiting.clear();
        while self.inbox.recv_batch(batch_size).is_some() {}
    }
}

impl QueueDepth {
    fn enqueued(&self) {
        let depth = self.current.fetch_add(1, Ordering::Relaxed) + 1;
        self.maximum.fetch_max(depth, Ordering::Relaxed);
    }

    fn dequeued(&self, count: usize) {
        let previous = self.current.fetch_sub(count, Ordering::Relaxed);
        debug_assert!(previous >= count);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        num::NonZeroUsize,
        sync::{mpsc, Arc, Barrier, Mutex},
        time::Duration,
        time::Instant,
    };

    use crate::{
        cuda_backend::{LibAflRunStatus, TaskResult, EDGES_MAP_SIZE},
        gpu_executor::PendingTask,
    };

    use super::{
        discard_queued_submissions, has_novel_bits, materialize_completions, merge_coverage_map,
        or_coverage_map, retirement_channel, sparse_has_novel_bits, sparse_words,
        AcquiredCompletion, CompletedTaskReleaser, OrderedPreparedInbox, PreparedCompletion,
        RetirementCredits, RetirementStats,
    };

    #[test]
    fn sparse_words_preserve_novelty_against_current_history() {
        let mut map = vec![0_u8; 19];
        map[1] = 0x80;
        map[8] = 0x04;
        map[18] = 0x20;
        let words = sparse_words(&map);

        assert_eq!(words.len(), 3);
        assert!(sparse_has_novel_bits(&words, &vec![0; map.len()]));
        assert!(!sparse_has_novel_bits(&words, &map));

        let mut partially_seen = map.clone();
        partially_seen[18] = 0;
        assert!(sparse_has_novel_bits(&words, &partially_seen));
    }

    fn prepared(sequence: u64) -> PreparedCompletion<()> {
        PreparedCompletion {
            sequence,
            task_id: sequence + 1,
            pending: PendingTask {
                task_id: sequence + 1,
                input: (),
                corpus_id: None,
                from_supply: false,
            },
            edge_map: Vec::new(),
            simt_memcov_map: Vec::new(),
            edge_words: Vec::new(),
            simt_words: Vec::new(),
            status: LibAflRunStatus::default(),
            exec_time_ns: 0,
        }
    }

    #[test]
    fn prepared_results_are_committed_in_sequence_order() {
        let (handoff, inbox) = retirement_channel(NonZeroUsize::new(4).unwrap());
        handoff.handoff(Ok(prepared(1))).unwrap();
        handoff.handoff(Ok(prepared(0))).unwrap();
        handoff.handoff(Ok(prepared(2))).unwrap();
        drop(handoff);

        let mut ordered = OrderedPreparedInbox::new(inbox, 0);
        let batch = ordered
            .recv_batch(NonZeroUsize::new(4).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            batch.iter().map(|item| item.sequence).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(ordered
            .recv_batch(NonZeroUsize::new(4).unwrap())
            .unwrap()
            .is_none());
    }

    struct OverwriteReleaser {
        maps: Vec<Arc<Mutex<Vec<u8>>>>,
        calls: Arc<Mutex<Vec<Vec<u64>>>>,
        released: usize,
    }

    impl CompletedTaskReleaser for OverwriteReleaser {
        fn release_completed_tasks(&self, task_ids: &[u64]) -> usize {
            self.calls.lock().unwrap().push(task_ids.to_vec());
            for map in &self.maps {
                map.lock().unwrap().fill(0);
            }
            self.released
        }
    }

    #[test]
    fn materialization_batches_release_after_copying_every_map() {
        let edge_1 = Arc::new(Mutex::new(vec![0x11; EDGES_MAP_SIZE]));
        let simt_1 = Arc::new(Mutex::new(vec![
            0x21;
            crate::cuda_backend::SIMT_MEMCOV_STORAGE_SIZE
        ]));
        let edge_2 = Arc::new(Mutex::new(vec![0x12; EDGES_MAP_SIZE]));
        let simt_2 = Arc::new(Mutex::new(vec![
            0x22;
            crate::cuda_backend::SIMT_MEMCOV_STORAGE_SIZE
        ]));
        let result = |task_id, edge: &Arc<Mutex<Vec<u8>>>, simt: &Arc<Mutex<Vec<u8>>>| TaskResult {
            task_id,
            edge_ptr: edge.lock().unwrap().as_ptr(),
            edge_size: EDGES_MAP_SIZE as u32,
            simt_memcov_ptr: simt.lock().unwrap().as_ptr(),
            simt_memcov_size: crate::cuda_backend::SIMT_MEMCOV_STORAGE_SIZE as u32,
            ..TaskResult::default()
        };
        let completion = |sequence, result| {
            AcquiredCompletion::with_sequence(
                sequence,
                result,
                PendingTask {
                    task_id: result.task_id,
                    input: (),
                    corpus_id: None,
                    from_supply: false,
                },
            )
        };
        let credits = RetirementCredits::new(NonZeroUsize::new(2).unwrap());
        assert!(credits.try_acquire());
        assert!(credits.try_acquire());
        let calls = Arc::new(Mutex::new(Vec::new()));

        let prepared = materialize_completions(
            vec![
                completion(0, result(7, &edge_1, &simt_1)),
                completion(1, result(8, &edge_2, &simt_2)),
            ],
            &OverwriteReleaser {
                maps: vec![
                    Arc::clone(&edge_1),
                    Arc::clone(&simt_1),
                    Arc::clone(&edge_2),
                    Arc::clone(&simt_2),
                ],
                calls: Arc::clone(&calls),
                released: 2,
            },
            &credits,
        );

        assert_eq!(*calls.lock().unwrap(), vec![vec![7, 8]]);
        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared[0].as_ref().unwrap().edge_map[0], 0x11);
        assert_eq!(prepared[0].as_ref().unwrap().simt_memcov_map[0], 0x21);
        assert_eq!(prepared[1].as_ref().unwrap().edge_map[0], 0x12);
        assert_eq!(prepared[1].as_ref().unwrap().simt_memcov_map[0], 0x22);
        assert_eq!(credits.in_flight(), 0);
    }

    #[test]
    fn materialization_reports_a_failed_release() {
        let edge = Arc::new(Mutex::new(vec![0; EDGES_MAP_SIZE]));
        let simt = Arc::new(Mutex::new(
            vec![0; crate::cuda_backend::SIMT_MEMCOV_STORAGE_SIZE],
        ));
        let result = TaskResult {
            task_id: 9,
            edge_ptr: edge.lock().unwrap().as_ptr(),
            edge_size: EDGES_MAP_SIZE as u32,
            simt_memcov_ptr: simt.lock().unwrap().as_ptr(),
            simt_memcov_size: crate::cuda_backend::SIMT_MEMCOV_STORAGE_SIZE as u32,
            ..TaskResult::default()
        };
        let credits = RetirementCredits::new(NonZeroUsize::new(1).unwrap());
        assert!(credits.try_acquire());
        let error = materialize_completions(
            vec![AcquiredCompletion::new(
                result,
                PendingTask {
                    task_id: 9,
                    input: (),
                    corpus_id: None,
                    from_supply: false,
                },
            )],
            &OverwriteReleaser {
                maps: vec![edge, simt],
                calls: Arc::new(Mutex::new(Vec::new())),
                released: 0,
            },
            &credits,
        )
        .pop()
        .unwrap()
        .unwrap_err();
        assert_eq!(error, "released 0 of 1 completed tasks");
        assert_eq!(credits.in_flight(), 1);
    }

    #[test]
    fn u64_coverage_merge_matches_bytewise_or() {
        let mut seed = 0x4d59_5df4_d0f3_3173_u64;
        let mut destination = vec![0_u8; EDGES_MAP_SIZE];
        let mut source = vec![0_u8; EDGES_MAP_SIZE];
        for byte in destination.iter_mut().chain(source.iter_mut()) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *byte = seed as u8;
        }
        let expected_added = destination
            .iter()
            .zip(&source)
            .map(|(left, right)| (right & !left).count_ones() as u64)
            .sum::<u64>();
        let expected = destination
            .iter()
            .zip(&source)
            .map(|(left, right)| left | right)
            .collect::<Vec<_>>();

        let added = merge_coverage_map(&source, &mut destination).unwrap();

        assert_eq!(destination, expected);
        assert_eq!(added, expected_added);

        let source_storage = (0_u8..19).collect::<Vec<_>>();
        let mut destination_storage = (19_u8..38).collect::<Vec<_>>();
        let source = &source_storage[1..18];
        let destination = &mut destination_storage[1..18];
        let expected = destination
            .iter()
            .zip(source)
            .map(|(left, right)| left | right)
            .collect::<Vec<_>>();
        assert_eq!(
            has_novel_bits(source, destination).unwrap(),
            source
                .iter()
                .zip(destination.iter())
                .any(|(current, seen)| current & !seen != 0)
        );
        or_coverage_map(source, destination).unwrap();
        assert_eq!(destination, expected);
    }

    #[test]
    fn handoff_does_not_wait_for_retirement_and_shutdown_drains() {
        let gate = Arc::new(Barrier::new(2));
        let processed = Arc::new(Mutex::new(Vec::new()));
        let worker_gate = Arc::clone(&gate);
        let worker_processed = Arc::clone(&processed);
        let credits = Arc::new(super::RetirementCredits::new(NonZeroUsize::new(4).unwrap()));
        let worker_credits = Arc::clone(&credits);
        let (handoff, inbox) = retirement_channel::<u64>(NonZeroUsize::new(4).unwrap());
        let worker = std::thread::spawn(move || {
            let started = Instant::now();
            let mut stats = RetirementStats::default();
            let mut first_batch = true;
            while let Some(batch) = inbox.recv_batch(NonZeroUsize::new(4).unwrap()) {
                if first_batch {
                    worker_gate.wait();
                    first_batch = false;
                }
                stats.record_batch(batch.len());
                worker_credits.release(batch.len());
                worker_processed.lock().unwrap().extend(batch);
            }
            stats.finish(started, inbox.max_queue_depth());
            stats
        });

        assert!(credits.try_acquire());
        handoff.handoff(1).unwrap();
        assert!(credits.try_acquire());
        handoff.handoff(2).unwrap();

        // If handoff waited for retirement, execution could not reach this gate.
        gate.wait();
        drop(handoff);
        let stats = worker.join().unwrap();
        assert_eq!(*processed.lock().unwrap(), vec![1, 2]);
        assert_eq!(stats.processed, 2);
        assert_eq!(credits.in_flight(), 0);
        assert!((1..=2).contains(&stats.max_queue_depth));
    }

    #[test]
    fn accepted_tasks_are_bounded_until_release() {
        let credits = super::RetirementCredits::new(NonZeroUsize::new(3).unwrap());
        assert!(credits.try_acquire());
        assert!(credits.try_acquire());
        assert!(credits.try_acquire());
        assert!(!credits.try_acquire());
        assert_eq!(credits.in_flight(), 3);

        credits.release(2);
        assert!(credits.try_acquire());
        assert_eq!(credits.in_flight(), 2);
        assert_eq!(credits.maximum(), 3);
    }

    #[test]
    fn full_retirement_batch_waits_for_the_configured_size() {
        let (handoff, inbox) = retirement_channel(NonZeroUsize::new(4).unwrap());
        let (done_sender, done_receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            done_sender
                .send(inbox.recv_full_batch(NonZeroUsize::new(3).unwrap()))
                .unwrap();
        });

        handoff.handoff(1).unwrap();
        assert!(done_receiver
            .recv_timeout(Duration::from_millis(50))
            .is_err());
        handoff.handoff(2).unwrap();
        handoff.handoff(3).unwrap();

        assert_eq!(
            done_receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            Some(vec![1, 2, 3])
        );
        worker.join().unwrap();
    }

    #[test]
    fn retirement_waits_only_when_the_owner_backlog_is_full() {
        let mut stats = RetirementStats::default();
        let capacity = NonZeroUsize::new(8).unwrap();

        stats.observe_in_flight(7);
        assert!(!stats.backlog_is_full(7, capacity));
        assert_eq!(stats.max_in_flight, 7);
        stats.observe_in_flight(8);
        assert!(stats.backlog_is_full(8, capacity));
        assert_eq!(stats.max_in_flight, 8);
        stats.record_batch(3);
        stats.observe_in_flight(10);
        assert!(!stats.backlog_is_full(10, capacity));
        assert_eq!(stats.max_in_flight, 8);
        stats.observe_in_flight(11);
        assert!(stats.backlog_is_full(11, capacity));
    }

    #[test]
    fn cancellation_discards_submissions_without_retirement_credits() {
        let credits = Arc::new(RetirementCredits::new(NonZeroUsize::new(64).unwrap()));
        for _ in 0..64 {
            assert!(credits.try_acquire());
        }
        let (sender, receiver) = mpsc::channel();
        for submission in 0..64 {
            credits.submission_enqueued();
            sender.send(submission).unwrap();
        }
        let worker_credits = Arc::clone(&credits);
        let (done_sender, done_receiver) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            discard_queued_submissions(&receiver, worker_credits.as_ref());
            done_sender.send(()).unwrap();
        });

        drop(sender);
        done_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("cancelled submission drain blocked on exhausted credits");
        worker.join().unwrap();
        assert_eq!(credits.queued_submissions(), 0);
        assert_eq!(credits.in_flight(), 64);
    }
}
