use std::{
    borrow::Cow,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use libafl::{
    events::{Event, EventFirer, EventWithStats},
    executors::ExitKind,
    feedbacks::{Feedback, StateInitializer},
    monitors::stats::{AggregatorOps, UserStats, UserStatsValue},
    state::HasExecutions,
    Error, HasMetadata,
};
use libafl_bolts::{impl_serdeany, Named};
use serde::{Deserialize, Serialize};

use crate::cuda_backend::SIMT_MEMCOV_STORAGE_SIZE;

pub const THREAD_ACTIVITY_BYTE_BASE: usize = 61_440 / 8;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct FeedbackStats {
    pub simt_memcov_bits: u64,
    pub thread_activity_bits: u64,
    pub submitted: u64,
    pub evaluated: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct BitsetNoveltiesMetadata {
    pub memory_index_buckets: Vec<u32>,
    pub thread_activity_buckets: Vec<u32>,
}

impl_serdeany!(BitsetNoveltiesMetadata);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BitsetNoveltyStateMetadata {
    pub seen: Vec<u8>,
    pub stats: FeedbackStats,
}

impl Default for BitsetNoveltyStateMetadata {
    fn default() -> Self {
        Self {
            seen: vec![0; SIMT_MEMCOV_STORAGE_SIZE],
            stats: FeedbackStats::default(),
        }
    }
}

impl_serdeany!(BitsetNoveltyStateMetadata);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BitsetObservation {
    pub interesting: bool,
    pub stats: FeedbackStats,
    pub novelties: BitsetNoveltiesMetadata,
}

#[derive(Debug)]
struct BitsetNoveltyState {
    submitted_since_sync: u64,
    pending_bits: Option<Vec<u8>>,
    pending: Option<BitsetObservation>,
    #[cfg(feature = "track_hit_feedbacks")]
    last_result: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct SimtMemCovFeedback {
    state: Arc<Mutex<BitsetNoveltyState>>,
    name: Cow<'static, str>,
}

impl SimtMemCovFeedback {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(BitsetNoveltyState {
                submitted_since_sync: 0,
                pending_bits: None,
                pending: None,
                #[cfg(feature = "track_hit_feedbacks")]
                last_result: None,
            })),
            name: Cow::Borrowed("bitset_novelty"),
        }
    }

    pub fn observe(&self, current: &[u8]) -> anyhow::Result<()> {
        if current.len() != SIMT_MEMCOV_STORAGE_SIZE {
            anyhow::bail!(
                "invalid feedback bitset length: expected {SIMT_MEMCOV_STORAGE_SIZE} bytes, got {}",
                current.len()
            );
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("feedback bitset state lock is poisoned"))?;
        state.pending_bits = Some(current.to_vec());
        state.pending = None;
        Ok(())
    }

    pub fn prepare_observation<S>(&self, state: &mut S, current: &[u8]) -> Result<(), Error>
    where
        S: HasMetadata,
    {
        if current.len() != SIMT_MEMCOV_STORAGE_SIZE {
            return Err(Error::illegal_argument(format!(
                "invalid feedback bitset length: expected {SIMT_MEMCOV_STORAGE_SIZE} bytes, got {}",
                current.len()
            )));
        }

        let mut inner = self
            .state
            .lock()
            .map_err(|_| Error::illegal_state("feedback bitset state lock is poisoned"))?;
        let submitted = std::mem::take(&mut inner.submitted_since_sync);
        let metadata = state.metadata_or_insert_with(BitsetNoveltyStateMetadata::default);
        let observation = evaluate_current_bits(current, metadata, submitted)?;
        inner.pending_bits = None;
        inner.pending = Some(observation);
        Ok(())
    }

    pub fn record_submission(&self) -> anyhow::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("feedback bitset state lock is poisoned"))?;
        state.submitted_since_sync += 1;
        Ok(())
    }

    pub fn record_known_uninteresting_batch<S>(
        &self,
        state: &mut S,
        evaluated: usize,
    ) -> Result<FeedbackStats, Error>
    where
        S: HasMetadata,
    {
        let mut inner = self
            .state
            .lock()
            .map_err(|_| Error::illegal_state("feedback bitset state lock is poisoned"))?;
        let submitted = std::mem::take(&mut inner.submitted_since_sync);
        inner.pending_bits = None;
        inner.pending = None;
        drop(inner);

        let metadata = state.metadata_or_insert_with(BitsetNoveltyStateMetadata::default);
        metadata.stats.submitted += submitted;
        metadata.stats.evaluated += evaluated as u64;
        let stats = metadata.stats;
        let mut inner = self
            .state
            .lock()
            .map_err(|_| Error::illegal_state("feedback bitset state lock is poisoned"))?;
        inner.pending = Some(BitsetObservation {
            interesting: false,
            stats,
            novelties: BitsetNoveltiesMetadata::default(),
        });
        Ok(stats)
    }

    pub fn publish_stats<EM, I, S>(
        state: &mut S,
        manager: &mut EM,
        stats: FeedbackStats,
    ) -> Result<(), Error>
    where
        EM: EventFirer<I, S>,
        S: HasExecutions,
    {
        for (name, value) in [
            ("simt_memcov_bits", stats.simt_memcov_bits),
            ("logical_thread_bits", stats.thread_activity_bits),
            ("submitted", stats.submitted),
            ("evaluated", stats.evaluated),
        ] {
            manager.fire(
                state,
                EventWithStats::with_current_time(
                    Event::UpdateUserStats {
                        name: Cow::Borrowed(name),
                        value: UserStats::new(UserStatsValue::Number(value), AggregatorOps::Max),
                        phantom: PhantomData,
                    },
                    *state.executions(),
                ),
            )?;
        }
        Ok(())
    }

    pub fn stats(&self) -> anyhow::Result<FeedbackStats> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("feedback bitset state lock is poisoned"))?;
        Ok(state
            .pending
            .as_ref()
            .map(|observation| observation.stats)
            .unwrap_or_else(FeedbackStats::default))
    }

    fn take_pending_observation(&self) -> anyhow::Result<BitsetObservation> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("feedback bitset state lock is poisoned"))?;
        state
            .pending
            .take()
            .ok_or_else(|| anyhow::anyhow!("no pending memory/index feedback observation"))
    }

    fn take_pending_bits(&self) -> anyhow::Result<(Vec<u8>, u64)> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("feedback bitset state lock is poisoned"))?;
        let bits = state
            .pending_bits
            .take()
            .ok_or_else(|| anyhow::anyhow!("no pending memory/index feedback bitset"))?;
        let submitted = state.submitted_since_sync;
        state.submitted_since_sync = 0;
        Ok((bits, submitted))
    }

    fn set_pending_observation(&self, observation: BitsetObservation) -> anyhow::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("feedback bitset state lock is poisoned"))?;
        state.pending = Some(observation);
        Ok(())
    }
}

fn collect_new_bucket_ids(
    byte_index: usize,
    new_bits: u8,
    novelties: &mut BitsetNoveltiesMetadata,
) {
    for bit_index in 0..8 {
        if new_bits & (1 << bit_index) == 0 {
            continue;
        }
        let bucket = (byte_index * 8 + bit_index) as u32;
        if bucket < 61_440 {
            novelties.memory_index_buckets.push(bucket);
        } else {
            novelties.thread_activity_buckets.push(bucket);
        }
    }
}

fn evaluate_current_bits(
    current: &[u8],
    metadata: &mut BitsetNoveltyStateMetadata,
    submitted_since_sync: u64,
) -> Result<BitsetObservation, Error> {
    if metadata.seen.len() != SIMT_MEMCOV_STORAGE_SIZE {
        return Err(Error::illegal_state(format!(
            "invalid persisted feedback bitset length: expected {SIMT_MEMCOV_STORAGE_SIZE} bytes, got {}",
            metadata.seen.len()
        )));
    }

    let mut novelties = BitsetNoveltiesMetadata::default();
    for (byte_index, (seen_byte, current_byte)) in metadata
        .seen
        .iter_mut()
        .zip(current.iter().copied())
        .enumerate()
    {
        let new_bits = current_byte & !*seen_byte;
        if new_bits != 0 {
            collect_new_bucket_ids(byte_index, new_bits, &mut novelties);
            let new_count = new_bits.count_ones() as u64;
            if byte_index < THREAD_ACTIVITY_BYTE_BASE {
                metadata.stats.simt_memcov_bits += new_count;
            } else {
                metadata.stats.thread_activity_bits += new_count;
            }
        }
        *seen_byte |= current_byte;
    }

    metadata.stats.submitted += submitted_since_sync;
    metadata.stats.evaluated += 1;

    let interesting =
        !novelties.memory_index_buckets.is_empty() || !novelties.thread_activity_buckets.is_empty();
    Ok(BitsetObservation {
        interesting,
        stats: metadata.stats,
        novelties,
    })
}

impl Named for SimtMemCovFeedback {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

impl<S> StateInitializer<S> for SimtMemCovFeedback
where
    S: HasMetadata,
{
    fn init_state(&mut self, state: &mut S) -> Result<(), Error> {
        let _ = state.metadata_or_insert_with(BitsetNoveltyStateMetadata::default);
        Ok(())
    }
}

impl<EM, I, OT, S> Feedback<EM, I, OT, S> for SimtMemCovFeedback
where
    EM: EventFirer<I, S>,
    S: HasExecutions + HasMetadata,
{
    fn is_interesting(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        _input: &I,
        _observers: &OT,
        _exit_kind: &ExitKind,
    ) -> Result<bool, Error> {
        let prepared = self
            .state
            .lock()
            .map_err(|_| Error::illegal_state("feedback bitset state lock is poisoned"))?
            .pending
            .clone();
        let observation = if let Some(observation) = prepared {
            observation
        } else {
            match self.take_pending_bits() {
                Ok((current, submitted_since_sync)) => {
                    let metadata =
                        state.metadata_or_insert_with(BitsetNoveltyStateMetadata::default);
                    let observation =
                        evaluate_current_bits(&current, metadata, submitted_since_sync)?;
                    self.set_pending_observation(observation.clone())
                        .map_err(|err| Error::illegal_state(err.to_string()))?;
                    observation
                }
                Err(_) => {
                    let metadata =
                        state.metadata_or_insert_with(BitsetNoveltyStateMetadata::default);
                    let observation = BitsetObservation {
                        interesting: false,
                        stats: metadata.stats,
                        novelties: BitsetNoveltiesMetadata::default(),
                    };
                    self.set_pending_observation(observation.clone())
                        .map_err(|err| Error::illegal_state(err.to_string()))?;
                    observation
                }
            }
        };

        #[cfg(feature = "track_hit_feedbacks")]
        {
            let mut inner = self
                .state
                .lock()
                .map_err(|_| Error::illegal_state("feedback bitset state lock is poisoned"))?;
            inner.last_result = Some(observation.interesting);
        }

        Self::publish_stats::<EM, I, S>(state, manager, observation.stats)?;

        Ok(observation.interesting)
    }

    fn append_metadata(
        &mut self,
        _state: &mut S,
        _manager: &mut EM,
        _observers: &OT,
        testcase: &mut libafl::corpus::Testcase<I>,
    ) -> Result<(), Error> {
        let observation = self
            .take_pending_observation()
            .map_err(|err| Error::illegal_state(err.to_string()))?;
        if !observation.novelties.memory_index_buckets.is_empty()
            || !observation.novelties.thread_activity_buckets.is_empty()
        {
            testcase.add_metadata(observation.novelties);
        }
        Ok(())
    }

    #[cfg(feature = "track_hit_feedbacks")]
    fn last_result(&self) -> Result<bool, Error> {
        let inner = self
            .state
            .lock()
            .map_err(|_| Error::illegal_state("feedback bitset state lock is poisoned"))?;
        inner
            .last_result
            .ok_or_else(|| Error::illegal_state("bitset feedback has not been evaluated"))
    }
}

impl Default for SimtMemCovFeedback {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libafl::{
        common::HasMetadata,
        corpus::Testcase,
        events::NopEventManager,
        executors::ExitKind,
        feedbacks::{Feedback, StateInitializer},
        inputs::BytesInput,
        state::NopState,
    };

    fn empty_bits() -> Vec<u8> {
        vec![0; SIMT_MEMCOV_STORAGE_SIZE]
    }

    fn evaluate(
        feedback: &mut SimtMemCovFeedback,
        state: &mut NopState<BytesInput>,
        bits: &[u8],
    ) -> bool {
        let mut manager = NopEventManager::new();
        let input = BytesInput::new(vec![1]);
        feedback.observe(bits).unwrap();
        feedback
            .is_interesting(state, &mut manager, &input, &(), &ExitKind::Ok)
            .unwrap()
    }

    #[test]
    fn only_unseen_bits_are_interesting() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        feedback.init_state(&mut state).unwrap();
        let mut bits = empty_bits();
        bits[0] = 0b0000_0010;

        assert!(evaluate(&mut feedback, &mut state, &bits));
        assert!(!evaluate(&mut feedback, &mut state, &bits));

        bits[0] = 0b0000_0110;
        assert!(evaluate(&mut feedback, &mut state, &bits));
    }

    #[test]
    fn known_uninteresting_batch_updates_counters_without_novelty() {
        let feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        feedback.record_submission().unwrap();
        feedback.record_submission().unwrap();

        feedback
            .record_known_uninteresting_batch(&mut state, 2)
            .unwrap();

        let metadata = state.metadata::<BitsetNoveltyStateMetadata>().unwrap();
        assert_eq!(metadata.stats.submitted, 2);
        assert_eq!(metadata.stats.evaluated, 2);
        assert_eq!(metadata.seen, empty_bits());
        assert_eq!(feedback.stats().unwrap(), metadata.stats);
    }

    #[test]
    fn counts_partitions_independently() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        feedback.init_state(&mut state).unwrap();
        let mut bits = empty_bits();
        bits[0] = 0b0000_0001;
        bits[THREAD_ACTIVITY_BYTE_BASE] = 0b0000_0001;

        assert!(evaluate(&mut feedback, &mut state, &bits));
        let stats = feedback.stats().unwrap();

        assert_eq!(stats.simt_memcov_bits, 1);
        assert_eq!(stats.thread_activity_bits, 1);
    }

    #[test]
    fn lower_valued_byte_with_an_unseen_bit_is_interesting() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        feedback.init_state(&mut state).unwrap();
        let mut bits = empty_bits();
        bits[0] = 0b1000_0000;
        assert!(evaluate(&mut feedback, &mut state, &bits));

        bits[0] = 0b0000_0001;
        assert!(evaluate(&mut feedback, &mut state, &bits));
    }

    #[test]
    fn prepared_observation_is_consumed_without_staging_bits() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        let mut manager = NopEventManager::new();
        let input = BytesInput::new(vec![1]);
        let mut bits = empty_bits();
        bits[7] = 0b0000_0100;

        feedback.init_state(&mut state).unwrap();
        feedback.record_submission().unwrap();
        feedback.prepare_observation(&mut state, &bits).unwrap();

        assert!(feedback
            .is_interesting(&mut state, &mut manager, &input, &(), &ExitKind::Ok)
            .unwrap());
        let stats = feedback.stats().unwrap();
        assert_eq!(stats.submitted, 1);
        assert_eq!(stats.evaluated, 1);
        assert_eq!(stats.simt_memcov_bits, 1);
    }

    #[test]
    fn rejects_wrong_bitset_length() {
        let feedback = SimtMemCovFeedback::new();

        let error = feedback.observe(&[0; 1]).unwrap_err();

        assert!(error.to_string().contains("expected 8192 bytes"));
    }

    #[test]
    fn consumes_each_pending_observation_once() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        feedback.init_state(&mut state).unwrap();
        let mut bits = empty_bits();
        bits[0] = 1;
        assert!(evaluate(&mut feedback, &mut state, &bits));

        assert!(feedback.take_pending_observation().unwrap().interesting);
        assert!(feedback
            .take_pending_observation()
            .unwrap_err()
            .to_string()
            .contains("no pending memory/index feedback"));
    }

    #[test]
    fn tracks_submitted_and_evaluated_counts() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        feedback.init_state(&mut state).unwrap();
        feedback.record_submission().unwrap();
        feedback.record_submission().unwrap();
        evaluate(&mut feedback, &mut state, &empty_bits());

        let stats = feedback.stats().unwrap();

        assert_eq!(stats.submitted, 2);
        assert_eq!(stats.evaluated, 1);
    }

    #[test]
    fn libafl_feedback_returns_the_pending_novelty() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut bits = empty_bits();
        bits[0] = 1;
        feedback.observe(&bits).unwrap();
        let mut state = NopState::<BytesInput>::new();
        let mut manager = NopEventManager::new();
        let input = BytesInput::new(vec![1]);

        let interesting = feedback
            .is_interesting(&mut state, &mut manager, &input, &(), &ExitKind::Ok)
            .unwrap();

        assert!(interesting);
    }

    #[test]
    fn submission_without_observation_is_not_interesting() {
        let mut feedback = SimtMemCovFeedback::new();
        feedback.record_submission().unwrap();
        let mut state = NopState::<BytesInput>::new();
        let mut manager = NopEventManager::new();
        let input = BytesInput::new(vec![1]);

        let interesting = feedback
            .is_interesting(&mut state, &mut manager, &input, &(), &ExitKind::Timeout)
            .unwrap();

        assert!(!interesting);
    }

    #[test]
    fn novelty_state_survives_state_serialization() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        let mut manager = NopEventManager::new();
        let input = BytesInput::new(vec![1]);
        let mut bits = empty_bits();
        bits[0] = 0b0000_0010;

        feedback.init_state(&mut state).unwrap();
        feedback.observe(&bits).unwrap();
        assert!(feedback
            .is_interesting(&mut state, &mut manager, &input, &(), &ExitKind::Ok)
            .unwrap());

        let serialized = postcard::to_allocvec(&state).unwrap();
        let mut restored: NopState<BytesInput> = postcard::from_bytes(&serialized).unwrap();
        let mut fresh_feedback = SimtMemCovFeedback::new();
        fresh_feedback.init_state(&mut restored).unwrap();
        fresh_feedback.observe(&bits).unwrap();

        assert!(!fresh_feedback
            .is_interesting(&mut restored, &mut manager, &input, &(), &ExitKind::Ok,)
            .unwrap());
    }

    #[test]
    fn appends_exact_new_bucket_metadata_to_testcase() {
        let mut feedback = SimtMemCovFeedback::new();
        let mut state = NopState::<BytesInput>::new();
        let mut manager = NopEventManager::new();
        let input = BytesInput::new(vec![1]);
        let mut bits = empty_bits();
        bits[0] = 0b0000_0011;
        bits[THREAD_ACTIVITY_BYTE_BASE] = 0b0000_0001;
        bits[THREAD_ACTIVITY_BYTE_BASE + 1] = 0b0000_0100;

        feedback.init_state(&mut state).unwrap();
        feedback.observe(&bits).unwrap();
        assert!(feedback
            .is_interesting(&mut state, &mut manager, &input, &(), &ExitKind::Ok)
            .unwrap());

        let mut testcase = Testcase::from(input);
        feedback
            .append_metadata(&mut state, &mut manager, &(), &mut testcase)
            .unwrap();
        let metadata = testcase.metadata::<BitsetNoveltiesMetadata>().unwrap();

        assert_eq!(metadata.memory_index_buckets, vec![0, 1]);
        assert_eq!(metadata.thread_activity_buckets, vec![61_440, 61_450]);
    }
}
