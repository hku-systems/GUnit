//| The [`AsyncMutationalStage`] is the async version of the mutational stage used during fuzzing.
//! For the current input, it will perform a range of random mutations, and then run them in the executor.

extern crate alloc;

use alloc::{
    borrow::{Cow, ToOwned},
    string::ToString,
};
use core::{marker::PhantomData, num::NonZeroUsize};

use libafl_bolts::{rands::Rand, Named};

#[cfg(feature = "introspection")]
use libafl::monitors::stats::PerfFeature;
use libafl::{
    corpus::{Corpus, CorpusId, HasCurrentCorpusId, Testcase},
    fuzzer::Evaluator,
    inputs::Input,
    mark_feature_time,
    mutators::{MutationResult, Mutator},
    nonzero,
    stages::{Restartable, RetryCountRestartHelper, Stage},
    start_timer,
    state::{HasCorpus, HasCurrentTestcase, HasExecutions, HasRand, MaybeHasClientPerfMonitor},
    Error, HasMetadata, HasNamedMetadata,
};

use crate::submission_context::{with_submission_context, SubmissionContext};

/// Action performed after the un-transformed input is executed (e.g., updating metadata)
pub trait MutatedTransformPost<S>: Sized {
    /// Perform any post-execution steps necessary for the transformed input (e.g., updating metadata)
    #[inline]
    fn post_exec(self, _state: &mut S, _new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        Ok(())
    }
}

impl<S> MutatedTransformPost<S> for () {}

/// A type which may both be transformed from and into a given input type, used to perform
/// mutations over inputs which are not necessarily performable on the underlying type
///
/// This trait is implemented such that all testcases inherently transform to their inputs, should
/// the input be cloneable.
pub trait MutatedTransform<I, S>: Sized {
    /// Type indicating actions to be taken after the post-transformation input is executed
    type Post: MutatedTransformPost<S>;

    /// Transform the provided testcase into this type
    fn try_transform_from(base: &mut Testcase<I>, state: &S) -> Result<Self, Error>;

    /// Transform this instance back into the original input type
    fn try_transform_into(self, state: &S) -> Result<(I, Self::Post), Error>;
}

// reflexive definition
impl<I, S> MutatedTransform<I, S> for I
where
    I: Clone,
    S: HasCorpus<I>,
{
    type Post = ();

    #[inline]
    fn try_transform_from(base: &mut Testcase<I>, state: &S) -> Result<Self, Error> {
        state.corpus().load_input_into(base)?;
        Ok(base.input().as_ref().unwrap().clone())
    }

    #[inline]
    fn try_transform_into(self, _state: &S) -> Result<(I, Self::Post), Error> {
        Ok((self, ()))
    }
}

/// A Mutational stage is the stage in a fuzzing run that mutates inputs.
/// Mutational stages will usually have a range of mutations that are
/// being applied to the input one by one, between executions.
pub trait MutationalStage<S> {
    /// The mutator of this stage
    type Mutator;

    /// The mutator registered for this stage
    fn mutator(&self) -> &Self::Mutator;

    /// The mutator registered for this stage (mutable)
    fn mutator_mut(&mut self) -> &mut Self::Mutator;

    /// Gets the number of iterations this mutator should run for.
    fn iterations(&self, state: &mut S) -> Result<usize, Error>;
}

/// Default value, how many iterations each stage gets, as an upper bound.
/// It may randomly continue earlier.
pub const DEFAULT_MUTATIONAL_MAX_ITERATIONS: usize = 128;

/// The async mutational stage for use with AsyncBatchFuzzer
#[derive(Debug, Clone)]
pub struct AsyncMutationalStage<E, EM, I1, I2, M, S, Z> {
    /// The name
    name: Cow<'static, str>,
    /// The mutator(s) to use
    mutator: M,
    /// The maximum amount of iterations we should do each round
    max_iterations: NonZeroUsize,
    /// Whether this stage's mutator guarantees canonical RAPID inputs.
    canonical_output: bool,
    phantom: PhantomData<(E, EM, I1, I2, S, Z)>,
}

impl<E, EM, I1, I2, M, S, Z> MutationalStage<S> for AsyncMutationalStage<E, EM, I1, I2, M, S, Z>
where
    S: HasRand,
{
    type Mutator = M;

    /// The mutator, added to this stage
    #[inline]
    fn mutator(&self) -> &Self::Mutator {
        &self.mutator
    }

    /// The list of mutators, added to this stage (as mutable ref)
    #[inline]
    fn mutator_mut(&mut self) -> &mut Self::Mutator {
        &mut self.mutator
    }

    /// Gets the number of iterations as a random number
    fn iterations(&self, state: &mut S) -> Result<usize, Error> {
        Ok(1 + state.rand_mut().below(self.max_iterations))
    }
}

/// The unique id for async mutational stage
static mut ASYNC_MUTATIONAL_STAGE_ID: usize = 0;
/// The name for async mutational stage
pub static ASYNC_MUTATIONAL_STAGE_NAME: &str = "async_mutational";

impl<E, EM, I1, I2, M, S, Z> Named for AsyncMutationalStage<E, EM, I1, I2, M, S, Z> {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

impl<E, EM, I1, I2, M, S, Z> Stage<E, EM, S, Z> for AsyncMutationalStage<E, EM, I1, I2, M, S, Z>
where
    I1: Clone + MutatedTransform<I2, S>,
    I2: Input,
    M: Mutator<I1, S>,
    S: HasRand
        + HasCorpus<I2>
        + HasMetadata
        + HasExecutions
        + HasNamedMetadata
        + HasCurrentCorpusId
        + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I2, S>,
{
    #[inline]
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), Error> {
        self.perform_mutational(fuzzer, executor, state, manager)
    }
}

impl<E, EM, I1, I2, M, S, Z> Restartable<S> for AsyncMutationalStage<E, EM, I1, I2, M, S, Z>
where
    S: HasMetadata + HasNamedMetadata + HasCurrentCorpusId,
{
    fn should_restart(&mut self, state: &mut S) -> Result<bool, Error> {
        RetryCountRestartHelper::should_restart(state, &self.name, 3)
    }

    fn clear_progress(&mut self, state: &mut S) -> Result<(), Error> {
        RetryCountRestartHelper::clear_progress(state, &self.name)
    }
}

impl<E, EM, I, M, S, Z> AsyncMutationalStage<E, EM, I, I, M, S, Z>
where
    M: Mutator<I, S>,
    I: MutatedTransform<I, S> + Input + Clone,
    S: HasCorpus<I> + HasRand + HasCurrentCorpusId + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I, S>,
{
    /// Creates a new default mutational stage
    pub fn new(mutator: M) -> Self {
        // Safe to unwrap: DEFAULT_MUTATIONAL_MAX_ITERATIONS is never 0.
        Self::transforming_with_max_iterations(mutator, nonzero!(DEFAULT_MUTATIONAL_MAX_ITERATIONS))
    }

    /// Creates a new mutational stage with the given max iterations
    #[inline]
    pub fn with_max_iterations(mutator: M, max_iterations: NonZeroUsize) -> Self {
        Self::transforming_with_max_iterations(mutator, max_iterations)
    }
}

impl<E, EM, I1, I2, M, S, Z> AsyncMutationalStage<E, EM, I1, I2, M, S, Z>
where
    I1: MutatedTransform<I2, S> + Clone,
    I2: Input,
    M: Mutator<I1, S>,
    S: HasCorpus<I2> + HasRand + HasCurrentCorpusId + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I2, S>,
{
    /// Creates a new transforming mutational stage with the default max iterations
    pub fn transforming(mutator: M) -> Self {
        // Safe to unwrap: DEFAULT_MUTATIONAL_MAX_ITERATIONS is never 0.
        Self::transforming_with_max_iterations(mutator, nonzero!(DEFAULT_MUTATIONAL_MAX_ITERATIONS))
    }

    /// Creates a new transforming mutational stage with the given max iterations
    #[inline]
    pub fn transforming_with_max_iterations(mutator: M, max_iterations: NonZeroUsize) -> Self {
        let stage_id = unsafe {
            let ret = ASYNC_MUTATIONAL_STAGE_ID;
            ASYNC_MUTATIONAL_STAGE_ID += 1;
            ret
        };
        let name = Cow::Owned(
            ASYNC_MUTATIONAL_STAGE_NAME.to_owned() + ":" + stage_id.to_string().as_str(),
        );
        Self {
            name,
            mutator,
            max_iterations,
            canonical_output: false,
            phantom: PhantomData,
        }
    }

    /// Mark outputs as canonical for the immediately following executor call.
    pub fn with_canonical_output(mut self) -> Self {
        self.canonical_output = true;
        self
    }
}

impl<E, EM, I1, I2, M, S, Z> AsyncMutationalStage<E, EM, I1, I2, M, S, Z>
where
    I1: MutatedTransform<I2, S> + Clone,
    I2: Input,
    M: Mutator<I1, S>,
    S: HasRand + HasCurrentTestcase<I2> + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I2, S>,
{
    /// Runs this (mutational) stage for the given testcase
    fn perform_mutational(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), Error> {
        start_timer!(state);

        let num = self.iterations(state)?;
        let mut testcase = state.current_testcase_mut()?;

        let Ok(input) = I1::try_transform_from(&mut testcase, state) else {
            return Ok(());
        };
        drop(testcase);
        mark_feature_time!(state, PerfFeature::GetInputFromCorpus);

        for _ in 0..num {
            let mut input = input.clone();

            start_timer!(state);
            let mutated = self.mutator_mut().mutate(state, &mut input)?;
            mark_feature_time!(state, PerfFeature::Mutate);

            if mutated == MutationResult::Skipped {
                continue;
            }

            let (untransformed, _post) = input.try_transform_into(state)?;
            let evaluation = if self.canonical_output {
                with_submission_context(SubmissionContext::Canonical, || {
                    fuzzer.evaluate_filtered(state, executor, manager, &untransformed)
                })
            } else {
                fuzzer.evaluate_filtered(state, executor, manager, &untransformed)
            };
            let (_, _corpus_id) = evaluation?;

            // AsyncBatchFuzzer runs post_exec after the GPU result arrives.
        }

        Ok(())
    }
}
