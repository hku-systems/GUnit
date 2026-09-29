use std::borrow::Cow;

use libafl::{
    inputs::{BytesInput, HasTargetBytes},
    mutators::{MutationResult, Mutator},
    Error,
};
use libafl_bolts::{AsSlice, Named};

use crate::arg_pack_v1::normalize_rapid_input_v1;

#[derive(Debug, Default)]
pub struct ArgPackNormalizeMutator;

impl Named for ArgPackNormalizeMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("ArgPackNormalizeMutator");
        &NAME
    }
}

impl<S> Mutator<BytesInput, S> for ArgPackNormalizeMutator {
    fn mutate(&mut self, _state: &mut S, input: &mut BytesInput) -> Result<MutationResult, Error> {
        let target = input.target_bytes();
        let original = target.as_slice();
        let normalized = normalize_rapid_input_v1(original);
        if normalized == original {
            Ok(MutationResult::Skipped)
        } else {
            *input = BytesInput::new(normalized);
            Ok(MutationResult::Mutated)
        }
    }

    fn post_exec(
        &mut self,
        _state: &mut S,
        _corpus_idx: Option<libafl::corpus::CorpusId>,
    ) -> Result<(), Error> {
        Ok(())
    }
}
