use std::borrow::Cow;

use libafl::{
    inputs::{BytesInput, HasTargetBytes},
    mutators::{MutationResult, Mutator},
    state::HasRand,
    Error,
};
use libafl_bolts::{rands::Rand, AsSlice, Named};

use crate::arg_pack_v1::{mutate_rapid_vconfig_unrepaired_v1, mutate_rapid_vconfig_v1};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum VConfigMode {
    Mutating,
    Fixed,
}

#[derive(Debug)]
pub struct VConfigMutator {
    mode: VConfigMode,
}

impl Default for VConfigMutator {
    fn default() -> Self {
        Self {
            mode: VConfigMode::Mutating,
        }
    }
}

impl VConfigMutator {
    pub fn fixed() -> Self {
        Self {
            mode: VConfigMode::Fixed,
        }
    }

    pub(super) fn mutate_unrepaired<S>(
        &mut self,
        state: &mut S,
        input: &mut BytesInput,
    ) -> Result<MutationResult, Error>
    where
        S: HasRand,
    {
        if self.mode == VConfigMode::Fixed {
            return Ok(MutationResult::Skipped);
        }

        let selector = state.rand_mut().next();
        let byte = (state.rand_mut().next() & 0xff) as u8;
        let mutated =
            mutate_rapid_vconfig_unrepaired_v1(input.target_bytes().as_slice(), selector, byte);
        if mutated == input.target_bytes().as_slice() {
            Ok(MutationResult::Skipped)
        } else {
            *input = BytesInput::new(mutated);
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for VConfigMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("VConfigMutator");
        &NAME
    }
}

impl<S> Mutator<BytesInput, S> for VConfigMutator
where
    S: HasRand,
{
    fn mutate(&mut self, state: &mut S, input: &mut BytesInput) -> Result<MutationResult, Error> {
        if self.mode == VConfigMode::Fixed {
            return Ok(MutationResult::Skipped);
        }

        let selector = state.rand_mut().next();
        let byte = (state.rand_mut().next() & 0xff) as u8;
        let mutated = mutate_rapid_vconfig_v1(input.target_bytes().as_slice(), selector, byte);
        if mutated == input.target_bytes().as_slice() {
            Ok(MutationResult::Skipped)
        } else {
            *input = BytesInput::new(mutated);
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
