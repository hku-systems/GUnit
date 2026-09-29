use std::borrow::Cow;

use libafl::{
    inputs::{BytesInput, HasTargetBytes},
    mutators::{MutationResult, Mutator},
    state::HasRand,
    Error,
};
use libafl_bolts::{rands::Rand, AsSlice, Named};

use crate::arg_pack_v1::{
    default_seed_rapid_input_v1, mutate_rapid_input_unrepaired_v1, mutate_rapid_input_v1,
    normalize_rapid_input_v1,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ArgPackStructureMode {
    Mutating,
    Fixed,
}

#[derive(Debug)]
pub struct ArgPackStructureMutator {
    mode: ArgPackStructureMode,
}

impl Default for ArgPackStructureMutator {
    fn default() -> Self {
        Self {
            mode: ArgPackStructureMode::Mutating,
        }
    }
}

impl ArgPackStructureMutator {
    pub fn fixed() -> Self {
        Self {
            mode: ArgPackStructureMode::Fixed,
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
        if self.mode == ArgPackStructureMode::Fixed {
            *input = BytesInput::new(default_seed_rapid_input_v1());
            return Ok(MutationResult::Mutated);
        }

        let selector = state.rand_mut().next();
        let byte = (state.rand_mut().next() & 0xff) as u8;
        let mutated =
            mutate_rapid_input_unrepaired_v1(input.target_bytes().as_slice(), selector, byte);
        if mutated == input.target_bytes().as_slice() {
            Ok(MutationResult::Skipped)
        } else {
            *input = BytesInput::new(mutated);
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for ArgPackStructureMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("ArgPackStructureMutator");
        &NAME
    }
}

impl<S> Mutator<BytesInput, S> for ArgPackStructureMutator
where
    S: HasRand,
{
    fn mutate(&mut self, state: &mut S, input: &mut BytesInput) -> Result<MutationResult, Error> {
        if self.mode == ArgPackStructureMode::Fixed {
            *input = BytesInput::new(normalize_rapid_input_v1(&default_seed_rapid_input_v1()));
            return Ok(MutationResult::Mutated);
        }

        let selector = state.rand_mut().next();
        let byte = (state.rand_mut().next() & 0xff) as u8;
        let mutated = mutate_rapid_input_v1(input.target_bytes().as_slice(), selector, byte);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arg_pack_v1::{
        default_seed_rapid_input_v1, init_arg_pack_manifest, normalize_rapid_input_v1,
    };
    use libafl_bolts::rands::StdRand;
    use std::marker::PhantomData;
    use std::time::{SystemTime, UNIX_EPOCH};

    const TEST_MANIFEST: &str = r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "scalar_kernel",
      "display_name": "scalar_kernel",
      "args": [
        {
          "index": 0,
          "name": "seed",
          "type": "uint32_t",
          "kind": "scalar",
          "size_bytes": 4,
          "align_bytes": 4,
          "domain": {
            "kind": "int_range",
            "min": "7",
            "signed": false
          }
        }
      ]
    }
  ]
}
"#;

    struct TestState<R>
    where
        R: Rand,
    {
        rand: R,
        phantom: PhantomData<R>,
    }

    impl<R> HasRand for TestState<R>
    where
        R: Rand,
    {
        type Rand = R;

        fn rand(&self) -> &Self::Rand {
            &self.rand
        }

        fn rand_mut(&mut self) -> &mut Self::Rand {
            &mut self.rand
        }
    }

    fn create_test_state() -> TestState<StdRand> {
        TestState {
            rand: StdRand::with_seed(42),
            phantom: PhantomData,
        }
    }

    fn init_test_manifest() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let manifest_path = std::env::temp_dir().join(format!(
            "rapid-arg-pack-structure-test-{}-{nanos}.json",
            std::process::id()
        ));
        std::fs::write(&manifest_path, TEST_MANIFEST).unwrap();
        let result = init_arg_pack_manifest(&manifest_path);
        let _ = std::fs::remove_file(manifest_path);
        if let Err(err) = result {
            assert_eq!(err, "manifest-driven arg-pack spec already initialized");
        }
    }

    #[test]
    fn fixed_mode_rewrites_to_canonical_seed_and_reports_mutated() {
        init_test_manifest();
        let expected = normalize_rapid_input_v1(&default_seed_rapid_input_v1());
        let mut input = BytesInput::new(vec![0xff, 0xee, 0xdd]);
        let mut state = create_test_state();
        let mut mutator = ArgPackStructureMutator::fixed();

        let result = mutator.mutate(&mut state, &mut input).unwrap();

        assert_eq!(result, MutationResult::Mutated);
        assert_eq!(input.target_bytes().as_slice(), expected.as_slice());

        let result = mutator.mutate(&mut state, &mut input).unwrap();

        assert_eq!(result, MutationResult::Mutated);
        assert_eq!(input.target_bytes().as_slice(), expected.as_slice());
    }
}
