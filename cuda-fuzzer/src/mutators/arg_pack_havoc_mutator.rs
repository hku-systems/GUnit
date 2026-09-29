use std::borrow::Cow;

use libafl::{
    inputs::{BytesInput, HasTargetBytes},
    mutators::{MutationResult, Mutator},
    state::HasRand,
    Error,
};
use libafl_bolts::{rands::Rand, AsSlice, Named};

use crate::arg_pack_v1::{
    mutate_rapid_input_havoc_unrepaired_v1, mutate_rapid_input_havoc_v1,
    ARG_PACK_HAVOC_MAX_STACKED_OPS,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ArgPackHavocMode {
    Mutating,
    Fixed,
}

#[derive(Debug)]
pub struct ArgPackHavocMutator {
    mode: ArgPackHavocMode,
}

impl Default for ArgPackHavocMutator {
    fn default() -> Self {
        Self {
            mode: ArgPackHavocMode::Mutating,
        }
    }
}

impl ArgPackHavocMutator {
    pub fn fixed() -> Self {
        Self {
            mode: ArgPackHavocMode::Fixed,
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
        if self.mode == ArgPackHavocMode::Fixed {
            return Ok(MutationResult::Skipped);
        }

        let operations = havoc_operations(state.rand_mut());
        let mutated =
            mutate_rapid_input_havoc_unrepaired_v1(input.target_bytes().as_slice(), &operations);
        if mutated == input.target_bytes().as_slice() {
            Ok(MutationResult::Skipped)
        } else {
            *input = BytesInput::new(mutated);
            Ok(MutationResult::Mutated)
        }
    }
}

impl Named for ArgPackHavocMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("ArgPackHavocMutator");
        &NAME
    }
}

fn havoc_operations<R>(rand: &mut R) -> Vec<(u64, u8)>
where
    R: Rand,
{
    let operation_count = (rand.next() % ARG_PACK_HAVOC_MAX_STACKED_OPS as u64) as usize + 1;
    let mut operations = Vec::with_capacity(operation_count);
    for _ in 0..operation_count {
        operations.push((rand.next(), (rand.next() & 0xff) as u8));
    }
    operations
}

impl<S> Mutator<BytesInput, S> for ArgPackHavocMutator
where
    S: HasRand,
{
    fn mutate(&mut self, state: &mut S, input: &mut BytesInput) -> Result<MutationResult, Error> {
        if self.mode == ArgPackHavocMode::Fixed {
            return Ok(MutationResult::Skipped);
        }

        let operations = havoc_operations(state.rand_mut());
        let mutated = mutate_rapid_input_havoc_v1(input.target_bytes().as_slice(), &operations);
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
    use super::havoc_operations;
    use crate::arg_pack_v1::{parse_rapid_input_envelope, ArgPackHavocTestSpec};
    use libafl_bolts::rands::StdRand;
    use std::collections::BTreeSet;

    const TEST_MANIFEST: &str = r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "havoc_kernel",
      "display_name": "havoc_kernel",
      "args": [
        {
          "index": 0,
          "name": "count",
          "type": "uint32_t",
          "kind": "scalar",
          "size_bytes": 4,
          "align_bytes": 4,
          "domain": {
            "kind": "int_range",
            "min": "1",
            "max": "16",
            "signed": false
          }
        },
        {
          "index": 1,
          "name": "variable",
          "type": "uint32_t *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "pointee_layout": {
            "index": "variable.*",
            "name": "$pointee",
            "type": "uint32_t",
            "kind": "scalar",
            "size_bytes": 4,
            "align_bytes": 4
          },
          "size_bytes": 8,
          "align_bytes": 8,
          "domain": {
            "kind": "bytes",
            "min_len": "8",
            "max_len": "64",
            "elem_size_bytes": 4
          }
        },
        {
          "index": 2,
          "name": "fixed",
          "type": "uint32_t *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "pointee_layout": {
            "index": "fixed.*",
            "name": "$pointee",
            "type": "uint32_t",
            "kind": "scalar",
            "size_bytes": 4,
            "align_bytes": 4
          },
          "size_bytes": 8,
          "align_bytes": 8,
          "domain": {
            "kind": "bytes",
            "min_len": "16",
            "max_len": "16",
            "elem_size_bytes": 4
          }
        },
        {
          "index": 3,
          "name": "tail_variable",
          "type": "uint32_t *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "pointee_layout": {
            "index": "tail_variable.*",
            "name": "$pointee",
            "type": "uint32_t",
            "kind": "scalar",
            "size_bytes": 4,
            "align_bytes": 4
          },
          "size_bytes": 8,
          "align_bytes": 8,
          "domain": {
            "kind": "bytes",
            "min_len": "8",
            "max_len": "64",
            "elem_size_bytes": 4
          }
        }
      ],
      "constraints": [
        {
          "kind": "count_fits_buffer",
          "count_arg": 0,
          "buffer_arg": 1,
          "elem_size_bytes": 4
        }
      ]
    }
  ]
}
"#;

    fn test_rng() -> StdRand {
        StdRand::with_seed(42)
    }

    #[test]
    fn havoc_outputs_are_valid_rapid_envelopes() {
        let spec = ArgPackHavocTestSpec::from_manifest_json(TEST_MANIFEST);
        let mut input = spec.default_rapid_input();
        let mut rng = test_rng();

        for _ in 0..256 {
            input = spec.mutate_rapid_input_havoc(&input, &havoc_operations(&mut rng));
            assert!(parse_rapid_input_envelope(&input).is_some());
            assert!(spec.is_canonical_rapid_input(&input));
        }
    }

    #[test]
    fn variable_buffer_length_changes_and_stays_in_bounds() {
        let spec = ArgPackHavocTestSpec::from_manifest_json(TEST_MANIFEST);
        let mut input = spec.default_rapid_input();
        let mut lengths = BTreeSet::from([spec.pointer_len(&input, 1)]);
        let mut rng = test_rng();

        for _ in 0..512 {
            input = spec.mutate_rapid_input_havoc(&input, &havoc_operations(&mut rng));
            let len = spec.pointer_len(&input, 1);
            assert!((8..=64).contains(&len));
            lengths.insert(len);
        }

        assert!(lengths.len() > 1, "variable buffer length never changed");
    }

    #[test]
    fn buffer_length_mutations_preserve_element_alignment() {
        let spec = ArgPackHavocTestSpec::from_manifest_json(TEST_MANIFEST);
        let input = spec.default_rapid_input();

        for selector in 0..256 {
            let mutated = spec.mutate_rapid_input_havoc(&input, &[(selector, 0xaa)]);
            let len = spec.pointer_len(&mutated, 3);
            assert_eq!(
                len % 4,
                0,
                "selector {selector} produced an unaligned buffer length {len}"
            );
        }
    }

    #[test]
    fn fixed_buffer_length_never_changes() {
        let spec = ArgPackHavocTestSpec::from_manifest_json(TEST_MANIFEST);
        let mut input = spec.default_rapid_input();
        let mut rng = test_rng();

        for _ in 0..512 {
            input = spec.mutate_rapid_input_havoc(&input, &havoc_operations(&mut rng));
            assert_eq!(spec.pointer_len(&input, 2), 16);
        }
    }

    #[test]
    fn count_fits_buffer_is_repaired_after_havoc() {
        let spec = ArgPackHavocTestSpec::from_manifest_json(TEST_MANIFEST);
        let mut input = spec.default_rapid_input();
        let mut rng = test_rng();

        for _ in 0..512 {
            input = spec.mutate_rapid_input_havoc(&input, &havoc_operations(&mut rng));
            let count = spec.scalar(&input, 0);
            let buffer_len = spec.pointer_len(&input, 1) as u64;
            assert!(count * 4 <= buffer_len);
        }
    }
}
