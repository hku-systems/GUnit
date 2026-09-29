use std::borrow::Cow;

use libafl::{
    corpus::CorpusId,
    inputs::BytesInput,
    mutators::{MutationResult, Mutator},
    state::HasRand,
    Error,
};
use libafl_bolts::{rands::Rand, Named};

use super::{
    ArgPackHavocMutator, ArgPackNormalizeMutator, ArgPackStructureMutator, VConfigMutator,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RapidInputMutationPlan {
    PayloadStructure,
    PayloadHavoc,
    VConfig,
    PayloadStructureAndVConfig,
    PayloadHavocAndVConfig,
}

impl RapidInputMutationPlan {
    fn mutates_vconfig(&self) -> bool {
        matches!(
            self,
            Self::VConfig | Self::PayloadStructureAndVConfig | Self::PayloadHavocAndVConfig
        )
    }

    fn uses_structure(&self) -> bool {
        matches!(
            self,
            Self::PayloadStructure | Self::PayloadStructureAndVConfig
        )
    }

    fn uses_havoc(&self) -> bool {
        matches!(self, Self::PayloadHavoc | Self::PayloadHavocAndVConfig)
    }
}

fn mutation_plan(selector: u64, vconfig_mutation: bool) -> RapidInputMutationPlan {
    if !vconfig_mutation {
        return if selector.is_multiple_of(2) {
            RapidInputMutationPlan::PayloadStructure
        } else {
            RapidInputMutationPlan::PayloadHavoc
        };
    }

    match selector % 5 {
        0 => RapidInputMutationPlan::PayloadStructure,
        1 => RapidInputMutationPlan::PayloadHavoc,
        2 => RapidInputMutationPlan::VConfig,
        3 => RapidInputMutationPlan::PayloadStructureAndVConfig,
        _ => RapidInputMutationPlan::PayloadHavocAndVConfig,
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RapidInputMutationMode {
    Mutating { vconfig_mutation: bool },
    Fixed,
}

#[derive(Debug)]
pub struct RapidInputMutator {
    mode: RapidInputMutationMode,
    structure: ArgPackStructureMutator,
    havoc: ArgPackHavocMutator,
    vconfig: VConfigMutator,
    normalize: ArgPackNormalizeMutator,
}

impl Default for RapidInputMutator {
    fn default() -> Self {
        Self::mutating(true)
    }
}

impl RapidInputMutator {
    pub fn mutating(vconfig_mutation: bool) -> Self {
        Self {
            mode: RapidInputMutationMode::Mutating { vconfig_mutation },
            structure: ArgPackStructureMutator::default(),
            havoc: ArgPackHavocMutator::default(),
            vconfig: VConfigMutator::default(),
            normalize: ArgPackNormalizeMutator,
        }
    }

    pub fn fixed() -> Self {
        Self {
            mode: RapidInputMutationMode::Fixed,
            structure: ArgPackStructureMutator::fixed(),
            havoc: ArgPackHavocMutator::fixed(),
            vconfig: VConfigMutator::fixed(),
            normalize: ArgPackNormalizeMutator,
        }
    }
}

impl Named for RapidInputMutator {
    fn name(&self) -> &Cow<'static, str> {
        static NAME: Cow<'static, str> = Cow::Borrowed("RapidInputMutator");
        &NAME
    }
}

fn combine_results(current: MutationResult, next: MutationResult) -> MutationResult {
    if current == MutationResult::Mutated || next == MutationResult::Mutated {
        MutationResult::Mutated
    } else {
        MutationResult::Skipped
    }
}

impl<S> Mutator<BytesInput, S> for RapidInputMutator
where
    S: HasRand,
{
    fn mutate(&mut self, state: &mut S, input: &mut BytesInput) -> Result<MutationResult, Error> {
        let mut result = MutationResult::Skipped;
        match self.mode {
            RapidInputMutationMode::Fixed => {
                result = self.structure.mutate_unrepaired(state, input)?;
            }
            RapidInputMutationMode::Mutating { vconfig_mutation } => {
                let plan = mutation_plan(state.rand_mut().next(), vconfig_mutation);
                if plan.uses_structure() {
                    let next = self.structure.mutate_unrepaired(state, input)?;
                    result = combine_results(result, next);
                }
                if plan.uses_havoc() {
                    let next = self.havoc.mutate_unrepaired(state, input)?;
                    result = combine_results(result, next);
                }
                if plan.mutates_vconfig() {
                    let next = self.vconfig.mutate_unrepaired(state, input)?;
                    result = combine_results(result, next);
                }
            }
        }
        let normalized = self.normalize.mutate(state, input)?;
        if matches!(
            self.mode,
            RapidInputMutationMode::Mutating {
                vconfig_mutation: false
            }
        ) {
            Ok(MutationResult::Mutated)
        } else {
            Ok(combine_results(result, normalized))
        }
    }

    fn post_exec(&mut self, state: &mut S, new_corpus_id: Option<CorpusId>) -> Result<(), Error> {
        self.structure.post_exec(state, new_corpus_id)?;
        self.havoc.post_exec(state, new_corpus_id)?;
        self.vconfig.post_exec(state, new_corpus_id)?;
        self.normalize.post_exec(state, new_corpus_id)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        marker::PhantomData,
        time::{SystemTime, UNIX_EPOCH},
    };

    use libafl::{
        inputs::{BytesInput, HasTargetBytes},
        mutators::{MutationResult, Mutator},
        state::HasRand,
    };
    use libafl_bolts::{
        rands::{Rand, StdRand},
        AsSlice,
    };

    use super::{mutation_plan, RapidInputMutator};
    use crate::arg_pack_v1::{
        default_seed_rapid_input_v1, init_arg_pack_manifest, normalize_rapid_input_v1,
        parse_rapid_input_envelope, test_normalize_calls_v1,
    };

    const TEST_MANIFEST: &str = r#"
{
  "schema_version": 1,
  "kernels": [
    {
      "symbol_name": "rapid_input_mutator_kernel",
      "display_name": "rapid_input_mutator_kernel",
      "args": [
        {
          "index": 0,
          "name": "count",
          "type": "uint32_t",
          "kind": "scalar",
          "size_bytes": 4,
          "align_bytes": 4,
          "domain": { "kind": "int_range", "min": "1", "max": "1", "signed": false }
        },
        {
          "index": 1,
          "name": "buffer",
          "type": "uint32_t *",
          "kind": "pointer",
          "pointer_role": "payload_buffer",
          "pointee_layout": {
            "index": "buffer.*",
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
            "min_len": "4",
            "max_len": "4",
            "elem_size_bytes": 4,
            "pattern_hex": "00000000"
          }
        }
      ],
      "constraints": [
        { "kind": "count_fits_buffer", "count_arg": 0, "buffer_arg": 1, "elem_size_bytes": 4 }
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
            "rapid-input-mutator-test-{}-{nanos}.json",
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
    fn vconfig_enabled_exposes_payload_vconfig_and_combined_plans() {
        let plans = (0..5)
            .map(|selector| mutation_plan(selector, true))
            .collect::<Vec<_>>();

        assert!(plans
            .iter()
            .any(|plan| plan.uses_structure() || plan.uses_havoc()));
        assert!(plans.iter().any(|plan| {
            !plan.uses_structure() && !plan.uses_havoc() && plan.mutates_vconfig()
        }));
        assert!(plans
            .iter()
            .any(|plan| (plan.uses_structure() || plan.uses_havoc()) && plan.mutates_vconfig()));
    }

    #[test]
    fn vconfig_disabled_exposes_only_payload_plans() {
        for selector in 0..32 {
            let plan = mutation_plan(selector, false);
            assert!(plan.uses_structure() || plan.uses_havoc());
            assert!(!plan.mutates_vconfig());
        }
    }

    #[test]
    fn fixed_mode_preserves_no_mutate_canonical_seed_contract() {
        init_test_manifest();
        let expected = normalize_rapid_input_v1(&default_seed_rapid_input_v1());
        let mut input = BytesInput::new(vec![0xff, 0xee, 0xdd]);
        let mut state = create_test_state();
        let mut mutator = RapidInputMutator::fixed();

        let result = mutator.mutate(&mut state, &mut input).unwrap();

        assert_eq!(result, MutationResult::Mutated);
        assert_eq!(input.target_bytes().as_slice(), expected.as_slice());
        assert!(parse_rapid_input_envelope(input.target_bytes().as_slice()).is_some());
    }

    #[test]
    fn payload_only_mode_reports_mutated_for_singleton_domain() {
        init_test_manifest();
        let seed = default_seed_rapid_input_v1();
        let initial_vconfig = parse_rapid_input_envelope(&seed).unwrap().vconfig;
        let mut input = BytesInput::new(seed);
        let mut state = create_test_state();
        let mut mutator = RapidInputMutator::mutating(false);

        for _ in 0..256 {
            let result = mutator.mutate(&mut state, &mut input).unwrap();
            let target = input.target_bytes();
            let bytes = target.as_slice();
            assert_eq!(result, MutationResult::Mutated);
            let envelope = parse_rapid_input_envelope(bytes).unwrap();
            assert_eq!(envelope.vconfig, initial_vconfig);
            assert_eq!(normalize_rapid_input_v1(bytes), bytes);
        }
    }

    #[test]
    fn each_mutation_finishes_with_exactly_one_normalize() {
        init_test_manifest();
        let seed = default_seed_rapid_input_v1();
        let mut state = create_test_state();
        let mut mutator = RapidInputMutator::default();

        for _ in 0..256 {
            let mut input = BytesInput::new(seed.clone());
            let before = test_normalize_calls_v1();

            let _ = mutator.mutate(&mut state, &mut input).unwrap();

            let after = test_normalize_calls_v1();
            assert_eq!(after - before, 1);
        }
    }
}
