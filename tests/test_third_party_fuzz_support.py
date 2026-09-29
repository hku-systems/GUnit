import json
import tempfile
import unittest
from pathlib import Path

from scripts.third_party_fuzz.support import (
    classify_kernel,
    load_support_registry,
    validate_support_coverage,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _rewrite_summary(*symbols: str) -> dict:
    return {
        "schema_version": 1,
        "results": [
            {
                "symbol_name": symbol,
                "status": "built",
            }
            for symbol in symbols
        ],
    }


def _registry(*items: dict) -> dict:
    return {
        "schema_version": 1,
        "project": "cudasift",
        "kernels": list(items),
    }


def _run_symbol(symbol: str = "_Z3runv") -> dict:
    return {
        "symbol_name": symbol,
        "decision": "run",
        "evidence": [
            {
                "kind": "kernel_body",
                "file": "third_party/example.cu",
                "line": 10,
                "note": "bounds checked one-dimensional kernel",
            }
        ],
    }


def _skip_symbol(symbol: str = "_Z4skipv") -> dict:
    return {
        "symbol_name": symbol,
        "decision": "skip",
        "reason_code": "vconfig_required",
        "detail": "uses threadIdx.y in a native 16x16 launch",
        "evidence": [
            {
                "kind": "launchsite",
                "file": "third_party/example.cu",
                "line": 20,
                "note": "dim3 threads(16, 16)",
            }
        ],
    }


class ThirdPartyFuzzSupportTest(unittest.TestCase):
    def test_loads_run_and_skip_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "support.json"
            _write_json(path, _registry(_run_symbol(), _skip_symbol()))

            decisions = load_support_registry(path)

            self.assertEqual(decisions["_Z3runv"].decision, "run")
            self.assertIsNone(decisions["_Z3runv"].reason_code)
            self.assertEqual(decisions["_Z4skipv"].reason_code, "vconfig_required")

    def test_rejects_duplicate_unknown_and_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            duplicate = root / "duplicate.json"
            _write_json(duplicate, _registry(_run_symbol(), _run_symbol()))
            with self.assertRaisesRegex(ValueError, "duplicate support decision symbol"):
                load_support_registry(duplicate)

            missing_evidence = root / "missing-evidence.json"
            item = _run_symbol()
            item["evidence"] = []
            _write_json(missing_evidence, _registry(item))
            with self.assertRaisesRegex(ValueError, "must include evidence"):
                load_support_registry(missing_evidence)

            run_with_reason = root / "run-with-reason.json"
            item = _run_symbol()
            item["reason_code"] = "vconfig_required"
            _write_json(run_with_reason, _registry(item))
            with self.assertRaisesRegex(ValueError, "run decision must not carry skip reason"):
                load_support_registry(run_with_reason)

    def test_validates_exact_phase2_built_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "support.json"
            _write_json(path, _registry(_run_symbol("_Z3onev")))
            decisions = load_support_registry(path)

            validate_support_coverage(
                Path(td),
                _rewrite_summary("_Z3onev"),
                decisions,
            )

            with self.assertRaisesRegex(ValueError, "missing support decisions"):
                validate_support_coverage(
                    Path(td),
                    _rewrite_summary("_Z3onev", "_Z3twov"),
                    decisions,
                )

            with self.assertRaisesRegex(ValueError, "unknown support decisions"):
                validate_support_coverage(
                    Path(td),
                    _rewrite_summary(),
                    decisions,
                )

    def test_rejects_rewrite_summary_without_results_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "rewrite summary results must be a list"):
                validate_support_coverage(
                    Path(td),
                    {
                        "schema_version": 1,
                        "kernels": [
                            {
                                "symbol_name": "_Z3onev",
                                "phase2_status": "built",
                            }
                        ],
                    },
                    {},
                )

    def test_classify_kernel_only_accepts_phase2_built_symbols(self) -> None:
        decisions = {"_Z3runv": load_support_registry_from_dict(_registry(_run_symbol()))["_Z3runv"]}

        self.assertEqual(
            classify_kernel("_Z3runv", "built", decisions).decision,
            "run",
        )
        with self.assertRaisesRegex(ValueError, "only accepts Phase 2-built"):
            classify_kernel("_Z3runv", "failed", decisions)
        with self.assertRaisesRegex(ValueError, "missing support decision"):
            classify_kernel("_Z7missingv", "built", decisions)

    def test_zero_coeff_count_kernel_is_recorded_as_target_bug_candidate(self) -> None:
        decisions = load_support_registry(Path("third_party/fuzz/support/phantom_fhe.json"))
        decision = decisions["_Z23zero_coeff_count_kernelPjPKmm__4fc80c15"]

        self.assertEqual(decision.decision, "skip")
        self.assertEqual(decision.reason_code, "target_bug_candidate_divergent_barrier")
        self.assertIn("tid < ct1_size", decision.detail or "")
        self.assertTrue(
            any(
                item["file"] == "third_party/phantom-fhe/src/polymath.cu"
                and item["line"] == 667
                for item in decision.evidence
            )
        )

    def test_real_tensor_rt_and_cutlass_support_use_call_site_evidence(self) -> None:
        cases = [
            (
                Path("third_party/fuzz/support/tensorrt_clip.json"),
                "_Z10clipKernelIffLj512EEviT_S0_PKT0_PS1___c1c27f18",
                "third_party/TensorRT/plugin/clipPlugin/clip.cu",
                62,
            ),
            (
                Path("third_party/fuzz/support/cutlass_basic.json"),
                "_Z20ReferenceGemm_kerneliiifPKfiS0_ifPfi__efdc1030",
                "third_party/cutlass/examples/00_basic_gemm/basic_gemm.cu",
                278,
            ),
        ]
        for registry_path, kernel_id, file, line in cases:
            with self.subTest(registry=registry_path, kernel_id=kernel_id):
                decisions = load_support_registry(registry_path)
                decision = decisions[kernel_id]

                self.assertEqual(decision.decision, "run")
                self.assertTrue(
                    any(
                        item["kind"] == "call_site"
                        and item["file"] == file
                        and item["line"] == line
                        for item in decision.evidence
                    )
                )

    def test_real_gpurir_support_distinguishes_extern_shared_from_vconfig(self) -> None:
        decisions = load_support_registry(Path("third_party/fuzz/support/gpurir.json"))

        self.assertEqual(decisions["_Z16reduceRIR_kernelPfS_iiii__a6bf7d18"].decision, "run")
        for kernel_id in (
            "_Z14diffRev_kernelPfS_S_S_iiii__e9523a2e",
            "_Z14envPred_kernelPfS_S_S_iiiffffffffff__4f38b2df",
            "_Z17calcAmpTau_kernelPfS_S_S_S_S_S_iifffffffffiiiiiff__169eb460",
            "_Z18generateRIR_kernelPfS_S_iiiiif__cd099106",
            "_Z24h2RIR_to_floatRIR_kernelP7__half2Pfii__d0f15d1b",
        ):
            with self.subTest(kernel_id=kernel_id):
                self.assertEqual(decisions[kernel_id].decision, "run")
                self.assertIsNone(decisions[kernel_id].reason_code)
        self.assertEqual(
            decisions["_Z19reduceRIR_mp_kernelP7__half2S0_iiii__ae859193"].reason_code,
            "protected_data_field",
        )

    def test_real_cudasift_support_runs_validated_vconfig_subset(self) -> None:
        decisions = load_support_registry(Path("third_party/fuzz/support/cudasift.json"))
        run_decisions = [item for item in decisions.values() if item.decision == "run"]
        skip_decisions = [item for item in decisions.values() if item.decision == "skip"]

        self.assertEqual(len(run_decisions), 14)
        self.assertEqual(len(skip_decisions), 1)
        self.assertEqual(
            skip_decisions[0].kernel_id,
            "_Z12FindMaxCorr3PfP9SiftPointS1_ii__079ca3bd",
        )
        self.assertEqual(skip_decisions[0].reason_code, "vconfig_barrier_unsupported")

        registry = json.loads(
            Path("third_party/fuzz/kernel_constraints/cudasift.json").read_text()
        )
        match_sift_points2 = next(
            item
            for item in registry["overrides"]
            if item["kernel_id"] == "_Z16MatchSiftPoints2P9SiftPointS0_Pfii__35c51f10"
        )
        self.assertTrue(match_sift_points2["launch_policy"]["vconfig_mutation"])

    def test_real_phantom_support_runs_radix2_opt_with_fixed_native_shape(self) -> None:
        decisions = load_support_registry(Path("third_party/fuzz/support/phantom_fhe.json"))
        decision = decisions["_Z23inplace_fnwt_radix2_optPmPKmS1_PK8DModulusm__39057492"]

        self.assertEqual(decision.decision, "run")
        self.assertIn("whole-warp block.x=32", decision.evidence[0]["note"])

    def test_real_phantom_support_runs_static_launch_policy_shapes_from_call_sites(self) -> None:
        decisions = load_support_registry(Path("third_party/fuzz/support/phantom_fhe.json"))
        expected = {
            "_Z29decompose_array_uint64_kernelPmPK7double2PK8DModulusj__60845b2f": (
                "third_party/phantom-fhe/src/rns_base.cu",
                157,
            ),
            "_Z30decompose_array_uint128_kernelPmPK7double2PK8DModulusj__4ca4c9ad": (
                "third_party/phantom-fhe/src/rns_base.cu",
                157,
            ),
            "_ZL29moddown_bconv_single_p_kernelPmPKmmPK8DModulusm__0c521208": (
                "third_party/phantom-fhe/src/rns_bconv.cu",
                796,
            ),
        }

        for kernel_id, (call_site_file, call_site_line) in expected.items():
            with self.subTest(kernel_id=kernel_id):
                decision = decisions[kernel_id]

                self.assertEqual(decision.decision, "run")
                self.assertIsNone(decision.reason_code)
                self.assertTrue(
                    any(
                        item["kind"] == "call_site"
                        and item["file"] == call_site_file
                        and item["line"] == call_site_line
                        for item in decision.evidence
                    )
                )


def load_support_registry_from_dict(data: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "support.json"
        _write_json(path, data)
        return load_support_registry(path)


if __name__ == "__main__":
    unittest.main()
