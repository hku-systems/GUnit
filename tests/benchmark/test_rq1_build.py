import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark.rq1.artifacts import (
    hash_backend_report,
    hash_shared_phase2,
    verify_shared_phase2_unchanged,
)
from benchmark.rq1.build import (
    EXPECTED_BACKENDS,
    _validate_backend_report,
    create_build_plan,
)
from benchmark.rq1.schema import load_catalog


REPO_ROOT = Path(__file__).resolve().parents[2]
RQ1_ROOT = REPO_ROOT / "benchmark" / "rq1"


class Rq1BuildTests(unittest.TestCase):
    def _expected_launch_config(self) -> dict:
        return {
            "grid": [1, 1, 1],
            "block_candidates": [256],
            "physical_block_max": 256,
            "target_dynamic_shared_bytes": 2048,
            "coverage_memory": "global",
            "vconfig_reserved": False,
            "logical_grid": [1, 1, 1],
            "logical_block": [256, 1, 1],
            "has_logical_vconfig_bounds": False,
            "vconfig_enabled": False,
            "vconfig_mutation": True,
            "vconfig_warp_aligned": False,
        }

    def _backend_report(self, phase2_dir: Path, launch_config: dict) -> dict:
        return {
            "phase2_dir": str(phase2_dir),
            "kernel_id": "kernel__abc",
            "feedback_instrumentation": "enabled",
            "launch_config": launch_config,
            "build_profile": "release",
            "optimization": {
                "profile": "release",
                "device_clang_flags": ["-O3", "-DNDEBUG"],
                "host_cxx_flags": ["-O3", "-DNDEBUG"],
                "llvm_opt_pipeline": "default<O3>",
                "llc_flags": ["-O3"],
            },
            "commands": [["clang++"]],
        }

    def test_backend_artifact_hashes_include_optimized_device_bitcode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq1-backend-hash-") as temp_dir:
            backend_dir = Path(temp_dir)
            optimized_bc = backend_dir / "optimized.bc"
            module_ptx = backend_dir / "module.ptx"
            shared_lib = backend_dir / "target.so"
            optimized_bc.write_bytes(b"optimized-device-bc")
            module_ptx.write_text("// ptx\n", encoding="utf-8")
            shared_lib.write_bytes(b"shared-library")
            report_path = backend_dir / "backend_build.json"
            report_path.write_text(
                json.dumps(
                    {
                        "selected_device_bc": str(optimized_bc),
                        "module_ptx": str(module_ptx),
                        "shared_lib": str(shared_lib),
                    }
                ),
                encoding="utf-8",
            )

            artifacts = hash_backend_report(report_path)

            self.assertEqual(
                artifacts["optimized_device_bitcode"]["path"],
                "optimized.bc",
            )
            self.assertEqual(
                artifacts["optimized_device_bitcode"]["size_bytes"],
                len(b"optimized-device-bc"),
            )

    def test_build_plan_captures_and_rewrites_once(self) -> None:
        workload = load_catalog(RQ1_ROOT).workloads[0]
        plan = create_build_plan(
            workload,
            rq1_root=RQ1_ROOT,
            out_root=RQ1_ROOT / "build" / "test-plan",
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
        )

        self.assertEqual(len(plan.capture_commands), 1)
        self.assertEqual(len(plan.phase1_commands), 1)
        self.assertEqual(len(plan.constraint_commands), 1)
        self.assertEqual(len(plan.phase2_commands), 1)
        self.assertEqual(tuple(item.name for item in plan.backends), tuple(EXPECTED_BACKENDS))
        self.assertEqual(
            tuple((item.directory, item.feedback) for item in plan.backends),
            tuple(EXPECTED_BACKENDS.values()),
        )

    def test_default_build_plan_commands_are_byte_exact(self) -> None:
        workload = load_catalog(RQ1_ROOT).workloads[0]
        plan = create_build_plan(
            workload,
            rq1_root=RQ1_ROOT,
            out_root=RQ1_ROOT / "build" / "test-plan",
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
        )
        payload = {
            "capture": plan.capture_commands,
            "phase1": plan.phase1_commands,
            "constraints": plan.constraint_commands,
            "phase2": plan.phase2_commands,
            "backends": tuple(
                (item.name, item.directory, item.feedback, item.command)
                for item in plan.backends
            ),
        }

        def normalize(value):
            if isinstance(value, str):
                if Path(value).is_absolute() and Path(value).name.startswith("python"):
                    return "$PYTHON"
                return value.replace(str(REPO_ROOT), "$REPO")
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        encoded = json.dumps(
            normalize(payload), sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "a4f98aa623c2010a31ae1d7164cad96a5717a8c80e7f8c4507e67646d88f0c72",
        )

    def test_build_plan_uses_explicit_release_o3_for_every_backend(self) -> None:
        workload = load_catalog(RQ1_ROOT).workloads[0]
        plan = create_build_plan(
            workload,
            rq1_root=RQ1_ROOT,
            out_root=RQ1_ROOT / "build" / "test-plan",
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
            build_profile="release",
        )

        self.assertEqual(plan.build_profile, "release")
        self.assertIn("-O3", plan.capture_commands[0])
        self.assertIn("-DNDEBUG", plan.capture_commands[0])
        for backend in plan.backends:
            with self.subTest(backend=backend.name):
                self.assertIn("--build-profile", backend.command)
                self.assertEqual(
                    backend.command[backend.command.index("--build-profile") + 1],
                    "release",
                )

    def test_rq1_build_plan_rejects_debug_profile(self) -> None:
        workload = load_catalog(RQ1_ROOT).workloads[0]

        with self.assertRaisesRegex(RuntimeError, "RQ1 requires release"):
            create_build_plan(
                workload,
                rq1_root=RQ1_ROOT,
                out_root=RQ1_ROOT / "build" / "test-plan",
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
                build_profile="debug",
            )

    def test_build_plan_rejects_explicit_empty_backend_selection(self) -> None:
        workload = load_catalog(RQ1_ROOT).workloads[0]

        with self.assertRaisesRegex(RuntimeError, "at least one backend"):
            create_build_plan(
                workload,
                rq1_root=RQ1_ROOT,
                out_root=RQ1_ROOT / "build" / "test-plan",
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
                backend_names=(),
            )

    def test_backend_matrix_has_exact_feedback_modes(self) -> None:
        self.assertEqual(
            EXPECTED_BACKENDS,
            {
                "cufuzz": ("cufuzz", "disabled"),
                "origin-no-feedback": ("origin", "disabled"),
                "origin": ("origin", "enabled"),
                "rapid-no-feedback": ("rapid", "disabled"),
                "rapid": ("rapid", "enabled"),
                "rapid2-no-feedback": ("rapid2", "disabled"),
                "rapid2": ("rapid2", "enabled"),
            },
        )

    def test_backend_report_launch_config_allows_known_vconfig_keys(self) -> None:
        phase2_dir = (RQ1_ROOT / "build" / "dummy-phase2").resolve()
        expected_launch = self._expected_launch_config()
        backend = SimpleNamespace(name="origin", feedback="enabled")
        report = self._backend_report(phase2_dir, dict(expected_launch))

        _validate_backend_report(
            report=report,
            backend=backend,
            phase2_dir=phase2_dir,
            kernel_id="kernel__abc",
            expected_launch=expected_launch,
            build_profile="release",
        )

    def test_backend_report_launch_config_rejects_contract_drift(self) -> None:
        phase2_dir = (RQ1_ROOT / "build" / "dummy-phase2").resolve()
        expected_launch = self._expected_launch_config()
        backend = SimpleNamespace(name="origin", feedback="enabled")
        report = self._backend_report(
            phase2_dir,
            {
                **expected_launch,
                "physical_block_max": 128,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "launch config drift"):
            _validate_backend_report(
                report=report,
                backend=backend,
                phase2_dir=phase2_dir,
                kernel_id="kernel__abc",
                expected_launch=expected_launch,
                build_profile="release",
            )

    def test_backend_report_launch_config_rejects_unknown_key(self) -> None:
        phase2_dir = (RQ1_ROOT / "build" / "dummy-phase2").resolve()
        expected_launch = self._expected_launch_config()
        backend = SimpleNamespace(name="origin", feedback="enabled")
        report = self._backend_report(
            phase2_dir,
            {**expected_launch, "future_extension": True},
        )

        with self.assertRaisesRegex(RuntimeError, "unknown launch config key"):
            _validate_backend_report(
                report=report,
                backend=backend,
                phase2_dir=phase2_dir,
                kernel_id="kernel__abc",
                expected_launch=expected_launch,
                build_profile="release",
            )

    def test_backend_report_launch_config_rejects_known_vconfig_drift(self) -> None:
        phase2_dir = (RQ1_ROOT / "build" / "dummy-phase2").resolve()
        expected_launch = self._expected_launch_config()
        backend = SimpleNamespace(name="origin", feedback="enabled")
        drifted = self._backend_report(
            phase2_dir,
            {**expected_launch, "vconfig_enabled": True},
        )
        with self.assertRaisesRegex(RuntimeError, "launch config drift"):
            _validate_backend_report(
                report=drifted,
                backend=backend,
                phase2_dir=phase2_dir,
                kernel_id="kernel__abc",
                expected_launch=expected_launch,
                build_profile="release",
            )

    def test_shared_phase2_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq1-phase2-drift-") as temp_dir:
            kernel_dir = Path(temp_dir) / "kernel"
            phase2_dir = kernel_dir / "phase2"
            gen_dir = phase2_dir / "gen"
            gen_dir.mkdir(parents=True)
            (kernel_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            (phase2_dir / "build_spec.json").write_text("{}\n", encoding="utf-8")
            (phase2_dir / "kernel.device.bc").write_bytes(b"device-bc")
            (gen_dir / "fuzzer_decode.v1.cuh").write_text("decode\n", encoding="utf-8")
            invoke = gen_dir / "fuzzer_invoke.v1.cuh"
            invoke.write_text("invoke\n", encoding="utf-8")
            expected = hash_shared_phase2(phase2_dir)

            invoke.write_text("drifted\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError,
                "shared Phase2 artifact drift: gen/fuzzer_invoke.v1.cuh",
            ):
                verify_shared_phase2_unchanged(phase2_dir, expected)

    def test_build_plan_backend_commands_share_one_phase2_placeholder(self) -> None:
        workload = load_catalog(RQ1_ROOT).workloads[3]
        plan = create_build_plan(
            workload,
            rq1_root=RQ1_ROOT,
            out_root=RQ1_ROOT / "build" / "test-plan",
            cuda_path="/cuda",
            cuda_arch="sm_86",
        )

        for backend in plan.backends:
            with self.subTest(backend=backend.name):
                self.assertEqual(backend.phase2_dir, plan.phase2_dir)
                self.assertIn("--phase2-dir", backend.command)
                self.assertEqual(
                    backend.command[backend.command.index("--phase2-dir") + 1],
                    str(plan.phase2_dir),
                )


if __name__ == "__main__":
    unittest.main()
