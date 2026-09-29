import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmark.rq2 import build as rq2_build
from benchmark.rq1.schema import sha256_file
from benchmark.rq2.build import (
    RQ2_BACKENDS,
    build_workload,
    create_build_plan,
    write_build_summary,
)
from benchmark.rq2.schema import load_catalog


REPO_ROOT = Path(__file__).resolve().parents[2]
RQ2_ROOT = REPO_ROOT / "benchmark" / "rq2"


class Rq2BuildTests(unittest.TestCase):
    def _write_mock_build_outputs(
        self,
        workload,
        out_root: Path,
        *,
        build_spec: dict,
        manifest: dict | None = None,
        registry_sha256: str | None = None,
    ) -> Path:
        report_path = out_root / workload.workload_id / "build_report.json"
        resolved_run_dir = report_path.parent / "resolved"
        phase2_dir = resolved_run_dir / "selected-kernel" / "phase2"
        phase2_dir.mkdir(parents=True)
        (phase2_dir / "build_spec.json").write_text(
            json.dumps(build_spec),
            encoding="utf-8",
        )
        if manifest is not None:
            (phase2_dir.parent / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
        (resolved_run_dir / "constraint_overrides.report.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "registry_sha256": registry_sha256
                    or f"sha256:{sha256_file(workload.constraints_path)}",
                }
            ),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workload_id": workload.workload_id,
                    "shared_phase2": {"phase2_dir": str(phase2_dir)},
                    "backends": {},
                }
            ),
            encoding="utf-8",
        )
        return report_path

    def _write_summary_reports(self, run_root: Path) -> tuple[Path, ...]:
        reports = tuple(
            run_root / workload.workload_id / "build_report.json"
            for workload in load_catalog(RQ2_ROOT).workloads
        )
        for workload, report_path in zip(
            load_catalog(RQ2_ROOT).workloads,
            reports,
        ):
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workload_id": workload.workload_id,
                    }
                ),
                encoding="utf-8",
            )
        return reports

    def test_plan_uses_rq2_inputs_four_backends_and_one_phase2(self) -> None:
        workload = load_catalog(RQ2_ROOT).workloads[0]
        plan = create_build_plan(
            workload,
            out_root=RQ2_ROOT / "build" / "test-plan",
            cuda_path="/usr/local/cuda",
            cuda_arch="sm_86",
        )

        capture = plan.capture_commands[0]
        self.assertEqual(capture[capture.index("-c") + 1], str(workload.kernel_path))
        if workload.vconfig_adaptation is None:
            self.assertNotEqual(workload.kernel_path, workload.kernel_path.resolve())
        else:
            self.assertEqual(workload.kernel_path, workload.kernel_path.resolve())
        constraints = plan.constraint_commands[0]
        self.assertEqual(
            constraints[constraints.index("--overrides") + 1],
            str(workload.constraints_path),
        )
        self.assertEqual(
            tuple(item.name for item in plan.backends),
            RQ2_BACKENDS,
        )
        self.assertEqual(
            tuple((item.directory, item.feedback) for item in plan.backends),
            (
                ("cufuzz", "disabled"),
                ("origin", "enabled"),
                ("rapid", "enabled"),
                ("rapid2", "enabled"),
            ),
        )
        self.assertNotIn("origin-no-feedback", RQ2_BACKENDS)
        self.assertNotIn("rapid-no-feedback", RQ2_BACKENDS)
        self.assertNotIn("rapid2-no-feedback", RQ2_BACKENDS)
        for backend in plan.backends:
            with self.subTest(backend=backend.name):
                self.assertEqual(backend.phase2_dir, plan.phase2_dir)
                self.assertEqual(
                    backend.command[backend.command.index("--phase2-dir") + 1],
                    str(plan.phase2_dir),
                )

    def test_flashattention_phase1_discovers_scalar_n(self) -> None:
        workload = next(
            item
            for item in load_catalog(RQ2_ROOT).workloads
            if item.workload_id == "flashattention_device1xn"
        )
        with tempfile.TemporaryDirectory(prefix="rq2-flash-phase1-") as temp_dir:
            plan = create_build_plan(
                workload,
                out_root=Path(temp_dir),
                cuda_path="/usr/local/cuda",
                cuda_arch="sm_86",
            )
            (plan.work_root / "capture").mkdir(parents=True)
            (plan.work_root / "capture-build").mkdir(parents=True)
            env = os.environ.copy()
            env["RAPID_CAPTURE_DIR"] = str(plan.work_root / "capture")
            for command in (*plan.capture_commands, *plan.phase1_commands):
                subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

            index = json.loads(
                (plan.raw_run_dir / "index.json").read_text(encoding="utf-8")
            )
            kernel_dir = plan.raw_run_dir / index["kernels"][0]["dir"]
            manifest = json.loads(
                (kernel_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [
                    (arg["index"], arg["name"], arg["type"], arg["kind"])
                    for arg in manifest["kernels"][0]["args"]
                ],
                [
                    (0, "input", "const float *", "pointer"),
                    (1, "output", "float *", "pointer"),
                    (2, "n", "const unsigned int", "scalar"),
                ],
            )

    def test_flashattention_build_contract_adapts_only_the_physical_envelope(
        self,
    ) -> None:
        workload = next(
            item
            for item in load_catalog(RQ2_ROOT).workloads
            if item.workload_id == "flashattention_device1xn"
        )
        original_path = workload.rq1_provenance_path
        original_sha256 = sha256_file(original_path)
        original = json.loads(original_path.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory(prefix="rq2-flash-build-contract-") as temp_dir:
            rq1_workload, rq1_root, adapter_path = rq2_build._rq1_build_contract(
                workload,
                Path(temp_dir),
            )
            adapted_path = rq1_root / rq1_workload.provenance_path
            adapted = json.loads(adapted_path.read_text(encoding="utf-8"))

            self.assertEqual(adapter_path, adapted_path)

        self.assertEqual(sha256_file(original_path), original_sha256)
        self.assertEqual(adapted["execution"]["block"], [256, 1, 1])
        self.assertEqual(adapted["execution"]["vconfig"]["block"], [128, 1, 1])
        adapted["execution"] = original["execution"]
        self.assertEqual(adapted, original)

    def test_flashattention_build_report_separates_rq1_and_rq2_contracts(self) -> None:
        workload = next(
            item
            for item in load_catalog(RQ2_ROOT).workloads
            if item.workload_id == "flashattention_device1xn"
        )
        rq1_provenance = json.loads(
            workload.rq1_provenance_path.read_text(encoding="utf-8")
        )
        constraints = json.loads(workload.constraints_path.read_text(encoding="utf-8"))
        override = constraints["overrides"][0]
        manifest_args = [
            {
                "index": domain["arg"],
                "name": domain["name"],
                "type": domain["type"],
                "kind": (
                    "pointer"
                    if domain["domain"]["kind"] == "bytes"
                    else "scalar"
                ),
                "domain": domain["domain"],
            }
            for domain in override["domains"]
        ]

        with tempfile.TemporaryDirectory(prefix="rq2-flash-build-report-") as temp_dir:
            out_root = Path(temp_dir)
            report_path = self._write_mock_build_outputs(
                workload,
                out_root,
                build_spec={"vconfig_requested": True, "vconfig_enabled": True},
                manifest={
                    "schema_version": 1,
                    "kernels": [
                        {
                            "symbol_name": override["symbol_name"],
                            "args": manifest_args,
                            "constraints": override["constraints"],
                            "launch_policy": override["launch_policy"],
                        }
                    ],
                },
            )
            report_path.write_text(
                json.dumps(
                    {
                        **json.loads(report_path.read_text(encoding="utf-8")),
                        "execution_contract": {
                            **rq1_provenance["execution"],
                            "block": [256, 1, 1],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "benchmark.rq2.build.rq1_build.build_workload",
                return_value=report_path,
            ) as rq1_build_workload:
                build_workload(
                    workload,
                    out_root=out_root,
                    cuda_path="/cuda",
                    cuda_arch="sm_86",
                )

            call = rq1_build_workload.call_args
            self.assertNotEqual(call.kwargs["rq1_root"], rq2_build.RQ1_ROOT)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["source_provenance"],
                str(workload.rq1_provenance_path.resolve()),
            )
            self.assertEqual(report["execution_contract"], rq1_provenance["execution"])
            self.assertEqual(
                report["rq2"]["execution_contract"],
                {
                    "entry": "rq1_flashattention_device1xn",
                    "arguments": manifest_args,
                    "constraints": override["constraints"],
                    "launch_policy": override["launch_policy"],
                },
            )
            adapter = report["rq2"]["build_contract_adapter"]
            self.assertEqual(adapter["field"], "execution.block")
            self.assertEqual(adapter["rq1_value"], [128, 1, 1])
            self.assertEqual(adapter["rq2_value"], [256, 1, 1])
            self.assertEqual(
                adapter["sha256"],
                f"sha256:{sha256_file(Path(adapter['path']))}",
            )

    def test_build_workload_enriches_report_and_passes_explicit_overrides(self) -> None:
        workload = load_catalog(RQ2_ROOT).workloads[0]
        with tempfile.TemporaryDirectory(prefix="rq2-build-report-") as temp_dir:
            out_root = Path(temp_dir)
            report_path = self._write_mock_build_outputs(
                workload,
                out_root,
                build_spec={
                    "vconfig_requested": True,
                    "vconfig_enabled": False,
                    "vconfig_disabled_reason": "vconfig_barrier_unsupported",
                },
            )

            with patch(
                "benchmark.rq2.build.rq1_build.build_workload",
                return_value=report_path,
            ) as rq1_build_workload:
                actual_path = build_workload(
                    workload,
                    out_root=out_root,
                    cuda_path="/cuda",
                    cuda_arch="sm_86",
                )

            self.assertEqual(actual_path, report_path)
            call = rq1_build_workload.call_args
            self.assertEqual(call.kwargs["source_path_override"], workload.kernel_path)
            self.assertEqual(
                call.kwargs["constraints_path_override"], workload.constraints_path
            )
            self.assertEqual(call.kwargs["backend_names"], RQ2_BACKENDS)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["rq2"],
                {
                    "constraints_path": str(workload.constraints_path),
                    "constraints_sha256": (
                        f"sha256:{sha256_file(workload.constraints_path)}"
                    ),
                    "source_path": str(workload.kernel_path),
                    "source_resolved_path": str(workload.kernel_path.resolve()),
                    "vconfig_disabled_reason": "vconfig_barrier_unsupported",
                    "vconfig_effective": False,
                    "vconfig_requested": True,
                },
            )

    def test_build_workload_rejects_applied_constraint_digest_mismatch(self) -> None:
        workload = load_catalog(RQ2_ROOT).workloads[0]
        with tempfile.TemporaryDirectory(prefix="rq2-build-stale-applied-") as temp_dir:
            out_root = Path(temp_dir)
            report_path = self._write_mock_build_outputs(
                workload,
                out_root,
                build_spec={
                    "vconfig_requested": True,
                    "vconfig_enabled": True,
                },
                registry_sha256="sha256:" + "0" * 64,
            )

            with patch(
                "benchmark.rq2.build.rq1_build.build_workload",
                return_value=report_path,
            ):
                with self.assertRaisesRegex(RuntimeError, "constraint digest"):
                    build_workload(
                        workload,
                        out_root=out_root,
                        cuda_path="/cuda",
                        cuda_arch="sm_86",
                    )

    def test_build_workload_rejects_constraint_changed_during_build(self) -> None:
        catalog_workload = load_catalog(RQ2_ROOT).workloads[0]
        with tempfile.TemporaryDirectory(prefix="rq2-build-mutated-constraint-") as temp_dir:
            out_root = Path(temp_dir)
            workload = replace(catalog_workload, root=out_root / "rq2")
            workload.constraints_path.parent.mkdir(parents=True)
            workload.constraints_path.write_text('{"version": 1}\n', encoding="utf-8")
            expected_digest = f"sha256:{sha256_file(workload.constraints_path)}"
            report_path = self._write_mock_build_outputs(
                workload,
                out_root,
                build_spec={
                    "vconfig_requested": True,
                    "vconfig_enabled": True,
                },
                registry_sha256=expected_digest,
            )

            def mutate_constraints(*_args, **_kwargs):
                workload.constraints_path.write_text(
                    '{"version": 2}\n',
                    encoding="utf-8",
                )
                return report_path

            with patch(
                "benchmark.rq2.build.rq1_build.build_workload",
                side_effect=mutate_constraints,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during build"):
                    build_workload(
                        workload,
                        out_root=out_root,
                        cuda_path="/cuda",
                        cuda_arch="sm_86",
                    )

    def test_build_workload_rejects_invalid_rq2_vconfig_states(self) -> None:
        workload = load_catalog(RQ2_ROOT).workloads[0]
        invalid_specs = (
            (
                "not-requested",
                {
                    "vconfig_requested": False,
                    "vconfig_enabled": False,
                    "vconfig_disabled_reason": "vconfig_barrier_unsupported",
                },
            ),
            (
                "enabled-with-disabled-reason",
                {
                    "vconfig_requested": True,
                    "vconfig_enabled": True,
                    "vconfig_disabled_reason": "vconfig_barrier_unsupported",
                },
            ),
            (
                "enabled-with-null-disabled-reason",
                {
                    "vconfig_requested": True,
                    "vconfig_enabled": True,
                    "vconfig_disabled_reason": None,
                },
            ),
            (
                "disabled-without-reason",
                {
                    "vconfig_requested": True,
                    "vconfig_enabled": False,
                },
            ),
            (
                "disabled-with-unknown-reason",
                {
                    "vconfig_requested": True,
                    "vconfig_enabled": False,
                    "vconfig_disabled_reason": "vconfig_unknown_unsupported",
                },
            ),
        )
        for label, build_spec in invalid_specs:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory(
                    prefix="rq2-build-vconfig-state-"
                ) as temp_dir:
                    out_root = Path(temp_dir)
                    report_path = self._write_mock_build_outputs(
                        workload,
                        out_root,
                        build_spec=build_spec,
                    )
                    with patch(
                        "benchmark.rq2.build.rq1_build.build_workload",
                        return_value=report_path,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "RQ2 VConfig"):
                            build_workload(
                                workload,
                                out_root=out_root,
                                cuda_path="/cuda",
                                cuda_arch="sm_86",
                            )

    def test_build_workload_accepts_supported_rq2_vconfig_states(self) -> None:
        workload = load_catalog(RQ2_ROOT).workloads[0]
        valid_specs = (
            (
                {"vconfig_requested": True, "vconfig_enabled": True},
                True,
                None,
            ),
            (
                {
                    "vconfig_requested": True,
                    "vconfig_enabled": False,
                    "vconfig_disabled_reason": "vconfig_inline_asm_unsupported",
                },
                False,
                "vconfig_inline_asm_unsupported",
            ),
        )
        for build_spec, expected_effective, expected_reason in valid_specs:
            with self.subTest(build_spec=build_spec):
                with tempfile.TemporaryDirectory(
                    prefix="rq2-build-valid-vconfig-state-"
                ) as temp_dir:
                    out_root = Path(temp_dir)
                    report_path = self._write_mock_build_outputs(
                        workload,
                        out_root,
                        build_spec=build_spec,
                    )
                    with patch(
                        "benchmark.rq2.build.rq1_build.build_workload",
                        return_value=report_path,
                    ):
                        build_workload(
                            workload,
                            out_root=out_root,
                            cuda_path="/cuda",
                            cuda_arch="sm_86",
                        )
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertIs(report["rq2"]["vconfig_requested"], True)
                    self.assertIs(
                        report["rq2"]["vconfig_effective"],
                        expected_effective,
                    )
                    self.assertEqual(
                        report["rq2"]["vconfig_disabled_reason"],
                        expected_reason,
                    )

    def test_build_summary_requires_and_records_all_seven_reports(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-build-summary-") as temp_dir:
            run_root = Path(temp_dir)
            reports = self._write_summary_reports(run_root)
            self.assertEqual(len(reports), 7)

            summary_path = write_build_summary(run_root, reports, "release")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["build_profile"], "release")
            self.assertEqual(summary["reports"], [str(path) for path in reports])
            with self.assertRaisesRegex(RuntimeError, "exactly 7 build reports"):
                write_build_summary(run_root, reports[:6], "release")

    def test_build_summary_rejects_incomplete_or_misordered_reports(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-build-summary-invalid-") as temp_dir:
            run_root = Path(temp_dir)
            reports = self._write_summary_reports(run_root)

            invalid_cases = (
                ("duplicate", (*reports[:-1], reports[0])),
                (
                    "missing",
                    (*reports[:-1], run_root / "missing" / "build_report.json"),
                ),
                ("out-of-order", tuple(reversed(reports))),
            )
            for label, invalid_reports in invalid_cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, "RQ2 summary"):
                        write_build_summary(run_root, invalid_reports, "release")

            final_report = reports[-1]
            final_report.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workload_id": "wrong-workload",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "RQ2 summary"):
                write_build_summary(run_root, reports, "release")


if __name__ == "__main__":
    unittest.main()
