import json
import tempfile
import unittest
from pathlib import Path

from scripts.third_party_fuzz.report import render_markdown, summarize_results, write_reports


class ThirdPartyFuzzReportTest(unittest.TestCase):
    def test_structural_skips_do_not_lower_fuzzer_normality(self) -> None:
        summary = summarize_results(
            [
                _record("run_kernel", support_decision="run", backend=True, fixed=True, mutation=True),
                _record("vconfig_kernel", support_decision="skip", skip_reason="vconfig_required"),
                _record("phase2_failed", phase2_status="failed", support_decision="skip", skip_reason="protected_data_field"),
            ]
        )

        self.assertTrue(summary["fuzzer_normal"])
        self.assertEqual(summary["runnable_kernels"], 1)
        self.assertEqual(summary["runtime_passed"], 1)
        self.assertEqual(summary["skipped_kernels"], 2)

    def test_any_runnable_runtime_failure_marks_fuzzer_not_normal(self) -> None:
        summary = summarize_results(
            [
                _record("ok", support_decision="run", backend=True, fixed=True, mutation=True),
                _record("bad", support_decision="run", backend=True, fixed=True, mutation=False),
            ]
        )

        self.assertFalse(summary["fuzzer_normal"])
        self.assertEqual(summary["runtime_failed"], 1)

    def test_summary_counts_runtime_skips_and_mutation_failures_separately(self) -> None:
        records = [
            _record("ok", support_decision="run", backend=True, fixed=True, mutation=True),
            _record(
                "runtime_skip",
                support_decision="run",
                backend=True,
                fixed=False,
                mutation=None,
                runtime_decision="skip",
                runtime_skip_reason="launch_out_of_resources",
            ),
            _record(
                "mutation_bad",
                support_decision="run",
                backend=True,
                fixed=True,
                mutation=False,
                mutation_failure_kind="cuda_illegal_memory_access",
            ),
        ]

        summary = summarize_results(records)

        self.assertEqual(summary["fixed_passed"], 2)
        self.assertEqual(summary["runtime_skipped"], 1)
        self.assertEqual(summary["runtime_skip_reasons"]["launch_out_of_resources"], 1)
        self.assertEqual(summary["mutation_passed"], 1)
        self.assertEqual(summary["mutation_failed"], 1)
        self.assertEqual(summary["mutation_failure_kinds"]["cuda_illegal_memory_access"], 1)

    def test_summary_separates_incomplete_mutation_from_runtime_failures(self) -> None:
        records = [
            _record("ok", support_decision="run", backend=True, fixed=True, mutation=True),
            _record("not_yet_mutated", support_decision="run", backend=True, fixed=True, mutation=None),
            _record(
                "mutation_bad",
                support_decision="run",
                backend=True,
                fixed=True,
                mutation=False,
                mutation_failure_kind="cuda_device_unavailable",
            ),
        ]

        summary = summarize_results(records)

        self.assertFalse(summary["fuzzer_normal"])
        self.assertEqual(summary["runtime_passed"], 1)
        self.assertEqual(summary["runtime_failed"], 1)
        self.assertEqual(summary["runtime_incomplete"], 1)
        self.assertEqual(summary["mutation_not_attempted"], 1)
        self.assertEqual(summary["mutation_failure_kinds"]["cuda_device_unavailable"], 1)

    def test_write_reports_persists_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            records = [
                _record("ok", support_decision="run", backend=True, fixed=True, mutation=True),
                _record("skip", support_decision="skip", skip_reason="vconfig_required"),
            ]

            write_reports(out_dir, records)

            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertTrue(summary["fuzzer_normal"])
            self.assertIn("ok", (out_dir / "SUMMARY.md").read_text())
            self.assertIn("vconfig_required", (out_dir / "SUMMARY.md").read_text())
            self.assertEqual(len(json.loads((out_dir / "kernel_results.json").read_text())["kernels"]), 2)

    def test_bug_candidates_are_reported_separately_from_fuzzer_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            records = [
                _record("ok", support_decision="run", backend=True, fixed=True, mutation=True),
                _record(
                    "zero_coeff_count_kernel",
                    support_decision="skip",
                    skip_reason="target_bug_candidate_divergent_barrier",
                ),
            ]

            summary = write_reports(out_dir, records)

            self.assertTrue(summary["fuzzer_normal"])
            self.assertEqual(summary["target_bug_candidates"], 1)
            self.assertEqual(
                summary["target_bug_candidate_reasons"],
                {"target_bug_candidate_divergent_barrier": 1},
            )
            bug_markdown = (out_dir / "BUG_CANDIDATES.md").read_text()
            self.assertIn("zero_coeff_count_kernel", bug_markdown)
            self.assertIn("target_bug_candidate_divergent_barrier", bug_markdown)

    def test_markdown_contains_per_kernel_evidence_status(self) -> None:
        markdown = render_markdown(
            [
                _record("ok", support_decision="run", backend=True, fixed=True, mutation=True),
                _record("skip", support_decision="skip", skip_reason="pointer_to_pointer_not_supported"),
                _record(
                    "mutation_bad",
                    support_decision="run",
                    backend=True,
                    fixed=True,
                    mutation=False,
                    mutation_failure_kind="cuda_illegal_memory_access",
                ),
            ],
            summarize_results(
                [
                    _record("ok", support_decision="run", backend=True, fixed=True, mutation=True),
                    _record("skip", support_decision="skip", skip_reason="pointer_to_pointer_not_supported"),
                    _record(
                        "mutation_bad",
                        support_decision="run",
                        backend=True,
                        fixed=True,
                        mutation=False,
                        mutation_failure_kind="cuda_illegal_memory_access",
                    ),
                ]
            ),
        )

        self.assertIn("| demo | ok | run | passed | passed |  | passed |", markdown)
        self.assertIn("pointer_to_pointer_not_supported", markdown)
        self.assertIn("| demo | mutation_bad | run | passed | passed |  | failed | cuda_illegal_memory_access |", markdown)

    def test_markdown_reports_backend_matrix_instead_of_stale_single_stage(
        self,
    ) -> None:
        record = _record(
            "matrix_built",
            support_decision="run",
            backend=None,
            fixed=None,
            mutation=None,
        )
        record["backend_results"] = {
            "rapid2": {"status": "passed"},
            "origin": {"status": "passed"},
            "rapid": {"status": "passed"},
        }

        summary = summarize_results([record])
        markdown = render_markdown([record], summary)

        self.assertEqual(
            summary["backend_matrix"],
            {
                "enabled": True,
                "cells_total": 3,
                "cells_passed": 3,
                "cells_failed": 0,
                "cells_incomplete": 0,
                "normal": True,
            },
        )
        self.assertIn(
            "Backend matrix builds passed: 3 / 3 (failed: 0, incomplete: 0)",
            markdown,
        )
        self.assertIn(
            "| demo | matrix_built | run | origin=passed, rapid=passed, rapid2=passed |",
            markdown,
        )
        self.assertNotIn("| demo | matrix_built | run | not_attempted |", markdown)

    def test_markdown_lists_incomplete_and_device_unavailable_counts(self) -> None:
        records = [
            _record("not_yet_mutated", support_decision="run", backend=True, fixed=True, mutation=None),
            _record(
                "driver_lost",
                support_decision="run",
                backend=True,
                fixed=True,
                mutation=False,
                mutation_failure_kind="cuda_device_unavailable",
            ),
        ]
        markdown = render_markdown(records, summarize_results(records))

        self.assertIn("Runtime incomplete: 1", markdown)
        self.assertIn("Mutation not attempted: 1", markdown)
        self.assertIn("cuda_device_unavailable", markdown)

    def test_coverage_summary_separates_applicability_skips_and_failures(
        self,
    ) -> None:
        first = _record(
            "barrier_disabled",
            support_decision="run",
            backend=True,
            fixed=True,
            mutation=True,
        )
        first["coverage_results"] = {
            "rapid2": {
                "off": _coverage_stage(cfg=9, memory=11, threads=4),
                "on": {
                    "status": "skipped",
                    "passed": True,
                    "skip_kind": "applicability",
                    "failure_reason": "vconfig_barrier_unsupported",
                },
            }
        }
        second = _record(
            "enabled",
            support_decision="run",
            backend=True,
            fixed=True,
            mutation=True,
        )
        second["coverage_results"] = {
            "rapid2": {
                "off": {"status": "failed", "passed": False},
                "on": _coverage_stage(cfg=7, memory=5, threads=3),
            }
        }

        summary = summarize_results([first, second])

        self.assertFalse(summary["coverage"]["normal"])
        self.assertFalse(summary["fuzzer_normal"])
        self.assertEqual(summary["coverage"]["cells_passed"], 2)
        self.assertEqual(summary["coverage"]["cells_failed"], 1)
        self.assertEqual(summary["coverage"]["applicability_skipped"], 1)
        self.assertEqual(summary["coverage"]["cells_incomplete"], 0)
        self.assertEqual(
            summary["coverage"]["final_by_cell"]["rapid2/off"],
            {
                "completed_kernels": 1,
                "executions_completed": 20,
                "cfg_sites": 9,
                "memory_features": 11,
                "thread_activity_features": 4,
            },
        )
        markdown = render_markdown([first, second], summary)
        self.assertIn(
            "Coverage passed / failed / applicability skipped / incomplete: 2 / 1 / 1 / 0",
            markdown,
        )
        self.assertIn("| rapid2/off | 1 | 20 | 9 | 11 | 4 |", markdown)

    def test_coverage_summary_excludes_runnable_kernels_outside_selected_matrix(
        self,
    ) -> None:
        selected = _record(
            "selected",
            support_decision="run",
            backend=True,
            fixed=True,
            mutation=True,
        )
        selected["coverage_results"] = {
            "rapid2": {
                "off": _coverage_stage(cfg=9, memory=11, threads=4),
                "on": _coverage_stage(cfg=7, memory=5, threads=3),
            }
        }
        unselected = _record(
            "unselected",
            support_decision="run",
            backend=True,
            fixed=True,
            mutation=True,
        )

        coverage = summarize_results([selected, unselected])["coverage"]

        self.assertEqual(coverage["cells_passed"], 2)
        self.assertEqual(coverage["cells_incomplete"], 0)
        self.assertEqual(
            coverage["final_by_cell"]["rapid2/off"]["completed_kernels"],
            1,
        )


def _record(
    kernel_id: str,
    *,
    phase2_status: str = "built",
    support_decision: str,
    skip_reason: str | None = None,
    backend: bool | None = None,
    fixed: bool | None = None,
    mutation: bool | None = None,
    runtime_decision: str | None = None,
    runtime_skip_reason: str | None = None,
    mutation_failure_kind: str | None = None,
) -> dict:
    record = {
        "project": "demo",
        "kernel_id": kernel_id,
        "symbol_name": f"_Z{len(kernel_id)}{kernel_id}v",
        "display_name": kernel_id,
        "phase2_status": phase2_status,
        "support_decision": support_decision,
        "skip_reason": skip_reason,
        "skip_detail": skip_reason,
        "backend": _stage(backend),
        "fixed": _stage(fixed),
        "mutation": _stage(mutation),
    }
    if runtime_decision:
        record["runtime_decision"] = runtime_decision
    if runtime_skip_reason:
        record["runtime_skip_reason"] = runtime_skip_reason
    if mutation_failure_kind:
        record["mutation"]["failure_kind"] = mutation_failure_kind
    return record


def _stage(passed: bool | None) -> dict:
    if passed is None:
        return {"status": "not_attempted"}
    return {"status": "passed" if passed else "failed", "passed": passed}


def _coverage_stage(*, cfg: int, memory: int, threads: int) -> dict:
    return {
        "status": "passed",
        "passed": True,
        "executions_completed": 20,
        "cfg_sites": cfg,
        "memory_features": memory,
        "thread_activity_features": threads,
    }


if __name__ == "__main__":
    unittest.main()
