from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.rq2.campaign import CONFIGURATIONS
from benchmark.rq2.plot_coverage import (
    CoverageCurve,
    CoveragePoint,
    build_saturation_rows,
    load_coverage_curves,
    render_coverage_figures,
)


WORKLOADS = (
    ("shoc_reduction", "SHOC Reduction"),
    ("shoc_radix_sort", "SHOC RadixSortBlock"),
    ("shoc_scan", "SHOC ScanSingleBlock"),
    ("cutlass_gemm", "CUTLASS ReferenceGemm"),
    ("flashattention_device1xn", "FlashAttention device1xN"),
    ("pytorch_batchnorm", "PyTorch BatchNorm"),
)


class Rq2CoveragePlotTests(unittest.TestCase):
    def test_plot_facade_exports_internal_module_contracts(self) -> None:
        from benchmark.rq2 import coverage_data, coverage_stats, plot_coverage

        self.assertIs(CoverageCurve, coverage_data.CoverageCurve)
        self.assertIs(build_saturation_rows, coverage_stats.build_saturation_rows)
        self.assertIs(plot_coverage.WORKLOAD_TITLES, coverage_data.WORKLOAD_TITLES)

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _result_fixture(self, root: Path, *, repetitions: int = 1) -> Path:
        result = root / "results"
        result.mkdir()
        (result / "environment.json").write_text(
            json.dumps(
                {
                    "repetitions": repetitions,
                    "coverage_seconds": 20,
                    "coverage_recording_mode": "completion-change-v1",
                    "workloads": [
                        {"workload_id": workload_id, "label": label}
                        for workload_id, label in WORKLOADS
                    ],
                    "configurations": [
                        configuration.configuration_id
                        for configuration in CONFIGURATIONS
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_jsonl(
            result / "configurations.jsonl",
            [
                {
                    "configuration_id": item.configuration_id,
                    "paper_label": item.paper_label,
                    "vconfig": item.vconfig,
                    "feedback_enabled": item.feedback_enabled,
                }
                for item in CONFIGURATIONS
            ],
        )
        trials: list[dict[str, object]] = []
        samples: list[dict[str, object]] = []
        plan_index = 0
        for repetition in range(repetitions):
            for workload_id, _label in WORKLOADS:
                for configuration in CONFIGURATIONS:
                    trial_id = (
                        f"r{repetition:03d}-{workload_id}-"
                        f"{configuration.configuration_id}"
                    )
                    trials.append(
                        {
                            "trial_id": trial_id,
                            "workload_id": workload_id,
                            "configuration_id": configuration.configuration_id,
                            "repetition": repetition,
                            "plan_index": plan_index,
                            "status": "completed",
                        }
                    )
                    plan_index += 1
                    if configuration.feedback_enabled:
                        samples.extend(
                            [
                                {
                                    "trial_id": trial_id,
                                    "workload_id": workload_id,
                                    "configuration_id": (
                                        configuration.configuration_id
                                    ),
                                    "paper_label": configuration.paper_label,
                                    "vconfig_effective": configuration.vconfig,
                                    "feedback_enabled": True,
                                    "instrumented_cfg_sites": 4,
                                    "memory_metric_version": "rapid-simt-memcov-v1",
                                    "memory_map_bits": 61_440,
                                    "sequence": 0,
                                    "timestamp_s": 0.05,
                                    "executions_completed": 10,
                                    "cfg_sites": 1,
                                    "memory_features": 2,
                                    "executions_pending": 1,
                                    "final_sample": False,
                                },
                                {
                                    "trial_id": trial_id,
                                    "workload_id": workload_id,
                                    "configuration_id": (
                                        configuration.configuration_id
                                    ),
                                    "paper_label": configuration.paper_label,
                                    "vconfig_effective": configuration.vconfig,
                                    "feedback_enabled": True,
                                    "instrumented_cfg_sites": 4,
                                    "memory_metric_version": "rapid-simt-memcov-v1",
                                    "memory_map_bits": 61_440,
                                    "sequence": 1,
                                    "timestamp_s": 0.08,
                                    "executions_completed": 18,
                                    "cfg_sites": 3,
                                    "memory_features": 4,
                                    "executions_pending": 1,
                                    "final_sample": False,
                                },
                                {
                                    "trial_id": trial_id,
                                    "workload_id": workload_id,
                                    "configuration_id": (
                                        configuration.configuration_id
                                    ),
                                    "paper_label": configuration.paper_label,
                                    "vconfig_effective": configuration.vconfig,
                                    "feedback_enabled": True,
                                    "instrumented_cfg_sites": 4,
                                    "memory_metric_version": "rapid-simt-memcov-v1",
                                    "memory_map_bits": 61_440,
                                    "sequence": 2,
                                    "timestamp_s": 0.10,
                                    "executions_completed": 20,
                                    "cfg_sites": 3,
                                    "memory_features": 4,
                                    "executions_pending": 0,
                                    "final_sample": True,
                                },
                            ]
                        )
                    else:
                        samples.append(
                            {
                                "trial_id": trial_id,
                                "workload_id": workload_id,
                                "configuration_id": configuration.configuration_id,
                                "paper_label": configuration.paper_label,
                                "vconfig_effective": configuration.vconfig,
                                "feedback_enabled": False,
                                "instrumented_cfg_sites": None,
                                "memory_metric_version": None,
                                "memory_map_bits": None,
                                "sequence": 0,
                                "timestamp_s": 0.10,
                                "executions_completed": 30,
                                "cfg_sites": None,
                                "memory_features": None,
                                "executions_pending": 0,
                                "final_sample": True,
                            }
                        )
        self._write_jsonl(result / "trials.jsonl", trials)
        self._write_jsonl(result / "samples.jsonl", samples)
        return result

    def _mark_vconfig_cell_skipped(self, result: Path) -> str:
        trial_id = "r000-shoc_reduction-origin-on"
        trials = [
            json.loads(line)
            for line in (result / "trials.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        trial = next(row for row in trials if row["trial_id"] == trial_id)
        trial.update(
            status="skipped",
            vconfig_effective="unsupported",
            vconfig_disabled_reason="vconfig_barrier_unsupported",
        )
        self._write_jsonl(result / "trials.jsonl", trials)
        samples = [
            row
            for row in (
                json.loads(line)
                for line in (result / "samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if row["trial_id"] != trial_id
        ]
        self._write_jsonl(result / "samples.jsonl", samples)
        return trial_id

    def test_loads_full_matrix_and_preserves_unavailable_cufuzz_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-plot-loader-") as temp_dir:
            result = self._result_fixture(Path(temp_dir))

            curves = load_coverage_curves(result)
            self.assertEqual(len(curves), 54)
            cufuzz = next(
                curve
                for curve in curves
                if curve.workload_id == "shoc_reduction"
                and curve.configuration_id == "cufuzz-off"
            )
            self.assertIsNone(cufuzz.instrumented_cfg_sites)
            self.assertIsNone(cufuzz.memory_metric_version)
            self.assertIsNone(cufuzz.memory_map_bits)
            self.assertIsNone(cufuzz.points[-1].cfg_sites)
            self.assertIsNone(cufuzz.points[-1].memory_features)
            origin = next(
                curve
                for curve in curves
                if curve.workload_id == "shoc_reduction"
                and curve.configuration_id == "origin-off"
            )
            self.assertEqual(
                [point.timestamp_ms for point in origin.points],
                [50.0, 80.0, 100.0],
            )
            self.assertEqual(
                [point.executions_completed for point in origin.points],
                [10, 18, 20],
            )
            self.assertEqual(origin.memory_metric_version, "rapid-simt-memcov-v1")
            self.assertEqual(origin.memory_map_bits, 61_440)
            self.assertEqual(
                origin.cfg_metric_version, "rapid-cfg-presence-v1"
            )

            rows = build_saturation_rows(curves)
            origin_row = next(
                row
                for row in rows
                if row["workload_id"] == "shoc_reduction"
                and row["configuration_id"] == "origin-off"
            )
            self.assertEqual(origin_row["final_cfg_sites"], 3)
            self.assertEqual(origin_row["cfg_saturation_ms"], 80.0)
            self.assertEqual(origin_row["cfg_change_count"], 2)
            self.assertEqual(origin_row["cfg_first_coverage_execution"], 10)
            self.assertEqual(origin_row["cfg_saturation_execution"], 18)
            self.assertEqual(origin_row["final_memory_features"], 4)
            self.assertEqual(origin_row["memory_saturation_ms"], 80.0)
            self.assertEqual(origin_row["memory_change_count"], 2)
            self.assertEqual(origin_row["memory_first_coverage_execution"], 10)
            self.assertEqual(origin_row["memory_saturation_execution"], 18)
            self.assertEqual(
                origin_row["execution_index_semantics"], "exact_completion_event"
            )
            self.assertIsNone(
                next(
                    row["cfg_time_to_50pct_censored"]
                    for row in rows
                    if row["workload_id"] == "shoc_reduction"
                    and row["configuration_id"] == "cufuzz-off"
                )
            )

    def test_loads_all_repetitions_in_campaign_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-plot-repetitions-") as temp_dir:
            result = self._result_fixture(Path(temp_dir), repetitions=2)

            curves = load_coverage_curves(result)

            self.assertEqual(len(curves), 108)
            self.assertEqual(curves[0].trial_id, "r000-shoc_reduction-cufuzz-off")
            self.assertEqual(curves[54].trial_id, "r001-shoc_reduction-cufuzz-off")
            self.assertEqual(len({curve.trial_id for curve in curves}), 108)

    def test_union_ceiling_fraction_times_include_right_censoring(self) -> None:
        def curve(
            trial_id: str,
            configuration_id: str,
            cfg_values: tuple[int, int, int],
            memory_values: tuple[int, int, int],
            timestamps: tuple[float, float, float],
            executions: tuple[int, int, int],
        ) -> CoverageCurve:
            return CoverageCurve(
                trial_id=trial_id,
                workload_id="workload",
                workload_label="Workload",
                configuration_id=configuration_id,
                paper_label="System",
                vconfig="off",
                feedback_enabled=True,
                instrumented_cfg_sites=10,
                memory_metric_version="rapid-simt-memcov-v1",
                memory_map_bits=61_440,
                points=tuple(
                    CoveragePoint(
                        sequence=index,
                        timestamp_ms=timestamp,
                        executions_completed=execution,
                        cfg_sites=cfg_value,
                        memory_features=memory_value,
                        executions_pending=0,
                        final_sample=index == 2,
                    )
                    for index, (
                        cfg_value,
                        memory_value,
                        timestamp,
                        execution,
                    ) in enumerate(
                        zip(cfg_values, memory_values, timestamps, executions)
                    )
                ),
            )

        lower = curve(
            "r000-workload-lower",
            "lower",
            (2, 5, 6),
            (4, 10, 12),
            (10.0, 20.0, 30.0),
            (1, 2, 3),
        )
        ceiling = curve(
            "r001-workload-ceiling",
            "ceiling",
            (5, 9, 10),
            (10, 18, 20),
            (12.0, 22.0, 32.0),
            (4, 8, 12),
        )

        rows = build_saturation_rows((lower, ceiling))
        lower_row, ceiling_row = rows

        self.assertEqual(lower_row["cfg_empirical_ceiling"], 10)
        self.assertEqual(lower_row["memory_empirical_ceiling"], 20)
        self.assertEqual(lower_row["cfg_time_to_50pct_ms"], 20.0)
        self.assertEqual(lower_row["cfg_time_to_50pct_executions_completed"], 2)
        self.assertFalse(lower_row["cfg_time_to_50pct_censored"])
        self.assertEqual(lower_row["memory_time_to_50pct_ms"], 20.0)
        self.assertEqual(lower_row["memory_time_to_50pct_executions_completed"], 2)
        self.assertFalse(lower_row["memory_time_to_50pct_censored"])
        for prefix in ("cfg", "memory"):
            for percent in (90, 95, 100):
                with self.subTest(metric=prefix, percent=percent):
                    self.assertIsNone(lower_row[f"{prefix}_time_to_{percent}pct_ms"])
                    self.assertIsNone(
                        lower_row[
                            f"{prefix}_time_to_{percent}pct_executions_completed"
                        ]
                    )
                    self.assertTrue(
                        lower_row[f"{prefix}_time_to_{percent}pct_censored"]
                    )

        self.assertEqual(ceiling_row["cfg_time_to_50pct_ms"], 12.0)
        self.assertEqual(ceiling_row["cfg_time_to_90pct_ms"], 22.0)
        self.assertEqual(ceiling_row["cfg_time_to_95pct_ms"], 32.0)
        self.assertEqual(ceiling_row["cfg_time_to_100pct_ms"], 32.0)
        self.assertEqual(ceiling_row["memory_time_to_95pct_ms"], 32.0)
        self.assertFalse(ceiling_row["memory_time_to_100pct_censored"])

    def test_skipped_vconfig_cell_is_excluded_and_recorded_as_applicability(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-plot-skip-") as temp_dir:
            result = self._result_fixture(Path(temp_dir))
            skipped_trial_id = self._mark_vconfig_cell_skipped(result)

            curves = load_coverage_curves(result)
            self.assertEqual(len(curves), 53)
            self.assertNotIn(skipped_trial_id, {curve.trial_id for curve in curves})

            render_coverage_figures(result)
            saturation = [
                json.loads(line)
                for line in (result / "coverage_saturation.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(saturation), 54)
            skipped = next(
                row for row in saturation if row["trial_id"] == skipped_trial_id
            )
            self.assertEqual(skipped["status"], "skipped")
            self.assertEqual(skipped["vconfig"], "on")
            self.assertEqual(
                skipped["vconfig_disabled_reason"],
                "vconfig_barrier_unsupported",
            )

    def test_rejects_nonmonotonic_or_incomplete_traces(self) -> None:
        mutations = {
            "sequence": lambda rows: rows[1].update(sequence=-1),
            "timestamp": lambda rows: rows[1].update(timestamp_s=0.01),
            "executions_completed": lambda rows: rows[1].update(
                executions_completed=9
            ),
            "cfg_sites": lambda rows: rows[1].update(cfg_sites=0),
            "memory_features": lambda rows: rows[1].update(memory_features=1),
            "final pending": lambda rows: rows[-1].update(executions_pending=2),
            "final sample": lambda rows: rows[-1].update(final_sample=False),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory(
                prefix="rq2-plot-invalid-"
            ) as temp_dir:
                result = self._result_fixture(Path(temp_dir))
                rows = [
                    json.loads(line)
                    for line in (result / "samples.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                target = [
                    row
                    for row in rows
                    if row["trial_id"] == "r000-shoc_reduction-origin-off"
                ]
                mutate(target)
                self._write_jsonl(result / "samples.jsonl", rows)
                with self.assertRaisesRegex(ValueError, expected):
                    load_coverage_curves(result)

    def test_rejects_numeric_cufuzz_internal_coverage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-plot-cufuzz-") as temp_dir:
            result = self._result_fixture(Path(temp_dir))
            rows = [
                json.loads(line)
                for line in (result / "samples.jsonl").read_text().splitlines()
            ]
            cufuzz = next(
                row
                for row in rows
                if row["trial_id"] == "r000-shoc_reduction-cufuzz-off"
            )
            cufuzz["cfg_sites"] = 0
            self._write_jsonl(result / "samples.jsonl", rows)

            with self.assertRaisesRegex(ValueError, "CuFuzz.*unavailable"):
                load_coverage_curves(result)

    def test_rejects_per_trial_and_cross_curve_metric_drift(self) -> None:
        mutations = {
            "memory metric version changed": lambda rows: rows[1].update(
                memory_metric_version="rapid-memory-bitset-v1"
            ),
            "memory map size changed": lambda rows: rows[1].update(
                memory_map_bits=60_000
            ),
            "unsupported CFG metric version": lambda rows: rows[1].update(
                cfg_metric_version="rapid-cfg-warpcount-v1",
            ),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory(
                prefix="rq2-plot-metric-drift-"
            ) as temp_dir:
                result = self._result_fixture(Path(temp_dir))
                rows = [
                    json.loads(line)
                    for line in (result / "samples.jsonl").read_text().splitlines()
                ]
                target = [
                    row
                    for row in rows
                    if row["trial_id"] == "r000-shoc_reduction-origin-off"
                ]
                mutate(target)
                self._write_jsonl(result / "samples.jsonl", rows)
                with self.assertRaisesRegex(ValueError, expected):
                    load_coverage_curves(result)

        with tempfile.TemporaryDirectory(prefix="rq2-plot-mixed-metric-") as temp_dir:
            result = self._result_fixture(Path(temp_dir))
            rows = [
                json.loads(line)
                for line in (result / "samples.jsonl").read_text().splitlines()
            ]
            for row in rows:
                if row["trial_id"] == "r000-shoc_reduction-origin-off":
                    row["memory_metric_version"] = "rapid-memory-bitset-v1"
            self._write_jsonl(result / "samples.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "mixed memory metric versions"):
                load_coverage_curves(result)

        with tempfile.TemporaryDirectory(prefix="rq2-plot-mixed-cfg-") as temp_dir:
            result = self._result_fixture(Path(temp_dir))
            rows = [
                json.loads(line)
                for line in (result / "samples.jsonl").read_text().splitlines()
            ]
            for row in rows:
                if row["trial_id"] == "r000-shoc_reduction-origin-off":
                    row["cfg_metric_version"] = "rapid-cfg-warpcount-v1"
            self._write_jsonl(result / "samples.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "unsupported CFG metric version"):
                load_coverage_curves(result)

    def test_renders_pdf_png_and_saturation_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rq2-plot-render-") as temp_dir:
            root = Path(temp_dir)
            result = self._result_fixture(root)
            review = root / "review"

            outputs = render_coverage_figures(result, review_dir=review)

            self.assertEqual(len(outputs), 4)
            for output in outputs:
                self.assertTrue(output.is_file(), output)
                self.assertGreater(output.stat().st_size, 1000)
                if output.suffix == ".pdf":
                    self.assertEqual(output.read_bytes()[:4], b"%PDF")
                else:
                    self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue((review / "rq2_cfg_coverage_executions.png").is_file())
            self.assertTrue((review / "rq2_memcov_executions.png").is_file())
            saturation = [
                json.loads(line)
                for line in (result / "coverage_saturation.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(saturation), 54)


if __name__ == "__main__":
    unittest.main()
