import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.rq2.coverage_delta import validate_delta_records
from benchmark.rq2.verify_results import SampleContext, validate_and_enrich_samples


class Rq2CoverageDeltaTests(unittest.TestCase):
    def _valid_records(self) -> list[dict[str, object]]:
        return [
            {
                "schema_version": 1,
                "sequence": 0,
                "timestamp_s": 0.0,
                "final_sample": False,
                "executions_submitted": 0,
                "executions_completed": 0,
                "cfg_sites": 0,
                "memory_features": 0,
                "thread_activity_features": 0,
                "feedback_features_total": 0,
                "cfg_sites_delta": 0,
                "memory_features_delta": 0,
                "thread_activity_features_delta": 0,
                "feedback_features_total_delta": 0,
            },
            {
                "schema_version": 1,
                "sequence": 1,
                "timestamp_s": 0.125,
                "final_sample": False,
                "executions_submitted": 2,
                "executions_completed": 1,
                "cfg_sites": 5,
                "memory_features": 7,
                "thread_activity_features": 2,
                "feedback_features_total": 14,
                "cfg_sites_delta": 5,
                "memory_features_delta": 7,
                "thread_activity_features_delta": 2,
                "feedback_features_total_delta": 14,
            },
            {
                "schema_version": 1,
                "sequence": 2,
                "timestamp_s": 1.0,
                "final_sample": True,
                "executions_submitted": 4,
                "executions_completed": 4,
                "cfg_sites": 5,
                "memory_features": 7,
                "thread_activity_features": 2,
                "feedback_features_total": 14,
                "cfg_sites_delta": 0,
                "memory_features_delta": 0,
                "thread_activity_features_delta": 0,
                "feedback_features_total_delta": 0,
            },
        ]

    def _context(self, *, feedback_enabled: bool) -> SampleContext:
        return SampleContext(
            trial_id="feedback" if feedback_enabled else "legacy-cufuzz",
            workload_id="workload",
            configuration_id="origin-off" if feedback_enabled else "cufuzz-off",
            paper_label="LibAFL+" if feedback_enabled else "CuFuzz-style",
            backend="origin" if feedback_enabled else "cufuzz",
            window=1,
            repetition=0,
            seed=1337,
            vconfig_requested="off",
            vconfig_effective="off",
            feedback_enabled=feedback_enabled,
            build_report_sha256="sha256:" + "1" * 64,
            manifest_sha256="sha256:" + "2" * 64,
            backend_sha256="sha256:" + "3" * 64,
            instrumentation_metadata_sha256=("sha256:" + "4" * 64)
            if feedback_enabled
            else None,
            instrumented_cfg_sites=5 if feedback_enabled else None,
            memory_metric_version="rapid-simt-memcov-v1"
            if feedback_enabled
            else None,
            memory_map_bits=7 if feedback_enabled else None,
            memory_sector_bytes=32 if feedback_enabled else None,
            thread_activity_map_bits=2 if feedback_enabled else None,
            memory_hash_contract_version="rapid-simt-memcov-hash-v1"
            if feedback_enabled
            else None,
            raw_telemetry_path="raw/telemetry.jsonl",
            raw_telemetry_sha256="sha256:" + "5" * 64,
            include_extended_memory_contract=feedback_enabled,
        )

    def test_accepts_sparse_baseline_change_and_final_records(self) -> None:
        validate_delta_records(self._valid_records())

    def test_accepts_coverage_discovered_only_at_final_drain(self) -> None:
        records = self._valid_records()
        records.pop(1)
        records[1].update(
            sequence=1,
            cfg_sites=5,
            memory_features=7,
            thread_activity_features=2,
            feedback_features_total=14,
            cfg_sites_delta=5,
            memory_features_delta=7,
            thread_activity_features_delta=2,
            feedback_features_total_delta=14,
        )

        validate_delta_records(records)

    def test_rejects_invalid_delta_traces(self) -> None:
        cases = {
            "schema_version": lambda rows: rows[1].update(schema_version=2),
            "sequence": lambda rows: rows[1].update(sequence=0),
            "timestamp": lambda rows: rows[1].update(timestamp_s=-1.0),
            "baseline": lambda rows: rows[0].update(cfg_sites=1, cfg_sites_delta=1),
            "cfg_sites_delta": lambda rows: rows[1].update(cfg_sites_delta=4),
            "memory_features_delta": lambda rows: rows[1].update(
                memory_features_delta=-1
            ),
            "thread_activity_features_delta": lambda rows: rows[1].update(
                thread_activity_features_delta=3
            ),
            "feedback_features_total_delta": lambda rows: rows[1].update(
                feedback_features_total_delta=13
            ),
            "completion event must advance": lambda rows: rows[1].update(
                executions_completed=0
            ),
            "unchanged nonfinal": lambda rows: rows.insert(
                2,
                {
                    **rows[1],
                    "sequence": 2,
                    "timestamp_s": 0.25,
                    "cfg_sites_delta": 0,
                    "memory_features_delta": 0,
                    "thread_activity_features_delta": 0,
                    "feedback_features_total_delta": 0,
                },
            ),
            "memory_map_hex": lambda rows: rows[1].update(memory_map_hex="00"),
            "final sample must be last": lambda rows: rows[1].update(final_sample=True),
            "exactly one final": lambda rows: rows[-1].update(final_sample=False),
            "pending executions": lambda rows: rows[-1].update(
                executions_submitted=5
            ),
        }

        for expected, mutate in cases.items():
            with self.subTest(expected=expected):
                records = copy.deepcopy(self._valid_records())
                mutate(records)
                if expected == "unchanged nonfinal":
                    records[-1]["sequence"] = 3
                with self.assertRaisesRegex(ValueError, expected):
                    validate_delta_records(records)

    def test_rejects_nonzero_execution_baselines(self) -> None:
        for submitted, completed in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(submitted=submitted, completed=completed):
                records = copy.deepcopy(self._valid_records())
                records[0]["executions_submitted"] = submitted
                records[0]["executions_completed"] = completed
                with self.assertRaisesRegex(ValueError, "baseline execution counters"):
                    validate_delta_records(records)

    def test_rq2_campaign_and_verifier_support_direct_script_invocation(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for relative_path in (
            "benchmark/rq2/campaign.py",
            "benchmark/rq2/verify_results.py",
        ):
            with self.subTest(script=relative_path):
                completed = subprocess.run(
                    [sys.executable, str(repo_root / relative_path), "--help"],
                    cwd=repo_root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_rejects_obsolete_full_bitmap_schema_one_records(self) -> None:
        record = {
            "schema_version": 1,
            "sequence": 0,
            "timestamp_s": 1.0,
            "final_sample": True,
            "executions_submitted": 0,
            "executions_completed": 0,
            "cfg_sites": 0,
            "memory_features": 0,
            "thread_activity_features": 0,
            "feedback_features_total": 0,
            "memory_map_hex": bytes(8192).hex(),
        }
        with tempfile.TemporaryDirectory(prefix="rq2-legacy-schema-") as temp_dir:
            path = Path(temp_dir) / "raw.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "memory_map_hex is forbidden"):
                validate_and_enrich_samples(
                    path,
                    self._context(feedback_enabled=False),
                )

    def test_feedback_enrichment_rejects_counts_beyond_declared_capacities(self) -> None:
        cases = {
            "cfg_sites": ("cfg_sites_delta", 6),
            "memory_features": ("memory_features_delta", 8),
            "thread_activity_features": ("thread_activity_features_delta", 3),
        }
        for field, (delta_field, impossible_value) in cases.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory(
                prefix="rq2-impossible-coverage-"
            ) as temp_dir:
                records = copy.deepcopy(self._valid_records())
                records[-1][field] = impossible_value
                records[-1][delta_field] = 1
                records[-1]["feedback_features_total"] = 15
                records[-1]["feedback_features_total_delta"] = 1
                path = Path(temp_dir) / "raw.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in records),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, f"{field} exceeds"):
                    validate_and_enrich_samples(
                        path,
                        self._context(feedback_enabled=True),
                    )


if __name__ == "__main__":
    unittest.main()
