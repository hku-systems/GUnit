import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RQ4ProfileDiagnosticTests(unittest.TestCase):
    def test_directory_input_skips_rows_without_profiling_records(self) -> None:
        from benchmark.rq4.profile import MAIN_LOOP_SEGMENTS
        from benchmark.rq4.profile_diagnostic import main

        records = [
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "main_loop",
                "segment": segment,
                "unit": "ns",
                "count": 1,
                "total": 1,
                "max": 1,
            }
            for segment in MAIN_LOOP_SEGMENTS
        ]
        records.extend(
            {
                "schema_version": 2,
                "record_type": "segment",
                "domain": "cpu_feedback",
                "segment": segment,
                "unit": "ns",
                "count": 1,
                "total": 1,
                "max": 1,
            }
            for segment in ("predicate", "metadata")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profiling_path = root / "complete.profiling.jsonl"
            profiling_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            profiles = [
                {
                    "mode": "mutating",
                    "workload_id": "incomplete",
                    "configuration": "gunit-sync",
                    "repetition": 1,
                    "capture_incomplete": True,
                    "profiling_record": None,
                },
                {
                    "mode": "mutating",
                    "workload_id": "complete",
                    "configuration": "gunit-sync",
                    "repetition": 1,
                    "profiling_record": str(profiling_path),
                },
            ]
            (root / "profiles.jsonl").write_text(
                "".join(json.dumps(profile) + "\n" for profile in profiles),
                encoding="utf-8",
            )
            output = root / "diagnostic.csv"
            with patch.object(
                sys,
                "argv",
                ["profile_diagnostic", "--input", str(root), "--output", str(output)],
            ):
                main()

            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(MAIN_LOOP_SEGMENTS))
            self.assertEqual({row["workload_id"] for row in rows}, {"complete"})


if __name__ == "__main__":
    unittest.main()
