import csv
import json
import tempfile
import unittest
from pathlib import Path


class RQ1PaperExportTests(unittest.TestCase):
    def test_paper_row_maps_backend_medians_to_figure_columns(self) -> None:
        from benchmark.rq1.export_paper import paper_row

        values = {
            "cufuzz": 1.0,
            "origin": 2.0,
            "origin-no-feedback": 3.0,
            "rapid-w4": 4.0,
            "rapid-no-feedback-w4": 5.0,
            "rapid2": 6.0,
            "rapid2-no-feedback": 7.0,
        }
        self.assertEqual(
            paper_row("kernel", "Kernel", values),
            {
                "workload_id": "kernel",
                "workload": "Kernel",
                "cufuzz": 1.0,
                "libafl_on": 2.0,
                "libafl_off": 3.0,
                "gunit_s_on": 4.0,
                "gunit_s_off": 5.0,
                "gunit_on": 6.0,
                "gunit_off": 7.0,
            },
        )

    def test_export_accepts_one_repetition_when_requested(self) -> None:
        from benchmark.rq1.export_paper import REQUIRED_BACKENDS, export

        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            with (input_dir / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "workload_id",
                        "backend",
                        "repetitions",
                        "median_exec_per_sec",
                    ),
                )
                writer.writeheader()
                for index, backend in enumerate(REQUIRED_BACKENDS, start=1):
                    writer.writerow(
                        {
                            "workload_id": "shoc_reduction",
                            "backend": backend,
                            "repetitions": 1,
                            "median_exec_per_sec": float(index),
                        }
                    )
            suite = root / "suite.json"
            suite.write_text(
                json.dumps({"workloads": ["shoc_reduction"]}), encoding="utf-8"
            )
            output = root / "paper.csv"

            export([input_dir], suite, output, expected_repetitions=1)

            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["workload"], "reduction")


if __name__ == "__main__":
    unittest.main()
