import csv
import tempfile
import unittest
from pathlib import Path


def write_breakdown(
    path: Path,
    *,
    omit_lane: str | None = None,
    crash_cell: tuple[str, int] | None = None,
) -> None:
    from benchmark.rq4.profile_plot import CATEGORIES, LANES

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "workload_id",
                "configuration",
                "lane",
                "repetition",
                "category",
                "value",
                "unit",
                "share",
                "measurement_domain",
                "status",
                "capture_incomplete",
                "has_crashes",
                "solutions",
            ),
        )
        writer.writeheader()
        for repetition, kernel_share in enumerate((0.2, 0.4, 0.9), start=1):
            for lane in LANES:
                if lane == omit_lane:
                    continue
                for category in CATEGORIES:
                    share = 0.0
                    if category == "kernel_exec":
                        share = kernel_share
                    elif category == "idle":
                        share = 1.0 - kernel_share
                    writer.writerow(
                        {
                            "workload_id": "k",
                            "configuration": "test",
                            "lane": lane,
                            "repetition": repetition,
                            "category": category,
                            "value": int(share * 100),
                            "unit": "ns",
                            "share": share,
                            "measurement_domain": "test",
                            "status": (
                                "crash" if (lane, repetition) == crash_cell else "complete"
                            ),
                            "capture_incomplete": False,
                            "has_crashes": (lane, repetition) == crash_cell,
                            "solutions": (
                                1 if (lane, repetition) == crash_cell else 0
                            ),
                        }
                    )


class RQ4ProfilePlotTests(unittest.TestCase):
    def test_uses_approved_lane_and_legend_order(self) -> None:
        from benchmark.rq4.profile_plot import CATEGORIES, CATEGORY_LABELS, LANES

        self.assertEqual(
            LANES,
            (
                "CuFuzz",
                "LibAFL",
                "LibAFL+",
                "GUnit-s-GPU",
                "GUnit-s-CPU",
                "GUnit-GPU",
                "GUnit-Coll",
                "GUnit-Disp",
            ),
        )
        self.assertEqual(
            CATEGORIES,
            (
                "kernel_exec",
                "feedback",
                "gpu_overhead",
                "memcpy",
                "free",
                "launch",
                "idle",
                "allocation",
            ),
        )
        self.assertEqual(
            CATEGORY_LABELS,
            (
                "Kernel Exec",
                "Feedback",
                "GPU Overhead",
                "DMemcpy",
                "DFree",
                "Kernel Launch",
                "Idle",
                "DMalloc",
            ),
        )

    def test_preserves_the_paper_plot_visual_contract(self) -> None:
        from benchmark.rq4.profile_plot import (
            CATEGORY_COLORS,
            GRID_SHAPE,
            KERNEL_TITLES,
            PLOT_CATEGORIES,
            ANNOTATION_MIN_PERCENT,
        )

        self.assertEqual(GRID_SHAPE, (4, 3))
        self.assertEqual(ANNOTATION_MIN_PERCENT, 7.0)
        self.assertEqual(
            PLOT_CATEGORIES,
            (
                "kernel_exec",
                "feedback",
                "memcpy",
                "launch",
                "allocation",
                "free",
                "gpu_overhead",
                "idle",
            ),
        )
        self.assertEqual(
            CATEGORY_COLORS,
            {
                "kernel_exec": "#7dcea0",
                "feedback": "#48c9b0",
                "memcpy": "#f5b041",
                "launch": "#5dade2",
                "allocation": "#ec7063",
                "free": "#f7dc6f",
                "gpu_overhead": "#af7ac5",
                "idle": "#bfc9ca",
            },
        )
        self.assertEqual(KERNEL_TITLES["shoc_reduction"], "reduction")
        self.assertEqual(KERNEL_TITLES["synth_complex"], "byte_fingerprint")

    def test_loader_takes_mean_share_across_repetitions(self) -> None:
        from benchmark.rq4.profile_plot import load_overhead_breakdown

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overhead.csv"
            write_breakdown(path)

            data = load_overhead_breakdown(path)

        self.assertAlmostEqual(data["k"]["CuFuzz"]["kernel_exec"], 0.5)
        self.assertAlmostEqual(data["k"]["CuFuzz"]["idle"], 0.5)

    def test_loader_excludes_crash_cells_from_lane_statistics(self) -> None:
        from benchmark.rq4.profile_plot import load_overhead_breakdown

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overhead.csv"
            write_breakdown(path, crash_cell=("CuFuzz", 3))

            data = load_overhead_breakdown(path)

        self.assertAlmostEqual(data["k"]["CuFuzz"]["kernel_exec"], 0.3)
        self.assertAlmostEqual(data["k"]["CuFuzz"]["idle"], 0.7)

    def test_loader_rejects_an_incomplete_eight_lane_matrix(self) -> None:
        from benchmark.rq4.profile_plot import load_overhead_breakdown

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overhead.csv"
            write_breakdown(path, omit_lane="GUnit-Disp")

            with self.assertRaisesRegex(ValueError, "incomplete lanes"):
                load_overhead_breakdown(path)


if __name__ == "__main__":
    unittest.main()
