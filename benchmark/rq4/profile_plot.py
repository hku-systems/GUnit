#!/usr/bin/env python3
"""Plot the eight independently normalized RQ4 CPU/GPU lanes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean

from benchmark.rq1.export_paper import PAPER_LABELS as KERNEL_TITLES
from benchmark.rq4.profile_aggregate import CATEGORIES, LANE_ORDER as LANES


CATEGORY_LABELS = (
    "Kernel Exec",
    "Feedback",
    "GPU Overhead",
    "DMemcpy",
    "DFree",
    "Kernel Launch",
    "Idle",
    "DMalloc",
)
PLOT_CATEGORIES = (
    "kernel_exec",
    "feedback",
    "memcpy",
    "launch",
    "allocation",
    "free",
    "gpu_overhead",
    "idle",
)
CATEGORY_COLORS = {
    "kernel_exec": "#7dcea0",
    "feedback": "#48c9b0",
    "memcpy": "#f5b041",
    "launch": "#5dade2",
    "allocation": "#ec7063",
    "free": "#f7dc6f",
    "gpu_overhead": "#af7ac5",
    "idle": "#bfc9ca",
}
CATEGORY_LABEL_BY_NAME = dict(zip(CATEGORIES, CATEGORY_LABELS))
LEGEND_CATEGORIES = (
    "kernel_exec",
    "feedback",
    "launch",
    "gpu_overhead",
    "idle",
    "memcpy",
    "allocation",
    "free",
)
GRID_SHAPE = (4, 3)
ANNOTATION_MIN_PERCENT = 7.0
LANE_POSITIONS = {
    "CuFuzz": 7.10,
    "LibAFL": 6.58,
    "LibAFL+": 6.06,
    "GUnit-s-GPU": 5.15,
    "GUnit-s-CPU": 4.63,
    "GUnit-GPU": 3.72,
    "GUnit-Coll": 3.20,
    "GUnit-Disp": 2.68,
}


def load_overhead_breakdown(
    path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty overhead breakdown: {path}")

    trials: dict[tuple[str, int, str], dict[str, float]] = {}
    excluded_trials: set[tuple[str, int, str]] = set()
    workload_order: list[str] = []
    for row in rows:
        workload = row["workload_id"]
        if workload not in workload_order:
            workload_order.append(workload)
        lane = row["lane"]
        if lane not in LANES:
            raise ValueError(f"unknown overhead lane: {lane}")
        repetition = int(row["repetition"])
        category = row["category"]
        key = (workload, repetition, lane)
        categories = trials.setdefault(key, {})
        status = row["status"]
        if status not in ("complete", "capture_incomplete", "crash"):
            raise ValueError(f"unknown profile status: {status}")
        unavailable = status != "complete"
        if unavailable:
            excluded_trials.add(key)
            if not category:
                continue
        if category not in CATEGORIES:
            raise ValueError(f"unknown overhead category: {category}")
        if category in categories:
            raise ValueError(
                f"duplicate overhead category: {workload} r{repetition} "
                f"{lane} {category}"
            )
        categories[category] = float(row["share"])

    repetitions_by_workload: dict[str, set[int]] = {}
    for workload, repetition, _lane in trials:
        repetitions_by_workload.setdefault(workload, set()).add(repetition)
    for workload, repetitions in repetitions_by_workload.items():
        for repetition in repetitions:
            actual_lanes = {
                lane
                for candidate, candidate_rep, lane in trials
                if candidate == workload and candidate_rep == repetition
            }
            if actual_lanes != set(LANES):
                raise ValueError(
                    f"incomplete lanes: {workload} r{repetition} "
                    f"missing={sorted(set(LANES) - actual_lanes)}"
                )
            for lane in LANES:
                key = (workload, repetition, lane)
                if key in excluded_trials:
                    continue
                categories = trials[key]
                if set(categories) != set(CATEGORIES):
                    raise ValueError(
                        f"incomplete categories: {workload} r{repetition} {lane}"
                    )
                if abs(sum(categories.values()) - 1.0) > 1e-6:
                    raise ValueError(
                        f"lane shares do not sum to one: "
                        f"{workload} r{repetition} {lane}"
                    )

    data: dict[str, dict[str, dict[str, float]]] = {}
    for workload in workload_order:
        repetitions = sorted(repetitions_by_workload[workload])
        data[workload] = {}
        for lane in LANES:
            complete_repetitions = [
                repetition
                for repetition in repetitions
                if (workload, repetition, lane) not in excluded_trials
            ]
            if not complete_repetitions:
                continue
            data[workload][lane] = {
                category: fmean(
                    trials[(workload, repetition, lane)][category]
                    for repetition in complete_repetitions
                )
                for category in CATEGORIES
            }
    return data


def plot_overhead_breakdown(
    data: dict[str, dict[str, dict[str, float]]], output: Path
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if len(data) != 12:
        raise ValueError(f"RQ4 profile figure requires 12 workloads, got {len(data)}")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(*GRID_SHAPE, figsize=(18.0, 17.0), sharex=True)
    for index, (axis, workload) in enumerate(zip(axes.flat, data)):
        for lane in LANES:
            if lane not in data[workload]:
                continue
            y = LANE_POSITIONS[lane]
            left = 0.0
            for category in PLOT_CATEGORIES:
                percentage = data[workload][lane][category] * 100.0
                axis.barh(
                    y,
                    percentage,
                    left=left,
                    height=0.46,
                    color=CATEGORY_COLORS[category],
                    edgecolor="black",
                    linewidth=1.0,
                )
                if percentage >= ANNOTATION_MIN_PERCENT:
                    center = left + percentage / 2.0
                    if left == 0.0:
                        center = max(center, 2.2)
                    axis.text(
                        center,
                        y - 0.015,
                        f"{percentage:.1f}",
                        ha="center",
                        va="center",
                        fontsize=15.0,
                        fontweight="bold",
                        color="black",
                    )
                left += percentage
        axis.set_yticks([LANE_POSITIONS[lane] for lane in LANES])
        axis.set_yticklabels(
            LANES,
            fontsize=18.0,
            fontweight="bold",
        )
        if index % GRID_SHAPE[1] != 0:
            axis.set_yticklabels([])
        axis.set_xlim(0, 100)
        axis.set_ylim(2.38, 7.40)
        axis.set_xticks((0, 20, 40, 60, 80, 100))
        axis.set_title(
            KERNEL_TITLES.get(workload, workload),
            fontsize=22.0,
            fontweight="bold",
            pad=7.0,
        )
        axis.grid(axis="x", linestyle="--", linewidth=0.65, alpha=0.3)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=15.0, width=1.0)
        for label in axis.get_xticklabels():
            label.set_fontweight("bold")
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)

    axes[-1, GRID_SHAPE[1] // 2].set_xlabel(
        "Overhead Percentage (%)", fontsize=20.0, fontweight="bold"
    )
    handles = [
        Patch(
            facecolor=CATEGORY_COLORS[category],
            edgecolor="black",
            label=CATEGORY_LABEL_BY_NAME[category],
        )
        for category in LEGEND_CATEGORIES
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=True,
        columnspacing=1.8,
        handlelength=2.2,
        handletextpad=0.65,
        prop={"weight": "bold", "size": 18.0},
    )
    fig.tight_layout(rect=(0.0, 0.095, 1.0, 0.995), h_pad=1.15, w_pad=0.75)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    plot_overhead_breakdown(load_overhead_breakdown(args.input), args.output)


if __name__ == "__main__":
    main()
