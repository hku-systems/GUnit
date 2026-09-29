#!/usr/bin/env python3
"""Plot and summarize the completed formal synth_complex RQ2 campaign."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rapid-matplotlib-cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = (
    REPO_ROOT
    / "benchmark/rq2/results/formal-synth-complex-9cfg-5seed-60s-20260801-gpu2"
)
FIGURE_PATH = (
    REPO_ROOT / "benchmark/rq2/reports/figures/synth-complex-coverage_curves.png"
)
sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq2.plot_coverage import CoverageCurve, load_coverage_curves  # noqa: E402
from benchmark.rq2.variants._common import (  # noqa: E402
    CLASSIC_RCPARAMS,
    DURATION_S,
    KEY_CONFIGS,
    METRICS,
    STYLES,
    configure_style,
    mean_t90,
    read_jsonl,
    step_curve,
)


def plot_curves(curves_by_config: dict[str, list[CoverageCurve]]) -> None:
    configure_style(CLASSIC_RCPARAMS)
    grid = np.linspace(0.0, DURATION_S, 601)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    for axis, (attribute, title, ylabel) in zip(axes, METRICS):
        for configuration_id in KEY_CONFIGS:
            samples = np.vstack(
                [
                    step_curve(curve, attribute, grid)
                    for curve in curves_by_config[configuration_id]
                ]
            )
            mean = samples.mean(axis=0)
            std = samples.std(axis=0)
            color, linestyle = STYLES[configuration_id]
            axis.plot(
                grid,
                mean,
                color=color,
                linestyle=linestyle,
                label=configuration_id,
                zorder=3,
            )
            axis.fill_between(
                grid,
                np.maximum(0.0, mean - std),
                mean + std,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        axis.set_title(title, pad=5)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, DURATION_S)
        axis.set_xticks(np.arange(0, 61, 10))
        axis.set_ylim(bottom=0.0)
        axis.tick_params(axis="both", which="major", pad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    figure.subplots_adjust(top=0.78, bottom=0.19, left=0.09, right=0.985, wspace=0.27)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(figure)


def format_metric(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_summary(
    configuration_order: list[str], curves_by_config: dict[str, list[CoverageCurve]]
) -> None:
    rows: dict[str, dict[str, float | None]] = {}
    for configuration_id in configuration_order:
        curves = curves_by_config[configuration_id]
        final_memory = [curve.points[-1].memory_features for curve in curves]
        final_cfg = [curve.points[-1].cfg_sites for curve in curves]
        rows[configuration_id] = {
            "memory": (
                statistics.fmean(final_memory)
                if all(value is not None for value in final_memory)
                else None
            ),
            "cfg": (
                statistics.fmean(final_cfg)
                if all(value is not None for value in final_cfg)
                else None
            ),
            "memory_t90": mean_t90(
                curves, "memory_features", allow_unavailable=True
            ),
            "cfg_t90": mean_t90(curves, "cfg_sites", allow_unavailable=True),
            "executions": statistics.fmean(
                curve.points[-1].executions_completed for curve in curves
            ),
        }

    table = [
        "| Configuration | Mean final MemCov features | Mean final CFG sites | Mean t90, MemCov / CFG (s) | Mean executions |",
        "|---|---:|---:|---:|---:|",
    ]
    for configuration_id in configuration_order:
        row = rows[configuration_id]
        t90 = (
            f"{format_metric(row['memory_t90'], 2)} / "
            f"{format_metric(row['cfg_t90'], 2)}"
        )
        table.append(
            f"| `{configuration_id}` | {format_metric(row['memory'])} | "
            f"{format_metric(row['cfg'])} | {t90} | {row['executions']:,.0f} |"
        )

    feedback_rows = [row for row in rows.values() if row["memory"] is not None]
    memory_t90s = [float(row["memory_t90"]) for row in feedback_rows]
    cfg_t90s = [float(row["cfg_t90"]) for row in feedback_rows]
    backend_names = ("origin", "rapid-w1", "rapid-w4", "rapid2")
    gaps = [
        100.0
        * (float(rows[f"{name}-on"]["memory"]) - float(rows[f"{name}-off"]["memory"]))
        / float(rows[f"{name}-off"]["memory"])
        for name in backend_names
    ]
    on_configs = [f"{name}-on" for name in backend_names]
    throughput_order = sorted(
        on_configs, key=lambda name: float(rows[name]["executions"]), reverse=True
    )
    memory_t90_order = sorted(
        on_configs, key=lambda name: float(rows[name]["memory_t90"])
    )
    cfg_t90_order = sorted(on_configs, key=lambda name: float(rows[name]["cfg_t90"]))

    def names(order: list[str]) -> str:
        return " > ".join(name.removesuffix("-on") for name in order)

    if memory_t90_order != cfg_t90_order:
        t90_ordering = (
            f"MemCov {names(memory_t90_order)}; CFG {names(cfg_t90_order)}"
        )
    else:
        t90_ordering = names(memory_t90_order)

    content = [
        "# Formal RQ2 coverage summary",
        "",
        "`synth_complex`, 60 s coverage window, five independent seeds per configuration. "
        "Each t90 is the mean across seeds of the first recorded time reaching "
        "90% of that seed's final value; "
        "`N/A` preserves CuFuzz's unavailable internal device coverage.",
        "",
        *table,
        "",
        "## Conclusions",
        "",
        f"- Time-to-plateau: MemCov t90 is {min(memory_t90s):.2f}-{max(memory_t90s):.2f} s; "
        f"CFG t90 is later at {min(cfg_t90s):.2f}-{max(cfg_t90s):.2f} s.",
        f"- VConfig on vs off: mean final MemCov rises by {min(gaps):.1f}-{max(gaps):.1f}% "
        "across origin, rapid-w1, rapid-w4, and rapid2.",
        f"- Backend ordering (VConfig on): throughput is {names(throughput_order)}; "
        f"earliest-to-latest t90 is {t90_ordering}.",
        "",
    ]
    (RESULT_DIR / "SUMMARY.md").write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    trial_rows = read_jsonl(RESULT_DIR / "trials.jsonl")
    if len(trial_rows) != 45 or any(row["status"] != "completed" for row in trial_rows):
        raise ValueError("expected 45 completed trials")

    configurations = read_jsonl(RESULT_DIR / "configurations.jsonl")
    configuration_order = [str(row["configuration_id"]) for row in configurations]
    missing = set(KEY_CONFIGS) - set(configuration_order)
    if missing:
        raise ValueError(f"missing key configurations: {sorted(missing)}")

    seeds: dict[str, set[int]] = defaultdict(set)
    for row in trial_rows:
        seeds[str(row["configuration_id"])].add(int(row["seed"]))
    if any(
        seeds[configuration_id] != set(range(1, 6))
        for configuration_id in configuration_order
    ):
        raise ValueError("expected seeds 1..5 for every configuration")

    curves_by_config: dict[str, list[CoverageCurve]] = defaultdict(list)
    for curve in load_coverage_curves(RESULT_DIR):
        curves_by_config[curve.configuration_id].append(curve)
    if any(
        len(curves_by_config[configuration_id]) != 5
        for configuration_id in configuration_order
    ):
        raise ValueError("expected five curves per configuration")

    plot_curves(curves_by_config)
    write_summary(configuration_order, curves_by_config)


if __name__ == "__main__":
    main()
