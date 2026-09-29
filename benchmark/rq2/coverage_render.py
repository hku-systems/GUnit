"""Render RQ2 coverage-growth figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from benchmark.rq2.coverage_data import CoverageCurve, WORKLOAD_TITLES


SYSTEM_COLORS = {
    "LibAFL+": "#56B4E9",
    "Sys-s (w1)": "#009E73",
    "Sys-s (w4)": "#E69F00",
    "Sys": "#D55E00",
}
SYSTEM_MARKERS = {
    "LibAFL+": "o",
    "Sys-s (w1)": "s",
    "Sys-s (w4)": "^",
    "Sys": "D",
}
VCONFIG_LINESTYLES = {"off": "--", "on": "-"}


def _metric_change_limit(curves: Sequence[CoverageCurve], attribute: str) -> float:
    last_change_execution = 0
    for curve in curves:
        previous = 0
        for point in curve.points:
            value = getattr(point, attribute)
            if value is not None and value != previous:
                last_change_execution = max(
                    last_change_execution, point.executions_completed
                )
                previous = value
    return float(max(32, math.ceil(last_change_execution / 10.0) * 10 + 10))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.45,
        }
    )


def _legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=SYSTEM_MARKERS[label],
            markersize=4,
            linestyle="-",
            label=label,
        )
        for label, color in SYSTEM_COLORS.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#333333", linestyle="--", label="VConfig off"),
            Line2D([0], [0], color="#333333", linestyle="-", label="VConfig on"),
            Line2D(
                [0],
                [0],
                color="#8C8C8C",
                linestyle=":",
                label="CuFuzz: internal coverage N/A",
            ),
        ]
    )
    return handles


def render_metric(
    curves: Sequence[CoverageCurve],
    *,
    attribute: str,
    ylabel: str,
    output_base: Path,
    normalize_cfg: bool = True,
) -> tuple[Path, Path]:
    _style()
    share_y = attribute == "cfg_sites"
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.15, 4.7),
        sharex=True,
        sharey=share_y,
    )
    workload_ids = list(dict.fromkeys(curve.workload_id for curve in curves))
    view_limit = _metric_change_limit(curves, attribute)
    for panel_index, (axis, workload_id) in enumerate(zip(axes.flat, workload_ids)):
        workload_curves = [
            curve
            for curve in curves
            if curve.workload_id == workload_id and curve.feedback_enabled
        ]
        for curve in workload_curves:
            values = [getattr(point, attribute) for point in curve.points]
            if attribute == "cfg_sites" and normalize_cfg:
                assert curve.instrumented_cfg_sites is not None
                values = [
                    100.0 * float(value) / curve.instrumented_cfg_sites
                    for value in values
                ]
            executions = [point.executions_completed for point in curve.points]
            plot_values = [float(value) for value in values]
            marker_every = max(1, len(executions) // 8)
            axis.step(
                executions,
                plot_values,
                where="post",
                color=SYSTEM_COLORS[curve.paper_label],
                linestyle=VCONFIG_LINESTYLES[curve.vconfig],
                marker=SYSTEM_MARKERS[curve.paper_label],
                markevery=marker_every,
                markersize=2.7,
                markeredgewidth=0.4,
                zorder=3 if curve.paper_label == "Sys" else 2,
            )
        title = WORKLOAD_TITLES.get(workload_id, workload_id)
        axis.set_title(f"({chr(ord('a') + panel_index)}) {title}", pad=4)
        axis.set_xlim(0.0, view_limit)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.tick_params(axis="both", which="major", pad=2)
        if attribute == "cfg_sites" and normalize_cfg:
            axis.set_ylim(0.0, 105.0)
        if panel_index % 3 == 0:
            axis.set_ylabel(ylabel)
        if panel_index >= 3:
            axis.set_xlabel("Completed executions")
    for axis in axes.flat[len(workload_ids) :]:
        axis.set_visible(False)
    figure.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handlelength=2.0,
    )
    figure.subplots_adjust(
        top=0.84,
        bottom=0.11,
        left=0.08,
        right=0.985,
        hspace=0.34,
        wspace=0.24,
    )
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    figure.savefig(pdf_path, format="pdf")
    figure.savefig(png_path, format="png", dpi=300)
    plt.close(figure)
    return pdf_path, png_path
