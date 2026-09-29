"""Shared data and plotting helpers for the RQ2 figure scripts."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from benchmark.rq2.plot_coverage import CoverageCurve


DURATION_S = 60.0
KEY_CONFIGS = ("rapid2-on", "rapid2-off", "origin-on", "rapid-w4-on")
METRICS = (
    ("memory_features", "(a) SIMT memory novelty", "SIMT MemCov features"),
    ("cfg_sites", "(b) Device CFG coverage", "Device CFG sites"),
)
STYLES = {
    "rapid2-on": ("#D55E00", "-"),
    "rapid2-off": ("#E69F00", "--"),
    "origin-on": ("#0072B2", "-."),
    "rapid-w4-on": ("#009E73", ":"),
}
BACKEND_STYLES = {
    "origin": ("Origin", "#0072B2"),
    "rapid": ("RAPID", "#009E73"),
    "rapid2": ("RAPID2", "#D55E00"),
}
VCONFIG_STYLES = {
    "off": ("VConfig off", "--"),
    "on": ("VConfig on", "-"),
}
CLASSIC_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
}
VARIANT_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.titlesize": 8.2,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.2,
}
MASTER_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.0,
    "axes.titlesize": 7.4,
    "axes.titleweight": "bold",
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.05,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
}


@dataclass(frozen=True)
class Curve:
    times: np.ndarray
    values: np.ndarray


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def step_curve(
    curve: CoverageCurve, attribute: str, grid: np.ndarray
) -> np.ndarray:
    times = np.asarray(
        [min(point.timestamp_ms / 1000.0, DURATION_S) for point in curve.points]
    )
    values = np.asarray([getattr(point, attribute) for point in curve.points])
    indices = np.searchsorted(times, grid, side="right") - 1
    if indices.min() < 0 or values.dtype == object:
        raise ValueError(f"{curve.trial_id}: incomplete {attribute} curve")
    return values[indices].astype(float)


def mean_t90(
    curves: list[CoverageCurve],
    attribute: str,
    *,
    allow_unavailable: bool = False,
) -> float | None:
    times = []
    for curve in curves:
        final = getattr(curve.points[-1], attribute)
        if final is None:
            if allow_unavailable:
                return None
            raise ValueError(f"{curve.trial_id}: unavailable {attribute}")
        hit = next(
            point
            for point in curve.points
            if 10 * getattr(point, attribute) >= 9 * final
        )
        times.append(min(hit.timestamp_ms / 1000.0, DURATION_S))
    return statistics.fmean(times)


def configure_style(parameters: Mapping[str, Any]) -> None:
    plt.rcParams.update(parameters)
