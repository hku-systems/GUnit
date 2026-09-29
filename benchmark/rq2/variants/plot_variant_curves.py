#!/usr/bin/env python3
"""Plot coverage growth for the derived CUDA variants."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rapid-matplotlib-cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq2.variants._common import (  # noqa: E402
    BACKEND_STYLES,
    VARIANT_RCPARAMS,
    VCONFIG_STYLES,
    Curve,
    configure_style,
)


CAMPAIGN_ROOT = REPO_ROOT / "build/e2e/third-party-fuzz/20260802-rq2-wave3"
FIGURE_ROOT = REPO_ROOT / "benchmark/rq2/reports/figures"
MEMCOV_OUTPUT = FIGURE_ROOT / "variants-coverage_curves.png"
CFG_OUTPUT = FIGURE_ROOT / "variants-cfg_curves.png"
MEMCOV_ZOOM_OUTPUT = FIGURE_ROOT / "variants-coverage_curves-zoom2s.png"
CFG_ZOOM_OUTPUT = FIGURE_ROOT / "variants-cfg_curves-zoom2s.png"
MEMCOV2_OUTPUT = FIGURE_ROOT / "variants2-coverage_curves.png"
CFG2_OUTPUT = FIGURE_ROOT / "variants2-cfg_curves.png"

BACKENDS = ("origin", "rapid", "rapid2")


@dataclass(frozen=True)
class Kernel:
    project_id: str
    project_label: str
    kernel_id: str
    title: str
    cfg_annotation_y: float | None = None

    @property
    def coverage_dir(self) -> Path:
        return (
            CAMPAIGN_ROOT
            / self.project_id
            / "kernels"
            / self.kernel_id
            / "coverage"
        )


KERNELS = (
    Kernel(
        "kaldi_variants",
        "Kaldi",
        "_ZN5kaldi37power_spectrum_kernel_blockdim_strideEiPKfiPfib__48ee4406",
        "power_spectrum_blockdim_stride",
    ),
    Kernel(
        "apex_variants",
        "Apex",
        "_Z23index_mul_2d_float_vgeoPfPKfS1_PKlll__88cdac70",
        "index_mul_2d_float_vgeo",
    ),
    Kernel(
        "cutlass_variants",
        "CUTLASS",
        "_Z30reference_syrk_dispatch_kerneljiidPKdidPdi__2f6a4695",
        "reference_syrk_dispatch",
    ),
)

NEW_KERNELS = (
    Kernel(
        "cutlass_variants",
        "CUTLASS",
        "_Z30reference_gemm_dispatch_kerneljiiifPKfiS0_ifPfi__efc18fc2",
        "ReferenceGemm dispatcher",
        0.58,
    ),
    Kernel(
        "gpujpeg_variants",
        "GPUJPEG",
        "_Z27gpujpeg_dct_dispatch_kerneljiiPhjPsiPKf__82472776",
        "DCT dispatcher",
        0.82,
    ),
    Kernel(
        "cuda_samples_variants",
        "CUDA Samples",
        "inverseCNDKernel_vgeo",
        "inverseCND direct",
    ),
    Kernel(
        "kaldi_variants/direct_vconfig",
        "Kaldi",
        "batched_extract_window_vgeo",
        "extract_window direct",
    ),
    Kernel(
        "cutlass_variants",
        "CUTLASS",
        "_Z30reference_trmm_dispatch_kerneljiidPKdiS0_iPdi__a79a6b09",
        "ReferenceTrmm dispatcher",
        0.58,
    ),
)


def load_curve(path: Path, metric: str) -> Curve:
    before = path.stat()
    times: list[float] = []
    values: list[float] = []
    final_sample = False

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                timestamp = float(row["timestamp_s"])
                value = float(row[metric])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid coverage row"
                ) from error
            if not math.isfinite(timestamp) or not math.isfinite(value):
                raise ValueError(f"{path}:{line_number}: non-finite coverage value")
            if timestamp < 0 or value < 0:
                raise ValueError(f"{path}:{line_number}: negative coverage value")
            if times and timestamp < times[-1]:
                raise ValueError(f"{path}:{line_number}: timestamps are not ordered")
            times.append(timestamp)
            values.append(value)
            final_sample = row.get("final_sample") is True

    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"{path}: changed while being read")
    if not times:
        raise ValueError(f"{path}: no coverage rows")
    if not final_sample:
        raise ValueError(f"{path}: final sample not present")
    return Curve(np.asarray(times), np.asarray(values))


def render(output: Path, metric: str, y_label: str, window_s: float = 60.0) -> None:
    configure_style(VARIANT_RCPARAMS)
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), squeeze=False)

    for axis, kernel in zip(axes.flat, KERNELS):
        axis.set_title(kernel.title, pad=5)
        axis.text(
            0.0,
            1.12,
            kernel.project_label,
            transform=axis.transAxes,
            color="#555555",
            fontsize=6.8,
            fontweight="bold",
            va="bottom",
        )

        ymax = 0.0
        for backend in BACKENDS:
            _, color = BACKEND_STYLES[backend]
            for vconfig in ("on", "off"):
                _, linestyle = VCONFIG_STYLES[vconfig]
                curve = load_curve(
                    kernel.coverage_dir
                    / backend
                    / vconfig
                    / "coverage.jsonl",
                    metric,
                )
                axis.plot(
                    curve.times,
                    curve.values,
                    color=color,
                    linestyle=linestyle,
                    drawstyle="steps-post",
                    solid_capstyle="round",
                    zorder=3 if vconfig == "on" else 4,
                )
                ymax = max(ymax, float(curve.values.max()))

        axis.set_xlim(0.0, window_s)
        axis.set_ylim(0.0, max(1.0, ymax * 1.08))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.yaxis.set_major_locator(
            MaxNLocator(nbins=4, min_n_ticks=3, integer=True)
        )

        if (
            metric == "cfg_sites"
            and kernel.project_id == "cutlass_variants"
            and window_s == 60.0
        ):
            axis.annotate(
                "On-CFG includes dispatcher\nswitch/guard sites\n(polluted by design)",
                xy=(58.0, 25.0),
                xytext=(25.0, 15.5),
                fontsize=6.2,
                color="#555555",
                arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 0.6},
            )

    handles = [
        *[
            Line2D([0], [0], color=color, label=label)
            for label, color in BACKEND_STYLES.values()
        ],
        *[
            Line2D([0], [0], color="#333333", linestyle=style, label=label)
            for label, style in VCONFIG_STYLES.values()
        ],
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=5,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.4,
    )
    window_label = (
        "60s cells"
        if window_s == 60.0
        else f"first {window_s:g}s of 60s cells"
    )
    figure.suptitle(
        f"derived variants, {window_label}, seed 1",
        fontsize=11.0,
        fontweight="bold",
        y=0.99,
    )
    figure.supxlabel("Time (s)", fontsize=9.0, y=0.03)
    figure.supylabel(y_label, fontsize=9.0, x=0.012)
    figure.subplots_adjust(
        top=0.71,
        bottom=0.17,
        left=0.07,
        right=0.99,
        wspace=0.30,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.resolve(), format="png", dpi=300)
    plt.close(figure)


def render_origin_only(output: Path, metric: str, y_label: str) -> None:
    configure_style(VARIANT_RCPARAMS)
    figure, axes = plt.subplots(1, 5, figsize=(13.8, 3.2), squeeze=False)
    _, origin_color = BACKEND_STYLES["origin"]

    for axis, kernel in zip(axes.flat, NEW_KERNELS):
        axis.set_title(kernel.title, pad=5)
        axis.text(
            0.0,
            1.12,
            kernel.project_label,
            transform=axis.transAxes,
            color="#555555",
            fontsize=6.8,
            fontweight="bold",
            va="bottom",
        )

        ymax = 0.0
        on_curve: Curve | None = None
        for vconfig in ("on", "off"):
            _, linestyle = VCONFIG_STYLES[vconfig]
            curve = load_curve(
                kernel.coverage_dir
                / "origin"
                / vconfig
                / "coverage.jsonl",
                metric,
            )
            axis.plot(
                np.minimum(curve.times, 60.0),
                curve.values,
                color=origin_color,
                linestyle=linestyle,
                drawstyle="steps-post",
                solid_capstyle="round",
                zorder=3 if vconfig == "on" else 4,
            )
            ymax = max(ymax, float(curve.values.max()))
            if vconfig == "on":
                on_curve = curve

        axis.set_xlim(0.0, 60.0)
        axis.set_ylim(0.0, max(1.0, ymax * 1.08))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.yaxis.set_major_locator(
            MaxNLocator(nbins=4, min_n_ticks=3, integer=True)
        )

        if metric == "cfg_sites" and kernel.cfg_annotation_y is not None:
            assert on_curve is not None
            axis.annotate(
                "On includes dispatcher\nswitch/guard sites",
                xy=(60.0, float(on_curve.values[-1])),
                xycoords="data",
                xytext=(0.04, kernel.cfg_annotation_y),
                textcoords="axes fraction",
                fontsize=5.8,
                color="#555555",
                arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 0.6},
            )

    handles = [
        Line2D(
            [0],
            [0],
            color=origin_color,
            linestyle=style,
            label=label,
        )
        for label, style in VCONFIG_STYLES.values()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=2,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.4,
    )
    figure.suptitle(
        "new derived variants, Origin only, 60s cells, seed 1",
        fontsize=11.0,
        fontweight="bold",
        y=0.99,
    )
    figure.supxlabel("Time (s)", fontsize=9.0, y=0.03)
    figure.supylabel(y_label, fontsize=9.0, x=0.008)
    figure.subplots_adjust(
        top=0.71,
        bottom=0.17,
        left=0.055,
        right=0.995,
        wspace=0.30,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.resolve(), format="png", dpi=300)
    plt.close(figure)


def main() -> int:
    render(MEMCOV_OUTPUT, "memory_features", "SIMT MemCov features")
    render(CFG_OUTPUT, "cfg_sites", "CFG sites")
    render(
        MEMCOV_ZOOM_OUTPUT,
        "memory_features",
        "SIMT MemCov features",
        window_s=2.0,
    )
    render(CFG_ZOOM_OUTPUT, "cfg_sites", "CFG sites", window_s=2.0)
    render_origin_only(
        MEMCOV2_OUTPUT,
        "memory_features",
        "SIMT MemCov features",
    )
    render_origin_only(CFG2_OUTPUT, "cfg_sites", "CFG sites")
    for output in (
        MEMCOV_OUTPUT,
        CFG_OUTPUT,
        MEMCOV_ZOOM_OUTPUT,
        CFG_ZOOM_OUTPUT,
        MEMCOV2_OUTPUT,
        CFG2_OUTPUT,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
