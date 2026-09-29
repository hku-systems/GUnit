#!/usr/bin/env python3
"""Render the two master RQ2 time-series coverage figures."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
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

from benchmark.rq2.plot_coverage import CoverageCurve, load_coverage_curves  # noqa: E402
from benchmark.rq2.variants._common import (  # noqa: E402
    BACKEND_STYLES,
    MASTER_RCPARAMS,
    VCONFIG_STYLES,
    Curve,
    configure_style,
)


RESULTS_ROOT = REPO_ROOT / "benchmark/rq2/results"
FIGURE_ROOT = REPO_ROOT / "benchmark/rq2/reports/figures"
OUTPUTS = {
    "memory_features": FIGURE_ROOT / "master-memcov.png",
    "cfg_sites": FIGURE_ROOT / "master-cfg.png",
}
BACKENDS = ("origin", "rapid", "rapid2")
FORMAL_CONFIGS = ("origin-on", "rapid-w4-on", "rapid2-off", "rapid2-on")
FAMILY_COLORS = {
    "Formal (5 seeds)": "#6B5B95",
    "Wave 1": "#0072B2",
    "Wave 2": "#009E73",
    "Wave 3": "#D55E00",
    "Variants": "#A05A2C",
}


@dataclass(frozen=True)
class Panel:
    family: str
    project_id: str
    project: str
    title: str
    window_s: float
    source: str
    root: str
    kernel_id: str
    expected_cells: int


def formal(
    campaign: str, workload_id: str, project: str, title: str
) -> Panel:
    return Panel(
        "Formal (5 seeds)", "", project, title, 60.0, "formal", campaign, workload_id, 4
    )


def coverage_panels(
    family: str,
    campaign: str,
    window_s: float,
    entries: tuple[tuple[str, str, str, str, int], ...],
) -> tuple[Panel, ...]:
    return tuple(
        Panel(
            family,
            project_id,
            project,
            title,
            window_s,
            "coverage",
            campaign,
            kernel,
            cells,
        )
        for project_id, project, kernel, title, cells in entries
    )


PANELS = (
    formal(
        "benchmark/rq2/results/formal-classic-shoc-9cfg-5seed-60s-20260802-gpu2",
        "shoc_reduction",
        "SHOC",
        "Reduction",
    ),
    formal(
        "benchmark/rq2/results/formal-classic-shoc-9cfg-5seed-60s-20260802-gpu2",
        "shoc_radix_sort",
        "SHOC",
        "RadixSortBlock",
    ),
    formal(
        "benchmark/rq2/results/formal-classic-shoc-9cfg-5seed-60s-20260802-gpu2",
        "shoc_scan",
        "SHOC",
        "ScanSingleBlock",
    ),
    formal(
        "benchmark/rq2/results/formal-classic-big3-9cfg-5seed-60s-20260802-gpu3",
        "cutlass_gemm",
        "CUTLASS",
        "ReferenceGemm",
    ),
    formal(
        "benchmark/rq2/results/formal-classic-big3-9cfg-5seed-60s-20260802-gpu3",
        "flashattention_device1xn",
        "FlashAttention",
        "device1xN",
    ),
    formal(
        "benchmark/rq2/results/formal-classic-big3-9cfg-5seed-60s-20260802-gpu3",
        "pytorch_batchnorm",
        "PyTorch",
        "BatchNorm",
    ),
    formal(
        "benchmark/rq2/results/formal-synth-complex-9cfg-5seed-60s-20260801-gpu2",
        "synth_complex",
        "RAPID",
        "synth_complex",
    ),
    *coverage_panels(
        "Wave 1",
        "build/e2e/third-party-fuzz/20260801-rq2-expanded",
        10.0,
        (
            ("gpurir", "gpuRIR", "_Z14diffRev_kernelPfS_S_S_iiii__e9523a2e", "diffRev", 6),
            ("gpurir", "gpuRIR", "_Z14envPred_kernelPfS_S_S_iiiffffffffff__4f38b2df", "envPred", 6),
            ("gpurir", "gpuRIR", "_Z17calcAmpTau_kernelPfS_S_S_S_S_S_iifffffffffiiiiiff__169eb460", "calcAmpTau", 6),
            ("gpurir", "gpuRIR", "_Z18generateRIR_kernelPfS_S_iiiiif__cd099106", "generateRIR", 6),
            ("gpurir", "gpuRIR", "_Z24h2RIR_to_floatRIR_kernelP7__half2Pfii__d0f15d1b", "h2RIR_to_floatRIR", 6),
            ("cudasift", "CudaSift", "_Z12FindMaxCorr4P9SiftPointS0_ii__8a70bde8", "FindMaxCorr4", 6),
            ("cudasift", "CudaSift", "_Z16MatchSiftPoints2P9SiftPointS0_Pfii__35c51f10", "MatchSiftPoints2", 6),
            ("phantom-fhe", "Phantom-FHE", "_Z23inplace_fnwt_radix2_optPmPKmS1_PK8DModulusm__39057492", "FNWT radix-2", 6),
            ("phantom-fhe", "Phantom-FHE", "_Z27bconv_matmul_unroll4_kernelPmPKmS1_PK8DModulusmS4_mm__f69f3327", "BConv matmul", 6),
            ("phantom-fhe", "Phantom-FHE", "_ZN7phantom38divide_and_round_ntt_inv_scalar_ker__60451525", "Divide and round", 6),
            ("phantom-fhe", "Phantom-FHE", "_ZN7phantom4util28apply_galois_ntt_permutationEPmPKmPKjmm__39e18b32", "Galois permute", 6),
        ),
    ),
    *coverage_panels(
        "Wave 2",
        "build/e2e/third-party-fuzz/20260802-rq2-wave2",
        60.0,
        (
            ("tensorrt_clip", "TensorRT", "_Z34modulatedDeformableIm2colGpuKernelIfEviPKT_S__418d0e4b", "DeformableIm2col", 6),
            ("tensorrt_clip", "TensorRT", "_ZN8nvinfer16plugin19cropAndResizeKernelIfEEviPKT_PKfiiiiiiifPf__1e27108f", "cropAndResize", 6),
            ("cutlass_basic", "CUTLASS", "_Z20ReferenceSyrk_kerneliidPKdidPdi__43f64bea", "ReferenceSyrk", 6),
            ("cutlass_basic", "CUTLASS", "_ZN7cutlass19nhwc_padding_kernelI6float2EEviiiiiT_PKS2_PS2___af58c0d1", "nhwc_padding", 6),
            ("llama_cpp", "llama.cpp", "_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout2EEvPKfPKiPvlllllii__b56d9d0f", "quantize_mmq_q8_1", 6),
            ("llama_cpp", "llama.cpp", "_ZL30upscale_f32_bilinear_antialiasPKfPfiiiiiiiiiifffff__e079a648", "upscale_f32_bilinear", 6),
            ("kaldi", "Kaldi", "_ZN5kaldi21power_spectrum_kernelEiPKfiPfib__7815ddf8", "power_spectrum", 6),
            ("kaldi", "Kaldi", "_ZN5kaldi29batched_extract_window_kernelEPKNS_8L__5231516c", "batched_extract_window", 6),
            ("kaldi", "Kaldi", "_ZN5kaldi29batched_process_window_kernelEPKNS_8L__abf3b5ce", "batched_process_window", 3),
            ("cuda_samples", "CUDA Samples", "_ZL16inverseCNDKernelPfPjj__168b4c09", "inverseCNDKernel", 6),
            ("cuda_samples", "CUDA Samples", "_Z11Mandelbrot1IfEvP6uchar4iiiT_S2_S2_S2_S2_S0_iiiib__deec3819", "Mandelbrot1", 6),
            ("gpujpeg", "GPUJPEG", "_Z20channel_remap_kernelIL20gpujpeg_pixel_format3EEvPhiiij__587fa403", "channel_remap", 6),
            ("gpujpeg", "GPUJPEG", "_Z22gpujpeg_dct_gpu_kernelILi4EEviiPhjPsiPKf__1d4723f3", "gpujpeg_dct", 6),
            ("lietorch", "LieTorch", "cholesky_solve6x6_forward_raw_kernel__3e2391fd", "cholesky_solve6x6", 6),
        ),
    ),
    *coverage_panels(
        "Wave 3",
        "build/e2e/third-party-fuzz/20260802-rq2-wave3",
        60.0,
        (
            ("apex", "Apex", "apex_adam_cuda_kernel__bbf82dcc", "adam", 6),
            ("apex", "Apex", "apex_maybe_cast_kernel__206ea46b", "maybe_cast", 6),
            ("deepspeed", "DeepSpeed", "deepspeed_lamb_cuda_kernel_part3__234a8314", "LAMB part3", 6),
            ("flash_attention", "FlashAttention", "flash_bwd_dot_do_o_pod_kernel__4271cecf", "bwd_dot_do_o", 6),
            ("flash_attention", "FlashAttention", "flash_bwd_convert_dq_pod_kernel__0ad34144", "bwd_convert_dq", 6),
            ("llama_cpp", "llama.cpp", "topk_moe_cuda_128_no_bias_adapter__fb60f880", "topk_moe_128", 6),
        ),
    ),
    *coverage_panels(
        "Variants",
        "build/e2e/third-party-fuzz/20260802-rq2-wave3",
        60.0,
        (
            ("kaldi_variants", "Kaldi", "_ZN5kaldi37power_spectrum_kernel_blockdim_strideEiPKfiPfib__48ee4406", "power_spectrum_blockdim", 6),
            ("apex_variants", "Apex", "_Z23index_mul_2d_float_vgeoPfPKfS1_PKlll__88cdac70", "index_mul_2d_vgeo", 6),
            ("cutlass_variants", "CUTLASS", "_Z30reference_syrk_dispatch_kerneljiidPKdidPdi__2f6a4695", "SYRK dispatcher", 6),
            ("cutlass_variants", "CUTLASS", "_Z30reference_gemm_dispatch_kerneljiiifPKfiS0_ifPfi__efc18fc2", "GEMM dispatcher", 6),
            ("cuda_samples_variants", "CUDA Samples", "inverseCNDKernel_vgeo", "inverseCND vgeo", 6),
            ("kaldi_variants/direct_vconfig", "Kaldi", "batched_extract_window_vgeo", "extract_window vgeo", 6),
            ("gpujpeg_variants", "GPUJPEG", "_Z27gpujpeg_dct_dispatch_kerneljiiPhjPsiPKf__82472776", "DCT dispatcher", 6),
            ("cutlass_variants", "CUTLASS", "_Z30reference_trmm_dispatch_kerneljiidPKdiS0_iPdi__a79a6b09", "TRMM dispatcher", 6),
        ),
    ),
)


def compute_t99(curve: Curve) -> float:
    final = float(curve.values[-1])
    if final <= 0.0:
        return 0.0
    index = int(np.flatnonzero(curve.values >= 0.99 * final)[0])
    return float(curve.times[index])


def compute_zoom_limit(curves: tuple[Curve, ...], window_s: float) -> float:
    t99 = max(compute_t99(curve) for curve in curves)
    return min(window_s, max(2.0, 1.3 * t99))


def load_jsonl_curve(path: Path, metric: str, window_s: float) -> Curve:
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
                raise ValueError(f"{path}:{line_number}: invalid coverage row") from error
            if not math.isfinite(timestamp) or not math.isfinite(value):
                raise ValueError(f"{path}:{line_number}: non-finite coverage value")
            if timestamp < 0.0 or value < 0.0:
                raise ValueError(f"{path}:{line_number}: negative coverage value")
            if times and (timestamp < times[-1] or value < values[-1]):
                raise ValueError(f"{path}:{line_number}: coverage trace regressed")
            times.append(min(timestamp, window_s))
            values.append(value)
            final_sample = row.get("final_sample") is True
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"{path}: changed while being read")
    if not times or not final_sample:
        raise ValueError(f"{path}: incomplete coverage trace")
    return Curve(np.asarray(times), np.asarray(values))


@lru_cache(maxsize=None)
def formal_campaign(path: str) -> tuple[CoverageCurve, ...]:
    return load_coverage_curves(REPO_ROOT / path)


def mean_formal_curve(
    curves: list[CoverageCurve], metric: str, window_s: float
) -> Curve:
    if len(curves) != 5:
        raise ValueError(f"expected five formal seeds, found {len(curves)}")
    grid = np.linspace(0.0, window_s, 601)
    samples = []
    for curve in curves:
        times = np.asarray(
            [min(point.timestamp_ms / 1000.0, window_s) for point in curve.points]
        )
        values = np.asarray([getattr(point, metric) for point in curve.points])
        indices = np.searchsorted(times, grid, side="right") - 1
        if indices.min() < 0 or values.dtype == object:
            raise ValueError(f"{curve.trial_id}: incomplete {metric} curve")
        samples.append(values[indices].astype(float))
    return Curve(grid, np.vstack(samples).mean(axis=0))


def load_panel(panel: Panel, metric: str) -> dict[str, Curve]:
    if panel.source == "formal":
        grouped: dict[str, list[CoverageCurve]] = {
            configuration: [] for configuration in FORMAL_CONFIGS
        }
        for curve in formal_campaign(panel.root):
            if (
                curve.workload_id == panel.kernel_id
                and curve.configuration_id in grouped
            ):
                grouped[curve.configuration_id].append(curve)
        curves = {
            configuration: mean_formal_curve(seeds, metric, panel.window_s)
            for configuration, seeds in grouped.items()
        }
    else:
        coverage_root = (
            REPO_ROOT
            / panel.root
            / panel.project_id
            / "kernels"
            / panel.kernel_id
            / "coverage"
        )
        curves = {}
        for backend in BACKENDS:
            for mode in VCONFIG_STYLES:
                path = coverage_root / backend / mode / "coverage.jsonl"
                if path.is_file():
                    curves[f"{backend}-{mode}"] = load_jsonl_curve(
                        path, metric, panel.window_s
                    )
    if len(curves) != panel.expected_cells:
        raise ValueError(
            f"{panel.family}/{panel.project}/{panel.title}: expected "
            f"{panel.expected_cells} curves, found {len(curves)}"
        )
    return curves


def line_style(configuration: str) -> tuple[str, str]:
    if configuration == "rapid-w4-on":
        backend, mode = "rapid", "on"
    else:
        backend, mode = configuration.split("-", maxsplit=1)
    return BACKEND_STYLES[backend][1], VCONFIG_STYLES[mode][1]


def render(metric: str, output: Path) -> tuple[int, tuple[float, ...]]:
    configure_style(MASTER_RCPARAMS)
    columns = 7
    rows = math.ceil(len(PANELS) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.2 * columns, 2.25 * rows), squeeze=False
    )
    zoom_limits = []
    for axis, panel in zip(axes.flat, PANELS):
        curves = load_panel(panel, metric)
        for configuration, curve in curves.items():
            color, linestyle = line_style(configuration)
            axis.step(
                curve.times,
                curve.values,
                where="post",
                color=color,
                linestyle=linestyle,
                solid_capstyle="round",
                zorder=3 if configuration.endswith("-on") else 2,
            )
        limit = compute_zoom_limit(tuple(curves.values()), panel.window_s)
        zoom_limits.append(limit)
        ymax = max(float(curve.values.max()) for curve in curves.values())
        wave_suffix = " [10s]" if panel.family == "Wave 1" else ""
        axis.set_title(f"{panel.project}\n{panel.title}{wave_suffix}", pad=5, linespacing=0.95)
        axis.text(
            0.5,
            1.26,
            panel.family,
            transform=axis.transAxes,
            color=FAMILY_COLORS[panel.family],
            fontsize=6.2,
            fontweight="bold",
            ha="center",
        )
        axis.set_xlim(0.0, limit)
        axis.set_ylim(0.0, max(1.0, 1.07 * ymax))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, integer=True))
        axis.tick_params(axis="both", direction="out", length=2.5, width=0.6, pad=1.5)
        if limit < panel.window_s:
            axis.text(
                0.98,
                0.04,
                f"x≤{limit:.1f}s",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.7,
                color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
            )
    for axis in axes.flat[len(PANELS) :]:
        axis.set_visible(False)

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
    metric_title = "SIMT MemCov features" if metric == "memory_features" else "Device CFG sites"
    figure.suptitle(f"RQ2 coverage growth — {metric_title}", fontsize=12, fontweight="bold", y=0.996)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.973),
        ncol=5,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.5,
    )
    figure.text(
        0.5,
        0.952,
        "Formal panels: five-seed mean; Origin-on, RAPID-w4-on, RAPID2-off/on",
        ha="center",
        va="top",
        fontsize=6.4,
        color="#444444",
    )
    figure.supxlabel("Time (s)", fontsize=8.5, y=0.012)
    figure.supylabel(metric_title, fontsize=8.5, x=0.006)
    figure.subplots_adjust(
        top=0.90,
        bottom=0.045,
        left=0.045,
        right=0.993,
        hspace=0.90,
        wspace=0.36,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.resolve(), format="png", dpi=300)
    plt.close(figure)
    zoomed = sum(limit < panel.window_s for panel, limit in zip(PANELS, zoom_limits))
    return zoomed, tuple(zoom_limits)


def main() -> int:
    if len(PANELS) != 46:
        raise ValueError(f"expected 46 panels, found {len(PANELS)}")
    for metric, output in OUTPUTS.items():
        zoomed, limits = render(metric, output)
        label = "MemCov" if metric == "memory_features" else "CFG"
        print(f"{label}: {len(PANELS)} panels, {zoomed} zoomed")
        print("x-limits: " + ", ".join(f"{limit:.2f}" for limit in limits))
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
