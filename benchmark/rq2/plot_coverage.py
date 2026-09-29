#!/usr/bin/env python3
"""Validate and plot RQ2 device-CFG and MemCov growth over completed executions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from benchmark.rq2.coverage_data import (
    WORKLOAD_TITLES,
    CoverageCurve,
    CoveragePoint,
    SkippedCoverageCell,
    load_coverage_curves,
    load_coverage_data,
)
from benchmark.rq2.coverage_render import render_metric
from benchmark.rq2.coverage_stats import build_saturation_rows


def _load_coverage_data(
    result_root: Path,
) -> tuple[tuple[CoverageCurve, ...], tuple[SkippedCoverageCell, ...]]:
    """Compatibility wrapper for the former private loader."""
    return load_coverage_data(result_root)


def _render_metric(
    curves: tuple[CoverageCurve, ...],
    *,
    attribute: str,
    ylabel: str,
    output_base: Path,
    normalize_cfg: bool = True,
) -> tuple[Path, Path]:
    """Compatibility wrapper for the former private renderer."""
    return render_metric(
        curves,
        attribute=attribute,
        ylabel=ylabel,
        output_base=output_base,
        normalize_cfg=normalize_cfg,
    )


def render_coverage_figures(
    result_root: Path,
    *,
    review_dir: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    result_root = result_root.resolve()
    curves, skipped_cells = _load_coverage_data(result_root)
    saturation_rows = build_saturation_rows(curves, skipped_cells)
    saturation_path = result_root / "coverage_saturation.jsonl"
    saturation_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in saturation_rows),
        encoding="utf-8",
    )
    cfg_pdf, cfg_png = _render_metric(
        curves,
        attribute="cfg_sites",
        ylabel="Device CFG coverage (%)",
        output_base=result_root / "rq2_cfg_coverage_executions",
        normalize_cfg=True,
    )
    mem_pdf, mem_png = _render_metric(
        curves,
        attribute="memory_features",
        ylabel="MemCov occupied buckets",
        output_base=result_root / "rq2_memcov_executions",
    )
    if review_dir is not None:
        review_dir = review_dir.resolve()
        review_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_png, review_dir / cfg_png.name)
        shutil.copy2(mem_png, review_dir / mem_png.name)
    return cfg_pdf, cfg_png, mem_pdf, mem_png


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--review-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    outputs = render_coverage_figures(args.result_root, review_dir=args.review_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "outputs": [str(path) for path in outputs],
                "saturation": str(
                    args.result_root.resolve() / "coverage_saturation.jsonl"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
