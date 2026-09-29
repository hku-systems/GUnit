#!/usr/bin/env python3
"""Export verified RQ1 medians to the paper-facing throughput CSV schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_LABELS = {
    "shoc_reduction": "reduction",
    "shoc_radix_sort": "sort",
    "shoc_scan": "scan",
    "cutlass_gemm": "gemm",
    "flashattention_device1xn": "device1xN",
    "pytorch_batchnorm": "batchnorm",
    "apex_maybe_cast": "maybe_cast",
    "llama_upscale_f32_bilinear": "upscale",
    "gpurir_generate_rir": "generateRIR",
    "apex_index_mul_2d_vgeo": "index_mul_2d",
    "cuda_samples_inverse_cnd": "inverseCND",
    "synth_complex": "byte_fingerprint",
}
REQUIRED_BACKENDS = (
    "cufuzz",
    "origin",
    "origin-no-feedback",
    "rapid-w4",
    "rapid-no-feedback-w4",
    "rapid2",
    "rapid2-no-feedback",
)


def paper_row(
    workload_id: str, label: str, values: dict[str, float]
) -> dict[str, str | float]:
    missing = set(REQUIRED_BACKENDS) - set(values)
    if missing:
        raise RuntimeError(f"missing RQ1 backend medians for {workload_id}: {sorted(missing)}")
    return {
        "workload_id": workload_id,
        "workload": label,
        "cufuzz": values["cufuzz"],
        "libafl_on": values["origin"],
        "libafl_off": values["origin-no-feedback"],
        "gunit_s_on": values["rapid-w4"],
        "gunit_s_off": values["rapid-no-feedback-w4"],
        "gunit_on": values["rapid2"],
        "gunit_off": values["rapid2-no-feedback"],
    }


def export(
    inputs: list[Path],
    suite_path: Path,
    output: Path,
    expected_repetitions: int = 2,
) -> None:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))["workloads"]
    medians: dict[str, dict[str, float]] = {}
    for input_dir in inputs:
        with (input_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                workload = row["workload_id"]
                if int(row["repetitions"]) != expected_repetitions:
                    raise RuntimeError(
                        "RQ1 paper export repetition mismatch: "
                        f"{workload} expected={expected_repetitions} "
                        f"actual={row['repetitions']}"
                    )
                if workload in medians and row["backend"] in medians[workload]:
                    raise RuntimeError(f"duplicate RQ1 summary row: {workload} {row['backend']}")
                medians.setdefault(workload, {})[row["backend"]] = float(
                    row["median_exec_per_sec"]
                )
    if set(medians) != set(suite):
        raise RuntimeError(
            f"RQ1 export workload mismatch: missing={sorted(set(suite) - set(medians))} "
            f"extra={sorted(set(medians) - set(suite))}"
        )
    rows = [paper_row(workload, PAPER_LABELS[workload], medians[workload]) for workload in suite]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--suite", type=Path, default=REPO_ROOT / "benchmark/rq1/suite.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    export(
        args.input,
        args.suite,
        args.output,
        expected_repetitions=args.repetitions,
    )


if __name__ == "__main__":
    main()
