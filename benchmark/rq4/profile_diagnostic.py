#!/usr/bin/env python3
"""Render the seven-segment host-loop view from unified profiling JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmark.rq4.profile import (
    MAIN_LOOP_SEGMENTS,
    load_profiling_records,
    validate_profiling_records,
)


def diagnostic_segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_profiling_records(records)
    by_key = {
        (record["domain"], record["segment"]): record for record in records
    }
    feedback_ns = sum(
        int(by_key[("cpu_feedback", segment)]["total"])
        for segment in ("predicate", "metadata")
    )
    rows = []
    for segment in MAIN_LOOP_SEGMENTS:
        record = by_key[("main_loop", segment)]
        exclusive_ns = int(record["total"])
        nested_ns = feedback_ns if segment == "evaluate" else 0
        rows.append(
            {
                "segment": segment,
                "count": int(record["count"]),
                "total_ns": exclusive_ns + nested_ns,
                "exclusive_ns": exclusive_ns,
                "nested_ns": nested_ns,
                "exclusive_max_ns": record["max"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.is_dir():
        profiles = [
            json.loads(line)
            for line in (args.input / "profiles.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        rows = []
        for profile in profiles:
            profiling_record = profile.get("profiling_record")
            if profiling_record is None:
                continue
            for row in diagnostic_segments(
                load_profiling_records(Path(profiling_record))
            ):
                rows.append(
                    {
                        "mode": profile["mode"],
                        "workload_id": profile["workload_id"],
                        "configuration": profile["configuration"],
                        "repetition": int(profile.get("repetition", 1)),
                        **row,
                    }
                )
    else:
        rows = diagnostic_segments(load_profiling_records(args.input))
    if not rows:
        raise RuntimeError("profiling diagnostic input is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
