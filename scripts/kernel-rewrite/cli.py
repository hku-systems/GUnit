#!/usr/bin/env python3
"""Kernel rewrite pipeline CLI.

Usage:
    python3 scripts/kernel-rewrite/cli.py --run-dir build/kernel-smoke/<run_id>
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from phase2 import run_phase2


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="kernel-rewrite",
        description="RAPID kernel rewrite post-processing",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to an existing Phase 1 run dir (contains index.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any kernel fails in rewrite stage",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    print("kernel-rewrite")
    print(f"  run_dir: {run_dir}")

    try:
        summary = run_phase2(run_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    counts = summary.get("counts", {})
    total = int(counts.get("total", 0))
    built = int(counts.get("built", 0))
    failed = int(counts.get("failed", 0))
    skipped = int(counts.get("skipped", 0))
    print(f"  Rewrite: {total} total, {built} built, {failed} failed, {skipped} skipped")
    print(f"  Output: {run_dir / 'rewrite_summary.json'}")

    if args.strict and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
