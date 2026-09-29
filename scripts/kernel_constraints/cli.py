#!/usr/bin/env python3
"""Create a resolved Phase 1 run with reviewed constraint overrides."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .overrides import apply_constraint_overrides


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="kernel-constraints",
        description="Apply symbol-keyed constraint overrides to a copied Phase 1 run",
    )
    parser.add_argument("--run-dir", required=True, help="Original Phase 1 run directory")
    parser.add_argument("--out-dir", required=True, help="New resolved Phase 1 run directory")
    parser.add_argument("--overrides", required=True, help="Constraint override registry JSON")
    args = parser.parse_args()

    try:
        report = apply_constraint_overrides(
            Path(args.run_dir),
            Path(args.out_dir),
            Path(args.overrides),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    counts = report["counts"]
    print(
        "kernel-constraints: "
        f"project={report['project']} applied={counts['applied']} "
        f"unmatched={counts['unmatched']} out={report['resolved_run_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
