#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

from shared_runtime import RuntimeSmokeRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Phase 2 runtime smoke cases for a given run dir.")
    parser.add_argument("--run-dir", required=True, help="Phase 1/Phase 2 run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    results = RuntimeSmokeRunner().run_all(run_dir=run_dir)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
