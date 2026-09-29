"""Shared command-line entry point for benchmark catalog verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def verify_catalog_main(
    *,
    description: str,
    default_root: Path,
    load_catalog: Callable[[Path], Any],
    verify_catalog: Callable[[Path], Sequence[str]],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="benchmark root (default: this script's directory)",
    )
    root = parser.parse_args().root.resolve()
    errors = verify_catalog(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    catalog = load_catalog(root)
    print(
        json.dumps(
            {
                "schema_version": catalog.schema_version,
                "status": "ok",
                "workload_ids": [item.workload_id for item in catalog.workloads],
            },
            sort_keys=True,
        )
    )
    return 0
