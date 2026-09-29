#!/usr/bin/env python3
"""Verify the RQ1 catalog and all source provenance artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.rq1.schema import load_catalog, verify_catalog
from benchmark.verify import verify_catalog_main


def main() -> int:
    return verify_catalog_main(
        description=__doc__ or "",
        default_root=Path(__file__).resolve().parent,
        load_catalog=load_catalog,
        verify_catalog=verify_catalog,
    )


if __name__ == "__main__":
    raise SystemExit(main())
