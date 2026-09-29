#!/usr/bin/env python3
"""Print manifest argument types and their resolved definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.type_resolution import load_manifest, resolve_manifest_arg_types


def _print_text(rows: list[dict[str, Any]]) -> None:
    current_kernel: tuple[str, str] | None = None
    for row in rows:
        kernel_key = (row["kernel"], row["symbol"])
        if kernel_key != current_kernel:
            current_kernel = kernel_key
            print(f"kernel: {row['kernel']}")
            print(f"  symbol: {row['symbol']}")
        lhs = f"  arg[{row['index']}] {row['name']}: {row['type']}"
        rhs_parts: list[str] = []
        if row.get("qualified_name"):
            kind = row.get("decl_kind")
            if kind:
                rhs_parts.append(f"{kind} {row['qualified_name']}")
            else:
                rhs_parts.append(str(row["qualified_name"]))
        if row.get("definition_status"):
            rhs_parts.append(f"definition={row['definition_status']}")
        if row.get("definition_loc"):
            rhs_parts.append(str(row["definition_loc"]))
        elif row.get("decl_loc"):
            rhs_parts.append(f"decl={row['decl_loc']}")
        if row.get("usr"):
            rhs_parts.append(f"usr={row['usr']}")
        if rhs_parts:
            print(lhs + " -> " + ", ".join(rhs_parts))
        else:
            print(lhs)
        if row.get("include_root"):
            print(f"    add include path: -I{row['include_root']}")
        if row.get("include_stmt"):
            print(f"    include with: {row['include_stmt']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to manifest.json")
    parser.add_argument("--json", action="store_true", help="Emit parsed rows as JSON")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    rows = resolve_manifest_arg_types(manifest, args.manifest)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_text(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
