"""Stable kernel id helpers for artifact-mode discovery."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def make_kernel_id(
    symbol: str,
    args: list[dict[str, Any]],
    line: int,
    source_path: str = "",
    source_label: str | None = None,
    variant_id: str | None = None,
) -> str:
    """Produce a stable filesystem-safe kernel_id."""
    label = source_label if source_label else source_path
    parts = [label, ":", symbol, "("]
    parts.append(",".join(str(a["type"]) for a in args))
    parts.append(")")
    parts.append(f"@{line}")
    if variant_id:
        parts.append(f"#v={variant_id}")
    digest = hashlib.sha256("".join(parts).encode()).hexdigest()[:8]

    safe_symbol = re.sub(r"[^A-Za-z0-9_]", "_", symbol)
    if len(safe_symbol) > 64:
        safe_symbol = safe_symbol[:48]

    return f"{safe_symbol}__{digest}"
