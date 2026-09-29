"""Shared helpers for the Phase 2 kernel-rewrite pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelPhase1Artifacts:
    kernel_id: str
    kernel_dir: Path
    phase2_dir: Path
    kernel_bc: Path
    manifest_path: Path
    manifest: dict[str, Any]
    metadata: dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def resolve_kernel_dir(run_dir: Path, entry_dir: str) -> Path:
    candidate = Path(entry_dir)
    if candidate.is_absolute():
        return candidate.resolve()
    return (run_dir / candidate).resolve()
