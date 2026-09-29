"""Artifact identity helpers for shared Phase2 and RQ1 backend builds."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .schema import sha256_file


SHARED_PHASE2_FILES = (
    "../manifest.json",
    "build_spec.json",
    "kernel.device.bc",
    "gen/fuzzer_decode.v1.cuh",
    "gen/fuzzer_invoke.v1.cuh",
)
TASK_ENVELOPE_HEADER_SIZE = 32


def envelope_payload_size(seed: bytes) -> int:
    if len(seed) < TASK_ENVELOPE_HEADER_SIZE:
        raise RuntimeError("canonical seed is shorter than the task envelope")
    payload_size = int.from_bytes(seed[24:32], "little")
    if payload_size != len(seed) - TASK_ENVELOPE_HEADER_SIZE:
        raise RuntimeError("canonical seed payload length is inconsistent")
    return payload_size


def dump_seed(fuzzer: Path, manifest: Path) -> bytes:
    completed = subprocess.run(
        [
            str(fuzzer.resolve()),
            "/dev/null",
            "--manifest",
            str(manifest.resolve()),
            "--dump-seed",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        text=True,
        capture_output=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            return bytes.fromhex(line.strip())
        except ValueError:
            continue
    raise RuntimeError(f"fuzzer produced no canonical seed for {manifest}")


def timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def artifact_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display_path = (
        str(resolved.relative_to(relative_to.resolve()))
        if relative_to is not None and resolved.is_relative_to(relative_to.resolve())
        else str(resolved)
    )
    return {
        "path": display_path,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def hash_shared_phase2(phase2_dir: Path) -> dict[str, dict[str, Any]]:
    phase2_dir = phase2_dir.resolve()
    result: dict[str, dict[str, Any]] = {}
    for relative in SHARED_PHASE2_FILES:
        path = (phase2_dir / relative).resolve()
        if not path.is_file():
            raise RuntimeError(f"missing shared Phase2 artifact: {relative}")
        key = "manifest.json" if relative == "../manifest.json" else relative
        result[key] = artifact_identity(path, relative_to=phase2_dir.parent)
    return result


def verify_shared_phase2_unchanged(
    phase2_dir: Path,
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    actual = hash_shared_phase2(phase2_dir)
    for relative, identity in expected.items():
        if actual.get(relative) != identity:
            raise RuntimeError(f"shared Phase2 artifact drift: {relative}")


def hash_backend_report(report_path: Path) -> dict[str, Any]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError(f"backend report must be an object: {report_path}")
    artifacts: dict[str, Any] = {
        "backend_build.json": artifact_identity(
            report_path,
            relative_to=report_path.parent,
        )
    }
    for report_key, artifact_key in (
        ("selected_device_bc", "optimized_device_bitcode"),
        ("module_ptx", "module.ptx"),
        ("shared_lib", "shared_library"),
    ):
        raw_path = report.get(report_key)
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(f"backend report missing {report_key}: {report_path}")
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            artifact_path = report_path.parent / artifact_path
        if not artifact_path.is_file():
            raise RuntimeError(f"backend artifact missing: {artifact_path}")
        artifacts[artifact_key] = artifact_identity(
            artifact_path,
            relative_to=report_path.parent,
        )
    return artifacts
