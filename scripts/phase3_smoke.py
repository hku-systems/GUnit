#!/usr/bin/env python3
"""Run manifest-driven Phase3 fixed-payload smoke checks for Phase2 outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("cufuzz", "origin", "rapid", "rapid2")
DUMP_SEED_UNUSED_LIB = Path("/tmp/rapid-phase3-dump-seed-unused.so")


@dataclass(frozen=True)
class SmokeTask:
    kernel_id: str
    display_name: str | None
    kernel_dir: Path
    phase2_dir: Path
    manifest_path: Path
    shared_lib: Path
    command: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_kernel_dir(run_dir: Path, raw_dir: str) -> Path:
    kernel_dir = Path(raw_dir)
    if kernel_dir.is_absolute():
        return kernel_dir
    return run_dir / kernel_dir


def _kernel_display_name(manifest_path: Path) -> str | None:
    manifest = _read_json(manifest_path)
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or not kernels or not isinstance(kernels[0], dict):
        return None
    value = kernels[0].get("display_name")
    return str(value) if isinstance(value, str) else None


def _phase2_built(kernel_dir: Path) -> bool:
    metadata = _read_json(kernel_dir / "metadata.json")
    phase2_metadata = _read_json(kernel_dir / "phase2" / "metadata.phase2.json")
    return metadata.get("build_status") == "built" and phase2_metadata.get("phase2_status") == "built"


def _shared_lib_from_backend(phase2_dir: Path, backend: str) -> Path | None:
    backends_root = phase2_dir / "backends"
    backend_dirs = [backends_root / backend]
    if backends_root.is_dir():
        backend_dirs.extend(
            path
            for path in sorted(backends_root.glob(f"{backend}*"))
            if path.is_dir() and path not in backend_dirs
        )

    for backend_dir in backend_dirs:
        shared_lib = _shared_lib_from_backend_dir(backend_dir)
        if shared_lib is not None:
            return shared_lib
    return None


def _shared_lib_from_backend_dir(backend_dir: Path) -> Path | None:
    backend_build = backend_dir / "backend_build.json"
    if backend_build.is_file():
        shared_lib = _read_json(backend_build).get("shared_lib")
        if isinstance(shared_lib, str) and shared_lib:
            path = Path(shared_lib)
            return path if path.is_absolute() else backend_dir / path
    return None


def _fuzzer_bin(repo_root: Path, backend: str, profile: str) -> Path:
    name = "fuzzer_async" if backend == "rapid2" else "fuzzer"
    return repo_root / "cuda-fuzzer" / "target" / profile / name


def build_fuzzer_command(*, backend: str, profile: str, repo_root: Path = REPO_ROOT) -> list[str]:
    name = "fuzzer_async" if backend == "rapid2" else "fuzzer"
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(repo_root / "cuda-fuzzer" / "Cargo.toml"),
        "--bin",
        name,
    ]
    if profile == "release":
        command.insert(2, "--release")
    return command


def collect_smoke_tasks(
    *,
    run_dir: Path,
    backend: str,
    display_name: str | None,
    limit: int | None,
    runs: int,
    repo_root: Path = REPO_ROOT,
    profile: str = "debug",
) -> list[SmokeTask]:
    index = _read_json(run_dir / "index.json")
    tasks: list[SmokeTask] = []
    for entry in index.get("kernels", []):
        if not isinstance(entry, dict):
            continue
        kernel_dir = _resolve_kernel_dir(run_dir, str(entry.get("dir", "")))
        manifest_path = kernel_dir / "manifest.json"
        phase2_dir = kernel_dir / "phase2"
        if not manifest_path.is_file() or not _phase2_built(kernel_dir):
            continue

        kernel_display_name = _kernel_display_name(manifest_path)
        if display_name is not None and kernel_display_name != display_name:
            continue

        shared_lib = _shared_lib_from_backend(phase2_dir, backend)
        if shared_lib is None:
            continue

        command = [
            str(_fuzzer_bin(repo_root, backend, profile)),
            str(shared_lib),
            "--manifest",
            str(manifest_path),
            "--no-mutate",
            "--runs",
            str(runs),
        ]
        tasks.append(
            SmokeTask(
                kernel_id=str(entry.get("kernel_id", kernel_dir.name)),
                display_name=kernel_display_name,
                kernel_dir=kernel_dir,
                phase2_dir=phase2_dir,
                manifest_path=manifest_path,
                shared_lib=shared_lib,
                command=command,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def collect_dump_seed_tasks(
    *,
    run_dir: Path,
    display_name: str | None,
    limit: int | None,
    repo_root: Path = REPO_ROOT,
    profile: str = "debug",
) -> list[SmokeTask]:
    index = _read_json(run_dir / "index.json")
    tasks: list[SmokeTask] = []
    for entry in index.get("kernels", []):
        if not isinstance(entry, dict):
            continue
        kernel_dir = _resolve_kernel_dir(run_dir, str(entry.get("dir", "")))
        manifest_path = kernel_dir / "manifest.json"
        phase2_dir = kernel_dir / "phase2"
        if not manifest_path.is_file() or not _phase2_built(kernel_dir):
            continue

        kernel_display_name = _kernel_display_name(manifest_path)
        if display_name is not None and kernel_display_name != display_name:
            continue

        command = [
            str(repo_root / "cuda-fuzzer" / "target" / profile / "fuzzer"),
            str(DUMP_SEED_UNUSED_LIB),
            "--manifest",
            str(manifest_path),
            "--dump-seed",
        ]
        tasks.append(
            SmokeTask(
                kernel_id=str(entry.get("kernel_id", kernel_dir.name)),
                display_name=kernel_display_name,
                kernel_dir=kernel_dir,
                phase2_dir=phase2_dir,
                manifest_path=manifest_path,
                shared_lib=DUMP_SEED_UNUSED_LIB,
                command=command,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def run_tasks(tasks: list[SmokeTask], *, dry_run: bool) -> int:
    for task in tasks:
        label = task.display_name or task.kernel_id
        print(f"[phase3] {label}: {' '.join(task.command)}", flush=True)
        if dry_run:
            continue
        subprocess.run(task.command, check=True, cwd=REPO_ROOT)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=BACKENDS, default="origin")
    parser.add_argument("--display-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--profile", choices=("debug", "release"), default="debug")
    parser.add_argument("--build-fuzzer", action="store_true")
    parser.add_argument(
        "--dump-seeds",
        action="store_true",
        help="Only validate manifest-driven seed generation; does not require backend .so artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if args.dump_seeds:
        tasks = collect_dump_seed_tasks(
            run_dir=args.run_dir.resolve(),
            display_name=args.display_name,
            limit=args.limit,
            profile=args.profile,
        )
    else:
        tasks = collect_smoke_tasks(
            run_dir=args.run_dir.resolve(),
            backend=args.backend,
            display_name=args.display_name,
            limit=args.limit,
            runs=args.runs,
            profile=args.profile,
        )
    if not tasks:
        if args.dump_seeds:
            print("[phase3] no built kernels with manifests", flush=True)
        else:
            print("[phase3] no built kernels with matching backend artifacts", flush=True)
        return 1
    if args.build_fuzzer and not args.dry_run:
        build_backend = "origin" if args.dump_seeds else args.backend
        subprocess.run(build_fuzzer_command(backend=build_backend, profile=args.profile), check=True, cwd=REPO_ROOT)
    return run_tasks(tasks, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
