"""Load compile-capture records and derive CUDA compile entries."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_capture_records(capture_dir: Path) -> list[dict[str, Any]]:
    """Read all JSON-line records from commands.jsonl."""
    path = capture_dir / "commands.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_cuda_record(record: dict[str, Any]) -> bool:
    compiler = os.path.basename(record.get("compiler", ""))
    if not compiler:
        return False

    # Main pipeline variants must come from toolchains that can replay into
    # device-aware preprocessing / AST / LLVM flows.
    is_replayable_compiler = "nvcc" in compiler or "clang" in compiler
    if not is_replayable_compiler:
        return False

    argv = record.get("argv", [])
    for arg in argv:
        if isinstance(arg, str) and arg.endswith(".cu"):
            return True
    for i, arg in enumerate(argv):
        if arg == "-x" and i + 1 < len(argv) and argv[i + 1] == "cuda":
            return True
    return False


def _find_source_file(record: dict[str, Any]) -> str | None:
    for arg in record.get("argv", []):
        if isinstance(arg, str) and arg.endswith(".cu"):
            return arg
    return None


def _compute_variant_id(abs_src: str, compiler: str, argv: list[str],
                        env_subset: dict[str, str], cwd: str) -> str:
    """Compute a normalized variant signature.

    variant_id = sha256(source_abs + compiler + relevant_flags + env_subset + cwd)[:12]

    Relevant flags: include/define/std/arch/optimization flags that affect
    compilation semantics (exclude output and linker flags).
    """
    relevant_prefixes = (
        "-I", "-D", "-U", "-std=", "--std=",
        "-O", "-g", "-f",
        "--cuda-gpu-arch", "--cuda-path",
        "-gencode", "-arch",
        "-isystem", "-include",
        "-target", "--target",
    )

    relevant_flags: list[str] = []
    two_token_flags = {
        "-I", "-D", "-U", "-isystem", "-include", "-target", "--target", "-arch", "-gencode"
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if not isinstance(arg, str):
            i += 1
            continue

        if arg in two_token_flags and i + 1 < len(argv):
            nxt = argv[i + 1]
            if isinstance(nxt, str):
                relevant_flags.append(f"{arg}={nxt}")
            i += 2
            continue

        for prefix in relevant_prefixes:
            if arg.startswith(prefix):
                relevant_flags.append(arg)
                break
        i += 1

    # Normalize: sort flags for stability
    relevant_flags.sort()

    # Normalize env subset: sort keys
    env_parts = sorted(f"{k}={v}" for k, v in env_subset.items()) if env_subset else []

    sig = "|".join([
        abs_src,
        os.path.basename(compiler),
        ",".join(relevant_flags),
        ",".join(env_parts),
        cwd,
    ])
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]


def collect_cuda_sources(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter records to CUDA compiles and return variant-aware entries.

    Unlike the previous deduplication by source only, this keeps multiple
    entries per source when they have different variant signatures (flags,
    env, etc.).  Within the same variant, prefer exit_code == 0 and the
    latest record.
    """
    # Build per-variant best-entry map
    best: dict[str, dict[str, Any]] = {}
    for idx, rec in enumerate(records):
        if not _is_cuda_record(rec):
            continue
        src = _find_source_file(rec)
        if src is None:
            continue
        cwd = rec.get("cwd", ".")
        abs_src = os.path.normpath(os.path.join(cwd, src))
        compiler = rec.get("compiler", "")
        argv = rec.get("argv", [])
        env_subset = rec.get("env_subset", {})

        variant_id = _compute_variant_id(abs_src, compiler, argv, env_subset, cwd)

        entry = {
            "source_file": abs_src,
            "cwd": cwd,
            "argv": argv,
            "compiler": compiler,
            "record_id": idx,
            "exit_code": rec.get("exit_code"),
            "env_subset": env_subset,
            "variant_id": variant_id,
            "selection_reason": "replayable_cuda_compile",
        }
        prev = best.get(variant_id)
        if prev is None:
            best[variant_id] = entry
        else:
            prev_ok = prev.get("exit_code") == 0
            curr_ok = entry.get("exit_code") == 0
            if curr_ok and not prev_ok:
                best[variant_id] = entry
            elif curr_ok == prev_ok and idx > prev["record_id"]:
                best[variant_id] = entry
    return list(best.values())
