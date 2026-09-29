from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a JSON object")
    kernels = data.get("kernels")
    if not isinstance(kernels, list):
        raise ValueError("manifest must contain a kernels array")
    return data


def format_loc(loc: dict[str, Any] | None) -> str | None:
    if not isinstance(loc, dict):
        return None
    file = loc.get("file")
    line = loc.get("line")
    column = loc.get("column")
    if not isinstance(file, str) or not file:
        return None
    if isinstance(line, int) and isinstance(column, int):
        return f"{file}:{line}:{column}"
    if isinstance(line, int):
        return f"{file}:{line}"
    return file


def _split_include_roots(argv: list[Any], cwd: str | None = None) -> list[tuple[Path, bool]]:
    roots: list[tuple[Path, bool]] = []
    base = Path(cwd).resolve() if isinstance(cwd, str) and cwd else None

    def add_cuda_roots(cuda_root: str) -> None:
        cuda_base = Path(cuda_root)
        if not cuda_base.is_absolute() and base is not None:
            cuda_base = base / cuda_base
        cuda_base = cuda_base.resolve()
        roots.append((cuda_base / "include", True))
        roots.append((cuda_base / "targets" / "x86_64-linux" / "include", True))

    i = 0
    while i < len(argv):
        item = argv[i]
        if not isinstance(item, str):
            i += 1
            continue
        if item == "-I" and i + 1 < len(argv) and isinstance(argv[i + 1], str):
            path = Path(argv[i + 1])
            if not path.is_absolute() and base is not None:
                path = base / path
            roots.append((path.resolve(), False))
            i += 2
            continue
        if item.startswith("-I") and len(item) > 2:
            path = Path(item[2:])
            if not path.is_absolute() and base is not None:
                path = base / path
            roots.append((path.resolve(), False))
            i += 1
            continue
        if item == "-isystem" and i + 1 < len(argv) and isinstance(argv[i + 1], str):
            path = Path(argv[i + 1])
            if not path.is_absolute() and base is not None:
                path = base / path
            roots.append((path.resolve(), True))
            i += 2
            continue
        if item == "--cuda-path" and i + 1 < len(argv) and isinstance(argv[i + 1], str):
            add_cuda_roots(argv[i + 1])
            i += 2
            continue
        if item.startswith("--cuda-path=") and len(item) > len("--cuda-path="):
            add_cuda_roots(item[len("--cuda-path=") :])
            i += 1
            continue
        i += 1
    return roots


def _find_run_dir(manifest_path: Path) -> Path | None:
    for parent in manifest_path.parents:
        candidate = parent / "index.json"
        if candidate.is_file():
            return parent
    return None


def _matching_source_paths(argv: list[Any], cwd: str | None) -> list[str]:
    matches: list[str] = []
    base = Path(cwd).resolve() if isinstance(cwd, str) and cwd else None
    for x in argv:
        if not isinstance(x, str):
            continue
        if "/" not in x and not x.endswith((".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc")):
            continue
        try:
            path = Path(x)
            if not path.is_absolute() and base is not None:
                path = base / path
            matches.append(str(path.resolve()))
        except OSError:
            continue
    return matches


def capture_include_roots(manifest_path: Path, source_file: str | None) -> list[tuple[Path, bool]]:
    if not source_file:
        return []
    run_dir = _find_run_dir(manifest_path)
    if run_dir is None:
        return []
    try:
        index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    capture_dir = index.get("capture_dir")
    if not isinstance(capture_dir, str) or not capture_dir:
        return []
    commands_path = Path(capture_dir) / "commands.jsonl"
    if not commands_path.is_file():
        return []

    roots: list[tuple[Path, bool]] = []
    source_path = str(Path(source_file).resolve())
    seen: set[tuple[str, bool]] = set()
    try:
        with commands_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                argv = entry.get("argv")
                if not isinstance(argv, list):
                    continue
                if source_path not in _matching_source_paths(argv, entry.get("cwd") if isinstance(entry, dict) else None):
                    continue
                for root, is_system in _split_include_roots(argv, entry.get("cwd") if isinstance(entry, dict) else None):
                    key = (str(root), is_system)
                    if key not in seen:
                        seen.add(key)
                        roots.append((root, is_system))
        return roots
    except OSError:
        return []


def best_include_hint(definition_file: str | None, include_roots: list[tuple[Path, bool]]) -> tuple[str | None, str | None]:
    if not definition_file:
        return (None, None)
    try:
        def_path = Path(definition_file).resolve()
    except OSError:
        return (None, None)

    candidates: list[tuple[int, int, str, str]] = []
    for root, is_system in include_roots:
        try:
            rel = def_path.relative_to(root)
        except ValueError:
            continue
        include_suffix = rel.as_posix()
        include_stmt = f"#include <{include_suffix}>" if is_system else f'#include "{include_suffix}"'
        score = (0 if is_system else 1, len(include_suffix), str(root), include_stmt)
        candidates.append(score)
    if not candidates:
        return (None, None)
    _, _, root, include_stmt = sorted(candidates)[0]
    return (root, include_stmt)


def include_roots_from_compile_args(argv: list[Any], cwd: str | None) -> list[tuple[Path, bool]]:
    return _split_include_roots(argv, cwd)


def _decl_from_type_info(type_info: dict[str, Any]) -> dict[str, Any]:
    if isinstance(type_info.get("kind"), str):
        return type_info
    return {}


def resolve_manifest_arg_types(manifest: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kernel in manifest.get("kernels", []):
        display_name = kernel.get("display_name") or kernel.get("symbol_name") or "<unknown>"
        symbol_name = kernel.get("symbol_name") or "<unknown>"
        others = kernel.get("others") if isinstance(kernel, dict) else None
        source_file = others.get("source_file") if isinstance(others, dict) else None
        include_roots = capture_include_roots(manifest_path, source_file if isinstance(source_file, str) else None)
        for arg in kernel.get("args", []):
            type_info = arg.get("type_info") if isinstance(arg, dict) else None
            resolved = _decl_from_type_info(type_info) if isinstance(type_info, dict) else {}
            definition = resolved.get("definition") if isinstance(resolved, dict) else None
            definition_loc = format_loc(definition.get("loc") if isinstance(definition, dict) else None)
            definition_file = None
            if isinstance(definition, dict) and isinstance(definition.get("loc"), dict):
                loc_obj = definition["loc"]
                if isinstance(loc_obj.get("file"), str):
                    definition_file = loc_obj["file"]
            include_root, include_stmt = best_include_hint(definition_file, include_roots)
            rows.append(
                {
                    "kernel": display_name,
                    "symbol": symbol_name,
                    "index": arg.get("index"),
                    "name": arg.get("name") or "unnamed",
                    "type": arg.get("type"),
                    "decl_kind": resolved.get("kind") if isinstance(resolved, dict) else None,
                    "qualified_name": resolved.get("qualified_name") if isinstance(resolved, dict) else None,
                    "usr": resolved.get("usr") if isinstance(resolved, dict) else None,
                    "decl_loc": format_loc(resolved.get("decl_loc") if isinstance(resolved, dict) else None),
                    "definition_status": definition.get("status") if isinstance(definition, dict) else None,
                    "definition_loc": definition_loc,
                    "include_root": include_root,
                    "include_stmt": include_stmt,
                }
            )
    return rows
