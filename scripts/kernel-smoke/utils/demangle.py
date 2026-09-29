"""Demangled C++ symbol helpers for artifact-mode AST filtering.

This module is intentionally small and reusable so the demangling and filter
derivation logic can be treated like an internal utility library.
"""

from __future__ import annotations

import re
import shutil
import subprocess

try:
    import cxxfilt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime dependency check
    cxxfilt = None


HAS_CXXFILT = cxxfilt is not None
_CXXFILT_BIN = shutil.which("c++filt")
HAS_DEMANGLER = HAS_CXXFILT or _CXXFILT_BIN is not None


def demangle_symbol(symbol: str) -> str:
    if cxxfilt is not None:
        try:
            return cxxfilt.demangle(symbol)
        except cxxfilt.Error as e:
            raise RuntimeError(f"demangle_failed: {symbol}: {e}") from e
    if _CXXFILT_BIN is not None:
        result = subprocess.run(
            [_CXXFILT_BIN, symbol],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or symbol
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"demangle_failed: {symbol}: {detail}")
    raise RuntimeError(
        "artifact demangling requires python package `cxxfilt` or system `c++filt`",
    )


def _ends_with_operator_kw(s: str) -> bool:
    return s.rstrip().endswith("operator")


def strip_template_args(name: str) -> str:
    """Remove top-level template argument lists ``<...>`` from *name*."""
    out: list[str] = []
    depth = 0
    i = 0
    n = len(name)
    in_operator_sym = False

    while i < n:
        ch = name[i]

        if in_operator_sym:
            if ch in "<>=!+-*/%^&|~,":
                out.append(ch)
                i += 1
                continue
            in_operator_sym = False

        if depth == 0 and ch in "<>" and _ends_with_operator_kw("".join(out)):
            in_operator_sym = True
            out.append(ch)
            i += 1
            continue

        if ch == "<":
            depth += 1
            i += 1
            continue
        if ch == ">":
            if depth > 0:
                depth -= 1
            else:
                out.append(ch)
            i += 1
            continue
        if depth == 0:
            out.append(ch)
        i += 1

    return "".join(out)


def strip_last_paren_group(s: str) -> str:
    """Strip the last top-level ``(...)`` parameter list from *s*."""
    stripped = s.rstrip()
    while True:
        changed = False
        for suffix in (" const", " volatile", " &", " &&", " noexcept"):
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].rstrip()
                changed = True
        if not changed:
            break

    if not stripped or stripped[-1] != ")":
        return s

    depth = 0
    for i in range(len(stripped) - 1, -1, -1):
        if stripped[i] == ")":
            depth += 1
        elif stripped[i] == "(":
            depth -= 1
        if depth == 0:
            return stripped[:i].rstrip()
    return s


def _extract_fp_return_name(text: str) -> str | None:
    idx = text.find("(*")
    if idx < 0:
        return None

    start = idx + 2
    while start < len(text) and text[start] == " ":
        start += 1

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        if depth > 0:
            i += 1
    if depth != 0:
        return None

    inner = text[start:i].strip()
    if not inner:
        return None
    return strip_last_paren_group(inner)


def qualified_name_from_demangled(demangled: str) -> str:
    text = demangled.strip()
    if not text:
        return demangled

    thunk_prefixes = [
        "non-virtual thunk to ",
        "virtual thunk to ",
        "covariant return thunk to ",
        "guard variable for ",
        "construction vtable for ",
    ]
    for pfx in thunk_prefixes:
        if text.startswith(pfx):
            text = text[len(pfx):]
            break

    fp_name = _extract_fp_return_name(text)
    if fp_name is not None:
        text = fp_name
    else:
        text = strip_last_paren_group(text)

    text = strip_template_args(text).strip()

    qual_seg = (
        r"(?:"
        r"~?[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\([^)]*\))?"
        r"(?:::\{[^}]*\})?"
        r"::"
        r")*"
    )

    op_match = re.search(
        r"("
        + qual_seg
        + r"operator\s*"
        + r"(?:"
        + r"\(\)"
        + r"|->(?:\*?)"
        + r"|[<>=!+\-*/%^&|~,]+[=]?"
        + r"|(?:new|delete)\s*(?:\[\])?"
        + r"|[A-Za-z_][A-Za-z0-9_:* &]*"
        + r")"
        + r")\s*$",
        text,
    )
    if op_match:
        return op_match.group(1).strip()

    normal_match = re.search(
        r"("
        + qual_seg
        + r"~?[A-Za-z_][A-Za-z0-9_]*"
        + r")\s*$",
        text,
    )
    if normal_match:
        return normal_match.group(1).strip()

    token = text.split()[-1] if " " in text else text
    token = token.strip()
    return token or demangled


__all__ = [
    "HAS_CXXFILT",
    "HAS_DEMANGLER",
    "demangle_symbol",
    "qualified_name_from_demangled",
    "strip_last_paren_group",
    "strip_template_args",
]
