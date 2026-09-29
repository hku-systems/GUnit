"""Rewrite execution strategies for Phase 2."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
VCONFIG_BUILD_HELPER = REPO_ROOT / "tools/rapid-vconfig-instrument/build.py"
_VCONFIG_UNSUPPORTED_REASONS = {
    "vconfig_barrier_unsupported",
    "vconfig_inline_asm_unsupported",
}
_DOTTED_GLOBAL_DEF_RE = re.compile(r"^(@\.[A-Za-z0-9_.$-]+)\s*=", re.MULTILINE)
_LLVM_GLOBAL_TOKEN_RE = re.compile(r"@[A-Za-z_.$][A-Za-z0-9_.$-]*")


def _find_llvm_tool(base: str) -> str | None:
    return shutil.which(f"{base}-22") or shutil.which(base)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("rewrite_ir_parse_fail")


def _extract_symbol_name(signature_text: str) -> str:
    at_index = signature_text.find("@")
    if at_index < 0:
        raise ValueError("rewrite_ir_parse_fail")
    open_paren = signature_text.find("(", at_index)
    if open_paren < 0:
        raise ValueError("rewrite_ir_parse_fail")
    return signature_text[at_index + 1 : open_paren]


def _extract_params_text(signature_text: str, function_name: str) -> str:
    marker = f"@{function_name}("
    start = signature_text.find(marker)
    if start < 0:
        raise ValueError("rewrite_ir_parse_fail")
    open_paren = start + len(marker) - 1
    close_paren = _find_matching_paren(signature_text, open_paren)
    return signature_text[open_paren + 1 : close_paren]


def _remove_kernel_only_attrs(signature_text: str) -> str:
    text = re.sub(r"\s+ptx_kernel\b", "", signature_text)
    text = re.sub(r"\s+comdat(?:\([^)]*\))?", "", text)
    return _normalize_space(text)


def _signature_key(signature_text: str, function_name: str) -> str:
    return _remove_kernel_only_attrs(signature_text.replace(f"@{function_name}(", "@__RAPID_FN__(", 1))


def _rewrite_signature(signature_text: str, *, original_name: str, new_name: str) -> str:
    rewritten = signature_text.replace(f"@{original_name}(", f"@{new_name}(", 1)
    if rewritten == signature_text:
        raise ValueError("rewrite_ir_parse_fail")
    marker = f"@{new_name}("
    open_paren = rewritten.find(marker) + len(marker) - 1
    close_paren = _find_matching_paren(rewritten, open_paren)
    params_text = rewritten[open_paren + 1 : close_paren].strip()
    if not params_text or params_text == "void":
        context_params = "ptr %__rapid_context"
    else:
        context_params = f"{params_text}, ptr %__rapid_context"
    rewritten = rewritten[: open_paren + 1] + context_params + rewritten[close_paren:]
    return _remove_kernel_only_attrs(rewritten)


@dataclass(frozen=True)
class LlvmFunction:
    name: str
    signature_text: str
    signature_key: str
    params: tuple[str, ...]
    param_count: int
    body_lines: tuple[str, ...]
    has_ptx_kernel: bool


@dataclass(frozen=True)
class VConfigRewriteResult:
    requested: bool
    enabled: bool
    reason: str | None = None
    warp_aligned: bool = False


def _count_params(signature_text: str, function_name: str) -> int:
    params_text = _extract_params_text(signature_text, function_name)
    count = 0
    for fragment in _split_top_level_commas(params_text):
        if fragment == "void":
            continue
        parts = fragment.rsplit(" ", 1)
        if len(parts) != 2 or not parts[1].startswith("%"):
            raise ValueError("rewrite_ir_parse_fail")
        count += 1
    return count


def _parse_functions(ll_text: str) -> list[LlvmFunction]:
    lines = ll_text.splitlines(keepends=True)
    functions: list[LlvmFunction] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.lstrip().startswith("define "):
            index += 1
            continue

        header_lines: list[str] = []
        brace_depth = 0
        saw_open = False
        while index < len(lines):
            current = lines[index]
            header_lines.append(current)
            brace_depth += current.count("{") - current.count("}")
            index += 1
            if "{" in current:
                saw_open = True
                break
        if not saw_open:
            raise ValueError("rewrite_ir_parse_fail")

        body_lines: list[str] = []
        while index < len(lines):
            current = lines[index]
            body_lines.append(current)
            brace_depth += current.count("{") - current.count("}")
            index += 1
            if brace_depth == 0:
                break
        if brace_depth != 0 or not body_lines:
            raise ValueError("rewrite_ir_parse_fail")

        signature_text = "".join(header_lines).rsplit("{", 1)[0].strip()
        function_name = _extract_symbol_name(signature_text)
        params = tuple(
            _normalize_space(fragment)
            for fragment in _split_top_level_commas(
                _extract_params_text(signature_text, function_name)
            )
            if fragment != "void"
        )
        functions.append(
            LlvmFunction(
                name=function_name,
                signature_text=signature_text,
                signature_key=_signature_key(signature_text, function_name),
                params=params,
                param_count=_count_params(signature_text, function_name),
                body_lines=tuple(body_lines[:-1]),
                has_ptx_kernel=bool(re.search(r"\bptx_kernel\b", signature_text)),
            )
        )
    return functions


def _find_target_function(functions: list[LlvmFunction], input_symbol: str) -> LlvmFunction:
    matches = [fn for fn in functions if fn.name == input_symbol and fn.has_ptx_kernel]
    if not matches:
        raise ValueError("rewrite_target_not_found")
    if len(matches) != 1:
        raise ValueError("rewrite_target_ambiguous")
    return matches[0]


def _render_function(signature_text: str, body_lines: tuple[str, ...]) -> str:
    body = "".join(body_lines)
    if body and not body.endswith("\n"):
        body += "\n"
    return f"{signature_text} {{\n{body}}}\n"


def _render_rewritten_entry_function(target: LlvmFunction, *, entry_symbol: str) -> str:
    entry_signature = _rewrite_signature(target.signature_text, original_name=target.name, new_name=entry_symbol)
    return _render_function(entry_signature, target.body_lines)


def _safe_nvptx_global_name(raw_name: str, used_names: set[str]) -> str:
    token = raw_name[1:].lstrip(".") or "global"
    token = re.sub(r"[^0-9A-Za-z_]+", "_", token).strip("_") or "global"
    if token[0].isdigit():
        token = f"g_{token}"
    base = f"@rapid_nvptx_global_{token}"
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _rename_dotted_globals_for_nvptx(ll_text: str) -> str:
    dotted_globals = sorted(set(_DOTTED_GLOBAL_DEF_RE.findall(ll_text)), key=len, reverse=True)
    if not dotted_globals:
        return ll_text

    used_names = set(_LLVM_GLOBAL_TOKEN_RE.findall(ll_text))
    replacements = {
        name: _safe_nvptx_global_name(name, used_names) for name in dotted_globals
    }
    rewritten = ll_text
    for old_name, new_name in replacements.items():
        rewritten = re.sub(
            rf"{re.escape(old_name)}(?![A-Za-z0-9_.$-])",
            new_name,
            rewritten,
        )
    return rewritten


def _rewrite_module_text(*, ll_text: str, rewrite_plan: dict[str, Any]) -> tuple[str, LlvmFunction]:
    input_symbol = str(rewrite_plan.get("input_symbol") or "")
    entry_symbol = str(rewrite_plan.get("entry_symbol") or "")
    payload_args = rewrite_plan.get("payload_args")
    if (
        not input_symbol
        or not entry_symbol
        or not isinstance(payload_args, list)
        or rewrite_plan.get("entry_abi_version") != 1
        or rewrite_plan.get("entry_context") != "rapid_kernel_context_v1"
    ):
        raise ValueError("rewrite_plan_invalid")

    functions = _parse_functions(ll_text)
    target = _find_target_function(functions, input_symbol)
    if target.param_count != len(payload_args):
        raise ValueError("input_signature_mismatch")

    entry_text = _render_rewritten_entry_function(target, entry_symbol=entry_symbol)
    rewritten = ll_text.rstrip() + "\n\n" + entry_text
    rewritten = _rename_dotted_globals_for_nvptx(rewritten)
    return rewritten, target


def _disassemble_bc(bc_path: Path) -> str:
    llvm_dis = _find_llvm_tool("llvm-dis")
    if llvm_dis is None:
        raise OSError("missing llvm-dis for rewrite")
    result = subprocess.run([llvm_dis, "-o", "-", str(bc_path)], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise OSError(f"llvm-dis failed: {result.stderr.strip()}")
    return result.stdout


def _assemble_ll(ll_text: str, output_bc: Path) -> None:
    llvm_as = _find_llvm_tool("llvm-as")
    if llvm_as is None:
        raise OSError("missing llvm-as for rewrite")
    with tempfile.TemporaryDirectory(prefix="kernel-rewrite-assemble-") as td:
        ll_path = Path(td) / "kernel.device.ll"
        ll_path.write_text(ll_text, encoding="utf-8")
        result = subprocess.run([llvm_as, str(ll_path), "-o", str(output_bc)], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise OSError(f"llvm-as failed: {result.stderr.strip()}")


def _verify_rewrite(
    *,
    input_function: LlvmFunction,
    output_ll_text: str,
    entry_symbol: str,
) -> None:
    output_functions = {fn.name: fn for fn in _parse_functions(output_ll_text)}
    entry_function = output_functions.get(entry_symbol)
    original_function = output_functions.get(input_function.name)

    if entry_function is None:
        raise ValueError("entry_symbol_missing")
    if (
        original_function is None
        or original_function.signature_key != input_function.signature_key
    ):
        raise ValueError("input_signature_changed")
    if not original_function.has_ptx_kernel:
        raise ValueError("input_symbol_not_kernel")
    if entry_function.params[:-1] != input_function.params:
        raise ValueError("entry_signature_mismatch")
    if entry_function.params[-1:] != ("ptr %__rapid_context",):
        raise ValueError("entry_signature_mismatch")
    if entry_function.has_ptx_kernel:
        raise ValueError("entry_symbol_still_kernel")


class RewriteExecutor:
    """Plan-driven `.ll` rewrite that emits a callable device entry symbol."""

    def __init__(self) -> None:
        self._vconfig_tool_dir: tempfile.TemporaryDirectory[str] | None = None
        self._vconfig_tool: Path | None = None

    def close(self) -> None:
        if self._vconfig_tool_dir is not None:
            self._vconfig_tool_dir.cleanup()
        self._vconfig_tool_dir = None
        self._vconfig_tool = None

    def __del__(self) -> None:
        self.close()

    def _get_vconfig_tool(self) -> Path:
        if self._vconfig_tool is not None:
            return self._vconfig_tool
        self._vconfig_tool_dir = tempfile.TemporaryDirectory(
            prefix="rapid-vconfig-instrument-"
        )
        tool = Path(self._vconfig_tool_dir.name) / "rapid-vconfig-instrument"
        result = subprocess.run(
            [sys.executable, str(VCONFIG_BUILD_HELPER), "--output", str(tool)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise OSError(f"VConfig tool build failed: {result.stderr.strip()}")
        self._vconfig_tool = tool
        return tool

    def _instrument_vconfig(
        self,
        *,
        device_bc: Path,
        entry_symbol: str,
        input_function: LlvmFunction,
    ) -> VConfigRewriteResult:
        with tempfile.TemporaryDirectory(
            prefix="kernel-rewrite-vconfig-", dir=device_bc.parent
        ) as temp_dir:
            transformed_bc = Path(temp_dir) / "kernel.device.bc"
            result = subprocess.run(
                [
                    str(self._get_vconfig_tool()),
                    "--input",
                    str(device_bc),
                    "--output",
                    str(transformed_bc),
                    "--entry-symbol",
                    entry_symbol,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                diagnostics = set(result.stderr.splitlines())
                reason = next(
                    (
                        candidate
                        for candidate in _VCONFIG_UNSUPPORTED_REASONS
                        if candidate in diagnostics
                    ),
                    None,
                )
                if reason is not None:
                    return VConfigRewriteResult(
                        requested=True,
                        enabled=False,
                        reason=reason,
                    )
                raise OSError(f"VConfig transform failed: {result.stderr.strip()}")

            transformed_ll = _disassemble_bc(transformed_bc)
            _verify_rewrite(
                input_function=input_function,
                output_ll_text=transformed_ll,
                entry_symbol=entry_symbol,
            )
            transformed_bc.replace(device_bc)
        return VConfigRewriteResult(
            requested=True,
            enabled=True,
            warp_aligned='"rapid.vconfig.warp_aligned"' in transformed_ll,
        )

    def execute(
        self,
        *,
        rewrite_plan: dict[str, Any],
        kernel_bc: Path,
        phase2_dir: Path,
    ) -> VConfigRewriteResult:
        entry_symbol = str(rewrite_plan.get("entry_symbol") or "")
        vconfig_requested = rewrite_plan.get("vconfig_requested")
        if not isinstance(vconfig_requested, bool):
            raise ValueError("rewrite_plan_invalid")
        out_device_bc = phase2_dir / "kernel.device.bc"

        input_ll = _disassemble_bc(kernel_bc)
        rewritten_ll, input_function = _rewrite_module_text(ll_text=input_ll, rewrite_plan=rewrite_plan)
        _assemble_ll(rewritten_ll, out_device_bc)

        output_ll = _disassemble_bc(out_device_bc)
        _verify_rewrite(
            input_function=input_function,
            output_ll_text=output_ll,
            entry_symbol=entry_symbol,
        )
        if not vconfig_requested:
            return VConfigRewriteResult(requested=False, enabled=False)
        return self._instrument_vconfig(
            device_bc=out_device_bc,
            entry_symbol=entry_symbol,
            input_function=input_function,
        )
