"""Exact-symbol runtime support decisions for third-party fuzz campaigns."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


VALID_DECISIONS = {"run", "skip"}


@dataclass(frozen=True)
class EligibilityDecision:
    symbol_name: str
    kernel_id: str | None
    decision: Literal["run", "skip"]
    reason_code: str | None
    detail: str | None
    evidence: Sequence[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "symbol_name": self.symbol_name,
            "kernel_id": self.kernel_id,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "evidence": list(self.evidence),
        }


def decision_key(symbol_name: str, kernel_id: str | None = None) -> str:
    return kernel_id or symbol_name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid support registry JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"support registry must be a JSON object: {path}")
    return value


def _require_evidence(symbol_name: str, evidence: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"support decision {symbol_name} must include evidence")
    normalized: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError(f"support decision {symbol_name} evidence entries must be objects")
        if not isinstance(item.get("kind"), str) or not item["kind"]:
            raise ValueError(f"support decision {symbol_name} evidence entry missing kind")
        if not isinstance(item.get("file"), str) or not item["file"]:
            raise ValueError(f"support decision {symbol_name} evidence entry missing file")
        if not isinstance(item.get("line"), int) or item["line"] <= 0:
            raise ValueError(f"support decision {symbol_name} evidence entry missing positive line")
        if not isinstance(item.get("note"), str) or not item["note"]:
            raise ValueError(f"support decision {symbol_name} evidence entry missing note")
        normalized.append(dict(item))
    return tuple(normalized)


def load_support_registry(path: Path) -> dict[str, EligibilityDecision]:
    data = _read_json(path)
    if data.get("schema_version") != 1:
        raise ValueError("support registry schema_version must be 1")
    if not isinstance(data.get("project"), str) or not data["project"]:
        raise ValueError("support registry missing project")
    items = data.get("kernels")
    if not isinstance(items, list):
        raise ValueError("support registry kernels must be a list")

    decisions: dict[str, EligibilityDecision] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("support registry kernel entry must be an object")
        symbol_name = item.get("symbol_name")
        if not isinstance(symbol_name, str) or not symbol_name:
            raise ValueError("support decision missing symbol_name")
        kernel_id = item.get("kernel_id")
        if kernel_id is not None and (not isinstance(kernel_id, str) or not kernel_id):
            raise ValueError(f"support decision {symbol_name} has invalid kernel_id")
        key = decision_key(symbol_name, kernel_id)
        if key in decisions:
            raise ValueError(f"duplicate support decision symbol {symbol_name}")
        decision = item.get("decision")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"invalid support decision for {symbol_name}: {decision}")
        reason_code = item.get("reason_code")
        detail = item.get("detail")
        if reason_code is not None and (not isinstance(reason_code, str) or not reason_code):
            raise ValueError(f"support decision {symbol_name} has invalid reason_code")
        if detail is not None and (not isinstance(detail, str) or not detail):
            raise ValueError(f"support decision {symbol_name} has invalid detail")
        evidence = _require_evidence(symbol_name, item.get("evidence"))

        if decision == "run" and (reason_code or detail):
            raise ValueError(f"run decision must not carry skip reason for {symbol_name}")
        if decision == "skip" and (not reason_code or not detail):
            raise ValueError(f"skip decision must include reason and detail for {symbol_name}")

        decisions[key] = EligibilityDecision(
            symbol_name=symbol_name,
            kernel_id=kernel_id,
            decision=decision,
            reason_code=reason_code,
            detail=detail,
            evidence=evidence,
        )
    return decisions


def _index_symbols_by_kernel_id(run_dir: Path) -> dict[str, str]:
    index_path = run_dir / "index.json"
    if not index_path.exists():
        return {}
    index = _read_json(index_path)
    kernels = index.get("kernels")
    if not isinstance(kernels, list):
        raise ValueError(f"Phase 1 index kernels must be a list: {index_path}")
    mapping: dict[str, str] = {}
    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        kernel_id = kernel.get("kernel_id")
        symbol_name = kernel.get("symbol_name")
        if isinstance(kernel_id, str) and isinstance(symbol_name, str):
            mapping[kernel_id] = symbol_name
    return mapping


def _rewrite_built_keys(run_dir: Path, rewrite_summary: Mapping[str, Any]) -> set[str]:
    results = rewrite_summary.get("results")
    if not isinstance(results, list):
        raise ValueError("rewrite summary results must be a list")
    by_kernel_id = _index_symbols_by_kernel_id(run_dir)
    built: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        status = result.get("status")
        symbol = result.get("symbol_name")
        kernel_id = result.get("kernel_id")
        if not isinstance(symbol, str) and isinstance(kernel_id, str):
            symbol = by_kernel_id.get(kernel_id)
        if status == "built" and isinstance(symbol, str) and symbol:
            built.add(decision_key(symbol, kernel_id if isinstance(kernel_id, str) else None))
    return built


def validate_support_coverage(
    run_dir: Path,
    rewrite_summary: Mapping[str, Any],
    decisions: Mapping[str, EligibilityDecision],
) -> None:
    built_keys = _rewrite_built_keys(run_dir, rewrite_summary)
    decision_keys = set(decisions)
    missing = sorted(built_keys.difference(decision_keys))
    extra = sorted(decision_keys.difference(built_keys))
    if missing:
        raise ValueError(f"missing support decisions: {missing}")
    if extra:
        raise ValueError(f"unknown support decisions: {extra}")


def classify_kernel(
    symbol_name: str,
    phase2_status: str,
    decisions: Mapping[str, EligibilityDecision],
    kernel_id: str | None = None,
) -> EligibilityDecision:
    if phase2_status != "built":
        raise ValueError(
            f"classify_kernel only accepts Phase 2-built symbols, got {phase2_status} for {symbol_name}"
        )
    try:
        return decisions[decision_key(symbol_name, kernel_id)]
    except KeyError as exc:
        raise ValueError(f"missing support decision for Phase 2-built symbol {symbol_name}") from exc
