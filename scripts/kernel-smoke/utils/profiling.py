"""Lightweight wall-clock profiling helpers for kernel-smoke.

This module intentionally stays small and standard-library-only. It records
aggregate wall-clock timings for stages, helper calls, and per-variant work,
and can be left disabled with near-zero call-site complexity.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Iterator


@dataclass
class CallStat:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_s": round(self.total_s, 6),
            "max_s": round(self.max_s, 6),
        }


@dataclass
class VariantProfiler:
    enabled: bool
    variant_id: str
    source_file: str
    helper_totals: dict[str, CallStat] = field(default_factory=dict)
    wall_time_s: float = 0.0

    @contextmanager
    def total_span(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.wall_time_s += time.perf_counter() - start

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.helper_totals.setdefault(name, CallStat()).add(elapsed)

    def as_dict(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "variant_id": self.variant_id,
            "source_file": self.source_file,
            "wall_time_s": round(self.wall_time_s, 6),
            "helper_totals": {
                name: stat.as_dict()
                for name, stat in sorted(self.helper_totals.items())
            },
        }


class Profiler:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.run_wall_time_s = 0.0
        self.stage_totals: dict[str, CallStat] = {}
        self.stage_meta: dict[str, dict[str, Any]] = {}
        self.helper_totals: dict[str, CallStat] = {}
        self.variant_profiles: list[dict[str, Any]] = []

    @contextmanager
    def run_span(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.run_wall_time_s += time.perf_counter() - start

    @contextmanager
    def stage_span(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stage_totals.setdefault(name, CallStat()).add(elapsed)

    def new_variant_profiler(self, variant_id: str, source_file: str) -> VariantProfiler:
        return VariantProfiler(self.enabled, variant_id=variant_id, source_file=source_file)

    def add_variant_profile(self, profile: dict[str, Any] | None) -> None:
        if not self.enabled or profile is None:
            return
        self.variant_profiles.append(profile)
        for name, stat_dict in profile.get("helper_totals", {}).items():
            stat = self.helper_totals.setdefault(name, CallStat())
            stat.count += int(stat_dict.get("count", 0))
            stat.total_s += float(stat_dict.get("total_s", 0.0))
            stat.max_s = max(stat.max_s, float(stat_dict.get("max_s", 0.0)))

    def set_stage_meta(self, name: str, **meta: Any) -> None:
        if not self.enabled:
            return
        self.stage_meta.setdefault(name, {}).update(meta)

    def as_dict(
        self,
        *,
        run_id: str,
        resume: bool,
        jobs: int,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        stages: dict[str, Any] = {}
        for name, stat in sorted(self.stage_totals.items()):
            stages[name] = {
                "wall_time_s": round(stat.total_s, 6),
                **self.stage_meta.get(name, {}),
            }
        return {
            "schema_version": 1,
            "run_id": run_id,
            "enabled": True,
            "resume": resume,
            "jobs": jobs,
            "wall_time_s": round(self.run_wall_time_s, 6),
            "stages": stages,
            "helper_totals": {
                name: stat.as_dict()
                for name, stat in sorted(self.helper_totals.items())
            },
            "variants": sorted(
                self.variant_profiles,
                key=lambda item: item.get("wall_time_s", 0.0),
                reverse=True,
            ),
        }


__all__ = ["Profiler", "VariantProfiler"]
