"""Compute RQ2 coverage saturation statistics."""

from __future__ import annotations

from typing import Sequence

from benchmark.rq2.coverage_data import (
    CoverageCurve,
    CoveragePoint,
    SkippedCoverageCell,
)


CEILING_PERCENTAGES = (50, 90, 95, 100)


def _first_final_point(
    points: Sequence[CoveragePoint], attribute: str
) -> CoveragePoint | None:
    final_value = getattr(points[-1], attribute)
    if final_value is None:
        return None
    return next(
        point
        for point in points
        if getattr(point, attribute) == final_value
    )


def _change_count(points: Sequence[CoveragePoint], attribute: str) -> int | None:
    if getattr(points[-1], attribute) is None:
        return None
    changes = 0
    previous = 0
    for point in points:
        value = getattr(point, attribute)
        assert isinstance(value, int)
        if value != previous:
            changes += 1
            previous = value
    return changes


def _first_coverage_execution(
    points: Sequence[CoveragePoint], attribute: str
) -> int | None:
    for point in points:
        value = getattr(point, attribute)
        if value is None:
            return None
        if value > 0:
            return point.executions_completed
    return None


def _empirical_workload_ceilings(
    curves: Sequence[CoverageCurve],
) -> dict[str, dict[str, int | None]]:
    ceilings: dict[str, dict[str, int | None]] = {}
    for curve in curves:
        workload_ceilings = ceilings.setdefault(
            curve.workload_id,
            {"cfg_sites": None, "memory_features": None},
        )
        for attribute in ("cfg_sites", "memory_features"):
            for point in curve.points:
                value = getattr(point, attribute)
                if value is not None:
                    current = workload_ceilings[attribute]
                    workload_ceilings[attribute] = (
                        value if current is None else max(current, value)
                    )
    return ceilings


def _first_ceiling_fraction_point(
    points: Sequence[CoveragePoint],
    attribute: str,
    ceiling: int,
    percentage: int,
) -> CoveragePoint | None:
    for point in points:
        value = getattr(point, attribute)
        if value is None:
            return None
        if value * 100 >= ceiling * percentage:
            return point
    return None


def _ceiling_fraction_fields(
    prefix: str,
    points: Sequence[CoveragePoint] | None,
    attribute: str,
    ceiling: int | None,
) -> dict[str, object]:
    available = (
        points is not None
        and ceiling is not None
        and getattr(points[-1], attribute) is not None
    )
    fields: dict[str, object] = {}
    for percentage in CEILING_PERCENTAGES:
        point = (
            _first_ceiling_fraction_point(
                points, attribute, ceiling, percentage
            )
            if available
            else None
        )
        field_prefix = f"{prefix}_time_to_{percentage}pct"
        fields[f"{field_prefix}_ms"] = (
            point.timestamp_ms if point is not None else None
        )
        fields[f"{field_prefix}_executions_completed"] = (
            point.executions_completed if point is not None else None
        )
        fields[f"{field_prefix}_censored"] = (
            point is None if available else None
        )
    return fields


def build_saturation_rows(
    curves: Sequence[CoverageCurve],
    skipped_cells: Sequence[SkippedCoverageCell] = (),
) -> tuple[dict[str, object], ...]:
    """Summarize self-saturation and workload-wide empirical ceiling times."""

    rows: list[dict[str, object]] = []
    ceilings = _empirical_workload_ceilings(curves)
    for curve in curves:
        final = curve.points[-1]
        cfg_saturation = _first_final_point(curve.points, "cfg_sites")
        memory_saturation = _first_final_point(curve.points, "memory_features")
        workload_ceilings = ceilings[curve.workload_id]
        rows.append(
            {
                "trial_id": curve.trial_id,
                "workload_id": curve.workload_id,
                "configuration_id": curve.configuration_id,
                "paper_label": curve.paper_label,
                "vconfig": curve.vconfig,
                "status": "completed",
                "execution_index_semantics": "exact_completion_event",
                "final_cfg_sites": final.cfg_sites,
                "cfg_change_count": _change_count(curve.points, "cfg_sites"),
                "cfg_first_coverage_execution": _first_coverage_execution(
                    curve.points, "cfg_sites"
                ),
                "cfg_saturation_ms": (
                    cfg_saturation.timestamp_ms if cfg_saturation is not None else None
                ),
                "cfg_saturation_execution": (
                    cfg_saturation.executions_completed
                    if cfg_saturation is not None
                    else None
                ),
                "cfg_empirical_ceiling": workload_ceilings["cfg_sites"],
                **_ceiling_fraction_fields(
                    "cfg",
                    curve.points,
                    "cfg_sites",
                    workload_ceilings["cfg_sites"],
                ),
                "final_memory_features": final.memory_features,
                "memory_change_count": _change_count(
                    curve.points, "memory_features"
                ),
                "memory_first_coverage_execution": _first_coverage_execution(
                    curve.points, "memory_features"
                ),
                "memory_saturation_ms": (
                    memory_saturation.timestamp_ms
                    if memory_saturation is not None
                    else None
                ),
                "memory_saturation_execution": (
                    memory_saturation.executions_completed
                    if memory_saturation is not None
                    else None
                ),
                "memory_empirical_ceiling": workload_ceilings["memory_features"],
                **_ceiling_fraction_fields(
                    "memory",
                    curve.points,
                    "memory_features",
                    workload_ceilings["memory_features"],
                ),
                "final_timestamp_ms": final.timestamp_ms,
            }
        )
    rows.extend(
        {
            "trial_id": cell.trial_id,
            "workload_id": cell.workload_id,
            "configuration_id": cell.configuration_id,
            "paper_label": cell.paper_label,
            "vconfig": cell.vconfig,
            "status": "skipped",
            "vconfig_disabled_reason": cell.vconfig_disabled_reason,
            "execution_index_semantics": None,
            "final_cfg_sites": None,
            "cfg_change_count": None,
            "cfg_first_coverage_execution": None,
            "cfg_saturation_ms": None,
            "cfg_saturation_execution": None,
            "cfg_empirical_ceiling": ceilings.get(cell.workload_id, {}).get(
                "cfg_sites"
            ),
            **_ceiling_fraction_fields(
                "cfg",
                None,
                "cfg_sites",
                ceilings.get(cell.workload_id, {}).get("cfg_sites"),
            ),
            "final_memory_features": None,
            "memory_change_count": None,
            "memory_first_coverage_execution": None,
            "memory_saturation_ms": None,
            "memory_saturation_execution": None,
            "memory_empirical_ceiling": ceilings.get(cell.workload_id, {}).get(
                "memory_features"
            ),
            **_ceiling_fraction_fields(
                "memory",
                None,
                "memory_features",
                ceilings.get(cell.workload_id, {}).get("memory_features"),
            ),
            "final_timestamp_ms": None,
        }
        for cell in skipped_cells
    )
    return tuple(rows)

