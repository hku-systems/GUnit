"""Rewrite plan generation for Phase 2 kernels."""

from __future__ import annotations

from contracts.kernel import KernelContract


class RewritePlanner:
    """Build minimal rewrite plans from normalized Phase 2 contracts."""

    def build_plan(self, contract: KernelContract) -> dict[str, object]:
        plan = contract.to_plan_dict()
        plan["schema_version"] = 1
        plan["kernel_id"] = contract.kernel_id
        return plan
