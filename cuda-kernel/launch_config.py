from __future__ import annotations

from pathlib import Path


DEFAULT_BLOCK_CANDIDATES = [1024, 512, 256, 128, 64, 32, 16]
KERNEL_ENTRY_FIELDS = {
    "symbol_name",
    "display_name",
    "args",
    "constraints",
    "launch_policy",
    "others",
}
LAUNCH_POLICY_FIELDS = {
    "grid",
    "block_candidates",
    "physical_block_max",
    "logical_grid",
    "logical_block",
    "logical_block_candidates",
    "target_dynamic_shared_bytes",
    "coverage_memory",
    "vconfig_reserved",
    "vconfig_mutation",
}


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"launch_policy.{field} must be a positive integer")
    return value


def _positive_dims(raw: object, *, field: str) -> list[int]:
    if not isinstance(raw, list) or len(raw) != 3:
        raise RuntimeError(f"launch_policy.{field} must contain exactly three dimensions")
    return [_positive_int(value, field=field) for value in raw]


def _dim_product(values: list[int]) -> int:
    product = 1
    for value in values:
        product *= value
    return product


def launch_config_from_kernel_entry(
    kernel_entry: dict,
    *,
    backend_name: str,
    require_warp_aligned_block: bool = False,
    vconfig_enabled: bool | None = None,
) -> dict:
    unknown_fields = sorted(set(kernel_entry) - KERNEL_ENTRY_FIELDS)
    if unknown_fields:
        raise RuntimeError(
            "unsupported manifest kernel fields: "
            + ", ".join(unknown_fields)
            + "; use canonical launch_policy"
        )
    raw = kernel_entry.get("launch_policy")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RuntimeError("launch_policy must be an object when present")
    unknown_policy_fields = sorted(set(raw) - LAUNCH_POLICY_FIELDS)
    if unknown_policy_fields:
        raise RuntimeError(
            "unsupported launch_policy fields: "
            + ", ".join(f"launch_policy.{field}" for field in unknown_policy_fields)
        )

    grid_values = _positive_dims(raw.get("grid", [1, 1, 1]), field="grid")
    if grid_values != [1, 1, 1]:
        raise RuntimeError(
            f"{backend_name} currently supports only physical grid [1, 1, 1]; "
            "multi-block physical grid support needs per-backend synchronization "
            "and logical grid mutation remains reserved for future VConfig Phase 2 "
            "rewriting"
        )

    raw_candidates = raw.get("block_candidates", DEFAULT_BLOCK_CANDIDATES)
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise RuntimeError("launch_policy.block_candidates must be a non-empty list")
    candidates: list[int] = []
    seen: set[int] = set()
    for value in raw_candidates:
        candidate = _positive_int(value, field="block_candidates")
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    candidates.sort(reverse=True)

    physical_block_max = raw.get("physical_block_max", candidates[0])
    physical_block_max = _positive_int(physical_block_max, field="physical_block_max")
    candidates = [candidate for candidate in candidates if candidate <= physical_block_max]
    if not candidates:
        raise RuntimeError("launch_policy.block_candidates are all above physical_block_max")
    if require_warp_aligned_block:
        candidates = [candidate for candidate in candidates if candidate % 32 == 0]
        if not candidates:
            raise RuntimeError(
                "launch_policy.block_candidates must include a warp-aligned "
                "candidate when VConfig warp adaptation is enabled"
            )

    has_logical_vconfig_bounds = "logical_grid" in raw or "logical_block" in raw
    logical_grid = (
        _positive_dims(raw["logical_grid"], field="logical_grid")
        if "logical_grid" in raw
        else list(grid_values)
    )
    if _dim_product(logical_grid) > _dim_product(grid_values):
        raise RuntimeError(
            "launch_policy.logical grid blocks exceed physical grid envelope"
        )

    logical_block = (
        _positive_dims(raw["logical_block"], field="logical_block")
        if "logical_block" in raw
        else [1, 1, 1]
    )
    if "logical_block" in raw:
        logical_block_threads = _dim_product(logical_block)
        if logical_block_threads > candidates[0]:
            raise RuntimeError(
                "launch_policy.logical block threads exceed physical block envelope"
            )
        candidates = [
            candidate for candidate in candidates if candidate >= logical_block_threads
        ]
        if not candidates:
            raise RuntimeError(
                "launch_policy.block_candidates are all below logical block threads"
            )

    target_dynamic_shared_bytes = raw.get("target_dynamic_shared_bytes", 0)
    if not isinstance(target_dynamic_shared_bytes, int) or target_dynamic_shared_bytes < 0:
        raise RuntimeError(
            "launch_policy.target_dynamic_shared_bytes must be a non-negative integer"
        )

    coverage_memory = raw.get("coverage_memory", "global")
    if coverage_memory != "global":
        raise RuntimeError(f"{backend_name} currently supports only launch_policy.coverage_memory=global")

    return {
        "grid": grid_values,
        "block_candidates": candidates,
        "physical_block_max": physical_block_max,
        "target_dynamic_shared_bytes": target_dynamic_shared_bytes,
        "coverage_memory": "global",
        "vconfig_reserved": bool(raw.get("vconfig_reserved", True)),
        "logical_grid": logical_grid,
        "logical_block": logical_block,
        "has_logical_vconfig_bounds": has_logical_vconfig_bounds,
        "vconfig_enabled": (
            bool(raw.get("vconfig_reserved", True))
            if vconfig_enabled is None
            else bool(vconfig_enabled)
        ),
        "vconfig_warp_aligned": bool(require_warp_aligned_block),
        "vconfig_mutation": bool(raw.get("vconfig_mutation", True)),
    }


def single_kernel_entry_from_manifest(manifest: dict, *, manifest_path: Path, backend_name: str) -> dict:
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 1 or not isinstance(kernels[0], dict):
        raise RuntimeError(f"{backend_name} build expects a single-kernel manifest: {manifest_path}")
    return kernels[0]


def write_launch_config_header(
    out_dir: Path,
    launch_config: dict,
    *,
    filename: str = "rapid_launch_config.v1.h",
) -> Path:
    candidates = launch_config["block_candidates"]
    grid = launch_config["grid"]
    logical_grid = launch_config.get("logical_grid", [1, 1, 1])
    logical_block = launch_config.get("logical_block", [1, 1, 1])
    header_path = out_dir / filename
    header_path.write_text(
        "#ifndef __RAPID_GENERATED_LAUNCH_CONFIG_V1_H__\n"
        "#define __RAPID_GENERATED_LAUNCH_CONFIG_V1_H__\n\n"
        "#include <array>\n"
        "#include <cstddef>\n\n"
        "namespace rapid_launch_config {\n\n"
        f"inline constexpr std::array<unsigned int, 3> kGrid{{{{{grid[0]}u, {grid[1]}u, {grid[2]}u}}}};\n"
        f"inline constexpr std::array<unsigned int, {len(candidates)}> kBlockCandidates{{{{"
        + ", ".join(f"{candidate}u" for candidate in candidates)
        + "}};\n"
        f"inline constexpr unsigned int kPhysicalBlockMax = {launch_config['physical_block_max']}u;\n"
        f"inline constexpr std::array<unsigned int, 3> kLogicalGrid{{{{{logical_grid[0]}u, {logical_grid[1]}u, {logical_grid[2]}u}}}};\n"
        f"inline constexpr std::array<unsigned int, 3> kLogicalBlock{{{{{logical_block[0]}u, {logical_block[1]}u, {logical_block[2]}u}}}};\n"
        f"inline constexpr bool kHasLogicalVConfigBounds = {'true' if launch_config.get('has_logical_vconfig_bounds', False) else 'false'};\n"
        f"inline constexpr size_t kTargetDynamicSharedBytes = {launch_config['target_dynamic_shared_bytes']}u;\n"
        'inline constexpr const char *kCoverageMemory = "global";\n'
        f"inline constexpr bool kVConfigReserved = {'true' if launch_config['vconfig_reserved'] else 'false'};\n\n"
        f"inline constexpr bool kVConfigEnabled = {'true' if launch_config.get('vconfig_enabled', launch_config['vconfig_reserved']) else 'false'};\n"
        f"inline constexpr bool kVConfigWarpAligned = {'true' if launch_config.get('vconfig_warp_aligned', False) else 'false'};\n"
        f"inline constexpr bool kVConfigMutation = {'true' if launch_config.get('vconfig_mutation', True) else 'false'};\n\n"
        "} // namespace rapid_launch_config\n\n"
        "#endif // __RAPID_GENERATED_LAUNCH_CONFIG_V1_H__\n",
        encoding="utf-8",
    )
    return header_path
