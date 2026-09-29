"""Generate per-target compile-time feedback layout constants."""

from __future__ import annotations

from pathlib import Path

from contracts.kernel import KernelContract


class TargetLayoutHeaderGenerator:
    """Generate the host/device-shared layout header for one kernel target."""

    def generate(self, *, contract: KernelContract, phase2_dir: Path) -> None:
        gen_dir = phase2_dir / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)
        count = len(contract.feedback_payload_slots)
        (gen_dir / "rapid_target_layout.v1.h").write_text(
            "#ifndef __RAPID_TARGET_LAYOUT_V1_H__\n"
            "#define __RAPID_TARGET_LAYOUT_V1_H__\n\n"
            f"#define RAPID_PAYLOAD_SLOT_COUNT {count}u\n"
            "#define RAPID_PAYLOAD_SLOT_STORAGE_COUNT \\\n  (RAPID_PAYLOAD_SLOT_COUNT == 0u ? 1u : RAPID_PAYLOAD_SLOT_COUNT)\n\n"
            "#endif // __RAPID_TARGET_LAYOUT_V1_H__\n",
            encoding="utf-8",
        )
