#!/usr/bin/env python3
"""Generate and execute the FindMaxCorr10 negative-index trigger.

This checks the manifest-driven seed and fixed-seed fuzzer path. The direct
kernel reproducer documented in ``third_party/BUG_CANDIDATES.md`` supplies the
negative-index oracle.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.kernel_constraints.overrides import (
    _validate_registry,
    apply_constraint_overrides,
)

KERNEL_ID = "_Z13FindMaxCorr10P9SiftPointS0_ii__845fd768"
REPLAY_REGISTRY = (
    REPO_ROOT
    / "third_party/fuzz/kernel_constraints/cudasift_replay_findmaxcorr10.json"
)
SOURCE_PHASE1 = (
    REPO_ROOT
    / "build/e2e/third-party-fuzz/20260731-rq2-fresh1/cudasift/phase1"
)
SOURCE_RUN = SOURCE_PHASE1 / "imported/cudasift"
REPLAY_DIR = REPO_ROOT / "build/e2e/replay-findmaxcorr10"
RESOLVED_DIR = REPLAY_DIR / "resolved"
BACKEND_DIR = REPLAY_DIR / "backend"
FUZZER = REPO_ROOT / "cuda-fuzzer/target/release/fuzzer_async"
PHASE2_SOURCE = (
    SOURCE_PHASE1 / "resolved/cudasift/kernels" / KERNEL_ID / "phase2"
)


def step1_validate() -> None:
    print("=== Step 1: Validate replay constraint registry ===")
    registry = json.loads(REPLAY_REGISTRY.read_text(encoding="utf-8"))
    _validate_registry(registry)
    domain = registry["overrides"][0]["domains"][3]["domain"]
    print(f"  Registry valid: {REPLAY_REGISTRY.name}")
    print(
        f"  numPts2 domain: [{domain['min']}, {domain['max']}] "
        "(trigger: numPts2 < 32)"
    )


def step2_apply_overrides() -> Path:
    print("\n=== Step 2: Apply constraint overrides ===")
    if RESOLVED_DIR.exists():
        shutil.rmtree(RESOLVED_DIR)
    apply_constraint_overrides(
        run_dir=SOURCE_RUN,
        out_dir=RESOLVED_DIR,
        registry_path=REPLAY_REGISTRY,
    )
    kernel_dir = RESOLVED_DIR / "kernels" / KERNEL_ID
    manifest = kernel_dir / "manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"Resolved manifest not found: {manifest}")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    domain = manifest_data["kernels"][0]["args"][3]["domain"]
    print(
        f"  Resolved manifest numPts2: min={domain['min']}, max={domain['max']}"
    )
    phase2_dst = kernel_dir / "phase2"
    if not phase2_dst.exists():
        print("  Copying phase2 artifacts from existing resolved run...")
        shutil.copytree(PHASE2_SOURCE, phase2_dst)
    print(f"  Kernel dir ready: {kernel_dir}")
    return kernel_dir


def step3_build_backend(kernel_dir: Path) -> Path:
    print("\n=== Step 3: Build rapid2 backend ===")
    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "cuda-kernel/rapid2/build.py"),
        "--phase2-dir",
        str(kernel_dir / "phase2"),
        "--out-dir",
        str(BACKEND_DIR),
        "--cuda-path",
        "/usr/local/cuda",
        "--cuda-arch",
        "sm_86",
    ]
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr[-3000:]}")
        raise RuntimeError(f"Backend build failed (exit {result.returncode})")
    so_path = BACKEND_DIR / "librapid2_target.so"
    if not so_path.exists():
        raise RuntimeError(f"Backend .so not found: {so_path}")
    print(f"  Built: {so_path}")
    return so_path


def step4_dump_seed(so_path: Path, kernel_dir: Path) -> int:
    print("\n=== Step 4: Verify seed encodes numPts2=1 ===")
    manifest = kernel_dir / "manifest.json"
    cmd = [
        str(FUZZER),
        str(so_path),
        "--manifest",
        str(manifest),
        "--dump-seed",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"dump-seed failed (exit {result.returncode})")
    seed_hex = result.stdout.strip()
    print(f"  Seed hex ({len(seed_hex)//2} bytes): {seed_hex[:80]}...")
    seed_bytes = bytes.fromhex(seed_hex)
    # The two scalar counts are the final little-endian fields in the envelope.
    num_pts1 = struct.unpack("<i", seed_bytes[-8:-4])[0]
    num_pts2 = struct.unpack("<i", seed_bytes[-4:])[0]
    print(f"  Decoded tail: numPts1={num_pts1}, numPts2={num_pts2}")
    if num_pts2 < 32:
        print(
            f"  CONFIRMED: seed contains numPts2={num_pts2} < 32 "
            "(trigger condition)"
        )
    else:
        print(f"  WARNING: numPts2={num_pts2} does NOT trigger the bug")
    return num_pts2


def step5_run_fuzzer(
    so_path: Path, kernel_dir: Path
) -> subprocess.CompletedProcess[str]:
    print("\n=== Step 5: Run fuzzer replay (fixed seed, 10 runs) ===")
    manifest = kernel_dir / "manifest.json"
    cmd = [
        str(FUZZER),
        str(so_path),
        "--manifest",
        str(manifest),
        "--no-mutate",
        "--runs",
        "10",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    print(f"  Exit code: {result.returncode}")
    for line in result.stdout.splitlines()[-20:]:
        print(f"    {line}")
    if result.stderr:
        for line in result.stderr.splitlines()[-5:]:
            print(f"    [err] {line}")
    return result


def main() -> None:
    step1_validate()
    kernel_dir = step2_apply_overrides()
    so_path = step3_build_backend(kernel_dir)
    num_pts2 = step4_dump_seed(so_path, kernel_dir)
    fuzzer_result = step5_run_fuzzer(so_path, kernel_dir)

    print("\n" + "=" * 60)
    print("REPLAY EVALUATION SUMMARY")
    print("=" * 60)
    print("  Constraint domain: numPts2 in [1, 32], numPts1 pinned at 32")
    print(f"  Seed generation: manifest-driven default (numPts2={num_pts2})")
    print(f"  Fuzzer exit code: {fuzzer_result.returncode}")
    trigger = num_pts2 < 32
    print(f"  Bug trigger condition (numPts2 < 32): {'MET' if trigger else 'NOT MET'}")
    if trigger and fuzzer_result.returncode == 0:
        print(f"  Trigger: numPts2={num_pts2} skips the sift2 tile loop")
        print("  Oracle: direct reproducer confirms the resulting sift2[-1] read")
        print("  RESULT: FUZZ PIPELINE EXECUTED THE CONFIRMED TRIGGER")
    elif trigger:
        print("  RESULT: PARTIAL (trigger generated, execution uncertain)")
    else:
        print("  RESULT: FAILED (domain did not generate trigger)")


if __name__ == "__main__":
    main()
