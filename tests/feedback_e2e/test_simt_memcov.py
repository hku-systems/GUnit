from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simt_memcov_kernel.cu"
SITE_ID = 0x12345678
MASK32 = 0xFFFFFFFF

ARG_SLOT_SALT = 0x9E3779B9
SECTOR_LOW_SALT = 0x85EBCA6B
SECTOR_HIGH_SALT = 0xC2B2AE35
ACCESS_KIND_SALT = 0x27D4EB2F
LOGICAL_BLOCK_SALT = 0x165667B1
WARP_IN_BLOCK_SALT = 0xD3A2646D
PATTERN_SALT = 0xFD7046C5

SINGLE = 0
FULL_BROADCAST = 1
FULL_CONTIGUOUS = 2
FULL_OTHER = 3
PARTIAL_BROADCAST = 4
PARTIAL_CONTIGUOUS = 5
PARTIAL_OTHER = 6


def _rotl32(value: int, amount: int) -> int:
    return ((value << amount) | (value >> (32 - amount))) & MASK32


def _mix32(value: int) -> int:
    value &= MASK32
    value ^= value >> 16
    value = (value * 0x7FEB352D) & MASK32
    value ^= value >> 15
    value = (value * 0x846CA68B) & MASK32
    value ^= value >> 16
    return value & MASK32


def _fold64(value: int) -> int:
    return (value & MASK32) ^ _rotl32((value >> 32) & MASK32, 16)


def _bucket(sector: int, warp: int, pattern: int) -> int:
    key = SITE_ID
    key ^= (0 * ARG_SLOT_SALT) & MASK32
    key ^= ((sector & MASK32) * SECTOR_LOW_SALT) & MASK32
    key ^= (((sector >> 32) & MASK32) * SECTOR_HIGH_SALT) & MASK32
    key ^= (0 * ACCESS_KIND_SALT) & MASK32
    key ^= (_fold64(0) * LOGICAL_BLOCK_SALT) & MASK32
    key ^= (warp * WARP_IN_BLOCK_SALT) & MASK32
    key ^= (pattern * PATTERN_SALT) & MASK32
    return _mix32(key) % 61_440


def _features(
    sectors: range | tuple[int, ...], pattern: int, warp: int = 0
) -> set[int]:
    return {_bucket(sector, warp, pattern) for sector in sectors}


class SimtMemCovGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("RAPID_RUN_GPU_TESTS") != "1":
            raise unittest.SkipTest("set RAPID_RUN_GPU_TESTS=1 to run CUDA tests")
        nvcc = Path("/usr/local/cuda/bin/nvcc")
        if not nvcc.is_file():
            raise unittest.SkipTest("nvcc is unavailable")
        cls.tempdir = tempfile.TemporaryDirectory(prefix="rapid-simt-memcov-")
        cls.executable = Path(cls.tempdir.name) / "simt_memcov"
        subprocess.run(
            [
                str(nvcc),
                "-std=c++17",
                "-O3",
                "-arch=sm_86",
                "-lineinfo",
                "-I",
                str(REPO_ROOT / "cuda-kernel"),
                "-I",
                str(REPO_ROOT / "cuda-kernel" / "utils"),
                str(FIXTURE),
                "-o",
                str(cls.executable),
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tempdir"):
            cls.tempdir.cleanup()

    def _run_with_atomic_count(self, case: str) -> tuple[set[int], int]:
        completed = subprocess.run(
            [str(self.executable), case],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = completed.stdout.strip()
        features = {int(value) for value in output.split(",")} if output else set()
        prefix = "atomic_calls="
        diagnostic = completed.stderr.strip()
        self.assertTrue(diagnostic.startswith(prefix), diagnostic)
        return features, int(diagnostic.removeprefix(prefix))

    def _run(self, case: str) -> set[int]:
        return self._run_with_atomic_count(case)[0]

    def test_all_frozen_warp_patterns_produce_exact_features(self) -> None:
        cases = {
            "single": _features((0,), SINGLE),
            "full_broadcast": _features((0,), FULL_BROADCAST),
            "full_contiguous": _features(range(4), FULL_CONTIGUOUS),
            "full_other": _features(range(8), FULL_OTHER),
            "partial_broadcast": _features((0,), PARTIAL_BROADCAST),
            "partial_contiguous": _features(range(2), PARTIAL_CONTIGUOUS),
            "partial_other": _features(range(4), PARTIAL_OTHER),
        }
        for case, expected in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self._run(case), expected)

    def test_cross_sector_invalid_lane_and_deduplication_semantics(self) -> None:
        self.assertEqual(self._run("cross_sector"), _features((0, 1), SINGLE))
        self.assertEqual(
            self._run("invalid_lanes"),
            _features((0, 1), PARTIAL_CONTIGUOUS),
        )
        self.assertEqual(self._run("deduplicate"), _features((0,), SINGLE))

    def test_vconfig_active_warps_use_preserved_linear_identity(self) -> None:
        warp_zero = _features(range(4), FULL_CONTIGUOUS, warp=0)
        warp_one = _features(range(4, 8), FULL_CONTIGUOUS, warp=1)
        self.assertEqual(self._run("vconfig32"), warp_zero)
        self.assertEqual(self._run("vconfig64"), warp_zero | warp_one)

    def test_other_path_emits_each_overlapping_sector_once(self) -> None:
        features, atomic_calls = self._run_with_atomic_count("overlapping_other")
        self.assertEqual(features, _features((0, 1), PARTIAL_OTHER))
        self.assertEqual(atomic_calls, 2)

    def test_other_path_emits_distinct_sectors_in_one_warp(self) -> None:
        features, atomic_calls = self._run_with_atomic_count(
            "distinct_sector_other"
        )
        self.assertEqual(features, _features(range(32), FULL_OTHER))
        self.assertEqual(atomic_calls, 32)


if __name__ == "__main__":
    unittest.main()
