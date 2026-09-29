"""Unit tests for benchmark/rq4/crash_replay.py (no GPU required)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from benchmark.rq4.crash_replay import (
    ReplayResult,
    _classify_exit,
    _detect_abi,
    collect_crash_inputs,
)


class ClassifyExitTests(unittest.TestCase):
    def test_status_codes_map_to_exit_kinds(self) -> None:
        expected = {
            0: "ok",
            1: "crash",
            2: "timeout",
            3: "invalid_input",
            99: "error",
        }
        for code, exit_kind in expected.items():
            with self.subTest(code=code):
                self.assertEqual(_classify_exit(code), exit_kind)


class CollectCrashInputsTests(unittest.TestCase):
    def test_single_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            f.write(b"\x00" * 32)
            f.flush()
            result = collect_crash_inputs(Path(f.name))
            self.assertEqual(result, [Path(f.name)])

    def test_directory_skips_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            (dp / "abc123").write_bytes(b"\x00" * 64)
            (dp / ".abc123").write_bytes(b"{}")
            (dp / ".abc123_1.metadata").write_bytes(b"{}")
            result = collect_crash_inputs(dp)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "abc123")


class DetectAbiTests(unittest.TestCase):
    def test_detects_sync_and_async_abis(self) -> None:
        mock_lib = MagicMock(spec=[])
        del mock_lib.libafl_submit_with_id
        self.assertEqual(_detect_abi(mock_lib), "sync")

        mock_lib = MagicMock()
        mock_lib.libafl_submit_with_id = MagicMock()
        self.assertEqual(_detect_abi(mock_lib), "async")


class ReplayResultTests(unittest.TestCase):
    def test_as_dict_format(self) -> None:
        r = ReplayResult(
            input_path="/tmp/crash",
            input_size=8288,
            status_code=1,
            status_name="CUDA_ERROR",
            stage=4,
            detail=0x2BC,
            exec_time_ns=1_500_000,
            exit_kind="crash",
        )
        d = r.as_dict()
        self.assertEqual(d["status_code"], 1)
        self.assertEqual(d["stage"], 4)
        self.assertEqual(d["detail"], "0x2bc")
        self.assertEqual(d["exit_kind"], "crash")
        self.assertEqual(d["input_size"], 8288)


if __name__ == "__main__":
    unittest.main()
