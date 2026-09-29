import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_CONFIG = REPO_ROOT / "cuda-kernel" / "launch_config.py"


def _load_launch_config_module():
    spec = importlib.util.spec_from_file_location(
        "launch_config_for_test", LAUNCH_CONFIG
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaunchConfigTest(unittest.TestCase):
    def test_warp_aligned_vconfig_filters_physical_block_candidates(self) -> None:
        launch = _load_launch_config_module()

        config = launch.launch_config_from_kernel_entry(
            {
                "launch_policy": {
                    "grid": [1, 1, 1],
                    "block_candidates": [96, 48, 32, 16],
                    "physical_block_max": 96,
                    "coverage_memory": "global",
                }
            },
            backend_name="origin",
            require_warp_aligned_block=True,
        )

        self.assertEqual(config["block_candidates"], [96, 32])
        self.assertTrue(all(candidate % 32 == 0 for candidate in config["block_candidates"]))
        self.assertIs(config["vconfig_warp_aligned"], True)

    def test_warp_aligned_vconfig_rejects_non_warp_physical_blocks(self) -> None:
        launch = _load_launch_config_module()

        with self.assertRaisesRegex(RuntimeError, "warp-aligned"):
            launch.launch_config_from_kernel_entry(
                {
                    "launch_policy": {
                        "grid": [1, 1, 1],
                        "block_candidates": [16, 8],
                        "physical_block_max": 16,
                        "coverage_memory": "global",
                    }
                },
                backend_name="rapid2",
                require_warp_aligned_block=True,
            )

    def test_launch_header_records_vconfig_runtime_flags(self) -> None:
        launch = _load_launch_config_module()
        config = {
            "grid": [1, 1, 1],
            "block_candidates": [64, 32],
            "physical_block_max": 64,
            "target_dynamic_shared_bytes": 0,
            "coverage_memory": "global",
            "vconfig_reserved": True,
            "vconfig_enabled": True,
            "vconfig_warp_aligned": True,
        }

        with tempfile.TemporaryDirectory() as td:
            header = launch.write_launch_config_header(Path(td), config)
            text = header.read_text(encoding="utf-8")

        self.assertIn("inline constexpr bool kVConfigEnabled = true;", text)
        self.assertIn("inline constexpr bool kVConfigWarpAligned = true;", text)

    def test_explicit_logical_vconfig_bounds_are_preserved(self) -> None:
        launch = _load_launch_config_module()

        config = launch.launch_config_from_kernel_entry(
            {
                "launch_policy": {
                    "grid": [1, 1, 1],
                    "block_candidates": [512, 256, 128],
                    "physical_block_max": 512,
                    "logical_grid": [1, 1, 1],
                    "logical_block": [16, 16, 1],
                    "coverage_memory": "global",
                    "vconfig_mutation": False,
                }
            },
            backend_name="rapid2",
        )

        self.assertEqual(config["logical_grid"], [1, 1, 1])
        self.assertEqual(config["logical_block"], [16, 16, 1])
        self.assertTrue(config["has_logical_vconfig_bounds"])
        self.assertFalse(config["vconfig_mutation"])
        with tempfile.TemporaryDirectory() as td:
            header = launch.write_launch_config_header(Path(td), config)
            text = header.read_text(encoding="utf-8")

        self.assertIn("inline constexpr bool kHasLogicalVConfigBounds = true;", text)
        self.assertIn("kLogicalBlock{{16u, 16u, 1u}}", text)

    def test_logical_block_must_fit_physical_candidate(self) -> None:
        launch = _load_launch_config_module()

        with self.assertRaisesRegex(RuntimeError, "logical block threads"):
            launch.launch_config_from_kernel_entry(
                {
                    "launch_policy": {
                        "grid": [1, 1, 1],
                        "block_candidates": [128, 64, 32],
                        "physical_block_max": 128,
                        "logical_block": [16, 16, 1],
                        "coverage_memory": "global",
                    }
                },
                backend_name="origin",
            )

    def test_discrete_logical_block_candidates_are_fuzzer_side_only(self) -> None:
        launch = _load_launch_config_module()

        config = launch.launch_config_from_kernel_entry(
            {
                "launch_policy": {
                    "grid": [1, 1, 1],
                    "block_candidates": [256],
                    "physical_block_max": 256,
                    "logical_block": [256, 1, 1],
                    "logical_block_candidates": [
                        [32, 1, 1],
                        [64, 1, 1],
                        [128, 1, 1],
                        [256, 1, 1],
                    ],
                    "coverage_memory": "global",
                }
            },
            backend_name="rapid2",
        )

        self.assertEqual(config["logical_block"], [256, 1, 1])
        self.assertNotIn("logical_block_candidates", config)


if __name__ == "__main__":
    unittest.main()
