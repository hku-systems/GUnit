import unittest
from pathlib import Path


class RQ4CampaignTests(unittest.TestCase):
    def test_benchmark_command_selects_window_only_for_gunit_variants(self) -> None:
        from benchmark.rq4.campaign import benchmark_command
        from benchmark.rq4.schema import configurations

        commands = {
            config.name: benchmark_command(
                executable=Path("fuzzer_async" if config.async_frontend else "fuzzer"),
                library=Path("target.so"),
                manifest=Path("manifest.json"),
                warmup_runs=100,
                benchmark_seconds=120,
                window_size=config.window_size,
            )
            for config in configurations(window_size=4)
        }
        self.assertNotIn("--window-size", commands["cufuzz"])
        self.assertNotIn("--window-size", commands["libafl"])
        self.assertNotIn("--window-size", commands["libafl-plus"])
        self.assertEqual(commands["gunit-sync"][-2:], ["--window-size", "4"])
        self.assertEqual(commands["gunit"][-2:], ["--window-size", "4"])
        for name in ("cufuzz", "libafl", "libafl-plus", "gunit-sync", "gunit"):
            self.assertNotIn("--retire-worker", commands[name])
            self.assertNotIn("--supply-threads", commands[name])


if __name__ == "__main__":
    unittest.main()
