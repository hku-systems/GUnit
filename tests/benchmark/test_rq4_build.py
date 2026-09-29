import unittest
from pathlib import Path


class RQ4BuildTests(unittest.TestCase):
    def test_timing_build_commands_enable_only_profile_instrumentation(self) -> None:
        from benchmark.rq4.build import timing_build_command

        commands = {
            backend: timing_build_command(
                backend=backend,
                python=Path("python"),
                phase2_dir=Path("phase2"),
                out_dir=Path(f"out/{backend}"),
                cuda_path="/cuda",
                cuda_arch="sm_86",
            )
            for backend in ("origin-no-feedback", "origin", "rapid", "rapid2")
        }
        for command in commands.values():
            self.assertIn("--enable-kernel-timing", command)
            self.assertEqual(command[command.index("--build-profile") + 1], "release")
        self.assertEqual(
            commands["origin-no-feedback"][commands["origin-no-feedback"].index("--feedback-instrumentation") + 1],
            "disabled",
        )
        for backend in ("origin", "rapid", "rapid2"):
            command = commands[backend]
            self.assertEqual(
                command[command.index("--feedback-instrumentation") + 1], "enabled"
            )
        self.assertTrue(commands["origin"][1].endswith("cuda-kernel/origin/build.py"))
        self.assertTrue(commands["rapid"][1].endswith("cuda-kernel/rapid/build.py"))
        self.assertTrue(commands["rapid2"][1].endswith("cuda-kernel/rapid2/build.py"))

    def test_timing_build_rejects_unknown_backend(self) -> None:
        from benchmark.rq4.build import timing_build_command

        with self.assertRaises(ValueError):
            timing_build_command(
                backend="unknown",
                python=Path("python"),
                phase2_dir=Path("phase2"),
                out_dir=Path("out"),
                cuda_path="/cuda",
                cuda_arch="sm_86",
            )


if __name__ == "__main__":
    unittest.main()
