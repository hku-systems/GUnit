import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase3_smoke import BACKENDS, build_fuzzer_command, collect_dump_seed_tasks, collect_smoke_tasks, main  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class Phase3SmokeTest(unittest.TestCase):
    def test_collect_smoke_tasks_uses_built_phase2_kernels_and_backend_build_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"
            shared_lib = phase2_dir / "backends" / "origin" / "libphase2_origin_target.so"
            shared_lib.parent.mkdir(parents=True)
            shared_lib.write_bytes(b"fake-so")

            _write_json(
                run_dir / "index.json",
                {"kernels": [{"kernel_id": "k0", "dir": "kernels/k0"}]},
            )
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(
                phase2_dir / "metadata.phase2.json",
                {"phase2_status": "built", "failure_reason": None, "failure_detail": None},
            )
            _write_json(
                shared_lib.parent / "backend_build.json",
                {"shared_lib": str(shared_lib)},
            )

            tasks = collect_smoke_tasks(
                run_dir=run_dir,
                backend="origin",
                display_name="target_kernel",
                limit=None,
                runs=7,
                repo_root=REPO_ROOT,
            )

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.kernel_id, "k0")
        self.assertEqual(task.display_name, "target_kernel")
        self.assertEqual(task.shared_lib, shared_lib)
        self.assertEqual(task.manifest_path, kernel_dir / "manifest.json")
        self.assertEqual(task.command[-3:], ["--no-mutate", "--runs", "7"])

    def test_collect_smoke_tasks_uses_async_fuzzer_for_rapid2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"
            shared_lib = phase2_dir / "backends" / "rapid2" / "librapid2_target.so"
            shared_lib.parent.mkdir(parents=True)
            shared_lib.write_bytes(b"fake-so")

            _write_json(run_dir / "index.json", {"kernels": [{"kernel_id": "k0", "dir": str(kernel_dir)}]})
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(phase2_dir / "metadata.phase2.json", {"phase2_status": "built"})
            _write_json(shared_lib.parent / "backend_build.json", {"shared_lib": str(shared_lib)})

            tasks = collect_smoke_tasks(
                run_dir=run_dir,
                backend="rapid2",
                display_name=None,
                limit=1,
                runs=3,
                repo_root=REPO_ROOT,
            )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].command[0], str(REPO_ROOT / "cuda-fuzzer" / "target" / "debug" / "fuzzer_async"))

    def test_collect_smoke_tasks_requires_backend_build_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"
            shared_lib = phase2_dir / "backends" / "origin" / "libphase2_origin_target.so"
            shared_lib.parent.mkdir(parents=True)
            shared_lib.write_bytes(b"fake-so")

            _write_json(run_dir / "index.json", {"kernels": [{"kernel_id": "k0", "dir": "kernels/k0"}]})
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(phase2_dir / "metadata.phase2.json", {"phase2_status": "built"})

            tasks = collect_smoke_tasks(
                run_dir=run_dir,
                backend="origin",
                display_name=None,
                limit=1,
                runs=3,
                repo_root=REPO_ROOT,
            )

        self.assertEqual(tasks, [])

    def test_collect_smoke_tasks_can_use_release_fuzzer_profile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"
            shared_lib = phase2_dir / "backends" / "origin" / "libphase2_origin_target.so"
            shared_lib.parent.mkdir(parents=True)
            shared_lib.write_bytes(b"fake-so")

            _write_json(run_dir / "index.json", {"kernels": [{"kernel_id": "k0", "dir": "kernels/k0"}]})
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(phase2_dir / "metadata.phase2.json", {"phase2_status": "built"})
            _write_json(shared_lib.parent / "backend_build.json", {"shared_lib": str(shared_lib)})

            tasks = collect_smoke_tasks(
                run_dir=run_dir,
                backend="origin",
                display_name=None,
                limit=1,
                runs=3,
                repo_root=REPO_ROOT,
                profile="release",
            )

        self.assertEqual(tasks[0].command[0], str(REPO_ROOT / "cuda-fuzzer" / "target" / "release" / "fuzzer"))

    def test_collect_smoke_tasks_accepts_backend_dir_with_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"
            shared_lib = phase2_dir / "backends" / "origin-phase3-e2e" / "libphase2_origin_target.so"
            shared_lib.parent.mkdir(parents=True)
            shared_lib.write_bytes(b"fake-so")

            _write_json(run_dir / "index.json", {"kernels": [{"kernel_id": "k0", "dir": "kernels/k0"}]})
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(phase2_dir / "metadata.phase2.json", {"phase2_status": "built"})
            _write_json(shared_lib.parent / "backend_build.json", {"shared_lib": str(shared_lib)})

            tasks = collect_smoke_tasks(
                run_dir=run_dir,
                backend="origin",
                display_name=None,
                limit=1,
                runs=3,
                repo_root=REPO_ROOT,
            )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].shared_lib, shared_lib)

    def test_build_fuzzer_command_uses_backend_and_profile(self) -> None:
        self.assertEqual(
            build_fuzzer_command(backend="origin", profile="debug", repo_root=REPO_ROOT),
            [
                "cargo",
                "build",
                "--manifest-path",
                str(REPO_ROOT / "cuda-fuzzer" / "Cargo.toml"),
                "--bin",
                "fuzzer",
            ],
        )

    def test_cufuzz_uses_the_synchronous_fuzzer(self) -> None:
        self.assertIn("cufuzz", BACKENDS)
        self.assertEqual(
            build_fuzzer_command(backend="cufuzz", profile="debug", repo_root=REPO_ROOT),
            [
                "cargo",
                "build",
                "--manifest-path",
                str(REPO_ROOT / "cuda-fuzzer" / "Cargo.toml"),
                "--bin",
                "fuzzer",
            ],
        )

    def test_main_dry_run_does_not_build_fuzzer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"
            shared_lib = phase2_dir / "backends" / "origin" / "libphase2_origin_target.so"
            shared_lib.parent.mkdir(parents=True)
            shared_lib.write_bytes(b"fake-so")

            _write_json(run_dir / "index.json", {"kernels": [{"kernel_id": "k0", "dir": "kernels/k0"}]})
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(phase2_dir / "metadata.phase2.json", {"phase2_status": "built"})
            _write_json(shared_lib.parent / "backend_build.json", {"shared_lib": str(shared_lib)})

            with mock.patch.object(
                sys,
                "argv",
                [
                    "phase3_smoke.py",
                    "--run-dir",
                    str(run_dir),
                    "--backend",
                    "origin",
                    "--build-fuzzer",
                    "--dry-run",
                ],
            ), mock.patch("phase3_smoke.subprocess.run") as run_mock:
                result = main()

        self.assertEqual(result, 0)
        run_mock.assert_not_called()
        self.assertEqual(
            build_fuzzer_command(backend="rapid2", profile="release", repo_root=REPO_ROOT),
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(REPO_ROOT / "cuda-fuzzer" / "Cargo.toml"),
                "--bin",
                "fuzzer_async",
            ],
        )

    def test_collect_dump_seed_tasks_does_not_require_backend_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"

            _write_json(run_dir / "index.json", {"kernels": [{"kernel_id": "k0", "dir": "kernels/k0"}]})
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(phase2_dir / "metadata.phase2.json", {"phase2_status": "built"})

            tasks = collect_dump_seed_tasks(
                run_dir=run_dir,
                display_name=None,
                limit=1,
                repo_root=REPO_ROOT,
                profile="debug",
            )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].shared_lib.name, "rapid-phase3-dump-seed-unused.so")
        self.assertEqual(tasks[0].command[0], str(REPO_ROOT / "cuda-fuzzer" / "target" / "debug" / "fuzzer"))
        self.assertEqual(tasks[0].command[-1], "--dump-seed")

    def test_main_dump_seeds_dry_run_does_not_require_backend_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            kernel_dir = run_dir / "kernels" / "k0"
            phase2_dir = kernel_dir / "phase2"

            _write_json(run_dir / "index.json", {"kernels": [{"kernel_id": "k0", "dir": "kernels/k0"}]})
            _write_json(
                kernel_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kernels": [
                        {
                            "display_name": "target_kernel",
                            "symbol_name": "target_kernel",
                            "args": [],
                        }
                    ],
                },
            )
            _write_json(kernel_dir / "metadata.json", {"build_status": "built"})
            _write_json(phase2_dir / "metadata.phase2.json", {"phase2_status": "built"})

            with mock.patch.object(
                sys,
                "argv",
                [
                    "phase3_smoke.py",
                    "--run-dir",
                    str(run_dir),
                    "--dump-seeds",
                    "--build-fuzzer",
                    "--dry-run",
                ],
            ), mock.patch("phase3_smoke.subprocess.run") as run_mock:
                result = main()

        self.assertEqual(result, 0)
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
