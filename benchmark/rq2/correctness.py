"""Build and run direct CUDA correctness checks for RQ2-adapted kernels."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


DRIVERS = {
    "shoc_scan": (32, 64, 128, 256),
    "shoc_reduction": (32, 64, 128, 256),
    "flashattention_device1xn": (32, 64, 128, 256),
    "synth_complex": (32, 64, 128, 256),
}


def _run_checked(command: list[str], *, env: dict[str, str] | None = None, label: str) -> None:
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit {result.returncode}: "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )


def build_and_run_correctness(
    *,
    repo_root: Path,
    cuda_path: str,
    gpu_device: int,
    build_dir: Path,
) -> None:
    rq2_root = repo_root / "benchmark" / "rq2"
    nvcc = str(Path(cuda_path) / "bin" / "nvcc")
    build_dir.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

    for workload_id, candidates in DRIVERS.items():
        executable = build_dir / f"{workload_id}_correctness"
        _run_checked(
            [
                nvcc,
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                str(rq2_root / "correctness" / f"{workload_id}_driver.cu"),
                str(rq2_root / "workloads" / workload_id / "kernel.cu"),
                "-o",
                str(executable),
            ],
            label=f"{workload_id} compile",
        )
        _run_checked(
            [str(executable), *(str(candidate) for candidate in candidates)],
            env=run_env,
            label=f"{workload_id} correctness",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-device", type=int, required=True)
    parser.add_argument("--cuda-path", default="/usr/local/cuda")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="rq2-correctness-") as temp_dir:
        build_and_run_correctness(
            repo_root=repo_root,
            cuda_path=args.cuda_path,
            gpu_device=args.gpu_device,
            build_dir=Path(temp_dir),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
