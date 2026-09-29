#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CUDA_KERNEL_DIR = Path(__file__).resolve().parent
if str(CUDA_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_KERNEL_DIR))

from backend_build import build_optimization_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_SMOKE_CLI = REPO_ROOT / "scripts" / "kernel-smoke" / "cli.py"
KERNEL_REWRITE_CLI = REPO_ROOT / "scripts" / "kernel-rewrite" / "cli.py"
KERNEL_SMOKE_WRAPPERS = REPO_ROOT / "scripts" / "kernel-smoke"
ORIGIN_BUILD_CLI = CUDA_KERNEL_DIR / "origin" / "build.py"
CUFUZZ_BUILD_CLI = CUDA_KERNEL_DIR / "cufuzz" / "build.py"
RAPID_BUILD_CLI = CUDA_KERNEL_DIR / "rapid" / "build.py"
RAPID2_BUILD_CLI = CUDA_KERNEL_DIR / "rapid2" / "build.py"


def _pick_python() -> str:
    for venv_name in (".venv", ".venv-ksmoke"):
        venv_python = REPO_ROOT / venv_name / "bin" / "python"
        if venv_python.is_file():
            return str(venv_python)
    return sys.executable


def _pick_wrapped_cuda_compiler() -> str:
    clang_wrapper = KERNEL_SMOKE_WRAPPERS / "clang++"
    nvcc_wrapper = KERNEL_SMOKE_WRAPPERS / "nvcc"
    if shutil.which("clang++") and clang_wrapper.is_file():
        return str(clang_wrapper)
    if shutil.which("nvcc") and nvcc_wrapper.is_file():
        return str(nvcc_wrapper)
    raise RuntimeError("no wrapped CUDA compiler available (clang++/nvcc)")


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _capture_compile_command(
    *,
    wrapped_compiler: str,
    source_path: Path,
    object_path: Path,
    cuda_path: str,
    cuda_arch: str,
    build_profile: str = "release",
) -> list[str]:
    optimization = build_optimization_contract(build_profile)
    return [
        wrapped_compiler,
        *optimization.device_clang_flags,
        "-x",
        "cuda",
        "-c",
        str(source_path),
        "-o",
        str(object_path),
        f"--cuda-path={cuda_path}",
        f"--cuda-gpu-arch={cuda_arch}",
        "-I",
        str(source_path.parent),
        "-I",
        str(CUDA_KERNEL_DIR),
        "-I",
        str(CUDA_KERNEL_DIR / "utils"),
    ]


def _build_backend(
    *,
    python: str,
    cli: Path,
    phase2_dir: Path,
    out_dir: Path,
    cuda_arch: str,
    cuda_path: str,
    env: dict[str, str],
    build_profile: str,
    extra_args: tuple[str, ...] = (),
) -> dict:
    _run(
        [
            python,
            str(cli),
            "--phase2-dir",
            str(phase2_dir),
            "--out-dir",
            str(out_dir),
            "--cuda-arch",
            cuda_arch,
            "--cuda-path",
            cuda_path,
            "--build-profile",
            build_profile,
            *extra_args,
        ],
        env=env,
    )
    return json.loads((out_dir / "backend_build.json").read_text(encoding="utf-8"))


def _kernel_dir_from_index_entry(run_dir: Path, entry: dict) -> Path:
    kernel_dir = Path(entry["dir"])
    if not kernel_dir.is_absolute():
        kernel_dir = run_dir / kernel_dir
    return kernel_dir


def run_builtin_phase_pipeline(
    *,
    out_root: Path,
    run_id: str,
    cuda_arch: str,
    cuda_path: str,
    source_path: Path | None = None,
    build_profile: str = "release",
    canary: str = "enabled",
) -> Path:
    capture_dir = out_root / "capture"
    out_dir = out_root / "out"
    object_path = out_root / "build" / "kernel.o"

    shutil.rmtree(capture_dir, ignore_errors=True)
    run_dir = out_dir / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RAPID_CAPTURE_DIR"] = str(capture_dir)
    wrapped_compiler = _pick_wrapped_cuda_compiler()
    python = _pick_python()
    source_path = (source_path or CUDA_KERNEL_DIR / "kernel.cu").resolve()

    compile_cmd = _capture_compile_command(
        wrapped_compiler=wrapped_compiler,
        source_path=source_path,
        object_path=object_path,
        cuda_path=cuda_path,
        cuda_arch=cuda_arch,
        build_profile=build_profile,
    )
    _run(compile_cmd, env=env)

    _run(
        [
            python,
            str(KERNEL_SMOKE_CLI),
            "run",
            "--capture-dir",
            str(capture_dir),
            "--out-root",
            str(out_dir),
            "--run-id",
            run_id,
            "--target-lib",
            "rapid_builtin_phase_pipeline",
            "--mode",
            "artifact",
            "--jobs",
            "1",
        ],
        env=env,
    )

    _run(
        [
            python,
            str(KERNEL_REWRITE_CLI),
            "--run-dir",
            str(run_dir),
        ],
        env=env,
    )

    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    kernels = index.get("kernels") or []
    canary_args = ("--canary", canary)
    for entry in kernels:
        kernel_dir = _kernel_dir_from_index_entry(run_dir, entry)
        phase2_dir = kernel_dir / "phase2"
        backends_dir = phase2_dir / "backends"
        backends_dir.mkdir(parents=True, exist_ok=True)
        _build_backend(
            python=python,
            cli=ORIGIN_BUILD_CLI,
            phase2_dir=phase2_dir,
            out_dir=backends_dir / "origin",
            cuda_arch=cuda_arch,
            cuda_path=cuda_path,
            env=env,
            build_profile=build_profile,
            extra_args=canary_args,
        )
        _build_backend(
            python=python,
            cli=ORIGIN_BUILD_CLI,
            phase2_dir=phase2_dir,
            out_dir=backends_dir / "origin-no-feedback",
            cuda_arch=cuda_arch,
            cuda_path=cuda_path,
            env=env,
            build_profile=build_profile,
            extra_args=("--feedback-instrumentation", "disabled")
            + canary_args,
        )
        _build_backend(
            python=python,
            cli=CUFUZZ_BUILD_CLI,
            phase2_dir=phase2_dir,
            out_dir=backends_dir / "cufuzz",
            cuda_arch=cuda_arch,
            cuda_path=cuda_path,
            env=env,
            build_profile=build_profile,
        )
        _build_backend(
            python=python,
            cli=RAPID_BUILD_CLI,
            phase2_dir=phase2_dir,
            out_dir=backends_dir / "rapid",
            cuda_arch=cuda_arch,
            cuda_path=cuda_path,
            env=env,
            build_profile=build_profile,
            extra_args=canary_args,
        )
        _build_backend(
            python=python,
            cli=RAPID2_BUILD_CLI,
            phase2_dir=phase2_dir,
            out_dir=backends_dir / "rapid2",
            cuda_arch=cuda_arch,
            cuda_path=cuda_path,
            env=env,
            build_profile=build_profile,
            extra_args=canary_args,
        )

    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the official Phase1/Phase2 pipeline for the in-repo builtin kernel example."
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--run-id", default="builtin-kernel")
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    parser.add_argument("--cuda-arch", default=os.environ.get("CUDA_ARCH", "sm_86"))
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--canary", choices=("enabled", "disabled"), default="enabled"
    )
    parser.add_argument(
        "--build-profile",
        choices=("release", "debug"),
        default="release",
    )
    args = parser.parse_args()

    run_dir = run_builtin_phase_pipeline(
        out_root=Path(args.out_root).resolve(),
        run_id=args.run_id,
        cuda_arch=args.cuda_arch,
        cuda_path=args.cuda_path,
        source_path=args.source,
        build_profile=args.build_profile,
        canary=args.canary,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
