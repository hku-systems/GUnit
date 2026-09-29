import atexit
import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "kernel_pipeline_smoke" / "fixtures"
WRAP_DIR = REPO_ROOT / "scripts" / "kernel-smoke"
CLI = REPO_ROOT / "scripts" / "kernel-smoke" / "cli.py"
RUN_ID = "fixtures-all-e2e"
TARGET_LIB = "fixtures_all_e2e"
CAPTURE_MODE = "artifact"
CAPTURE_JOBS = 1


@dataclass(frozen=True)
class CaptureE2EContext:
    temp_root: Path
    capture_dir: Path
    out_root: Path
    run_id: str
    run_dir: Path
    commands_jsonl: Path
    fixtures_build_dir: Path
    fixtures_runner: Path
    mini_project_bin: Path


_CTX: CaptureE2EContext | None = None


def cuda_device_skip_reason() -> str | None:
    """Return a unittest skip reason when no CUDA device is available."""
    lib_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
    try:
        libcuda = ctypes.CDLL(lib_name)
    except OSError as exc:
        return f"CUDA driver library not available: {exc}"

    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int
    status = cu_init(0)
    if status != 0:
        return f"cuInit failed with CUDA driver status {status}"

    count = ctypes.c_int()
    cu_device_get_count = libcuda.cuDeviceGetCount
    cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_device_get_count.restype = ctypes.c_int
    status = cu_device_get_count(ctypes.byref(count))
    if status != 0:
        return f"cuDeviceGetCount failed with CUDA driver status {status}"
    if count.value <= 0:
        return "no CUDA-capable device is available"

    return None


def _pick_cuda_compiler() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc

    clang = shutil.which("clang++")
    if clang and _is_cuda_capable_clang(clang):
        return clang

    return clang or ""


def _is_cuda_capable_clang(clang_path: str) -> bool:
    cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    cuda_arch = os.environ.get("CUDA_ARCH", "sm_86")
    test_src = "__global__ void _ksmoke_probe() {}\n"
    with tempfile.TemporaryDirectory(prefix="ksmoke-clang-probe-") as td:
        src_path = Path(td) / "probe.cu"
        out_path = Path(td) / "probe.o"
        src_path.write_text(test_src, encoding="utf-8")
        cmd = [
            clang_path,
            "-x",
            "cuda",
            "-c",
            str(src_path),
            "-o",
            str(out_path),
            f"--cuda-path={cuda_path}",
            f"--cuda-gpu-arch={cuda_arch}",
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
    return result.returncode == 0

def get_capture_e2e_context() -> CaptureE2EContext:
    global _CTX
    if _CTX is not None:
        return _CTX

    compiler = _pick_cuda_compiler()
    if not compiler:
        raise RuntimeError("no CUDA compiler found (clang++/nvcc)")
    wrapper = WRAP_DIR / Path(compiler).name
    if not wrapper.is_file():
        raise RuntimeError(f"compiler wrapper not found: {wrapper}")
    if not CLI.is_file():
        raise RuntimeError(f"cli.py not found: {CLI}")
    if not FIXTURES_DIR.is_dir():
        raise RuntimeError(f"fixtures dir not found: {FIXTURES_DIR}")

    temp_root = Path(tempfile.mkdtemp(prefix="kernel-smoke-e2e-"))
    capture_dir = temp_root / "capture"
    out_root = temp_root / "out"
    build_dir = Path(tempfile.mkdtemp(prefix="build-fixtures-", dir=temp_root))
    run_id = RUN_ID
    run_dir = out_root / run_id
    commands_jsonl = capture_dir / "commands.jsonl"

    env = os.environ.copy()
    env["RAPID_CAPTURE_DIR"] = str(capture_dir)
    build_dir_arg = f"BUILD_DIR={build_dir}"

    subprocess.run(
        ["make", "-C", str(FIXTURES_DIR), "clean", build_dir_arg],
        capture_output=True,
        env=env,
    )
    build = subprocess.run(
        ["make", "-C", str(FIXTURES_DIR), "all", f"CUDA_COMPILER={wrapper}", build_dir_arg],
        capture_output=True,
        timeout=600,
        env=env,
    )
    if build.returncode != 0:
        raise RuntimeError(
            "make all failed:\n"
            f"stdout={build.stdout.decode()}\n"
            f"stderr={build.stderr.decode()}"
        )

    if not commands_jsonl.is_file():
        raise RuntimeError(f"missing capture log: {commands_jsonl}")

    run = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "run",
            "--capture-dir",
            str(capture_dir),
            "--out-root",
            str(out_root),
            "--run-id",
            run_id,
            "--target-lib",
            TARGET_LIB,
            "--mode",
            CAPTURE_MODE,
            "--jobs",
            str(CAPTURE_JOBS),
        ],
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=600,
        env=env,
    )
    if run.returncode != 0:
        raise RuntimeError(
            "cli run --capture-dir failed:\n"
            f"stdout={run.stdout.decode()}\n"
            f"stderr={run.stderr.decode()}"
        )

    _CTX = CaptureE2EContext(
        temp_root=temp_root,
        capture_dir=capture_dir,
        out_root=out_root,
        run_id=run_id,
        run_dir=run_dir,
        commands_jsonl=commands_jsonl,
        fixtures_build_dir=build_dir,
        fixtures_runner=FIXTURES_DIR / "fixtures_runner",
        mini_project_bin=FIXTURES_DIR / "project" / "mini_project",
    )

    def _cleanup() -> None:
        subprocess.run(
            ["make", "-C", str(FIXTURES_DIR), "clean", build_dir_arg],
            capture_output=True,
        )
        shutil.rmtree(temp_root, ignore_errors=True)

    atexit.register(_cleanup)
    return _CTX
