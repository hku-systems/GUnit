"""Replay captured compile commands to produce LLVM device bitcode.

Transforms captured compiler invocations into clang device-only
emit-llvm commands, supporting both clang and nvcc-based captures.

Also provides preprocessing replay for macro/include-expanded source
analysis (Phase A).
"""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


_REPLAY_TIMEOUT_SEC = int(os.environ.get("KSMOKE_REPLAY_TIMEOUT_SEC", "600"))
_BUILD_DEVICE_BC_TIMEOUT_SEC = int(
    os.environ.get(
        "KSMOKE_BUILD_DEVICE_BC_TIMEOUT_SEC",
        os.environ.get("KSMOKE_REPLAY_TIMEOUT_SEC", "1200"),
    ),
)

_CUDA_VERSION_CACHE: dict[str, tuple[str, str] | None] = {}


# Flags that are linker-only or compile-driver wrappers (drop these)
_DROP_FLAGS = frozenset([
    "-shared", "-static", "-rdynamic", "-pie", "-no-pie",
    "-Wl,", "-s", "--as-needed", "--no-as-needed",
])

# Prefixes for flags that take a following argument (linker paths, libs)
_DROP_FLAG_PREFIXES = ("-L", "-l", "-Wl,", "-rpath", "--soname")

# Flags to preserve (include/define/std/arch related)
_KEEP_PREFIXES = (
    "-I", "-D", "-U", "-std=", "--std=",
    "-isystem", "-include", "-isysroot",
    "--cuda-gpu-arch", "--cuda-path",
    "-O", "-g", "-f", "-W",
    "-target", "--target",
)

# Flags relevant for preprocessing (include/define/std/language)
_PREPROCESS_KEEP_PREFIXES = (
    "-I", "-D", "-U", "-std=", "--std=",
    "-isystem", "-include", "-isysroot",
    "--cuda-gpu-arch", "--cuda-path",
    "-target", "--target",
)


def _should_drop_flag(flag: str) -> bool:
    """Return True if the flag should be dropped for device analysis."""
    if flag in _DROP_FLAGS:
        return True
    for prefix in _DROP_FLAG_PREFIXES:
        if flag.startswith(prefix):
            return True
    return False


def _is_source_file(arg: str) -> bool:
    """Check if an argument looks like a source file."""
    return arg.endswith((".cu", ".cpp", ".cxx", ".cc", ".c"))


def _is_object_file(arg: str) -> bool:
    """Check if an argument looks like an object/output file."""
    return arg.endswith((".o", ".obj", ".so", ".a", ".out"))


def _extract_sm_from_codegen_spec(spec: str) -> str | None:
    """Extract a replayable sm_XX token from nvcc gencode/generate-code spec."""
    m = re.search(r"sm_\d+", spec)
    if m is not None:
        return m.group(0)

    m = re.search(r"compute_(\d+)", spec)
    if m is None:
        return None
    return f"sm_{m.group(1)}"


def _normalize_sm_token(token: str) -> str | None:
    """Normalize arch token into canonical sm_XX form."""
    t = token.strip().lower()
    if not t:
        return None
    if t.startswith("sm_"):
        suffix = t[3:]
        if suffix.isdigit():
            return f"sm_{suffix}"
        return None
    if t.isdigit():
        return f"sm_{t}"
    return None


def _detect_local_sm() -> str | None:
    """Detect local GPU arch (sm_XX), prefer explicit env override."""
    forced = os.environ.get("KSMOKE_CUDA_ARCH", "")
    if forced:
        norm = _normalize_sm_token(forced)
        if norm is not None:
            return norm

    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        return None

    try:
        result = subprocess.run(
            [nvsmi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    line = ""
    for raw in (result.stdout or "").splitlines():
        raw = raw.strip()
        if raw:
            line = raw
            break
    if not line:
        return None

    # Typical nvidia-smi output: "9.0"
    m = re.search(r"(\d+)(?:\.(\d+))?", line)
    if m is None:
        return None

    major = m.group(1)
    minor = m.group(2) or "0"
    return f"sm_{major}{minor}"


def _extract_cuda_version_macros(cuda_path_arg: str | None) -> tuple[str, str] | None:
    """Best-effort CUDA version macros for clang CUDA replay compatibility."""
    cache_key = cuda_path_arg or ""
    cached = _CUDA_VERSION_CACHE.get(cache_key)
    if cached is not None or cache_key in _CUDA_VERSION_CACHE:
        return cached

    nvcc = shutil.which("nvcc")
    if cuda_path_arg and cuda_path_arg.startswith("--cuda-path="):
        candidate = Path(cuda_path_arg.split("=", 1)[1]) / "bin" / "nvcc"
        if candidate.exists():
            nvcc = str(candidate)

    if nvcc is None:
        _CUDA_VERSION_CACHE[cache_key] = None
        return None

    try:
        result = subprocess.run(
            [nvcc, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        _CUDA_VERSION_CACHE[cache_key] = None
        return None

    if result.returncode != 0:
        _CUDA_VERSION_CACHE[cache_key] = None
        return None

    output = f"{result.stdout}\n{result.stderr}"
    m = re.search(r"release\s+(\d+)\.(\d+)", output)
    if m is None:
        _CUDA_VERSION_CACHE[cache_key] = None
        return None

    version = (m.group(1), m.group(2))
    _CUDA_VERSION_CACHE[cache_key] = version
    return version


def _pick_single_sm(candidates: list[str]) -> str | None:
    """Pick one arch for replay to avoid multi-output clang mode."""
    if not candidates:
        return None

    local_sm = _detect_local_sm()
    if local_sm and local_sm in candidates:
        return local_sm

    return candidates[0]


def _read_rsp_tokens(rsp_path: Path, seen: set[Path]) -> list[str]:
    """Recursively read nvcc response/options files and return expanded tokens."""
    resolved = rsp_path.resolve()
    if resolved in seen:
        return []
    seen.add(resolved)

    try:
        content = resolved.read_text(encoding="utf-8")
    except OSError:
        return []

    tokens = shlex.split(content, posix=True)
    return _expand_nvcc_options_files(tokens, resolved.parent, seen)


def _expand_nvcc_options_files(
    argv: list[str], cwd: str | Path, seen: set[Path] | None = None,
) -> list[str]:
    """Expand nvcc options-file arguments into normal argv tokens."""
    if seen is None:
        seen = set()

    expanded: list[str] = []
    cwd_path = Path(cwd)
    i = 0
    while i < len(argv):
        arg = argv[i]

        rsp_value: str | None = None
        if arg in ("--options-file", "-optf") and i + 1 < len(argv):
            rsp_value = argv[i + 1]
            i += 2
        elif arg.startswith("--options-file="):
            rsp_value = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("-optf="):
            rsp_value = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("@") and len(arg) > 1:
            rsp_value = arg[1:]
            i += 1
        else:
            expanded.append(arg)
            i += 1
            continue

        rsp_path = Path(rsp_value)
        if not rsp_path.is_absolute():
            rsp_path = cwd_path / rsp_path

        expanded.extend(_read_rsp_tokens(rsp_path, seen))

    return expanded


def _extract_cuda_path_arg(argv: list[str]) -> str | None:
    """Extract a normalized --cuda-path argument from argv."""
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--cuda-path="):
            return arg
        if arg == "--cuda-path" and i + 1 < len(argv):
            return f"--cuda-path={argv[i + 1]}"
        i += 1
    return None


def _transform_clang_argv(
    argv: list[str], source_file: str, out_bc: Path,
) -> list[str]:
    """Transform captured clang argv for device-only LLVM BC emission."""
    compiler = argv[0]
    new_argv = [compiler]
    kept_cuda_arch = False

    i = 1
    while i < len(argv):
        arg = argv[i]

        # Skip output flag and its argument
        if arg == "-o":
            i += 2
            continue

        # Skip object file inputs (linked .o files)
        if _is_object_file(arg) and not arg.startswith("-"):
            i += 1
            continue

        # Drop linker-only flags
        if _should_drop_flag(arg):
            # If this is a prefix-style flag that takes next arg, skip it too
            if arg in ("-L", "-l", "-rpath", "--soname"):
                i += 2
            else:
                i += 1
            continue

        # Skip source files (we add ours explicitly)
        if _is_source_file(arg) and not arg.startswith("-"):
            i += 1
            continue

        # Device-only replay writes one output file, so keep at most one
        # CUDA arch and drop per-arch PTX packaging flags.
        if arg.startswith("--no-cuda-include-ptx="):
            i += 1
            continue
        if arg.startswith("--cuda-gpu-arch="):
            if kept_cuda_arch:
                i += 1
                continue
            kept_cuda_arch = True
            new_argv.append(arg)
            i += 1
            continue

        # Pass -isystem <path> with its argument
        if arg in ("-isystem", "-include", "-isysroot", "-target", "--target"):
            if i + 1 < len(argv):
                new_argv.append(arg)
                new_argv.append(argv[i + 1])
                i += 2
                continue

        # Keep the flag
        new_argv.append(arg)
        i += 1

    # Remove any existing -c (we add our own)
    new_argv = [a for a in new_argv if a != "-c"]

    # Remove any existing -emit-llvm or --cuda-device-only
    new_argv = [
        a for a in new_argv
        if a not in ("-emit-llvm", "--cuda-device-only",
                     "-S", "-E", "-fsyntax-only")
    ]

    # Add device-only LLVM BC flags
    new_argv.extend([
        "--cuda-device-only",
        "-emit-llvm",
        "-c",
        "-o", str(out_bc),
        source_file,
    ])

    return new_argv


def _map_nvcc_to_clang(
    argv: list[str], source_file: str, out_bc: Path, cwd: str,
) -> list[str]:
    """Map nvcc flags to a clang++ device-only command."""
    clang = shutil.which("clang++")
    if clang is None:
        clang = "clang++"

    argv = _expand_nvcc_options_files(argv, cwd)

    new_argv = [clang]
    captured_cuda_path = _extract_cuda_path_arg(argv)
    if captured_cuda_path is not None:
        new_argv.append(captured_cuda_path)
    else:
        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        new_argv.append(f"--cuda-path={cuda_path}")

    new_argv.extend([
        "-D__NV_NO_HOST_COMPILER_CHECK=1",
        "-Wno-unknown-cuda-version",
        "-Wno-unused-command-line-argument",
    ])
    cuda_version = _extract_cuda_version_macros(captured_cuda_path)
    if cuda_version is not None:
        new_argv.extend([
            f"-D__CUDACC_VER_MAJOR__={cuda_version[0]}",
            f"-D__CUDACC_VER_MINOR__={cuda_version[1]}",
        ])

    mapped_sms: list[str] = []

    def remember_sm(sm: str | None) -> None:
        if sm is None:
            return
        if sm not in mapped_sms:
            mapped_sms.append(sm)

    i = 1
    while i < len(argv):
        arg = argv[i]

        # Preserve normalized --cuda-path once, skip captured forms here.
        if arg == "--cuda-path" and i + 1 < len(argv):
            i += 2
            continue
        if arg.startswith("--cuda-path="):
            i += 1
            continue

        # Skip output flag
        if arg == "-o":
            i += 2
            continue

        # Skip source files
        if _is_source_file(arg) and not arg.startswith("-"):
            i += 1
            continue

        # Map include paths
        if arg == "-I" and i + 1 < len(argv):
            new_argv.extend(["-I", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("-I"):
            new_argv.append(arg)
            i += 1
            continue

        # Preserve flags that take a following argument.
        if arg in ("-isystem", "-include", "-isysroot", "-target", "--target"):
            if i + 1 < len(argv):
                new_argv.extend([arg, argv[i + 1]])
                i += 2
                continue
        if arg.startswith("--target=") or arg.startswith("-target="):
            new_argv.append(arg)
            i += 1
            continue

        # Map defines/undefines
        if arg == "-D" and i + 1 < len(argv):
            new_argv.extend(["-D", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("-D") or arg.startswith("-U"):
            new_argv.append(arg)
            i += 1
            continue

        # Map -std=
        if arg.startswith("-std=") or arg.startswith("--std="):
            new_argv.append(arg)
            i += 1
            continue

        # Map nvcc gencode forms -> --cuda-gpu-arch=sm_XX
        if arg in ("-gencode", "--generate-code") and i + 1 < len(argv):
            sm = _extract_sm_from_codegen_spec(argv[i + 1])
            remember_sm(sm)
            i += 2
            continue
        if arg.startswith("-gencode=") or arg.startswith("--generate-code="):
            sm = _extract_sm_from_codegen_spec(arg.split("=", 1)[1])
            remember_sm(sm)
            i += 1
            continue

        # Map -arch sm_XX -> --cuda-gpu-arch=sm_XX
        if arg == "-arch" and i + 1 < len(argv):
            remember_sm(_normalize_sm_token(argv[i + 1]))
            i += 2
            continue
        if arg.startswith("-arch="):
            remember_sm(_normalize_sm_token(arg.split("=", 1)[1]))
            i += 1
            continue

        # Map optimization flags
        if arg in ("-O0", "-O1", "-O2", "-O3", "-Os"):
            new_argv.append(arg)
            i += 1
            continue

        # Skip nvcc-specific flags
        if arg in ("-c", "-dc", "-dw", "--device-c", "--device-w",
                   "-rdc=true", "-rdc=false", "--relocatable-device-code=true",
                   "--relocatable-device-code=false"):
            i += 1
            continue

        # Skip -Xcompiler and its argument
        if arg in ("-Xcompiler", "-Xlinker", "-Xptxas"):
            i += 2
            continue

        # Skip linker flags
        if _should_drop_flag(arg):
            i += 1
            continue

        # Skip object inputs
        if _is_object_file(arg):
            i += 1
            continue

        # Skip unknown nvcc flags
        i += 1

    selected_sm = _pick_single_sm(mapped_sms)
    if selected_sm is None:
        selected_sm = _detect_local_sm()
    if selected_sm is not None:
        new_argv.append(f"--cuda-gpu-arch={selected_sm}")

    new_argv.extend([
        "--cuda-device-only",
        "-emit-llvm",
        "-c",
        "-o", str(out_bc),
        source_file,
    ])

    return new_argv


def build_device_bc(
    capture_entry: dict[str, Any], out_bc: Path,
) -> tuple[bool, str | None, list[str], str | None]:
    """Replay a captured compile to produce device LLVM bitcode.

    Args:
        capture_entry: dict with source_file, cwd, argv, compiler, record_id.
        out_bc: output path for the .bc file.

    Returns:
        (ok, failure_reason, command_used, failure_detail) where:
        - ok: True if the .bc was successfully produced.
        - failure_reason: None on success, or a string like
          'toolchain_mismatch', 'replay_failed', 'clang_not_found'.
        - command_used: the command list that was executed (or attempted).
        - failure_detail: stderr/exception context on failure, else None.
    """
    compiler = os.path.basename(capture_entry.get("compiler", ""))
    argv = capture_entry.get("argv", [])
    source_file = capture_entry.get("source_file", "")
    cwd = capture_entry.get("cwd", ".")

    out_bc = out_bc.resolve()
    out_bc.parent.mkdir(parents=True, exist_ok=True)

    if "clang" in compiler:
        cmd = _transform_clang_argv(argv, source_file, out_bc)
    elif "nvcc" in compiler:
        # Need clang++ available for nvcc mapping
        if shutil.which("clang++") is None:
            return (False, "clang_not_found", [], "clang++ not found in PATH")
        cmd = _map_nvcc_to_clang(argv, source_file, out_bc, cwd)
    else:
        return (False, "toolchain_mismatch", [], f"unsupported compiler: {compiler}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_BUILD_DEVICE_BC_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        return (False, "clang_not_found", cmd, "clang executable not found")
    except subprocess.TimeoutExpired:
        return (
            False,
            "replay_timeout",
            cmd,
            f"replay timed out after {_BUILD_DEVICE_BC_TIMEOUT_SEC}s",
        )
    except OSError as e:
        return (False, f"replay_error: {e}", cmd, str(e))

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return (False, "replay_failed", cmd, stderr[:4000] or None)

    if not out_bc.exists() or out_bc.stat().st_size == 0:
        return (False, "replay_failed", cmd, "replay produced empty or missing .bc output")

    return (True, None, cmd, None)


# ---------------------------------------------------------------------------
# Phase A: Preprocessing replay
# ---------------------------------------------------------------------------

def _transform_clang_preprocess(
    argv: list[str], source_file: str, out_ii: Path,
) -> list[str]:
    """Transform captured clang argv to a preprocessing-only command (-E)."""
    compiler = argv[0]
    new_argv = [compiler]

    i = 1
    while i < len(argv):
        arg = argv[i]

        # Skip output flag and its argument
        if arg == "-o":
            i += 2
            continue

        # Skip object file inputs
        if _is_object_file(arg) and not arg.startswith("-"):
            i += 1
            continue

        # Drop linker-only flags
        if _should_drop_flag(arg):
            if arg in ("-L", "-l", "-rpath", "--soname"):
                i += 2
            else:
                i += 1
            continue

        # Skip source files (we add ours explicitly)
        if _is_source_file(arg) and not arg.startswith("-"):
            i += 1
            continue

        # Pass flags that take a following argument
        if arg in ("-isystem", "-include", "-isysroot", "-target", "--target"):
            if i + 1 < len(argv):
                new_argv.append(arg)
                new_argv.append(argv[i + 1])
                i += 2
                continue

        # Keep preprocessing-relevant flags
        keep = False
        for prefix in _PREPROCESS_KEEP_PREFIXES:
            if arg.startswith(prefix):
                keep = True
                break
        if keep:
            new_argv.append(arg)
            i += 1
            continue

        # Drop compile/link/emit/device flags not relevant for preprocessing
        if arg in ("-c", "-emit-llvm", "--cuda-device-only",
                   "-S", "-fsyntax-only"):
            i += 1
            continue

        # Keep other flags (optimization, warning, etc.) for compat
        new_argv.append(arg)
        i += 1

    # Remove any leftover -c, -S, -emit-llvm, --cuda-device-only
    new_argv = [
        a for a in new_argv
        if a not in ("-c", "-S", "-emit-llvm", "--cuda-device-only",
                     "-fsyntax-only", "-E")
    ]

    new_argv.extend([
        "-E",
        "-o", str(out_ii),
        source_file,
    ])

    return new_argv


def _map_nvcc_preprocess(
    argv: list[str], source_file: str, out_ii: Path, cwd: str,
) -> list[str]:
    """Map nvcc capture to clang++ preprocessing command."""
    clang = shutil.which("clang++")
    if clang is None:
        clang = "clang++"

    argv = _expand_nvcc_options_files(argv, cwd)

    new_argv = [clang]
    captured_cuda_path = _extract_cuda_path_arg(argv)
    if captured_cuda_path is not None:
        new_argv.append(captured_cuda_path)
    else:
        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")
        new_argv.append(f"--cuda-path={cuda_path}")

    mapped_sms: list[str] = []

    def remember_sm(sm: str | None) -> None:
        if sm is None:
            return
        if sm not in mapped_sms:
            mapped_sms.append(sm)

    i = 1
    while i < len(argv):
        arg = argv[i]

        # Preserve normalized --cuda-path once, skip captured forms here.
        if arg == "--cuda-path" and i + 1 < len(argv):
            i += 2
            continue
        if arg.startswith("--cuda-path="):
            i += 1
            continue

        # Skip output flag
        if arg == "-o":
            i += 2
            continue

        # Skip source files
        if _is_source_file(arg) and not arg.startswith("-"):
            i += 1
            continue

        # Map include paths
        if arg == "-I" and i + 1 < len(argv):
            new_argv.extend(["-I", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("-I"):
            new_argv.append(arg)
            i += 1
            continue

        # Preserve flags that take a following argument.
        if arg in ("-isystem", "-include", "-isysroot", "-target", "--target"):
            if i + 1 < len(argv):
                new_argv.extend([arg, argv[i + 1]])
                i += 2
                continue
        if arg.startswith("--target=") or arg.startswith("-target="):
            new_argv.append(arg)
            i += 1
            continue

        # Map defines/undefines
        if arg == "-D" and i + 1 < len(argv):
            new_argv.extend(["-D", argv[i + 1]])
            i += 2
            continue
        if arg.startswith("-D") or arg.startswith("-U"):
            new_argv.append(arg)
            i += 1
            continue

        # Map -std=
        if arg.startswith("-std=") or arg.startswith("--std="):
            new_argv.append(arg)
            i += 1
            continue

        # Map nvcc gencode forms -> --cuda-gpu-arch=sm_XX
        if arg in ("-gencode", "--generate-code") and i + 1 < len(argv):
            sm = _extract_sm_from_codegen_spec(argv[i + 1])
            remember_sm(sm)
            i += 2
            continue
        if arg.startswith("-gencode=") or arg.startswith("--generate-code="):
            sm = _extract_sm_from_codegen_spec(arg.split("=", 1)[1])
            remember_sm(sm)
            i += 1
            continue

        # Map -arch sm_XX -> --cuda-gpu-arch=sm_XX
        if arg == "-arch" and i + 1 < len(argv):
            remember_sm(_normalize_sm_token(argv[i + 1]))
            i += 2
            continue
        if arg.startswith("-arch="):
            remember_sm(_normalize_sm_token(arg.split("=", 1)[1]))
            i += 1
            continue

        # Skip nvcc-specific flags
        if arg in ("-c", "-dc", "-dw", "--device-c", "--device-w",
                   "-rdc=true", "-rdc=false", "--relocatable-device-code=true",
                   "--relocatable-device-code=false"):
            i += 1
            continue

        # Skip -Xcompiler and its argument
        if arg in ("-Xcompiler", "-Xlinker", "-Xptxas"):
            i += 2
            continue

        # Skip linker flags
        if _should_drop_flag(arg):
            i += 1
            continue

        # Skip object inputs
        if _is_object_file(arg):
            i += 1
            continue

        # Skip unknown nvcc flags
        i += 1

    selected_sm = _pick_single_sm(mapped_sms)
    if selected_sm is not None:
        new_argv.append(f"--cuda-gpu-arch={selected_sm}")

    new_argv.extend([
        "-x", "cuda",
        "-E",
        "-o", str(out_ii),
        source_file,
    ])

    return new_argv


def build_preprocessed(
    capture_entry: dict[str, Any], out_ii: Path,
) -> tuple[bool, str | None, list[str], str | None]:
    """Replay a captured compile as a preprocessing-only command.

    Produces a preprocessed output file (.ii) with all macros and
    includes expanded, suitable for kernel discovery on the expanded
    source text.

    Args:
        capture_entry: dict with source_file, cwd, argv, compiler.
        out_ii: output path for the .ii file (absolute recommended).

    Returns:
        (ok, reason, cmd, detail) where:
        - ok: True if the .ii was successfully produced.
        - reason: None on success, or a stable code on failure.
        - cmd: the command list that was executed (or attempted).
        - detail: stderr / exception context on failure, else None.
    """
    compiler = os.path.basename(capture_entry.get("compiler", ""))
    argv = capture_entry.get("argv", [])
    source_file = capture_entry.get("source_file", "")
    cwd = capture_entry.get("cwd", ".")

    out_ii = out_ii.resolve()
    out_ii.parent.mkdir(parents=True, exist_ok=True)

    if "clang" in compiler:
        cmd = _transform_clang_preprocess(argv, source_file, out_ii)
    elif "nvcc" in compiler:
        if shutil.which("clang++") is None:
            return (False, "clang_not_found", [], "clang++ not found in PATH")
        cmd = _map_nvcc_preprocess(argv, source_file, out_ii, cwd)
    else:
        return (False, "toolchain_mismatch", [], f"unsupported compiler: {compiler}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_REPLAY_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        return (False, "clang_not_found", cmd, "clang executable not found")
    except subprocess.TimeoutExpired:
        return (
            False,
            "preprocess_timeout",
            cmd,
            f"preprocess timed out after {_REPLAY_TIMEOUT_SEC}s",
        )
    except OSError as e:
        return (False, "preprocess_error", cmd, str(e))

    if result.returncode != 0:
        return (False, "preprocess_failed", cmd, (result.stderr or "").strip()[:4000] or None)

    if not out_ii.exists() or out_ii.stat().st_size == 0:
        return (False, "preprocess_empty", cmd, "preprocess produced empty or missing .ii output")

    return (True, None, cmd, None)
