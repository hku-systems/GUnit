"""Shared feedback instrumentation support for CUDA backend builders."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_DIR = REPO_ROOT / "tools" / "rapid-feedback-instrument"
FeedbackInstrumentationMode = Literal["enabled", "disabled"]
BuildProfile = Literal["release", "debug"]


def kernel_timing_defines(enabled: bool) -> tuple[str, ...]:
    """Return the opt-in define shared by device and host timing code."""

    return ("-DENABLE_KERNEL_TIMING=1",) if enabled else ()


def canary_defines(enabled: bool) -> tuple[str, ...]:
    return (f"-DRAPID_ENABLE_CANARY={int(enabled)}",)


def add_canary_argument(parser) -> None:
    parser.add_argument(
        "--canary",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Enable the L1 payload canary (default: enabled).",
    )


@dataclass(frozen=True)
class BuildOptimizationContract:
    profile: BuildProfile
    device_clang_flags: tuple[str, ...]
    host_cxx_flags: tuple[str, ...]
    llvm_opt_pipeline: str | None
    llc_flags: tuple[str, ...]

    def as_metadata(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "device_clang_flags": list(self.device_clang_flags),
            "host_cxx_flags": list(self.host_cxx_flags),
            "llvm_opt_pipeline": self.llvm_opt_pipeline,
            "llc_flags": list(self.llc_flags),
        }


def build_optimization_contract(profile: str) -> BuildOptimizationContract:
    if profile == "release":
        return BuildOptimizationContract(
            profile="release",
            device_clang_flags=("-O3", "-DNDEBUG"),
            host_cxx_flags=("-O3", "-DNDEBUG"),
            llvm_opt_pipeline="default<O3>",
            llc_flags=("-O3",),
        )
    if profile == "debug":
        return BuildOptimizationContract(
            profile="debug",
            device_clang_flags=("-O0", "-g"),
            host_cxx_flags=("-O0", "-g"),
            llvm_opt_pipeline=None,
            llc_flags=("-O0",),
        )
    raise RuntimeError(f"unsupported build profile: {profile}")


def add_build_profile_argument(parser) -> None:
    parser.add_argument(
        "--build-profile",
        choices=("release", "debug"),
        default="release",
        help="Build with explicit O3 release flags or O0 debug flags.",
    )


def run_build_command(
    command: list[str],
    *,
    command_log: list[list[str]],
) -> None:
    recorded = [str(value) for value in command]
    command_log.append(recorded)
    subprocess.run(recorded, check=True, cwd=REPO_ROOT)


def add_optimization_build_metadata(
    result: dict,
    *,
    optimization: BuildOptimizationContract,
    optimized_device_bc: Path,
    command_log: list[list[str]],
) -> None:
    result["build_profile"] = optimization.profile
    result["optimization"] = optimization.as_metadata()
    result["optimized_device_bc"] = str(optimized_device_bc)
    result["selected_device_bc"] = str(optimized_device_bc)
    result["commands"] = command_log


def _require_llvm_tool(*names: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(f"required LLVM tool not found: {'/'.join(names)}")


def optimize_device_bitcode(
    *,
    input_bc: Path,
    output_bc: Path,
    optimization: BuildOptimizationContract,
    command_log: list[list[str]],
) -> Path:
    """Apply the selected module pipeline and enforce the release IR contract."""
    if not input_bc.is_file():
        raise RuntimeError(f"missing device bitcode for optimization: {input_bc}")
    output_bc.parent.mkdir(parents=True, exist_ok=True)
    if optimization.llvm_opt_pipeline is None:
        shutil.copy2(input_bc, output_bc)
    else:
        llvm_opt = _require_llvm_tool("opt-22", "opt")
        command = [
            llvm_opt,
            f"-passes={optimization.llvm_opt_pipeline}",
            str(input_bc),
            "-o",
            str(output_bc),
        ]
        command_log.append(command)
        subprocess.run(command, check=True, cwd=REPO_ROOT)

    if optimization.profile == "release":
        llvm_dis = _require_llvm_tool("llvm-dis-22", "llvm-dis")
        command = [llvm_dis, str(output_bc), "-o", "-"]
        command_log.append(command)
        result = subprocess.run(
            command,
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if "optnone" in result.stdout:
            raise RuntimeError(
                f"release device bitcode still contains optnone: {output_bc}"
            )
    return output_bc


@dataclass(frozen=True)
class FeedbackInstrumentationArtifacts:
    instrumented_bc: Path
    metadata: Path
    executable: Path


@dataclass(frozen=True)
class FeedbackDeviceBitcodeSelection:
    mode: FeedbackInstrumentationMode
    selected_bc: Path
    instrumented_bc: Path | None
    metadata: Path | None
    executable: Path | None


def validate_phase2_context_abi(build_spec_path: Path) -> dict:
    """Require the hidden RapidKernelContext ABI consumed by all backends."""
    build_spec = json.loads(build_spec_path.read_text(encoding="utf-8"))
    if (
        build_spec.get("entry_abi_version") != 1
        or build_spec.get("entry_context") != "rapid_kernel_context_v1"
    ):
        raise RuntimeError("backend build requires Phase2 entry ABI rapid_kernel_context_v1")
    if not isinstance(build_spec.get("feedback_payload_slots"), list):
        raise RuntimeError("backend build requires feedback payload slot metadata")
    return build_spec


def _feedback_executable(
    intermediates_dir: Path,
    *,
    command_log: list[list[str]] | None = None,
) -> Path:
    override = os.environ.get("RAPID_FEEDBACK_INSTRUMENT")
    if override:
        executable = Path(override).expanduser().resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"RAPID_FEEDBACK_INSTRUMENT is not executable: {executable}")
        return executable

    executable = intermediates_dir / "rapid-feedback-instrument"
    command = [
        sys.executable,
        str(TOOL_DIR / "build.py"),
        "--output",
        str(executable),
    ]
    if command_log is None:
        subprocess.run(command, check=True, cwd=REPO_ROOT)
    else:
        run_build_command(command, command_log=command_log)
    return executable


def instrument_feedback_module(
    *,
    linked_bc: Path,
    build_spec_path: Path,
    intermediates_dir: Path,
    instrument_selects: bool = False,
    command_log: list[list[str]] | None = None,
) -> FeedbackInstrumentationArtifacts:
    validate_phase2_context_abi(build_spec_path)
    if not linked_bc.is_file():
        raise RuntimeError(f"missing linked device bitcode: {linked_bc}")

    intermediates_dir.mkdir(parents=True, exist_ok=True)
    executable = _feedback_executable(intermediates_dir, command_log=command_log)
    instrumented_bc = intermediates_dir / "instrumented.bc"
    metadata = intermediates_dir / "feedback_metadata.json"
    command = [
        str(executable),
        "--input",
        str(linked_bc),
        "--output",
        str(instrumented_bc),
        "--build-spec",
        str(build_spec_path),
        "--metadata-out",
        str(metadata),
    ]
    if instrument_selects:
        command.append("--instrument-selects")
    if command_log is None:
        subprocess.run(command, check=True, cwd=REPO_ROOT)
    else:
        run_build_command(command, command_log=command_log)
    if not instrumented_bc.is_file() or not metadata.is_file():
        raise RuntimeError("feedback instrumentation did not produce required artifacts")
    return FeedbackInstrumentationArtifacts(
        instrumented_bc=instrumented_bc,
        metadata=metadata,
        executable=executable,
    )


def select_feedback_device_bitcode(
    *,
    linked_bc: Path,
    build_spec_path: Path,
    intermediates_dir: Path,
    mode: str,
    instrument_selects: bool = False,
    command_log: list[list[str]] | None = None,
) -> FeedbackDeviceBitcodeSelection:
    if mode not in ("enabled", "disabled"):
        raise RuntimeError(f"unsupported feedback instrumentation mode: {mode}")
    validate_phase2_context_abi(build_spec_path)
    if not linked_bc.is_file():
        raise RuntimeError(f"missing linked device bitcode: {linked_bc}")

    if mode == "disabled":
        if instrument_selects:
            raise RuntimeError(
                "select instrumentation requires feedback instrumentation"
            )
        return FeedbackDeviceBitcodeSelection(
            mode="disabled",
            selected_bc=linked_bc,
            instrumented_bc=None,
            metadata=None,
            executable=None,
        )

    feedback = instrument_feedback_module(
        linked_bc=linked_bc,
        build_spec_path=build_spec_path,
        intermediates_dir=intermediates_dir,
        instrument_selects=instrument_selects,
        command_log=command_log,
    )
    return FeedbackDeviceBitcodeSelection(
        mode="enabled",
        selected_bc=feedback.instrumented_bc,
        instrumented_bc=feedback.instrumented_bc,
        metadata=feedback.metadata,
        executable=feedback.executable,
    )


def add_feedback_instrumentation_argument(parser) -> None:
    parser.add_argument(
        "--feedback-instrumentation",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Run LLVM feedback instrumentation before PTX lowering, or lower linked bitcode directly.",
    )
    parser.add_argument(
        "--instrument-selects",
        action="store_true",
        help="Instrument evaluated LLVM select instructions as pseudo-sites.",
    )


def add_feedback_build_metadata(result: dict, feedback: FeedbackDeviceBitcodeSelection) -> None:
    result["feedback_instrumentation"] = feedback.mode
    result["selected_device_bc"] = str(feedback.selected_bc)
    if feedback.instrumented_bc is not None:
        result["instrumented_bc"] = str(feedback.instrumented_bc)
    if feedback.metadata is not None:
        result["feedback_metadata"] = str(feedback.metadata)
