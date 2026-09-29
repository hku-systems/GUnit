from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .feedback_runtime import FeedbackArtifact, discover_feedback_artifact
    from .run_campaign import CampaignResult, run_backend_campaign
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from feedback_runtime import FeedbackArtifact, discover_feedback_artifact
    from run_campaign import CampaignResult, run_backend_campaign


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    executions: int
    observed_seconds: float
    executions_per_second: float
    client_joined: bool
    library: str | None
    command: list[str]
    broker_log: str
    client_log: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_case_from_campaign(
    label: str,
    result: CampaignResult,
    *,
    observed_seconds: float,
    library: str | None = None,
) -> BenchmarkCase:
    if result.backend != "rapid2":
        raise RuntimeError(f"{label} benchmark must use rapid2, got {result.backend}")
    if observed_seconds <= 0:
        raise RuntimeError(f"{label} observed_seconds must be positive")
    if not result.client_joined:
        raise RuntimeError(f"{label} client did not join")
    if result.executions <= 0:
        raise RuntimeError(f"{label} requires positive executions")

    return BenchmarkCase(
        label=label,
        executions=result.executions,
        observed_seconds=observed_seconds,
        executions_per_second=result.executions / observed_seconds,
        client_joined=result.client_joined,
        library=library,
        command=result.command,
        broker_log=result.broker_log,
        client_log=result.client_log,
    )


def validate_build_identity(
    instrumented_build: dict[str, Any],
    uninstrumented_build: dict[str, Any],
) -> dict[str, str]:
    if instrumented_build.get("feedback_instrumentation") != "enabled":
        raise RuntimeError("instrumented build must record feedback_instrumentation=enabled")
    if uninstrumented_build.get("feedback_instrumentation") != "disabled":
        raise RuntimeError("uninstrumented build must record feedback_instrumentation=disabled")

    for key in ("kernel_id", "display_name", "phase2_dir"):
        if instrumented_build.get(key) != uninstrumented_build.get(key):
            raise RuntimeError(f"RAPID2 A/B benchmark must use the same {key}")

    return {
        "kernel_id": str(instrumented_build["kernel_id"]),
        "display_name": str(instrumented_build.get("display_name", "")),
        "phase2_dir": str(instrumented_build["phase2_dir"]),
        "instrumented_library": str(instrumented_build.get("shared_lib", "")),
        "uninstrumented_library": str(uninstrumented_build.get("shared_lib", "")),
    }


def build_benchmark_report(
    *,
    instrumented_build: dict[str, Any],
    uninstrumented_build: dict[str, Any],
    instrumented: CampaignResult,
    uninstrumented: CampaignResult,
    observed_seconds: float,
) -> dict[str, Any]:
    identity = validate_build_identity(instrumented_build, uninstrumented_build)
    instrumented_case = benchmark_case_from_campaign(
        "instrumented",
        instrumented,
        observed_seconds=observed_seconds,
        library=identity["instrumented_library"],
    )
    uninstrumented_case = benchmark_case_from_campaign(
        "uninstrumented",
        uninstrumented,
        observed_seconds=observed_seconds,
        library=identity["uninstrumented_library"],
    )
    slowdown = 1.0 - (
        instrumented_case.executions_per_second
        / uninstrumented_case.executions_per_second
    )
    return {
        **identity,
        "observed_seconds": observed_seconds,
        "instrumented": asdict(instrumented_case),
        "uninstrumented": asdict(uninstrumented_case),
        "slowdown": slowdown,
        "slowdown_percent": slowdown * 100.0,
    }


def _rapid2_artifact(base_artifact: FeedbackArtifact, rapid2_library: Path) -> FeedbackArtifact:
    return FeedbackArtifact(
        kernel_dir=base_artifact.kernel_dir,
        manifest=base_artifact.manifest,
        origin_library=base_artifact.origin_library,
        rapid_library=base_artifact.rapid_library,
        rapid2_library=rapid2_library,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a same-kernel RAPID2 instrumentation A/B benchmark."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--kernel-id")
    parser.add_argument("--instrumented-build", required=True, type=Path)
    parser.add_argument("--uninstrumented-build", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--fuzzer-root", type=Path, default=REPO_ROOT / "cuda-fuzzer/target/release")
    parser.add_argument("--runs", type=int, default=100_000)
    parser.add_argument("--observe-seconds", type=float, default=20.0)
    parser.add_argument("--gpu-device", default="0")
    args = parser.parse_args()

    instrumented_build = _read_json(args.instrumented_build)
    uninstrumented_build = _read_json(args.uninstrumented_build)
    identity = validate_build_identity(instrumented_build, uninstrumented_build)

    base_artifact = discover_feedback_artifact(args.run_dir, kernel_id=args.kernel_id)
    root = args.out_dir or args.run_dir / "rapid2_benchmark"
    root.mkdir(parents=True, exist_ok=True)

    instrumented_campaign = run_backend_campaign(
        backend="rapid2",
        artifact=_rapid2_artifact(base_artifact, Path(identity["instrumented_library"])),
        fuzzer_root=args.fuzzer_root,
        root=root / "instrumented",
        runs=args.runs,
        observe_seconds=args.observe_seconds,
        gpu_devices=[args.gpu_device],
    )
    uninstrumented_campaign = run_backend_campaign(
        backend="rapid2",
        artifact=_rapid2_artifact(base_artifact, Path(identity["uninstrumented_library"])),
        fuzzer_root=args.fuzzer_root,
        root=root / "uninstrumented",
        runs=args.runs,
        observe_seconds=args.observe_seconds,
        gpu_devices=[args.gpu_device],
    )
    report = build_benchmark_report(
        instrumented_build=instrumented_build,
        uninstrumented_build=uninstrumented_build,
        instrumented=instrumented_campaign,
        uninstrumented=uninstrumented_campaign,
        observed_seconds=args.observe_seconds,
    )
    report_path = root / "report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
