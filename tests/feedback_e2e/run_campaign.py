from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

if __package__:
    from .feedback_runtime import FeedbackArtifact, discover_feedback_artifact
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from feedback_runtime import FeedbackArtifact, discover_feedback_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("origin", "rapid", "rapid2")


@dataclass(frozen=True)
class CampaignResult:
    backend: str
    command: list[str]
    cuda_visible_devices: str
    broker_returncode: int | None
    client_returncode: int | None
    client_joined: bool
    executions: int
    cfg_sites: int
    simt_memcov_bits: int
    logical_thread_bits: int
    broker_log: str
    client_log: str
    timed_out: bool
    rapid_submitted: int = 0
    rapid_completed: int = 0
    rapid_pending: int = 0
    rapid_failed: int = 0


def build_fuzzer_command(
    *,
    backend: str,
    fuzzer_root: Path,
    library: Path,
    manifest: Path,
    runs: int,
) -> list[str]:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if runs <= 0:
        raise ValueError(f"runs must be positive, got {runs}")
    binary = fuzzer_root / ("fuzzer_async" if backend == "rapid2" else "fuzzer")
    return [
        str(binary),
        str(library),
        "--manifest",
        str(manifest),
        "--runs",
        str(runs),
    ]


def backend_environment(backend: str, gpu_devices: list[str]) -> dict[str, str]:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if not gpu_devices:
        raise ValueError("at least one CUDA device must be supplied")
    environment = os.environ.copy()
    backend_index = BACKENDS.index(backend)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_devices[backend_index % len(gpu_devices)]
    environment.setdefault("RUST_LOG", "info")
    return environment


def backend_library(artifact: FeedbackArtifact, backend: str) -> Path:
    return {
        "origin": artifact.origin_library,
        "rapid": artifact.rapid_library,
        "rapid2": artifact.rapid2_library,
    }[backend]


def _read_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _max_counter(logs: Iterable[str], name: str) -> int:
    values: list[int] = []
    pattern = re.compile(rf"\b{re.escape(name)}\s*[:=]\s*(\d+)")
    for log in logs:
        values.extend(int(value) for value in pattern.findall(log))
    return max(values, default=0)


def _parse_rapid_counters(logs: Iterable[str]) -> dict[str, int]:
    names = (
        "rapid_submitted",
        "rapid_completed",
        "rapid_pending",
        "rapid_failed",
    )
    snapshots = [
        {name: _max_counter((line,), name) for name in names}
        for log in logs
        for line in log.splitlines()
        if "rapid_submitted" in line
    ]
    if not snapshots:
        return {name: 0 for name in names}
    return max(
        snapshots,
        key=lambda snapshot: (
            snapshot["rapid_submitted"],
            snapshot["rapid_completed"],
        ),
    )


def throughput_execution_count(result: CampaignResult) -> int:
    if result.backend == "rapid":
        return result.rapid_completed
    return result.executions


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_backend_campaign(
    *,
    backend: str,
    artifact: FeedbackArtifact,
    fuzzer_root: Path,
    root: Path,
    runs: int,
    observe_seconds: float,
    gpu_devices: list[str],
) -> CampaignResult:
    if observe_seconds <= 0:
        raise ValueError(f"observe_seconds must be positive, got {observe_seconds}")
    workdir = root / backend
    corpus_dir = workdir / "corpus"
    crashes_dir = workdir / "crashes"
    workdir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(exist_ok=True)
    crashes_dir.mkdir(exist_ok=True)

    command = build_fuzzer_command(
        backend=backend,
        fuzzer_root=fuzzer_root,
        library=backend_library(artifact, backend),
        manifest=artifact.manifest,
        runs=runs,
    )
    if "--no-mutate" in command:
        raise AssertionError("full campaign command must enable mutation")
    environment = backend_environment(backend, gpu_devices)
    broker_log = workdir / "broker.log"
    client_log = workdir / "client.log"
    timed_out = False

    with broker_log.open("w", encoding="utf-8") as broker_output, client_log.open(
        "w", encoding="utf-8"
    ) as client_output:
        broker = subprocess.Popen(
            command,
            cwd=workdir,
            env=environment,
            stdout=broker_output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        client: subprocess.Popen[str] | None = None
        try:
            time.sleep(1.0)
            if broker.poll() is not None:
                raise RuntimeError(
                    f"{backend} broker exited before client startup: {_read_log(broker_log)}"
                )
            client = subprocess.Popen(
                command,
                cwd=workdir,
                env=environment,
                stdout=client_output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + observe_seconds
            while time.monotonic() < deadline:
                if broker.poll() is not None and client.poll() is not None:
                    break
                time.sleep(0.2)
            timed_out = time.monotonic() >= deadline
        finally:
            if client is not None:
                _terminate_process_group(client)
            _terminate_process_group(broker)

    broker_text = _read_log(broker_log)
    client_text = _read_log(client_log)
    logs = (broker_text, client_text)
    rapid_counters = _parse_rapid_counters(logs)
    return CampaignResult(
        backend=backend,
        command=command,
        cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"],
        broker_returncode=broker.returncode,
        client_returncode=client.returncode if client is not None else None,
        client_joined=bool(re.search(r"clients:\s*[1-9]", broker_text)),
        executions=_max_counter(logs, "executions"),
        cfg_sites=_max_counter(logs, "cfg_sites"),
        simt_memcov_bits=_max_counter(logs, "simt_memcov_bits"),
        logical_thread_bits=_max_counter(logs, "logical_thread_bits"),
        broker_log=str(broker_log),
        client_log=str(client_log),
        timed_out=timed_out,
        **rapid_counters,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--kernel-id")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--fuzzer-root", type=Path, default=REPO_ROOT / "cuda-fuzzer/target/release")
    parser.add_argument("--runs", type=int, default=100_000)
    parser.add_argument("--observe-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-devices", default="0,1,2")
    args = parser.parse_args()

    requested_backends = tuple(filter(None, args.backends.split(",")))
    if not requested_backends or any(backend not in BACKENDS for backend in requested_backends):
        parser.error(f"--backends must be a subset of {','.join(BACKENDS)}")
    gpu_devices = tuple(filter(None, args.gpu_devices.split(",")))
    if not gpu_devices:
        parser.error("--gpu-devices must contain at least one device")

    artifact = discover_feedback_artifact(args.run_dir, kernel_id=args.kernel_id)
    root = args.out_dir or args.run_dir / "feedback_campaigns"
    root.mkdir(parents=True, exist_ok=True)
    results = [
        run_backend_campaign(
            backend=backend,
            artifact=artifact,
            fuzzer_root=args.fuzzer_root,
            root=root,
            runs=args.runs,
            observe_seconds=args.observe_seconds,
            gpu_devices=list(gpu_devices),
        )
        for backend in requested_backends
    ]
    report = {"run_dir": str(args.run_dir), "results": [asdict(result) for result in results]}
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
