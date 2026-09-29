"""Process and resource lifecycle helpers for RQ4 profile cells."""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path


MUTATING_BROKER_ADDRESS = ("127.0.0.1", 1337)
PROCESS_CLEANUP_TIMEOUT = 5.0
LOCK_HOLDER_TERM_TIMEOUT = 5.0
LOCK_HOLDER_POLL_INTERVAL = 0.1
LOCK_HOLDER_RESCAN_INTERVAL = 5.0
LOCK_HOLDER_TOTAL_TIMEOUT = 60.0
LOGGER = logging.getLogger("benchmark.rq4.profile")


def run_profile_command(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as error:
        _signal_process_group(process.pid, signal.SIGKILL)
        try:
            stdout, stderr = process.communicate()
        finally:
            process.wait()
        if isinstance(error, subprocess.TimeoutExpired):
            error.stdout = stdout
            error.stderr = stderr
        raise
    return subprocess.CompletedProcess(
        process.args, process.returncode, stdout=stdout, stderr=stderr
    )


def _signal_process_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    _signal_process_group(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=PROCESS_CLEANUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        process.wait(timeout=PROCESS_CLEANUP_TIMEOUT)


def mutating_broker_port_is_available() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(MUTATING_BROKER_ADDRESS)
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return False
            raise RuntimeError(
                f"failed to probe mutating broker port "
                f"{MUTATING_BROKER_ADDRESS[1]}: {error}"
            ) from error
    return True


def _wait_for_mutating_broker_port(*, available: bool, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mutating_broker_port_is_available() == available:
            return True
        time.sleep(0.05)
    return False


def gpu_client_lock_path(gpu_device: str) -> Path:
    identity = gpu_device.split(",")[0].strip() or "default"
    encoded = "".join(f"{byte:02x}" for byte in identity.encode())
    return Path(tempfile.gettempdir()) / f"rapid-fuzzer-async-gpu-{encoded}.lock"


def gpu_client_lock_is_available(gpu_device: str) -> bool:
    path = gpu_client_lock_path(gpu_device)
    if not path.exists():
        return True
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def find_cell_residual_pids(pgid: int) -> list[int]:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return []
    return [
        pid
        for pid, _ppid, process_group in _process_table()
        if process_group == pgid
    ]


def _process_table() -> list[tuple[int, int, int]]:
    """Read (pid, parent pid, process group) relationships from procfs."""
    processes: list[tuple[int, int, int]] = []
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                fields = stat.split()
                processes.append((int(fields[0]), int(fields[3]), int(fields[4])))
            except (FileNotFoundError, IndexError, ValueError, PermissionError):
                continue
    except OSError:
        pass
    return processes


class CellResidualError(Exception):
    def __init__(self, *, port_busy: bool, lock_busy: bool, residual_pids: list[int]) -> None:
        parts: list[str] = []
        if port_busy:
            parts.append(f"port {MUTATING_BROKER_ADDRESS[1]} still occupied")
        if lock_busy:
            parts.append("GPU client lock still held")
        if residual_pids:
            parts.append(f"residual PIDs: {residual_pids}")
        super().__init__(f"cell cleanup verification failed: {'; '.join(parts)}")
        self.port_busy = port_busy
        self.lock_busy = lock_busy
        self.residual_pids = residual_pids


def find_lock_fd_holders(gpu_device: str) -> list[int]:
    """Scan procfs for all processes holding an fd to the lock file inode."""
    path = gpu_client_lock_path(gpu_device)
    if not path.exists():
        return []
    try:
        lock_stat = os.stat(str(path))
    except OSError:
        return []
    target_dev = lock_stat.st_dev
    target_ino = lock_stat.st_ino
    my_pid = os.getpid()
    holders: list[int] = []
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == my_pid:
                continue
            fd_dir = entry / "fd"
            try:
                for fd_link in fd_dir.iterdir():
                    try:
                        stat = os.stat(str(fd_link))
                        if stat.st_dev == target_dev and stat.st_ino == target_ino:
                            holders.append(pid)
                            break
                    except (OSError, ValueError):
                        continue
            except (OSError, PermissionError):
                continue
    except OSError:
        pass
    return holders


def _proc_info(pid: int) -> dict[str, str]:
    """Read comm and cmdline for a PID from procfs."""
    info: dict[str, str] = {"pid": str(pid)}
    try:
        info["comm"] = Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        info["comm"] = "<gone>"
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        info["cmdline"] = (
            raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        )
    except OSError:
        info["cmdline"] = "<gone>"
    return info


def _collect_process_tree(root_pids: list[int]) -> list[int]:
    """Collect all descendants of root_pids via procfs ppid traversal."""
    my_pid = os.getpid()
    all_pids: set[int] = set(root_pids)
    parent_map = {
        pid: ppid for pid, ppid, _pgid in _process_table() if pid != my_pid
    }
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_map.items():
            if pid not in all_pids and ppid in all_pids:
                all_pids.add(pid)
                changed = True
    return list(all_pids)


def _signal_processes(pids: list[int], sig: signal.Signals) -> int:
    signaled = 0
    for pid in pids:
        try:
            os.kill(pid, sig)
            signaled += 1
        except (ProcessLookupError, PermissionError):
            pass
    return signaled


def _signal_lock_holders(gpu_device: str, sig: signal.Signals) -> int:
    holders = find_lock_fd_holders(gpu_device)
    tree = _collect_process_tree(holders) if holders else []
    return _signal_processes(tree, sig)


def _kill_all_lock_holders(
    gpu_device: str, *, deadline: float, term_timeout: float
) -> None:
    """Terminate all lock holders and their process trees with TERM then KILL."""
    holders = find_lock_fd_holders(gpu_device)
    if holders:
        tree_pids = _collect_process_tree(holders)
        holder_details = [_proc_info(pid) for pid in tree_pids]
        LOGGER.warning(
            "GPU lock holders + process tree (%d pids): %s",
            len(tree_pids),
            "; ".join(
                f"pid={item['pid']} comm={item['comm']} cmd={item['cmdline'][:120]}"
                for item in holder_details
            ),
        )
        _signal_processes(tree_pids, signal.SIGTERM)

    term_deadline = min(deadline, time.monotonic() + term_timeout)
    while time.monotonic() < term_deadline:
        if gpu_client_lock_is_available(gpu_device):
            return
        remaining = max(0.0, term_deadline - time.monotonic())
        time.sleep(min(LOCK_HOLDER_POLL_INTERVAL, remaining))

    _signal_lock_holders(gpu_device, signal.SIGKILL)
    next_rescan = time.monotonic() + LOCK_HOLDER_RESCAN_INTERVAL
    while time.monotonic() < deadline:
        if gpu_client_lock_is_available(gpu_device):
            return
        now = time.monotonic()
        if now >= next_rescan:
            killed = _signal_lock_holders(gpu_device, signal.SIGKILL)
            if killed:
                LOGGER.info(
                    "lock rescan: re-KILLed %d pids, waiting for driver teardown",
                    killed,
                )
            next_rescan = now + LOCK_HOLDER_RESCAN_INTERVAL
        time.sleep(min(LOCK_HOLDER_POLL_INTERVAL, max(0.0, deadline - now)))

    still_holding = find_lock_fd_holders(gpu_device)
    if still_holding:
        details = [_proc_info(pid) for pid in still_holding]
        LOGGER.error(
            "GPU lock STILL held after KILL escalation: %s",
            "; ".join(
                f"pid={item['pid']} comm={item['comm']} cmd={item['cmdline'][:120]}"
                for item in details
            ),
        )


def _cleanup_cell_process_group(pgid: int) -> None:
    _signal_process_group(pgid, signal.SIGKILL)
    deadline = time.monotonic() + PROCESS_CLEANUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            break
        if not find_cell_residual_pids(pgid):
            break
        time.sleep(0.05)


def verify_cell_boundary(
    *,
    gpu_device: str,
    pgid: int | None = None,
    lock_holder_total_timeout: float = LOCK_HOLDER_TOTAL_TIMEOUT,
    lock_holder_term_timeout: float = LOCK_HOLDER_TERM_TIMEOUT,
) -> str | None:
    """Terminate lock holders, wait for release, then verify the cell boundary."""
    if not gpu_client_lock_is_available(gpu_device):
        deadline = time.monotonic() + lock_holder_total_timeout
        _kill_all_lock_holders(
            gpu_device,
            deadline=deadline,
            term_timeout=lock_holder_term_timeout,
        )

    residuals = find_cell_residual_pids(pgid) if pgid is not None else []
    port_busy = not mutating_broker_port_is_available()
    lock_busy = not gpu_client_lock_is_available(gpu_device)
    if port_busy or residuals:
        raise CellResidualError(
            port_busy=port_busy,
            lock_busy=lock_busy,
            residual_pids=residuals,
        )
    if lock_busy:
        holders = find_lock_fd_holders(gpu_device)
        details = [_proc_info(pid) for pid in holders]
        LOGGER.warning(
            "lock_residual: GPU lock still held after %.0fs timeout; "
            "downgrading to warning (next cell preflight will re-attempt). "
            "Remaining holders: %s",
            lock_holder_total_timeout,
            "; ".join(
                f"pid={item['pid']} comm={item['comm']} cmd={item['cmdline'][:120]}"
                for item in details
            )
            or "<holders vanished but lock still busy>",
        )
        return "lock_residual"
    return None


def teardown_cell(
    pgid: int | None,
    *,
    gpu_device: str,
    lock_holder_total_timeout: float = LOCK_HOLDER_TOTAL_TIMEOUT,
    lock_holder_term_timeout: float = LOCK_HOLDER_TERM_TIMEOUT,
) -> str | None:
    """Teardown a cell, including respawners outside its process group."""
    if pgid is not None:
        _cleanup_cell_process_group(pgid)
    return verify_cell_boundary(
        gpu_device=gpu_device,
        pgid=pgid,
        lock_holder_total_timeout=lock_holder_total_timeout,
        lock_holder_term_timeout=lock_holder_term_timeout,
    )


class CrashLoopTimeout(Exception):
    """Raised when the nsys client exceeds its budget, indicating a crash loop."""

    def __init__(
        self, stdout: str, stderr: str, *, elapsed: float, budget: float
    ) -> None:
        super().__init__(
            f"mutating client exceeded budget ({elapsed:.1f}s > {budget:.1f}s)"
        )
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed
        self.budget = budget


def mutating_cell_timeout(mutate_seconds: int, *, base_timeout: int) -> float:
    return float(max(base_timeout, mutate_seconds * 3 + 60))


def run_mutating_profile(
    command: list[str],
    *,
    broker_command: list[str],
    cwd: Path,
    env: dict[str, str],
    broker_log: Path,
    timeout: float,
    budget_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run a mutating profile cell and return its result and broker PGID."""
    broker_port = MUTATING_BROKER_ADDRESS[1]
    if not mutating_broker_port_is_available():
        raise RuntimeError(
            f"mutating broker port {broker_port} is already in use; "
            "stop the stale broker before profiling"
        )
    broker_env = env.copy()
    broker_env.pop("RAPID_PROFILE", None)
    broker_env.pop("RAPID_PROFILE_OUTPUT", None)
    with broker_log.open("w", encoding="utf-8") as broker_error:
        broker = subprocess.Popen(
            broker_command,
            cwd=cwd,
            env=broker_env,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=broker_error,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + min(timeout, 30.0)
            while time.monotonic() < deadline:
                if broker.poll() is not None:
                    raise RuntimeError(
                        f"mutating broker exited before client start ({broker.returncode})"
                    )
                if not mutating_broker_port_is_available():
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("mutating broker did not become ready")
            started_at = time.monotonic()
            try:
                result = run_profile_command(
                    command, cwd=cwd, env=env, timeout=timeout
                )
                return result, broker.pid
            except subprocess.TimeoutExpired as error:
                elapsed = time.monotonic() - started_at
                if elapsed >= budget_seconds:
                    raise CrashLoopTimeout(
                        stdout=error.stdout or "",
                        stderr=error.stderr or "",
                        elapsed=elapsed,
                        budget=budget_seconds,
                    ) from error
                raise
        finally:
            _terminate_process_group(broker)
            if not _wait_for_mutating_broker_port(
                available=True, timeout=PROCESS_CLEANUP_TIMEOUT
            ):
                raise RuntimeError(
                    f"mutating broker port {broker_port} was not released after shutdown"
                )
