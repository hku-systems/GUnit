"""Configuration matrices shared by release diagnostics and RQ4 profiling."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_WINDOW_SIZE = 2


@dataclass(frozen=True)
class Configuration:
    name: str
    label: str
    artifact_backend: str
    async_frontend: bool
    window_size: int | None
    feedback_enabled: bool


def configurations(window_size: int) -> tuple[Configuration, ...]:
    if not 1 <= window_size <= 32:
        raise ValueError("window_size must be in 1..=32")
    return (
        Configuration(
            "cufuzz", "CuFuzz-style", "cufuzz", False, None, feedback_enabled=False
        ),
        Configuration(
            "libafl",
            "LibAFL",
            "origin-no-feedback",
            False,
            None,
            feedback_enabled=False,
        ),
        Configuration(
            "libafl-plus", "LibAFL+", "origin", False, None, feedback_enabled=True
        ),
        Configuration(
            "gunit-sync",
            "GUnit-s",
            "rapid",
            False,
            window_size,
            feedback_enabled=True,
        ),
        Configuration(
            "gunit", "GUnit", "rapid2", True, window_size, feedback_enabled=True
        ),
    )
