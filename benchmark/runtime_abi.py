"""Shared ctypes mirrors for the CUDA backend runtime ABI."""

from __future__ import annotations

import ctypes


class RunStatus(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("stage", ctypes.c_uint16),
        ("detail", ctypes.c_uint32),
    ]


class TaskResult(ctypes.Structure):
    _fields_ = [
        ("task_id", ctypes.c_uint64),
        ("input_ptr", ctypes.c_size_t),
        ("edge_ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("simt_memcov_ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("edge_size", ctypes.c_uint32),
        ("simt_memcov_size", ctypes.c_uint32),
        ("status", RunStatus),
        ("exec_time_ns", ctypes.c_uint64),
    ]


def stop_library(target: ctypes.CDLL) -> None:
    """Stop a backend that exports the optional persistent-worker hook."""
    try:
        stop = target.libafl_stop
    except AttributeError:
        return
    stop.argtypes = []
    stop.restype = None
    stop()
