"""
GPU introspection and budget enforcement.

Deliberately shells out to nvidia-smi rather than importing torch. This module
is imported by the orchestrator, which must NOT initialise CUDA - the whole
point of the subprocess design is that the parent process never touches the
GPU and therefore never holds a context.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from .logging import get_logger

log = get_logger("gpu")


@dataclass
class GPUState:
    name: str
    total_mb: int
    used_mb: int
    free_mb: int

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.used_mb} MB used / {self.total_mb} MB total "
            f"({self.free_mb} MB free)"
        )


def query() -> Optional[GPUState]:
    """Current VRAM state, or None if nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("nvidia-smi failed: %s", exc)
        return None

    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        return GPUState(
            name=parts[0],
            total_mb=int(float(parts[1])),
            used_mb=int(float(parts[2])),
            free_mb=int(float(parts[3])),
        )
    except ValueError:
        return None


def assert_headroom(required_mb: int, stage: str, hard: bool = False) -> None:
    """
    Verify enough VRAM is free before a stage allocates.

    Default is a warning, not a failure: on Windows the driver will spill to
    shared system memory rather than raise, so the failure mode we are
    guarding against is a silent 50x slowdown. Set hard=True to abort instead.
    """
    state = query()
    if state is None:
        log.debug("no NVIDIA GPU visible; skipping headroom check for %s", stage)
        return

    if state.free_mb < required_mb:
        msg = (
            f"Stage {stage!r} wants ~{required_mb} MB but only "
            f"{state.free_mb} MB is free ({state}). Something is still "
            f"resident - check `ollama ps` and confirm the previous stage "
            f"process actually exited."
        )
        if hard:
            raise RuntimeError(msg)
        log.warning("%s", msg)
    else:
        log.info(
            "%s: %d MB free, need ~%d MB - ok", stage, state.free_mb, required_mb
        )


def log_state(label: str) -> None:
    state = query()
    if state:
        log.info("VRAM @ %s -> %s", label, state)
