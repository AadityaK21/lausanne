"""
GPU introspection and budget enforcement.

Deliberately shells out to nvidia-smi rather than importing torch. This module
is imported by the orchestrator, which must NOT initialise CUDA - the whole
point of the subprocess design is that the parent process never touches the
GPU and therefore never holds a context.
"""

from __future__ import annotations

import os
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


# --------------------------------------------------------------------------
# Windows CUDA DLL discovery
# --------------------------------------------------------------------------

# The single most common Windows failure in this project, and it has nothing
# to do with the GPU being wrong. CTranslate2 (faster-whisper's backend) links
# cuDNN 9 and cuBLAS at load time and resolves them through the OS loader. The
# CUDA-enabled PyTorch wheel already ships those exact DLLs - but on Windows it
# bundles them inside `torch\lib`, which is not on PATH and which the loader
# therefore never searches. The card is fine, the driver is fine, the DLLs are
# on disk, and the import fails.
#
# Worth being precise about why this is Windows-specific, because the usual
# advice found online is written for Linux and cannot work here. On Linux,
# `pip install torch --index-url .../cu126` pulls the CUDA runtime in as
# separate distributions - nvidia-cudnn-cu12 and friends - so `import
# nvidia.cudnn` resolves and its `lib` directory is the answer. The Windows
# wheel has no such dependency: there is no `nvidia` package to import, only
# `torch\lib`. An instruction to import it can only ever raise
# ModuleNotFoundError on a correctly installed Windows environment.
#
# Rather than asking the user to edit PATH, the directories are registered
# with os.add_dll_directory at stage start. Python 3.8+ deliberately stopped
# honouring PATH for extension-module dependencies on Windows; this is the
# supported mechanism, it affects only this process, and it leaves the user's
# environment untouched.

_CUDA_DLL_DIRS_ADDED: list = []


def _candidate_dll_dirs():
    """
    Directories that may hold cuDNN/cuBLAS, located WITHOUT importing torch.

    `find_spec` on a top-level package locates it through the path finders and
    returns a spec without executing the module, so this costs nothing and -
    critically - does not initialise a CUDA context in a process whose entire
    purpose is to not hold one.
    """
    import importlib.util
    import pathlib

    dirs = []

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        lib = pathlib.Path(spec.origin).parent / "lib"
        if lib.is_dir():
            dirs.append(lib)

    # Also cover an environment that has the standalone runtime packages,
    # whether because it is Linux-shaped or because they were installed by
    # hand. Harmless when absent.
    try:
        nv = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        nv = None
    if nv is not None:
        for loc in (nv.submodule_search_locations or []):
            base = pathlib.Path(loc)
            for sub in ("cudnn", "cublas"):
                for leaf in ("bin", "lib"):
                    d = base / sub / leaf
                    if d.is_dir():
                        dirs.append(d)

    return dirs


def add_cuda_dll_dirs() -> list:
    """
    Make torch's bundled CUDA DLLs loadable by CTranslate2. Windows only.

    Returns the directories registered, for logging. Idempotent, and a no-op
    everywhere except Windows - on Linux the loader finds these through
    RPATH and needs no help.
    """
    import sys

    if sys.platform != "win32":
        return []
    if _CUDA_DLL_DIRS_ADDED:
        return list(_CUDA_DLL_DIRS_ADDED)

    added = []
    for d in _candidate_dll_dirs():
        try:
            os.add_dll_directory(str(d))
        except (OSError, AttributeError) as exc:
            log.debug("could not add DLL directory %s: %s", d, exc)
            continue
        added.append(str(d))

    _CUDA_DLL_DIRS_ADDED.extend(added)
    if added:
        log.debug("registered CUDA DLL directories: %s", "; ".join(added))
    return added


def cudnn_status():
    """
    (found, directory, missing) for the cuDNN libraries CTranslate2 needs.

    Reported by `doctor` so the failure is visible before a run rather than
    as a load-time error three minutes into transcription.
    """
    import pathlib
    import sys

    needed = ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cublas64_12.dll"] \
        if sys.platform == "win32" else []
    if not needed:
        return True, None, []

    for d in _candidate_dll_dirs():
        present = {p.name for p in pathlib.Path(d).glob("*.dll")}
        missing = [n for n in needed if n not in present]
        if not missing:
            return True, str(d), []
    return False, None, needed
