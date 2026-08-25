"""Atomic JSON I/O and run-directory management."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_json(path: str | Path, model: BaseModel | dict) -> Path:
    """
    Write JSON atomically.

    Atomicity matters here because a stage that crashes partway through must
    not leave a half-written artifact that the next stage happily parses.
    We write to a temp file in the same directory, then os.replace, which is
    atomic on both POSIX and NTFS.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, BaseModel):
        payload = model.model_dump(mode="json")
    else:
        payload = model

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_json(path: str | Path, model_cls: Type[T]) -> T:
    """Read and validate a stage artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing stage artifact: {path}. "
            f"Run the preceding stage, or use run.py which sequences them."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return model_cls.model_validate(raw)


def read_raw(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(base: str | Path, audio_path: str | Path) -> Path:
    """
    One working directory per input file, so concurrent or repeated runs do
    not clobber each other's intermediates.
    """
    stem = Path(audio_path).stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return ensure_dir(Path(base) / safe)
