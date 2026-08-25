"""Minimal structured logging shared by every stage."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("LAUSANNE_LOG", "INFO").upper()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("lausanne")
        root.setLevel(level)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"lausanne.{name}")


@contextmanager
def timed(logger: logging.Logger, label: str):
    """Log how long a block took. Every stage reports its own wall clock."""
    start = time.perf_counter()
    logger.info("%s ...", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s done in %.1fs", label, elapsed)
