"""
medlingo/utils/logging_utils.py

Structured logging with rich formatting for the MedLingo pipeline.
"""

from __future__ import annotations

import sys
import logging
import time
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    _RICH = True
except ImportError:
    _RICH = False


def setup_logging(level: str = "INFO", rich: bool = True) -> None:
    """Configure root logger with optional Rich formatting."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    if rich and _RICH:
        handlers = [RichHandler(rich_tracebacks=True, markup=True)]
        fmt = "%(message)s"
    else:
        handlers = [logging.StreamHandler(sys.stdout)]
        fmt = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("transformers", "datasets", "tokenizers", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class StageTimer:
    """Simple context manager for logging stage wall-clock times."""

    def __init__(self, stage_name: str, logger: Optional[logging.Logger] = None):
        self._name = stage_name
        self._log = logger or logging.getLogger(__name__)
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        self._log.info("▶  Starting stage: %s", self._name)
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        self._log.info(
            "✓  Stage '%s' completed in %.1f s (%.1f min)",
            self._name, elapsed, elapsed / 60,
        )
