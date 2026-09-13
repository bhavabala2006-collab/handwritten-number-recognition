"""
logger.py
---------
A single, reusable logging setup used across the whole project so every
script prints consistent, timestamped, leveled messages instead of relying
on bare `print()` calls (which makes debugging real-time webcam issues much
harder).

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded successfully")
    logger.error("Webcam could not be opened")
"""

import logging
import sys
from pathlib import Path

try:
    from config import CONFIG
except ImportError:  # pragma: no cover - fallback if run outside project root
    CONFIG = None


_CONFIGURED = False


def _configure_root_logger() -> None:
    """Configure the root logger exactly once (console + optional file)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler (always on, so it works even if file logging fails)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler - useful for debugging long training runs.
    if CONFIG is not None:
        try:
            CONFIG.LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                CONFIG.LOG_DIR / "project.log", encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError:
            # If we cannot write logs to disk (e.g. read-only filesystem),
            # continue with console-only logging rather than crashing.
            pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    _configure_root_logger()
    return logging.getLogger(name)
