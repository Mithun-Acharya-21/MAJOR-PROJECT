"""
Logger — Consistent logging setup for the project.

Exports:
  setup_logger — create a configured logger instance
"""

from __future__ import annotations

import logging
import sys


def setup_logger(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name.
        level: Logging level.
        fmt: Log format string.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
