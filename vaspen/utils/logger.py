"""Centralized logging configuration for VASPen."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "vaspen",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        name: Logger name.
        level: Logging level (default INFO).
        log_file: Optional path to a log file. If None, logs only to stderr.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (once; repeated calls must not duplicate it)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    # File handler (added on request even if the console handler
    # already exists — main() re-invokes setup_logger with a file
    # destination after the module-level default call)
    if log_file is not None and not any(
            isinstance(h, logging.FileHandler) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# Module-level convenience logger
logger = setup_logger()
