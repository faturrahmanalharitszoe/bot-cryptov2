"""
Structured logging for Bot-CryptoV2.

Provides a consistent logging interface using Python's logging module
with rich formatting for terminal output and file rotation.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from rich.logging import RichHandler
from rich.console import Console


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logger(
    name: str = "bot_crypto",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with rich terminal output and optional file logging.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (relative to project root). None = no file logging.
        max_bytes: Max size per log file before rotation
        backup_count: Number of rotated log files to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Rich terminal handler
    rich_handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    rich_format = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(rich_format)
    logger.addHandler(rich_handler)

    # File handler (optional)
    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOG_DIR / log_file if not Path(log_file).is_absolute() else Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # File always gets DEBUG
        file_format = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "bot_crypto") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)
