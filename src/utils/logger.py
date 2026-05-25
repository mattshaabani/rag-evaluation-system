"""
src/utils/logger.py

Centralized logger for the entire project.
Uses Python's built-in logging module with structured output.

Usage in any file:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"chunk_id": 5})
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# 1. Log format
# ─────────────────────────────────────────────

# This is what each log line looks like:
# 2024-01-15 14:32:01 | INFO     | src.data.chunker | Processing chunk
LOG_FORMAT = "{asctime} | {levelname:<8} | {name} | {message}"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ─────────────────────────────────────────────
# 2. Color codes for terminal output
# ─────────────────────────────────────────────

COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[41m",   # red background
    "RESET":    "\033[0m",    # reset to default
}


class ColorFormatter(logging.Formatter):
    """
    Custom formatter that adds color to terminal log output.
    Colors make it easy to spot warnings and errors at a glance.
    Only applied to the terminal handler — log files stay plain text.
    """

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]

        # Temporarily colorize the level name
        record.levelname = f"{color}{record.levelname}{reset}"
        formatted = super().format(record)

        # Restore original level name (record objects are reused)
        record.levelname = record.levelname.replace(color, "").replace(reset, "")
        return formatted


# ─────────────────────────────────────────────
# 3. Logger factory function
# ─────────────────────────────────────────────

def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name:     Logger name. Always pass __name__ so logs show which
                  module they came from. e.g. "src.data.chunker"
        level:    Log level string: DEBUG, INFO, WARNING, ERROR, CRITICAL.
                  Defaults to value from settings (or INFO if not set).
        log_file: Optional path to write logs to a file as well.

    Returns:
        A configured Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Chunking started")
        logger.error("Failed to load document", extra={"path": str(path)})
    """

    # ── Import here to avoid circular imports ──
    # config imports nothing from logger, but we import config here
    # so logger can read the log level from settings
    try:
        from src.utils.config import settings
        default_level = settings.env.log_level
    except Exception:
        default_level = "INFO"

    # ── Resolve level ──
    level_str = (level or default_level).upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    # ── Get or create logger ──
    # logging.getLogger() returns the same object if called with the
    # same name — so calling get_logger(__name__) twice in the same
    # module is safe and efficient
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # ── Avoid adding duplicate handlers ──
    # If this logger already has handlers (e.g. called twice), skip setup
    if logger.handlers:
        return logger

    # ── Terminal handler (with color) ──
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setLevel(numeric_level)
    terminal_formatter = ColorFormatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        style="{"        # use {message} style instead of %s style
    )
    terminal_handler.setFormatter(terminal_formatter)
    logger.addHandler(terminal_handler)

    # ── File handler (optional, plain text) ──
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            style="{"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # ── Prevent logs from bubbling up to the root logger ──
    # Without this, logs can appear twice in some setups
    logger.propagate = False

    return logger