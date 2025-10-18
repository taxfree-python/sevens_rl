"""
Logging utilities for Sevens RL project

Provides standardized logging configuration for training, evaluation, and debugging.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "sevens_rl",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name (typically module name or 'sevens_rl')
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("training", level=logging.DEBUG, log_file="logs/train.log")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Default format: timestamp - name - level - message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "sevens_rl") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> from src.utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Message from module")
    """
    logger = logging.getLogger(name)

    # If logger not configured, set up with defaults
    if not logger.handlers:
        setup_logger(name)

    return logger
