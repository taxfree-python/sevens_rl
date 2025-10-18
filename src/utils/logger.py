"""
Logging utilities for Sevens RL project

Provides standardized logging configuration for training, evaluation, and debugging.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_log_level(level: int | str) -> int:
    """
    Convert log level string to logging constant.

    Args:
        level: Log level as int (logging.INFO) or string ("INFO")

    Returns:
        Logging level constant (int)

    Raises:
        ValueError: If level string is invalid
    """
    if isinstance(level, int):
        return level

    level_str = level.upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level_str not in level_map:
        raise ValueError(
            f"Invalid log level: {level}. Must be one of {list(level_map.keys())}"
        )

    return level_map[level_str]


def setup_logger(
    name: str = "sevens_rl",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str | None = None,
    use_rotation: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name (typically module name or 'sevens_rl')
        level: Logging level (int or string: DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Custom format string for log messages
        use_rotation: Use RotatingFileHandler instead of FileHandler
        max_bytes: Maximum bytes per log file (used if use_rotation=True)
        backup_count: Number of backup files to keep (used if use_rotation=True)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("training", level="INFO", log_file="logs/train.log")
        >>> logger.info("Training started")
        >>> # With rotation
        >>> logger = setup_logger("training", level="DEBUG", log_file="logs/train.log",
        ...                       use_rotation=True, max_bytes=10*1024*1024)
    """
    # Convert level string to int if needed
    level = get_log_level(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs from parent loggers

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

        if use_rotation:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
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
