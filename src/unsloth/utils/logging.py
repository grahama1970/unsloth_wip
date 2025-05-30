"""Logging utilities."""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
        rotation: Log rotation setting
        retention: Log retention setting
        format: Log format string
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
        
    logger.info(f"Logging initialized at level {level}")