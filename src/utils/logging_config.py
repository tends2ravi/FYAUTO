"""Logging configuration for the application."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Union

def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Set up logging configuration for the application.
    
    Args:
        log_file: Path to the log file. If None, logs to stderr only.
        log_level: Logging level to use.
        max_bytes: Maximum size of each log file.
        backup_count: Number of backup files to keep.
        log_format: Format string for log messages.
    """
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)

class LoggerMixin:
    """Mixin class to add logging capabilities to a class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger for the class.
        
        Returns:
            Logger instance named after the class.
        """
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger 