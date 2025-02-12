"""Error handling utilities for the application."""

import logging
import functools
from typing import Callable, TypeVar, ParamSpec, Any

# Type variables for generic function signatures
P = ParamSpec('P')
T = TypeVar('T')

logger = logging.getLogger(__name__)

class ApplicationError(Exception):
    """Base exception class for application-specific errors."""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ResourceNotFoundError(ApplicationError):
    """Raised when a required resource is not found."""
    pass

class ValidationError(ApplicationError):
    """Raised when input validation fails."""
    pass

class ProcessingError(ApplicationError):
    """Raised when processing of data fails."""
    pass

def handle_errors(max_retries: int = 3) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for handling errors with retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts.
        
    Returns:
        Decorated function with error handling.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.error(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}"
                    )
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {func.__name__}")
                        raise RuntimeError(
                            f"Operation failed after {max_retries} attempts"
                        ) from last_error
            return None  # This line should never be reached
        return wrapper
    return decorator

def log_error(error: Exception, context: dict[str, Any] = None) -> None:
    """Log an error with optional context information.
    
    Args:
        error: The exception to log.
        context: Optional dictionary of context information.
    """
    error_message = f"Error: {str(error)}"
    if context:
        error_message += f" Context: {context}"
    logger.error(error_message, exc_info=True) 