"""
Error handling and recovery system.
"""
from typing import Any, Callable, Dict, Optional, Type, Union
from loguru import logger
import traceback
import time
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor

class VideoProductionError(Exception):
    """Base exception class for video production errors."""
    def __init__(self, message: str, error_code: str, details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class GenerationError(VideoProductionError):
    """Error during content generation."""
    pass

class ProcessingError(VideoProductionError):
    """Error during content processing."""
    pass

class ResourceError(VideoProductionError):
    """Error related to resource management."""
    pass

class ErrorHandler:
    """Handles error recovery and logging for the video production system."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts: Dict[str, int] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    @staticmethod
    def with_retry(
        retry_on: Union[Type[Exception], tuple] = Exception,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        error_code: str = "GENERAL_ERROR"
    ):
        """
        Decorator for retrying operations on failure.
        
        Args:
            retry_on: Exception type(s) to retry on
            max_retries: Maximum number of retries (defaults to class max_retries)
            retry_delay: Delay between retries (defaults to class retry_delay)
            error_code: Error code for logging
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                retries = 0
                max_attempts = max_retries or self.max_retries
                delay = retry_delay or self.retry_delay
                
                while retries < max_attempts:
                    try:
                        return await func(self, *args, **kwargs)
                    except retry_on as e:
                        retries += 1
                        if retries == max_attempts:
                            logger.error(f"Max retries ({max_attempts}) reached for {func.__name__}")
                            raise VideoProductionError(
                                str(e),
                                error_code,
                                {
                                    "function": func.__name__,
                                    "retries": retries,
                                    "traceback": traceback.format_exc()
                                }
                            )
                        
                        logger.warning(
                            f"Retry {retries}/{max_attempts} for {func.__name__} "
                            f"due to {type(e).__name__}: {str(e)}"
                        )
                        await asyncio.sleep(delay * (2 ** (retries - 1)))  # Exponential backoff
            
            @wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                retries = 0
                max_attempts = max_retries or self.max_retries
                delay = retry_delay or self.retry_delay
                
                while retries < max_attempts:
                    try:
                        return func(self, *args, **kwargs)
                    except retry_on as e:
                        retries += 1
                        if retries == max_attempts:
                            logger.error(f"Max retries ({max_attempts}) reached for {func.__name__}")
                            raise VideoProductionError(
                                str(e),
                                error_code,
                                {
                                    "function": func.__name__,
                                    "retries": retries,
                                    "traceback": traceback.format_exc()
                                }
                            )
                        
                        logger.warning(
                            f"Retry {retries}/{max_attempts} for {func.__name__} "
                            f"due to {type(e).__name__}: {str(e)}"
                        )
                        time.sleep(delay * (2 ** (retries - 1)))  # Exponential backoff
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def with_fallback(
        self,
        fallback_function: Callable,
        error_code: str = "FALLBACK_ERROR"
    ):
        """
        Decorator for providing fallback behavior on failure.
        
        Args:
            fallback_function: Function to call on failure
            error_code: Error code for logging
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Function {func.__name__} failed, using fallback")
                    self._log_error(error_code, e, 0, 0)
                    return await fallback_function(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Function {func.__name__} failed, using fallback")
                    self._log_error(error_code, e, 0, 0)
                    return fallback_function(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def with_cleanup(self, cleanup_function: Callable):
        """
        Decorator for ensuring cleanup after function execution.
        
        Args:
            cleanup_function: Function to call for cleanup
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                finally:
                    await cleanup_function(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                finally:
                    cleanup_function(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def _log_error(
        self,
        error_code: str,
        error: Exception,
        attempt: int,
        max_retries: int
    ) -> None:
        """Log error details with context."""
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        error_context = {
            "error_code": error_code,
            "error_type": type(error).__name__,
            "attempt": attempt,
            "max_retries": max_retries,
            "stack_trace": traceback.format_exc()
        }
        
        if attempt < max_retries:
            logger.warning(
                f"Error occurred (attempt {attempt + 1}/{max_retries})",
                error_context
            )
        else:
            logger.error(
                f"Final error occurred after {max_retries} retries",
                error_context
            )
    
    def _should_retry(self, error_code: str, attempt: int, max_retries: int) -> bool:
        """Determine if another retry attempt should be made."""
        # Check attempt count
        if attempt >= max_retries:
            return False
        
        # Check error frequency
        error_count = self.error_counts.get(error_code, 0)
        if error_count > max_retries * 2:
            logger.warning(f"Too many errors of type {error_code}, stopping retries")
            return False
        
        return True
    
    def _wrap_error(
        self,
        original_error: Exception,
        error_code: str,
        function_name: str
    ) -> VideoProductionError:
        """Wrap the original error with more context."""
        error_type = type(original_error).__name__
        message = f"Error in {function_name}: {str(original_error)}"
        
        if isinstance(original_error, (IOError, OSError)):
            return ResourceError(message, error_code, {
                "original_error": error_type,
                "details": str(original_error)
            })
        elif "generation" in function_name.lower():
            return GenerationError(message, error_code, {
                "original_error": error_type,
                "details": str(original_error)
            })
        else:
            return ProcessingError(message, error_code, {
                "original_error": error_type,
                "details": str(original_error)
            })
    
    async def run_with_timeout(
        self,
        func: Callable,
        timeout: float,
        *args,
        **kwargs
    ) -> Any:
        """
        Run a function with a timeout.
        
        Args:
            func: Function to run
            timeout: Timeout in seconds
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            TimeoutError if the function takes too long
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    func,
                    *args,
                    **kwargs
                )
        except asyncio.TimeoutError:
            logger.error(f"Function {func.__name__} timed out after {timeout} seconds")
            raise TimeoutError(f"Operation timed out: {func.__name__}")
    
    def handle_background_errors(self, task: asyncio.Task) -> None:
        """Handle errors from background tasks."""
        try:
            error = task.exception()
            if error is not None:
                logger.error(
                    f"Background task failed: {task.get_name()}",
                    error=str(error),
                    stack_trace=traceback.format_exc()
                )
        except asyncio.CancelledError:
            logger.warning(f"Background task was cancelled: {task.get_name()}")
    
    async def cleanup_resources(self) -> None:
        """Clean up resources used by the error handler."""
        self.executor.shutdown(wait=True)
        self.error_counts.clear() 