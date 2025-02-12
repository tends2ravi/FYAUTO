"""
Enhanced error handling and recovery system for video production.
Combines robust error handling, logging, and resource management.
"""
from typing import Any, Callable, Dict, Optional, Type, Union
from loguru import logger
import traceback
import time
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from pathlib import Path
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp

from . import config

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

class ValidationError(VideoProductionError):
    """Error during input validation."""
    pass

class APIError(VideoProductionError):
    """Error during API calls."""
    pass

class AudioGenerationError(VideoProductionError):
    """Error in audio generation."""
    pass

class CacheError(VideoProductionError):
    """Error in cache operations."""
    pass

class NetworkError(VideoProductionError):
    """Error in network operations."""
    pass

class TimeoutError(VideoProductionError):
    """Error when operation times out."""
    pass

class ConfigurationError(VideoProductionError):
    """Error in configuration."""
    pass

class ErrorHandler:
    """Handles error recovery and logging for the video production system."""
    
    def __init__(self, max_retries: int = None, retry_delay: float = None):
        self.max_retries = max_retries or config.MAX_RETRIES
        self.retry_delay = retry_delay or config.RETRY_DELAY
        self.error_counts: Dict[str, int] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Set up logging directory
        self.log_dir = Path(config.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure loguru logger
        logger.add(
            self.log_dir / "app.log",
            rotation="1 day",
            retention="30 days",
            level=config.LOG_LEVEL,
            format=config.LOG_FORMAT,
            backtrace=True,
            diagnose=True
        )
        
        # Add error-specific log file
        logger.add(
            self.log_dir / "errors.log",
            rotation="1 day",
            retention="30 days",
            level="ERROR",
            format=config.LOG_FORMAT,
            filter=lambda record: record["level"].name == "ERROR",
            backtrace=True,
            diagnose=True
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error with detailed context and stack trace."""
        error_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'context': context or {}
        }
        
        # Add error code and details if available
        if isinstance(error, VideoProductionError):
            error_data.update({
                'error_code': error.error_code,
                'error_details': error.details
            })
        
        # Add system info
        error_data['system_info'] = {
            'memory_available': psutil.virtual_memory().available,
            'disk_available': psutil.disk_usage('/').free,
            'cpu_percent': psutil.cpu_percent()
        }
        
        logger.error(json.dumps(error_data, indent=2))
        
        # Save detailed error info to file
        error_file = self.log_dir / 'errors' / f"error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)
    
    @staticmethod
    def with_retry(
        retry_on: Union[Type[Exception], tuple] = Exception,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        error_code: str = "GENERAL_ERROR"
    ):
        """Decorator for retrying operations on failure with exponential backoff."""
        def decorator(func):
            @retry(
                stop=stop_after_attempt(max_retries or config.MAX_RETRIES),
                wait=wait_exponential(multiplier=retry_delay or config.RETRY_DELAY, min=4, max=10),
                retry=lambda e: isinstance(e, retry_on)
            )
            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                try:
                    return await func(self, *args, **kwargs)
                except retry_on as e:
                    self.log_error(e, {
                        'function': func.__name__,
                        'args': args,
                        'kwargs': kwargs,
                        'error_code': error_code
                    })
                    raise
            
            @retry(
                stop=stop_after_attempt(max_retries or config.MAX_RETRIES),
                wait=wait_exponential(multiplier=retry_delay or config.RETRY_DELAY, min=4, max=10),
                retry=lambda e: isinstance(e, retry_on)
            )
            @wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except retry_on as e:
                    self.log_error(e, {
                        'function': func.__name__,
                        'args': args,
                        'kwargs': kwargs,
                        'error_code': error_code
                    })
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def with_fallback(
        self,
        fallback_function: Callable,
        error_code: str = "FALLBACK_ERROR"
    ):
        """Decorator for providing fallback behavior on failure."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.log_error(e, {
                        'function': func.__name__,
                        'fallback': fallback_function.__name__,
                        'error_code': error_code
                    })
                    return await fallback_function(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.log_error(e, {
                        'function': func.__name__,
                        'fallback': fallback_function.__name__,
                        'error_code': error_code
                    })
                    return fallback_function(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def validate_input(self, validation_func: Callable):
        """Decorator to validate input parameters."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    validation_func(*args, **kwargs)
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.log_error(e, {
                        'function': func.__name__,
                        'validation': validation_func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    })
                    raise ValidationError(
                        str(e),
                        "VALIDATION_ERROR",
                        {'validation_function': validation_func.__name__}
                    )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    validation_func(*args, **kwargs)
                    return func(*args, **kwargs)
                except Exception as e:
                    self.log_error(e, {
                        'function': func.__name__,
                        'validation': validation_func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    })
                    raise ValidationError(
                        str(e),
                        "VALIDATION_ERROR",
                        {'validation_function': validation_func.__name__}
                    )
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def check_resources(
        self,
        min_memory_mb: float = None,
        min_disk_mb: float = None
    ):
        """Decorator to check system resources before executing function."""
        min_memory = min_memory_mb or config.MIN_MEMORY_MB
        min_disk = min_disk_mb or config.MIN_DISK_MB
        
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Check memory
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                if available_memory < min_memory:
                    raise ResourceError(
                        f"Insufficient memory: {available_memory:.2f}MB available, {min_memory}MB required",
                        "MEMORY_ERROR",
                        {
                            'available': available_memory,
                            'required': min_memory
                        }
                    )
                
                # Check disk space
                available_disk = psutil.disk_usage('/').free / (1024 * 1024)
                if available_disk < min_disk:
                    raise ResourceError(
                        f"Insufficient disk space: {available_disk:.2f}MB available, {min_disk}MB required",
                        "DISK_ERROR",
                        {
                            'available': available_disk,
                            'required': min_disk
                        }
                    )
                
                return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Same resource checks for sync functions
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                if available_memory < min_memory:
                    raise ResourceError(
                        f"Insufficient memory: {available_memory:.2f}MB available, {min_memory}MB required",
                        "MEMORY_ERROR",
                        {
                            'available': available_memory,
                            'required': min_memory
                        }
                    )
                
                available_disk = psutil.disk_usage('/').free / (1024 * 1024)
                if available_disk < min_disk:
                    raise ResourceError(
                        f"Insufficient disk space: {available_disk:.2f}MB available, {min_disk}MB required",
                        "DISK_ERROR",
                        {
                            'available': available_disk,
                            'required': min_disk
                        }
                    )
                
                return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def run_with_timeout(
        self,
        func: Callable,
        timeout: float,
        *args,
        **kwargs
    ) -> Any:
        """Run a function with timeout."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(*args, **kwargs), timeout)
            else:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    func,
                    *args,
                    **kwargs
                )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Operation timed out after {timeout} seconds",
                "TIMEOUT_ERROR",
                {
                    'timeout': timeout,
                    'function': func.__name__
                }
            )
    
    def handle_background_errors(self, task: asyncio.Task) -> None:
        """Handle errors from background tasks."""
        try:
            task.result()
        except Exception as e:
            self.log_error(e, {
                'task': task.get_name(),
                'error_code': "BACKGROUND_ERROR"
            })
    
    async def cleanup_resources(self) -> None:
        """Clean up resources."""
        try:
            self.executor.shutdown(wait=True)
            await asyncio.sleep(0)  # Let other tasks finish
        except Exception as e:
            self.log_error(e, {
                'error_code': "CLEANUP_ERROR"
            })
    
    def handle_api_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Handle API-specific errors."""
        if isinstance(error, aiohttp.ClientError):
            error_type = "NETWORK_ERROR"
        elif isinstance(error, asyncio.TimeoutError):
            error_type = "API_TIMEOUT"
        else:
            error_type = "API_ERROR"
        
        self.log_error(error, {
            'error_code': error_type,
            'context': context or {}
        })
        
        raise APIError(
            str(error),
            error_type,
            context
        )
    
    def handle_cache_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Handle cache-specific errors."""
        self.log_error(error, {
            'error_code': "CACHE_ERROR",
            'context': context or {}
        })
        
        raise CacheError(
            str(error),
            "CACHE_ERROR",
            context
        ) 