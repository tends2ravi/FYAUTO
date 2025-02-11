"""
Tests for the error handling system.
"""
import pytest
import asyncio
import time
from src.error_handler import (
    ErrorHandler,
    VideoProductionError,
    GenerationError,
    ProcessingError,
    ResourceError
)

@pytest.mark.asyncio
class TestErrorHandler:
    """Test suite for ErrorHandler."""
    
    async def test_retry_decorator_success(self):
        """Test successful retry decorator execution."""
        handler = ErrorHandler(max_retries=3)
        counter = 0
        
        @handler.with_retry(retry_on=Exception)
        async def test_function():
            nonlocal counter
            counter += 1
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert counter == 1  # Should succeed on first try
    
    async def test_retry_decorator_failure(self):
        """Test retry decorator with failing function."""
        handler = ErrorHandler(max_retries=3)
        counter = 0
        
        @handler.with_retry(retry_on=ValueError)
        async def test_function():
            nonlocal counter
            counter += 1
            raise ValueError("Test error")
        
        with pytest.raises(VideoProductionError) as exc_info:
            await test_function()
        
        assert counter == 4  # Initial try + 3 retries
        assert "Test error" in str(exc_info.value)
    
    async def test_retry_with_eventual_success(self):
        """Test retry decorator with eventual success."""
        handler = ErrorHandler(max_retries=3)
        counter = 0
        
        @handler.with_retry(retry_on=ValueError)
        async def test_function():
            nonlocal counter
            counter += 1
            if counter < 3:
                raise ValueError("Test error")
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert counter == 3  # Should succeed on third try
    
    async def test_fallback_decorator(self):
        """Test fallback decorator."""
        handler = ErrorHandler()
        
        async def fallback_func(*args, **kwargs):
            return "fallback"
        
        @handler.with_fallback(fallback_func)
        async def test_function():
            raise ValueError("Test error")
        
        result = await test_function()
        assert result == "fallback"
    
    async def test_cleanup_decorator(self):
        """Test cleanup decorator."""
        handler = ErrorHandler()
        cleaned = False
        
        async def cleanup_func(*args, **kwargs):
            nonlocal cleaned
            cleaned = True
        
        @handler.with_cleanup(cleanup_func)
        async def test_function():
            return "success"
        
        result = await test_function()
        assert result == "success"
        assert cleaned is True
    
    async def test_run_with_timeout_success(self):
        """Test successful execution with timeout."""
        handler = ErrorHandler()
        
        async def quick_function():
            return "success"
        
        result = await handler.run_with_timeout(quick_function, timeout=1.0)
        assert result == "success"
    
    async def test_run_with_timeout_failure(self):
        """Test timeout failure."""
        handler = ErrorHandler()
        
        async def slow_function():
            await asyncio.sleep(2.0)
            return "success"
        
        with pytest.raises(TimeoutError):
            await handler.run_with_timeout(slow_function, timeout=1.0)
    
    async def test_error_wrapping(self):
        """Test error wrapping functionality."""
        handler = ErrorHandler()
        
        # Test generation error
        error = handler._wrap_error(
            ValueError("Test error"),
            "TEST_ERROR",
            "generate_content"
        )
        assert isinstance(error, GenerationError)
        
        # Test processing error
        error = handler._wrap_error(
            ValueError("Test error"),
            "TEST_ERROR",
            "process_data"
        )
        assert isinstance(error, ProcessingError)
        
        # Test resource error
        error = handler._wrap_error(
            IOError("Test error"),
            "TEST_ERROR",
            "any_function"
        )
        assert isinstance(error, ResourceError)
    
    async def test_background_error_handling(self):
        """Test background task error handling."""
        handler = ErrorHandler()
        
        async def failing_task():
            raise ValueError("Background error")
        
        # Create and start task
        task = asyncio.create_task(failing_task())
        task.set_name("test_task")
        
        # Wait for task to complete
        try:
            await task
        except ValueError:
            pass
        
        # Check error handling
        handler.handle_background_errors(task)
        # No assertion needed, just checking it doesn't raise
    
    async def test_error_counting(self):
        """Test error counting functionality."""
        handler = ErrorHandler(max_retries=3)
        counter = 0
        
        @handler.with_retry(retry_on=ValueError, error_code="TEST_ERROR")
        async def test_function():
            nonlocal counter
            counter += 1
            raise ValueError("Test error")
        
        try:
            await test_function()
        except VideoProductionError:
            pass
        
        assert handler.error_counts["TEST_ERROR"] == 4  # Initial + 3 retries
    
    async def test_retry_delay_backoff(self):
        """Test retry delay with backoff."""
        handler = ErrorHandler(max_retries=2, retry_delay=0.1)
        counter = 0
        start_time = time.time()
        
        @handler.with_retry(retry_on=ValueError)
        async def test_function():
            nonlocal counter
            counter += 1
            raise ValueError("Test error")
        
        try:
            await test_function()
        except VideoProductionError:
            pass
        
        duration = time.time() - start_time
        # Should take at least: 0.1 + 0.2 seconds (with backoff)
        assert duration >= 0.3
    
    async def test_cleanup_resources(self):
        """Test resource cleanup."""
        handler = ErrorHandler()
        
        # Add some error counts
        handler.error_counts["TEST1"] = 1
        handler.error_counts["TEST2"] = 2
        
        await handler.cleanup_resources()
        
        assert len(handler.error_counts) == 0
        # Executor should be shut down
        assert handler.executor._shutdown 