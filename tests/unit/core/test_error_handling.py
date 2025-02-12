"""
Tests for the error handling and logging system.
"""

import pytest
import asyncio
import aiohttp
from pathlib import Path
import json
import logging
from unittest.mock import patch, MagicMock

from src.error_handling import (
    VideoGenerationError,
    APIError,
    ResourceError,
    ValidationError,
    log_error,
    handle_api_error,
    validate_input,
    check_resources,
    setup_error_handling
)

@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir

def test_video_generation_error():
    """Test VideoGenerationError creation and attributes."""
    details = {"step": "video_generation", "status": "failed"}
    error = VideoGenerationError("Test error", details)
    
    assert str(error) == "Test error"
    assert error.details == details
    assert isinstance(error, Exception)

@pytest.mark.asyncio
async def test_handle_api_error_decorator():
    """Test API error handling decorator with retries."""
    
    # Mock function that fails twice then succeeds
    call_count = 0
    @handle_api_error
    async def mock_api_call():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise aiohttp.ClientError("API Error")
        return "success"
    
    result = await mock_api_call()
    assert result == "success"
    assert call_count == 3

def test_log_error(temp_log_dir):
    """Test error logging functionality."""
    error = ValueError("Test error")
    context = {"test": "context"}
    
    with patch('logging.Logger.error') as mock_logger:
        log_error(error, context)
        
        # Verify logger was called
        mock_logger.assert_called_once()
        
        # Verify error file was created
        error_files = list(temp_log_dir.glob("error_*.json"))
        assert len(error_files) == 1
        
        # Verify error file content
        with open(error_files[0]) as f:
            error_data = json.load(f)
            assert error_data["error_type"] == "ValueError"
            assert error_data["error_message"] == "Test error"
            assert error_data["context"] == context

@pytest.mark.asyncio
async def test_validate_input_decorator():
    """Test input validation decorator."""
    def validation_func(*args, **kwargs):
        if not args or not isinstance(args[0], str):
            raise ValueError("First argument must be a string")
    
    @validate_input(validation_func)
    async def test_func(text: str):
        return text.upper()
    
    # Test valid input
    result = await test_func("hello")
    assert result == "HELLO"
    
    # Test invalid input
    with pytest.raises(ValidationError):
        await test_func(123)

@pytest.mark.asyncio
async def test_check_resources_decorator():
    """Test resource checking decorator."""
    @check_resources(min_memory_mb=1, min_disk_mb=1)
    async def test_func():
        return "success"
    
    # Test with sufficient resources
    result = await test_func()
    assert result == "success"
    
    # Test with insufficient resources
    with patch('psutil.virtual_memory') as mock_memory:
        mock_memory.return_value = MagicMock(available=0)
        with pytest.raises(ResourceError):
            await test_func()

def test_setup_error_handling(temp_log_dir):
    """Test error handling setup."""
    log_file = temp_log_dir / "test.log"
    setup_error_handling(str(log_file))
    
    # Verify log file was created
    assert log_file.exists()
    
    # Test logging
    logger = logging.getLogger(__name__)
    test_message = "Test log message"
    logger.error(test_message)
    
    # Verify message was logged
    with open(log_file) as f:
        log_content = f.read()
        assert test_message in log_content 