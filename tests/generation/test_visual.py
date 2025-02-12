"""
Tests for the visual generation module.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock
import json
import pickle

from src.generation.visual import VisualGenerator
from src.core.errors import APIError, ResourceError

@pytest.mark.asyncio
async def test_visual_generator_init(
    error_handler,
    mock_redis,
    mock_aiohttp_session,
    temp_dir
):
    """Test visual generator initialization."""
    with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
        generator = VisualGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        assert generator.error_handler == error_handler
        assert generator.output_dir == temp_dir
        assert generator.model_version == "v1.0"
        assert generator.device in ["cuda", "cpu"]

@pytest.mark.asyncio
async def test_generate_image_success(
    error_handler,
    mock_redis,
    mock_aiohttp_session,
    temp_dir
):
    """Test successful image generation."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "image": b"fake_image_data"
    }
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response
    
    with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session), \
         patch('PIL.Image.open', return_value=Mock()):
        generator = VisualGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        result = await generator.generate_image(
            prompt="Test prompt",
            style="standard"
        )
        
        assert isinstance(result, Path)
        assert result.parent == temp_dir
        assert mock_aiohttp_session.post.called

@pytest.mark.asyncio
async def test_generate_image_primary_failure_fallback_success(
    error_handler,
    mock_redis,
    mock_aiohttp_session,
    temp_dir
):
    """Test fallback to dev model when primary model fails."""
    # Mock primary model failure
    mock_failure = AsyncMock()
    mock_failure.status = 500
    
    # Mock fallback model success
    mock_success = AsyncMock()
    mock_success.status = 200
    mock_success.json.return_value = {
        "image": b"fake_image_data"
    }
    
    mock_aiohttp_session.post.side_effect = [
        AsyncMock(return_value=mock_failure),
        AsyncMock(return_value=mock_success)
    ]
    
    with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session), \
         patch('PIL.Image.open', return_value=Mock()):
        generator = VisualGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        result = await generator.generate_image(
            prompt="Test prompt",
            style="standard"
        )
        
        assert isinstance(result, Path)
        assert result.parent == temp_dir
        assert mock_aiohttp_session.post.call_count == 2

@pytest.mark.asyncio
async def test_generate_image_all_failures(
    error_handler,
    mock_redis,
    mock_aiohttp_session,
    temp_dir
):
    """Test error when all models fail."""
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response
    
    with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
        generator = VisualGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        with pytest.raises(APIError) as exc_info:
            await generator.generate_image(
                prompt="Test prompt",
                style="standard"
            )
        
        assert "All visual generation attempts failed" in str(exc_info.value)
        assert mock_aiohttp_session.post.call_count == 2

@pytest.mark.asyncio
async def test_generate_sequence(
    error_handler,
    mock_redis,
    mock_aiohttp_session,
    temp_dir
):
    """Test generating a sequence of images."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "image": b"fake_image_data"
    }
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response
    
    with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session), \
         patch('PIL.Image.open', return_value=Mock()):
        generator = VisualGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        prompts = ["Test 1", "Test 2", "Test 3"]
        results = await generator.generate_sequence(prompts)
        
        assert len(results) == len(prompts)
        assert all(isinstance(p, Path) for p in results)
        assert mock_aiohttp_session.post.call_count == len(prompts)

@pytest.mark.asyncio
async def test_cache_operations(
    error_handler,
    redis_client,
    mock_aiohttp_session,
    temp_dir
):
    """Test caching of generated images."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "image": b"fake_image_data"
    }
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response
    
    with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session), \
         patch('PIL.Image.open', return_value=Mock()):
        generator = VisualGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        generator.redis_client = redis_client
        
        # First generation
        result1 = await generator.generate_visual(
            prompt="Test prompt",
            output_path=temp_dir / "test.png"
        )
        
        # Second generation (should use cache)
        result2 = await generator.generate_visual(
            prompt="Test prompt",
            output_path=temp_dir / "test2.png"
        )
        
        assert result1["cached"] is False
        assert result2["cached"] is True
        assert mock_aiohttp_session.post.call_count == 1

@pytest.mark.asyncio
async def test_cleanup(
    error_handler,
    mock_redis,
    mock_aiohttp_session,
    temp_dir
):
    """Test resource cleanup."""
    with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
        generator = VisualGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        await generator.cleanup()
        
        assert mock_aiohttp_session.close.called 