"""
Tests for the script generation module.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock
import json
import pickle
from typing import Dict, Any

from src.generation.script import ScriptGenerator
from src.core.errors import APIError, ValidationError
from src.features.preferences import FormatSettings

@pytest.fixture
def sample_script_data() -> Dict[str, Any]:
    """Create sample script data for testing."""
    return {
        "title": "Test Video",
        "hook": "Test hook",
        "scenes": [
            {
                "id": "scene_1",
                "title": "Test Scene",
                "content": "Test content",
                "duration": 60.0,
                "visuals": [
                    {
                        "description": "Test visual",
                        "duration": 20.0
                    }
                ]
            }
        ],
        "call_to_action": "Test CTA",
        "metadata": {
            "estimated_duration": 300,
            "key_points": ["point 1"],
            "target_keywords": ["keyword1"]
        }
    }

@pytest.mark.asyncio
async def test_script_generator_init(
    error_handler,
    mock_redis,
    temp_dir
):
    """Test script generator initialization."""
    with patch('redis.Redis', return_value=mock_redis):
        generator = ScriptGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        assert generator.error_handler == error_handler
        assert generator.output_dir == temp_dir
        assert generator.redis_client == mock_redis

@pytest.mark.asyncio
async def test_generate_script_success(
    error_handler,
    mock_redis,
    temp_dir,
    sample_script_data
):
    """Test successful script generation."""
    mock_llm = AsyncMock()
    mock_llm.generate_text.return_value = json.dumps(sample_script_data)
    
    with patch('redis.Redis', return_value=mock_redis), \
         patch('src.providers.llm.LLMProviderManager', return_value=mock_llm):
        generator = ScriptGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        result = await generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        assert result["title"] == "Test Video"
        assert result["hook"] == "Test hook"
        assert len(result["scenes"]) == 1
        assert mock_llm.generate_text.called

@pytest.mark.asyncio
async def test_generate_script_from_cache(
    error_handler,
    mock_redis,
    temp_dir,
    sample_script_data
):
    """Test script generation with cache hit."""
    mock_redis.get.return_value = pickle.dumps(sample_script_data)
    mock_llm = AsyncMock()
    
    with patch('redis.Redis', return_value=mock_redis), \
         patch('src.providers.llm.LLMProviderManager', return_value=mock_llm):
        generator = ScriptGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        result = await generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        assert result["title"] == "Test Video"
        assert result["hook"] == "Test hook"
        assert not mock_llm.generate_text.called

@pytest.mark.asyncio
async def test_generate_script_invalid_response(
    error_handler,
    mock_redis,
    temp_dir
):
    """Test script generation with invalid response."""
    mock_llm = AsyncMock()
    mock_llm.generate_text.return_value = "Invalid JSON"
    
    with patch('redis.Redis', return_value=mock_redis), \
         patch('src.providers.llm.LLMProviderManager', return_value=mock_llm):
        generator = ScriptGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await generator.generate_script(
                topic="Test topic",
                style="informative",
                duration_minutes=5.0
            )
        
        assert "No JSON found in response" in str(exc_info.value)

@pytest.mark.asyncio
async def test_generate_script_missing_fields(
    error_handler,
    mock_redis,
    temp_dir
):
    """Test script generation with missing required fields."""
    mock_llm = AsyncMock()
    mock_llm.generate_text.return_value = json.dumps({
        "title": "Test Video"
    })
    
    with patch('redis.Redis', return_value=mock_redis), \
         patch('src.providers.llm.LLMProviderManager', return_value=mock_llm):
        generator = ScriptGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await generator.generate_script(
                topic="Test topic",
                style="informative",
                duration_minutes=5.0
            )
        
        assert "Missing required fields" in str(exc_info.value)

@pytest.mark.asyncio
async def test_generate_script_with_format_settings(
    error_handler,
    mock_redis,
    temp_dir,
    sample_script_data
):
    """Test script generation with format settings."""
    mock_llm = AsyncMock()
    mock_llm.generate_text.return_value = json.dumps(sample_script_data)
    
    format_settings = FormatSettings(
        duration_range={"min": 180.0, "max": 1200.0, "recommended": 600.0},
        resolution={"width": 1920, "height": 1080},
        fps=30,
        bitrate="6M",
        codec="h264"
    )
    
    with patch('redis.Redis', return_value=mock_redis), \
         patch('src.providers.llm.LLMProviderManager', return_value=mock_llm):
        generator = ScriptGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        
        result = await generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0,
            format_settings=format_settings
        )
        
        assert result["title"] == "Test Video"
        assert result["hook"] == "Test hook"
        assert mock_llm.generate_text.called
        
        # Verify format settings were included in prompt
        prompt = mock_llm.generate_text.call_args[0][0]
        assert "1920" in prompt
        assert "1080" in prompt
        assert "h264" in prompt

@pytest.mark.asyncio
async def test_cache_operations(
    error_handler,
    redis_client,
    temp_dir,
    sample_script_data
):
    """Test caching of generated scripts."""
    mock_llm = AsyncMock()
    mock_llm.generate_text.return_value = json.dumps(sample_script_data)
    
    with patch('src.providers.llm.LLMProviderManager', return_value=mock_llm):
        generator = ScriptGenerator(
            error_handler=error_handler,
            output_dir=temp_dir
        )
        generator.redis_client = redis_client
        
        # First generation
        result1 = await generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        # Second generation (should use cache)
        result2 = await generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        assert mock_llm.generate_text.call_count == 1
        assert result1 == result2 