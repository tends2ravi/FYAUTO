"""
Unit tests for the script generation module.
"""
import pytest
from pathlib import Path
import json
import pickle
from typing import Dict, Any

from src.generation.script import ScriptGenerator
from src.core.errors import APIError, ValidationError
from src.features.preferences import FormatSettings
from tests.utils import BaseGenerationTest, MockClient

class TestScriptGenerator(BaseGenerationTest):
    """Test cases for ScriptGenerator class."""
    
    @pytest.fixture(autouse=True)
    def setup_script_generator(self, mock_redis):
        """Set up script generator test environment."""
        self.sample_script = {
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
        
        # Set up mock LLM responses
        self.mock_client = MockClient({
            "generate": MockResponse(json_data=self.sample_script)
        })
        
        # Initialize generator
        self.generator = ScriptGenerator(
            error_handler=self.error_handler,
            output_dir=self.output_dir
        )
        self.generator.redis_client = mock_redis
        
        yield
    
    @pytest.mark.asyncio
    async def test_script_generation_success(self):
        """Test successful script generation."""
        result = await self.generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        assert result["title"] == "Test Video"
        assert result["hook"] == "Test hook"
        assert len(result["scenes"]) == 1
        assert result["scenes"][0]["id"] == "scene_1"
    
    @pytest.mark.asyncio
    async def test_script_generation_from_cache(self, mock_redis):
        """Test script generation with cache hit."""
        # Set up cache hit
        mock_redis.get.return_value = pickle.dumps(self.sample_script)
        
        result = await self.generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        assert result["title"] == "Test Video"
        assert result["hook"] == "Test hook"
        assert not self.mock_client.requests  # No API calls made
    
    @pytest.mark.asyncio
    async def test_script_generation_invalid_response(self):
        """Test script generation with invalid response."""
        # Set up invalid response
        self.mock_client.responses["generate"] = MockResponse(
            text="Invalid JSON"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await self.generator.generate_script(
                topic="Test topic",
                style="informative",
                duration_minutes=5.0
            )
        
        assert "No JSON found in response" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_script_generation_missing_fields(self):
        """Test script generation with missing required fields."""
        # Set up response with missing fields
        self.mock_client.responses["generate"] = MockResponse(
            json_data={"title": "Test Video"}
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await self.generator.generate_script(
                topic="Test topic",
                style="informative",
                duration_minutes=5.0
            )
        
        assert "Missing required fields" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_script_generation_with_format_settings(self):
        """Test script generation with format settings."""
        format_settings = FormatSettings(
            duration_range={"min": 180.0, "max": 1200.0, "recommended": 600.0},
            resolution={"width": 1920, "height": 1080},
            fps=30,
            bitrate="6M",
            codec="h264"
        )
        
        result = await self.generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0,
            format_settings=format_settings
        )
        
        assert result["title"] == "Test Video"
        assert result["hook"] == "Test hook"
        
        # Verify format settings were included in prompt
        last_request = self.mock_client.requests[-1]
        prompt = last_request[2]["json"]["prompt"]
        assert "1920" in prompt
        assert "1080" in prompt
        assert "h264" in prompt
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, redis_client):
        """Test caching of generated scripts."""
        self.generator.redis_client = redis_client
        
        # First generation
        result1 = await self.generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        # Second generation (should use cache)
        result2 = await self.generator.generate_script(
            topic="Test topic",
            style="informative",
            duration_minutes=5.0
        )
        
        assert len(self.mock_client.requests) == 1  # Only one API call
        assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during script generation."""
        # Set up error response
        self.mock_client.responses["generate"] = MockResponse(
            status=500,
            error=APIError("API Error", "TEST_ERROR")
        )
        
        with pytest.raises(APIError) as exc_info:
            await self.generator.generate_script(
                topic="Test topic",
                style="informative",
                duration_minutes=5.0
            )
        
        assert "API Error" in str(exc_info.value)
        assert exc_info.value.error_code == "TEST_ERROR" 