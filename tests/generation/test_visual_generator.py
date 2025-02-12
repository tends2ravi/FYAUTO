"""
Unit tests for the visual generation module.
"""
import pytest
from pathlib import Path
import json
import pickle
from typing import Dict, Any

from src.generation.visual import VisualGenerator
from src.core.errors import VisualGenerationError, ValidationError
from tests.utils import (
    BaseGenerationTest,
    MockClient,
    create_test_image,
    create_test_script,
    create_test_preferences
)

class TestVisualGenerator(BaseGenerationTest):
    """Test cases for VisualGenerator class."""
    
    @pytest.fixture(autouse=True)
    def setup_visual_generator(self, mock_redis, mock_image_client):
        """Set up visual generator test environment."""
        # Set up mock responses
        self.mock_client = mock_image_client
        
        # Initialize generator
        self.generator = VisualGenerator(
            error_handler=self.error_handler,
            output_dir=self.output_dir
        )
        self.generator.redis_client = mock_redis
        
        # Create test image
        self.test_image = create_test_image(self.temp_dir)
        
        # Set up test data
        self.test_script = create_test_script()
        self.test_preferences = create_test_preferences()
        
        yield
    
    @pytest.mark.asyncio
    async def test_generate_image_success(self):
        """Test successful image generation."""
        result = await self.generator.generate_image(
            prompt="Test prompt",
            style="minimalist",
            width=1920,
            height=1080
        )
        
        assert "output_path" in result
        assert Path(result["output_path"]).exists()
        assert not result["cached"]
    
    @pytest.mark.asyncio
    async def test_generate_image_from_cache(self, mock_redis):
        """Test image generation with cache hit."""
        # Set up cache hit
        cached_result = {
            "output_path": str(self.test_image),
            "validation": {
                "width": 1920,
                "height": 1080,
                "format": "PNG"
            }
        }
        mock_redis.get.return_value = pickle.dumps(cached_result)
        
        result = await self.generator.generate_image(
            prompt="Test prompt",
            style="minimalist",
            width=1920,
            height=1080
        )
        
        assert result["output_path"] == str(self.test_image)
        assert result["cached"]
        assert not self.mock_client.generate.called  # No API calls made
    
    @pytest.mark.asyncio
    async def test_generate_image_validation_error(self):
        """Test image generation with validation error."""
        # Create invalid image file
        invalid_image = self.temp_dir / "invalid.png"
        invalid_image.write_bytes(b"invalid data")
        
        with pytest.raises(ValidationError) as exc_info:
            await self.generator.generate_image(
                prompt="Test prompt",
                style="minimalist",
                width=1920,
                height=1080,
                output_path=invalid_image
            )
        
        assert "Failed to validate image" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_visuals_for_script(self):
        """Test generating visuals for multiple scenes."""
        results = await self.generator.generate_visuals_for_script(
            script=self.test_script,
            preferences=self.test_preferences
        )
        
        assert len(results) == len(self.test_script["scenes"])
        for scene_id, scene_visuals in results.items():
            assert isinstance(scene_visuals, list)
            assert all("output_path" in visual for visual in scene_visuals)
            assert all(Path(visual["output_path"]).exists() for visual in scene_visuals)
    
    @pytest.mark.asyncio
    async def test_style_transfer(self):
        """Test style transfer functionality."""
        result = await self.generator.apply_style_transfer(
            image_path=self.test_image,
            style="minimalist",
            strength=0.8
        )
        
        assert "output_path" in result
        assert Path(result["output_path"]).exists()
        assert result["output_path"] != str(self.test_image)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during image generation."""
        # Set up error response
        self.mock_client.generate.side_effect = VisualGenerationError(
            "Generation Error",
            "TEST_ERROR"
        )
        
        with pytest.raises(VisualGenerationError) as exc_info:
            await self.generator.generate_image(
                prompt="Test prompt",
                style="minimalist",
                width=1920,
                height=1080
            )
        
        assert "Generation Error" in str(exc_info.value)
        assert exc_info.value.error_code == "TEST_ERROR"
    
    @pytest.mark.asyncio
    async def test_batch_generation(self):
        """Test batch image generation."""
        prompts = [
            "Test prompt 1",
            "Test prompt 2",
            "Test prompt 3"
        ]
        
        results = await self.generator.generate_batch(
            prompts=prompts,
            style="minimalist",
            width=1920,
            height=1080
        )
        
        assert len(results) == len(prompts)
        assert all("output_path" in result for result in results)
        assert all(Path(result["output_path"]).exists() for result in results)
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test resource cleanup."""
        await self.generator.cleanup()
        
        # Verify cleanup actions
        assert self.mock_client.close.called 