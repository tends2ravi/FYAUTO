"""
Tests for the visual generation system.
"""
import pytest
from pathlib import Path
import aiohttp
from unittest.mock import patch, MagicMock
import json
from PIL import Image
import numpy as np

from src.visual_generator import VisualGenerator
from src.error_handler import APIError, ResourceError

@pytest.mark.asyncio
class TestVisualGenerator:
    """Test suite for VisualGenerator."""
    
    @pytest.fixture
    def visual_generator(self):
        return VisualGenerator(api_key="test_key")

    @pytest.fixture
    def mock_client(self, monkeypatch):
        mock = MagicMock()
        mock.images.generate.return_value = MagicMock(data=[{"b64_json": "fake_image_data"}])
        monkeypatch.setattr("together.Together", lambda *args, **kwargs: mock)
        return mock

    @pytest.fixture
    def sample_script_data(self):
        """Create sample script data for testing."""
        return {
            "title": "Test Video",
            "hook": "This is an attention-grabbing hook.",
            "sections": [
                {
                    "title": "Introduction",
                    "content": "This is the introduction section about key concept.",
                    "duration_seconds": 30
                },
                {
                    "title": "Main Points",
                    "content": "These are the main points with visual examples.",
                    "duration_seconds": 60
                }
            ],
            "call_to_action": "Like and subscribe!",
            "metadata": {
                "estimated_duration_seconds": 120
            }
        }
    
    async def test_generate_visuals_for_script(self, visual_generator, sample_script_data, test_dir):
        """Test visual generation for entire script."""
        output_dir = test_dir / "visuals"
        
        # Generate visuals
        visuals = visual_generator.generate_visuals_for_script(
            script_data=sample_script_data,
            output_dir=output_dir
        )
        
        # Verify output
        assert isinstance(visuals, dict)
        assert "title" in visuals
        assert len(visuals) >= len(sample_script_data["sections"]) + 2  # sections + title + end
        
        # Check file existence and format
        for paths in visuals.values():
            assert isinstance(paths, list)
            for path in paths:
                assert path.exists()
                assert path.suffix == ".png"
    
    async def test_generate_images(self, visual_generator, test_dir):
        """Test image generation."""
        output_dir = test_dir / "test_images"
        
        image_paths = visual_generator._generate_images(
            prompt="Test image prompt",
            output_dir=output_dir,
            prefix="test",
            count=1,
            resolution=(512, 512)
        )
        
        assert len(image_paths) == 1
        assert all(path.exists() for path in image_paths)
        assert all(path.suffix == ".png" for path in image_paths)
    
    async def test_create_title_prompt(self, visual_generator):
        """Test title prompt creation."""
        prompt = visual_generator._create_title_prompt(
            title="Test Title",
            style="modern minimalist"
        )
        
        assert "Test Title" in prompt
        assert "modern minimalist" in prompt
        assert "title card" in prompt.lower()
        assert "no text" in prompt.lower()
    
    async def test_create_section_prompt(self, visual_generator):
        """Test section prompt creation."""
        prompt = visual_generator._create_section_prompt(
            concept="key concept",
            style="modern minimalist",
            section_title="Test Section"
        )
        
        assert "key concept" in prompt
        assert "Test Section" in prompt
        assert "modern minimalist" in prompt
        assert "no text" in prompt.lower()
    
    async def test_extract_key_concepts(self, visual_generator):
        """Test key concept extraction."""
        text = "First important point. Second key concept. Third main idea."
        concepts = visual_generator._extract_key_concepts(text, max_concepts=3)
        
        assert len(concepts) <= 3
        assert all(isinstance(concept, str) for concept in concepts)
        assert concepts[0] in text
    
    def test_error_handling(self, visual_generator, mock_client):
        mock_client.images.generate.side_effect = Exception("API Error")
        with pytest.raises(Exception) as exc_info:
            visual_generator._generate_images("Test prompt", 1)
        assert "API Error" in str(exc_info.value)
    
    async def test_resolution_handling(self, visual_generator, test_dir):
        """Test handling of different resolutions."""
        resolutions = [(512, 512), (1024, 1024), (768, 768)]
        
        for width, height in resolutions:
            image_paths = visual_generator._generate_images(
                prompt="Test prompt",
                output_dir=test_dir,
                prefix=f"res_{width}x{height}",
                resolution=(width, height)
            )
            
            assert len(image_paths) > 0
            # In a real scenario, we would verify image dimensions
    
    async def test_style_variations(self, visual_generator, test_dir):
        """Test different visual styles."""
        styles = ["modern minimalist", "realistic", "artistic"]
        
        for style in styles:
            visuals = visual_generator.generate_visuals_for_script(
                script_data={"title": "Test", "sections": [], "call_to_action": "Test"},
                style=style,
                output_dir=test_dir / style
            )
            
            assert len(visuals) > 0
    
    async def test_concurrent_generation(self, visual_generator, sample_script_data, test_dir):
        """Test concurrent generation of multiple images."""
        output_dir = test_dir / "concurrent_test"
        
        visuals = visual_generator.generate_visuals_for_script(
            script_data=sample_script_data,
            output_dir=output_dir
        )
        
        # Verify all files were generated
        assert all(path.exists() for paths in visuals.values() for path in paths)
    
    def test_api_key_validation(self):
        with pytest.raises(ValueError) as exc_info:
            VisualGenerator(api_key="")
        assert "API key cannot be empty" in str(exc_info.value)
    
    async def test_output_directory_creation(self, visual_generator, test_dir):
        """Test output directory creation."""
        output_dir = test_dir / "new_visuals_dir"
        
        visual_generator.generate_visuals_for_script(
            script_data={"title": "Test", "sections": [], "call_to_action": "Test"},
            output_dir=output_dir
        )
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    async def test_file_naming(self, visual_generator, test_dir):
        """Test file naming convention."""
        output_dir = test_dir / "naming_test"
        
        image_paths = visual_generator._generate_images(
            prompt="Test prompt",
            output_dir=output_dir,
            prefix="test_prefix",
            count=2
        )
        
        for i, path in enumerate(image_paths):
            assert path.name.startswith("test_prefix")
            assert path.name.endswith(".png")
            assert f"{i:02d}" in path.name
    
    def test_empty_prompt_handling(self, visual_generator):
        with pytest.raises(ValueError) as exc_info:
            visual_generator._generate_images("", 1)
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    def test_image_saving(self, visual_generator, tmp_path):
        test_image = Image.new('RGB', (100, 100), color='red')
        path = tmp_path / "test_image.png"
        visual_generator._save_image(test_image, path)
        assert path.exists()
        assert path.stat().st_size > 0

    async def test_generate_image_basic(
        self,
        visual_generator: VisualGenerator,
        output_dir: Path
    ):
        """Test basic image generation."""
        # Mock successful API response
        mock_response = {
            "image": Image.new("RGB", (512, 512), "black").tobytes()
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = \
                lambda: mock_response
            
            image_path = await visual_generator.generate_image(
                prompt="Test image",
                width=512,
                height=512
            )
            
            assert image_path.exists()
            assert image_path.parent == output_dir
            assert image_path.suffix == ".png"
    
    async def test_generate_image_with_style(
        self,
        visual_generator: VisualGenerator
    ):
        """Test image generation with different styles."""
        mock_response = {
            "image": Image.new("RGB", (512, 512), "black").tobytes()
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = \
                lambda: mock_response
            
            for style in ["standard", "minimal", "dramatic"]:
                image_path = await visual_generator.generate_image(
                    prompt="Test image",
                    style=style
                )
                
                assert image_path.exists()
                assert style in visual_generator._enhance_prompt("test", style)
    
    async def test_generate_image_with_fallback(
        self,
        visual_generator: VisualGenerator
    ):
        """Test fallback to FLUX.1-dev model."""
        mock_response = {
            "image": Image.new("RGB", (512, 512), "black").tobytes()
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            # Primary model fails
            mock_post.return_value.__aenter__.return_value.status = 500
            mock_post.return_value.__aenter__.return_value.text = \
                lambda: "Primary model error"
            
            # Fallback model succeeds
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = \
                lambda: mock_response
            
            image_path = await visual_generator.generate_image(
                prompt="Test image"
            )
            
            assert image_path.exists()
            assert "flux_dev" in image_path.name
    
    async def test_generate_sequence(
        self,
        visual_generator: VisualGenerator
    ):
        """Test generation of image sequences."""
        mock_response = {
            "image": Image.new("RGB", (512, 512), "black").tobytes()
        }
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.json = \
                lambda: mock_response
            
            prompts = [
                "First test image",
                "Second test image",
                "Third test image"
            ]
            
            image_paths = await visual_generator.generate_sequence(
                prompts=prompts,
                style="standard"
            )
            
            assert len(image_paths) == len(prompts)
            for path in image_paths:
                assert path.exists()
    
    async def test_error_handling(
        self,
        visual_generator: VisualGenerator
    ):
        """Test error handling in visual generation."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            # Test API error
            mock_post.return_value.__aenter__.return_value.status = 500
            mock_post.return_value.__aenter__.return_value.text = \
                lambda: "API error"
            
            with pytest.raises(APIError):
                await visual_generator.generate_image(
                    prompt="Test image"
                )
            
            # Test resource error
            with patch("psutil.virtual_memory") as mock_memory:
                mock_memory.return_value = MagicMock(available=0)
                
                with pytest.raises(ResourceError):
                    await visual_generator.generate_image(
                        prompt="Test image"
                    )
    
    async def test_prompt_enhancement(
        self,
        visual_generator: VisualGenerator
    ):
        """Test prompt enhancement with style modifiers."""
        # Test standard style
        enhanced = visual_generator._enhance_prompt(
            "A cat",
            "standard"
        )
        assert "high quality" in enhanced
        assert "detailed" in enhanced
        
        # Test minimal style
        enhanced = visual_generator._enhance_prompt(
            "A cat",
            "minimal"
        )
        assert "minimalist" in enhanced
        assert "clean" in enhanced
        
        # Test dramatic style
        enhanced = visual_generator._enhance_prompt(
            "A cat",
            "dramatic"
        )
        assert "cinematic" in enhanced
        assert "dramatic lighting" in enhanced
    
    async def test_cleanup(
        self,
        visual_generator: VisualGenerator
    ):
        """Test cleanup of resources."""
        # Create a mock session
        session = MagicMock()
        visual_generator.session = session
        
        await visual_generator.cleanup()
        
        # Verify session was closed
        session.close.assert_called_once()
    
    async def test_context_manager(
        self,
        error_handler,
        output_dir: Path
    ):
        """Test visual generator as context manager."""
        async with VisualGenerator(
            error_handler=error_handler,
            output_dir=output_dir
        ) as generator:
            # Mock successful API response
            mock_response = {
                "image": Image.new("RGB", (512, 512), "black").tobytes()
            }
            
            with patch("aiohttp.ClientSession.post") as mock_post:
                mock_post.return_value.__aenter__.return_value.status = 200
                mock_post.return_value.__aenter__.return_value.json = \
                    lambda: mock_response
                
                image_path = await generator.generate_image(
                    prompt="Test image"
                )
                
                assert image_path.exists()
        
        # Verify session was closed after context exit
        assert generator.session.closed 