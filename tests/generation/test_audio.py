"""
Unit tests for the audio generation module.
"""
import pytest
from pathlib import Path
import json
import pickle
from typing import Dict, Any

from src.generation.audio import AudioGenerator
from src.core.errors import AudioGenerationError, ValidationError
from tests.utils import BaseGenerationTest, MockClient, create_test_audio

class TestAudioGenerator(BaseGenerationTest):
    """Test cases for AudioGenerator class."""
    
    @pytest.fixture(autouse=True)
    def setup_audio_generator(self, mock_redis):
        """Set up audio generator test environment."""
        # Set up mock responses
        self.mock_client = MockClient({
            "generate": MockResponse(content=b"fake_audio_data")
        })
        
        # Initialize generator
        self.generator = AudioGenerator(
            error_handler=self.error_handler,
            output_dir=self.output_dir
        )
        self.generator.redis_client = mock_redis
        
        # Create test audio file
        self.test_audio = create_test_audio(self.temp_dir)
        
        yield
    
    @pytest.mark.asyncio
    async def test_generate_speech_success(self):
        """Test successful speech generation."""
        result = await self.generator.generate_speech(
            text="Test text",
            voice_id="test_voice",
            language="en"
        )
        
        assert "output_path" in result
        assert Path(result["output_path"]).exists()
        assert not result["cached"]
    
    @pytest.mark.asyncio
    async def test_generate_speech_from_cache(self, mock_redis):
        """Test speech generation with cache hit."""
        # Set up cache hit
        cached_result = {
            "output_path": str(self.test_audio),
            "validation": {
                "duration": 1.0,
                "sample_rate": 44100
            }
        }
        mock_redis.get.return_value = pickle.dumps(cached_result)
        
        result = await self.generator.generate_speech(
            text="Test text",
            voice_id="test_voice",
            language="en"
        )
        
        assert result["output_path"] == str(self.test_audio)
        assert result["cached"]
        assert not self.mock_client.requests  # No API calls made
    
    @pytest.mark.asyncio
    async def test_generate_speech_validation_error(self):
        """Test speech generation with validation error."""
        # Create invalid audio file
        invalid_audio = self.temp_dir / "invalid.wav"
        invalid_audio.write_bytes(b"invalid data")
        
        with pytest.raises(ValidationError) as exc_info:
            await self.generator.generate_speech(
                text="Test text",
                voice_id="test_voice",
                language="en",
                output_path=invalid_audio
            )
        
        assert "Failed to validate audio" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_speech_for_script(self):
        """Test generating speech for multiple scenes."""
        script = [
            {"id": "scene_1", "text": "Scene 1 text"},
            {"id": "scene_2", "text": "Scene 2 text"}
        ]
        
        results = await self.generator.generate_speech_for_script(
            script=script,
            voice_id="test_voice",
            language="en"
        )
        
        assert len(results) == 2
        assert all("output_path" in result for result in results)
        assert all(Path(result["output_path"]).exists() for result in results)
    
    @pytest.mark.asyncio
    async def test_get_available_voices(self):
        """Test getting available voices."""
        voices = await self.generator.get_available_voices(language="en")
        
        assert isinstance(voices, dict)
        assert all(isinstance(provider_voices, list) for provider_voices in voices.values())
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during speech generation."""
        # Set up error response
        self.mock_client.responses["generate"] = MockResponse(
            status=500,
            error=AudioGenerationError("Generation Error", "TEST_ERROR")
        )
        
        with pytest.raises(AudioGenerationError) as exc_info:
            await self.generator.generate_speech(
                text="Test text",
                voice_id="test_voice",
                language="en"
            )
        
        assert "Generation Error" in str(exc_info.value)
        assert exc_info.value.error_code == "TEST_ERROR"
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test resource cleanup."""
        await self.generator.cleanup()
        
        # Verify cleanup actions
        assert self.mock_client.closed 