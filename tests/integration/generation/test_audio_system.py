"""
Tests for the audio generation system.
"""
import pytest
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock
import json
import asyncio
from typing import Generator

from src.audio_system import AudioSystem
from src.error_handler import AudioGenerationError

@pytest.mark.asyncio
class TestAudioSystem:
    """Test suite for AudioSystem."""
    
    @pytest.fixture
    async def audio_system(self, error_handler, output_dir):
        """Create an audio system instance for testing."""
        system = AudioSystem(
            error_handler=error_handler,
            output_dir=output_dir,
            api_key="test_key"
        )
        yield system
        await system.cleanup()

    async def test_generate_speech_basic(
        self,
        audio_system,
        output_dir: Path
    ):
        """Test basic speech generation."""
        mock_response = b"fake_audio_data"
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.read = \
                lambda: mock_response
            
            audio_path = await audio_system.generate_speech(
                text="Test speech",
                voice_id="test_voice"
            )
            
            assert audio_path.exists()
            assert audio_path.parent == output_dir
            assert audio_path.suffix == ".mp3"

    async def test_generate_speech_with_style(
        self,
        audio_system
    ):
        """Test speech generation with different styles."""
        mock_response = b"fake_audio_data"
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.read = \
                lambda: mock_response
            
            for style in ["neutral", "excited", "dramatic"]:
                audio_path = await audio_system.generate_speech(
                    text="Test speech",
                    voice_id="test_voice",
                    style=style
                )
                
                assert audio_path.exists()
                assert style in audio_system._enhance_prompt("test", style)

    async def test_generate_speech_with_effects(
        self,
        audio_system
    ):
        """Test speech generation with audio effects."""
        mock_response = b"fake_audio_data"
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200
            mock_post.return_value.__aenter__.return_value.read = \
                lambda: mock_response
            
            effects = {
                "reverb": 0.5,
                "pitch": 1.2,
                "speed": 1.1
            }
            
            audio_path = await audio_system.generate_speech(
                text="Test speech",
                voice_id="test_voice",
                effects=effects
            )
            
            assert audio_path.exists()
            # In a real scenario, we would verify audio properties

    async def test_generate_sound_effects(
        self,
        audio_system
    ):
        """Test generation of sound effects."""
        effects = [
            "whoosh",
            "impact",
            "transition"
        ]
        
        for effect in effects:
            audio_path = await audio_system.generate_sound_effect(
                effect_type=effect,
                duration=1.0
            )
            
            assert audio_path.exists()
            assert audio_path.suffix == ".wav"

    async def test_mix_audio(
        self,
        audio_system,
        sample_audio
    ):
        """Test mixing multiple audio tracks."""
        tracks = [
            sample_audio,
            sample_audio  # Use same audio for testing
        ]
        
        mixed_path = await audio_system.mix_audio(
            audio_paths=tracks,
            output_name="mixed_audio"
        )
        
        assert mixed_path.exists()
        assert mixed_path.suffix == ".mp3"

    async def test_error_handling(
        self,
        audio_system
    ):
        """Test error handling in audio generation."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            # Test API error
            mock_post.return_value.__aenter__.return_value.status = 500
            mock_post.return_value.__aenter__.return_value.text = \
                lambda: "API error"
            
            with pytest.raises(AudioGenerationError):
                await audio_system.generate_speech(
                    text="Test speech",
                    voice_id="test_voice"
                )

    async def test_cleanup(
        self,
        audio_system
    ):
        """Test cleanup of resources."""
        # Create some temporary files
        temp_files = [
            audio_system.output_dir / f"temp_{i}.mp3"
            for i in range(3)
        ]
        
        for temp_file in temp_files:
            temp_file.touch()
            audio_system._temp_files.append(temp_file)
        
        await audio_system.cleanup()
        
        # Verify files are deleted
        for temp_file in temp_files:
            assert not temp_file.exists()
        assert len(audio_system._temp_files) == 0

    async def test_background_music(
        self,
        audio_system
    ):
        """Test background music generation."""
        music_path = await audio_system.generate_background_music(
            style="superhero",
            duration=5.0
        )
        
        assert music_path.exists()
        assert music_path.suffix == ".mp3"

    async def test_adjust_audio_length(
        self,
        audio_system,
        sample_audio
    ):
        """Test adjusting audio length."""
        target_duration = 2.0
        
        adjusted_path = await audio_system.adjust_audio_length(
            audio_path=sample_audio,
            target_duration=target_duration
        )
        
        assert adjusted_path.exists()
        # In a real scenario, we would verify the duration 