"""
Tests for the audio generation system.
"""
import pytest
from pathlib import Path
import json
import os
from unittest.mock import patch, MagicMock
from google.cloud import texttospeech
from src.audio_generator import AudioGenerator
from src.error_handler import ErrorHandler

@pytest.mark.asyncio
class TestAudioGenerator:
    """Test suite for AudioGenerator."""
    
    @pytest.fixture
    def error_handler(self):
        return ErrorHandler()

    @pytest.fixture
    def audio_generator(self, error_handler):
        return AudioGenerator(api_key="test_key", error_handler=error_handler)
    
    @pytest.fixture
    def sample_script_data(self):
        """Create sample script data for testing."""
        return {
            "title": "Test Video",
            "hook": "This is an attention-grabbing hook.",
            "sections": [
                {
                    "title": "Introduction",
                    "content": "This is the introduction section.",
                    "duration_seconds": 30
                },
                {
                    "title": "Main Content",
                    "content": "This is the main content section.",
                    "duration_seconds": 60
                }
            ],
            "call_to_action": "Like and subscribe!",
            "metadata": {
                "estimated_duration_seconds": 120
            }
        }
    
    async def test_generate_audio_for_script(self, audio_generator, sample_script_data, test_dir):
        """Test audio generation for entire script."""
        output_dir = test_dir / "audio"
        
        # Generate audio files
        audio_files = audio_generator.generate_audio_for_script(
            script_data=sample_script_data,
            output_dir=output_dir
        )
        
        # Verify output
        assert isinstance(audio_files, dict)
        assert "hook" in audio_files
        assert len(audio_files) == len(sample_script_data["sections"]) + 2  # sections + hook + CTA
        
        # Check file existence and format
        for path in audio_files.values():
            assert path.exists()
            assert path.suffix == ".mp3"
    
    async def test_generate_audio_segment(self, audio_generator, test_dir):
        """Test audio generation for a single segment."""
        output_path = test_dir / "test_segment.mp3"
        
        result_path = audio_generator._generate_audio_segment(
            text="Test audio content",
            voice_name="en-US-Neural2-D",
            output_path=output_path
        )
        
        assert result_path.exists()
        assert result_path == output_path
    
    async def test_long_text_chunking(self, audio_generator, test_dir):
        """Test handling of long text that needs chunking."""
        # Create long text that exceeds the character limit
        long_text = "Test content. " * 1000  # Will exceed 5000 character limit
        output_path = test_dir / "long_text.mp3"
        
        result_path = audio_generator._generate_audio_segment(
            text=long_text,
            voice_name="en-US-Neural2-D",
            output_path=output_path
        )
        
        assert result_path.exists()
        assert result_path == output_path
    
    async def test_voice_configuration(self, audio_generator):
        """Test voice configuration settings."""
        with patch('google.cloud.texttospeech.VoiceSelectionParams') as mock_voice:
            audio_generator._generate_audio_segment(
                text="Test content",
                voice_name="en-US-Neural2-D",
                output_path=Path("test.mp3")
            )
            
            mock_voice.assert_called_once_with(
                language_code="en-US",
                name="en-US-Neural2-D"
            )
    
    async def test_audio_configuration(self, audio_generator):
        """Test audio configuration settings."""
        with patch('google.cloud.texttospeech.AudioConfig') as mock_config:
            audio_generator._generate_audio_segment(
                text="Test content",
                voice_name="en-US-Neural2-D",
                output_path=Path("test.mp3")
            )
            
            mock_config.assert_called_once_with(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
                volume_gain_db=0.0
            )
    
    async def test_error_handling(self, audio_generator, test_dir):
        """Test error handling during audio generation."""
        with patch('google.cloud.texttospeech.TextToSpeechClient.synthesize_speech') as mock_synth:
            mock_synth.side_effect = Exception("API Error")
            
            with pytest.raises(Exception) as exc_info:
                audio_generator._generate_audio_segment(
                    text="Test content",
                    voice_name="en-US-Neural2-D",
                    output_path=test_dir / "error_test.mp3"
                )
            
            assert "Error generating audio" in str(exc_info.value)
    
    async def test_temp_file_cleanup(self, audio_generator, test_dir):
        """Test cleanup of temporary files."""
        output_path = test_dir / "cleanup_test.mp3"
        temp_pattern = "temp_*_cleanup_test.mp3"
        
        # Generate audio
        audio_generator._generate_audio_segment(
            text="Test content",
            voice_name="en-US-Neural2-D",
            output_path=output_path
        )
        
        # Check that no temp files remain
        temp_files = list(test_dir.glob(temp_pattern))
        assert len(temp_files) == 0
    
    async def test_file_name_sanitization(self, audio_generator):
        """Test filename sanitization."""
        test_cases = [
            ("Test Title", "test_title"),
            ("Test & Special * Chars", "test_special_chars"),
            ("Multiple     Spaces", "multiple_spaces"),
            ("Mixed CASE", "mixed_case")
        ]
        
        for input_name, expected in test_cases:
            result = audio_generator._sanitize_filename(input_name)
            assert result == expected
    
    async def test_concurrent_generation(self, audio_generator, sample_script_data, test_dir):
        """Test concurrent generation of multiple audio segments."""
        output_dir = test_dir / "concurrent_test"
        output_dir.mkdir(exist_ok=True)
        
        # Generate multiple audio files concurrently
        audio_files = audio_generator.generate_audio_for_script(
            script_data=sample_script_data,
            output_dir=output_dir
        )
        
        # Verify all files were generated
        assert all(path.exists() for path in audio_files.values())
    
    async def test_api_key_validation(self, monkeypatch):
        """Test API key validation during initialization."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        
        with pytest.raises(ValueError) as exc_info:
            AudioGenerator()
        
        assert "Google API key not found" in str(exc_info.value)
    
    async def test_output_directory_creation(self, audio_generator, test_dir):
        """Test output directory creation."""
        output_dir = test_dir / "new_audio_dir"
        
        audio_generator.generate_audio_for_script(
            script_data={"hook": "Test", "sections": [], "call_to_action": "Test"},
            output_dir=output_dir
        )
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    async def test_empty_text_handling(self, audio_generator, test_dir):
        """Test handling of empty text."""
        output_path = test_dir / "empty_test.mp3"
        
        with pytest.raises(ValueError) as exc_info:
            audio_generator._generate_audio_segment(
                text="",
                voice_name="en-US-Neural2-D",
                output_path=output_path
            )
        
        assert "Empty text" in str(exc_info.value) 