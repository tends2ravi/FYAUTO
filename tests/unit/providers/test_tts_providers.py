"""
Tests for the TTS providers module.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
import tempfile
from pathlib import Path

from src.tts_providers import (
    TTSProvider,
    GoogleTTSProvider,
    ElevenLabsProvider,
    CoquiTTSProvider,
    TTSProviderManager,
    AudioGenerationError
)
from src.error_handler import ErrorHandler

@pytest.fixture
def error_handler():
    """Create a mock error handler."""
    return Mock(spec=ErrorHandler)

@pytest.fixture
def mock_process():
    """Create a mock asyncio subprocess."""
    mock = AsyncMock()
    mock.communicate = AsyncMock(return_value=(b"Success", b""))
    mock.returncode = 0
    return mock

@pytest.fixture
def mock_subprocess(mock_process):
    """Mock asyncio.create_subprocess_exec."""
    with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock:
        yield mock

@pytest.fixture
def temp_output_file():
    """Create a temporary output file."""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.wav', delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()

@pytest.fixture
def mock_voices_response():
    """Create a mock voices response."""
    return [
        {
            "id": "voice1",
            "name": "Test Voice 1",
            "language": "en",
            "gender": "female"
        },
        {
            "id": "voice2",
            "name": "Test Voice 2",
            "language": "en",
            "gender": "male"
        }
    ]

@pytest.mark.asyncio
async def test_google_tts_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful speech generation with Google TTS."""
    # Create the output file to simulate successful generation
    temp_output_file.touch()
    
    provider = GoogleTTSProvider(error_handler)
    output = await provider.generate_speech(
        text="Test text",
        voice_id="test_voice",
        output_path=temp_output_file
    )
    
    assert output == temp_output_file
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_google_tts_failure(error_handler, mock_process, temp_output_file):
    """Test failed speech generation with Google TTS."""
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"", b"API Error")
    
    provider = GoogleTTSProvider(error_handler)
    
    with pytest.raises(AudioGenerationError) as exc_info:
        await provider.generate_speech(
            text="Test text",
            voice_id="test_voice",
            output_path=temp_output_file
        )
    
    assert "Google TTS error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_elevenlabs_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful speech generation with ElevenLabs."""
    # Create the output file to simulate successful generation
    temp_output_file.touch()
    
    provider = ElevenLabsProvider(error_handler)
    output = await provider.generate_speech(
        text="Test text",
        voice_id="test_voice",
        output_path=temp_output_file
    )
    
    assert output == temp_output_file
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_coqui_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful speech generation with Coqui TTS."""
    # Create the output file to simulate successful generation
    temp_output_file.touch()
    
    provider = CoquiTTSProvider(error_handler)
    output = await provider.generate_speech(
        text="Test text",
        voice_id="test_voice",
        output_path=temp_output_file
    )
    
    assert output == temp_output_file
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_provider_manager_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful speech generation with provider manager."""
    # Create the output file to simulate successful generation
    temp_output_file.touch()
    
    manager = TTSProviderManager(error_handler)
    output = await manager.generate_speech(
        text="Test text",
        voice_id="test_voice",
        output_path=temp_output_file
    )
    
    assert output == temp_output_file
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_provider_manager_fallback(error_handler, temp_output_file):
    """Test provider manager fallback behavior."""
    # Mock providers to fail in sequence
    with patch('src.tts_providers.GoogleTTSProvider.generate_speech', side_effect=AudioGenerationError("Google failed", "TEST_ERROR")), \
         patch('src.tts_providers.ElevenLabsProvider.generate_speech', side_effect=AudioGenerationError("ElevenLabs failed", "TEST_ERROR")), \
         patch('src.tts_providers.CoquiTTSProvider.generate_speech', side_effect=AudioGenerationError("Coqui failed", "TEST_ERROR")):
        
        manager = TTSProviderManager(error_handler)
        
        with pytest.raises(AudioGenerationError) as exc_info:
            await manager.generate_speech(
                text="Test text",
                voice_id="test_voice",
                output_path=temp_output_file
            )
        
        assert "All TTS providers failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_provider_manager_partial_fallback(error_handler, mock_subprocess, temp_output_file):
    """Test provider manager partial fallback behavior."""
    # Create the output file to simulate successful generation
    temp_output_file.touch()
    
    # Mock first provider to fail, second to succeed
    with patch('src.tts_providers.GoogleTTSProvider.generate_speech', side_effect=AudioGenerationError("Google failed", "TEST_ERROR")):
        manager = TTSProviderManager(error_handler)
        output = await manager.generate_speech(
            text="Test text",
            voice_id="test_voice",
            output_path=temp_output_file
        )
        
        assert output == temp_output_file
        mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_get_available_voices(error_handler, mock_subprocess, mock_voices_response):
    """Test getting available voices from providers."""
    # Mock subprocess to return voice list
    mock_subprocess.return_value.communicate = AsyncMock(return_value=(json.dumps(mock_voices_response).encode(), b""))
    
    manager = TTSProviderManager(error_handler)
    voices = await manager.get_available_voices()
    
    assert "google" in voices
    assert "elevenlabs" in voices
    assert "coqui" in voices
    assert len(voices["google"]) == 2
    assert voices["google"][0]["id"] == "voice1"

@pytest.mark.asyncio
async def test_get_available_voices_partial_failure(error_handler, mock_subprocess, mock_voices_response):
    """Test getting available voices with some providers failing."""
    # Mock Google TTS to succeed, others to fail
    with patch('src.tts_providers.GoogleTTSProvider.get_available_voices', return_value=mock_voices_response), \
         patch('src.tts_providers.ElevenLabsProvider.get_available_voices', side_effect=AudioGenerationError("Failed", "TEST_ERROR")), \
         patch('src.tts_providers.CoquiTTSProvider.get_available_voices', side_effect=AudioGenerationError("Failed", "TEST_ERROR")):
        
        manager = TTSProviderManager(error_handler)
        voices = await manager.get_available_voices()
        
        assert len(voices["google"]) == 2
        assert voices["elevenlabs"] == []
        assert voices["coqui"] == []

def test_provider_names():
    """Test provider name getters."""
    assert GoogleTTSProvider(None).get_provider_name() == "google"
    assert ElevenLabsProvider(None).get_provider_name() == "elevenlabs"
    assert CoquiTTSProvider(None).get_provider_name() == "coqui" 