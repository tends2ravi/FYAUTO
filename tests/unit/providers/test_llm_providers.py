"""
Tests for the LLM providers module.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
import tempfile
from pathlib import Path

from src.llm_providers import (
    LLMProvider,
    GeminiProvider,
    DeepSeekProvider,
    OpenAIProvider,
    LocalHuggingFaceProvider,
    LLMProviderManager,
    APIError
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
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump({"response": "Test response"}, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()

@pytest.mark.asyncio
async def test_gemini_provider_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful text generation with Gemini provider."""
    provider = GeminiProvider(error_handler)
    response = await provider.generate_text("Test prompt")
    
    assert response == '{"response": "Test response"}'
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_gemini_provider_failure(error_handler, mock_process):
    """Test failed text generation with Gemini provider."""
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"", b"API Error")
    
    provider = GeminiProvider(error_handler)
    
    with pytest.raises(APIError) as exc_info:
        await provider.generate_text("Test prompt")
    
    assert "Gemini API error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_deepseek_provider_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful text generation with DeepSeek provider."""
    provider = DeepSeekProvider(error_handler)
    response = await provider.generate_text("Test prompt")
    
    assert response == '{"response": "Test response"}'
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_openai_provider_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful text generation with OpenAI provider."""
    provider = OpenAIProvider(error_handler)
    response = await provider.generate_text("Test prompt")
    
    assert response == '{"response": "Test response"}'
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_local_provider_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful text generation with local provider."""
    provider = LocalHuggingFaceProvider(error_handler)
    response = await provider.generate_text("Test prompt")
    
    assert response == '{"response": "Test response"}'
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_provider_manager_success(error_handler, mock_subprocess, temp_output_file):
    """Test successful text generation with provider manager."""
    manager = LLMProviderManager(error_handler)
    response = await manager.generate_text("Test prompt")
    
    assert response == '{"response": "Test response"}'
    mock_subprocess.assert_called_once()

@pytest.mark.asyncio
async def test_provider_manager_fallback(error_handler):
    """Test provider manager fallback behavior."""
    # Mock providers to fail in sequence
    with patch('src.llm_providers.GeminiProvider.generate_text', side_effect=APIError("Gemini failed", "TEST_ERROR")), \
         patch('src.llm_providers.DeepSeekProvider.generate_text', side_effect=APIError("DeepSeek failed", "TEST_ERROR")), \
         patch('src.llm_providers.OpenAIProvider.generate_text', side_effect=APIError("OpenAI failed", "TEST_ERROR")), \
         patch('src.llm_providers.LocalHuggingFaceProvider.generate_text', side_effect=APIError("Local failed", "TEST_ERROR")):
        
        manager = LLMProviderManager(error_handler)
        
        with pytest.raises(APIError) as exc_info:
            await manager.generate_text("Test prompt")
        
        assert "All LLM providers failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_provider_manager_partial_fallback(error_handler, mock_subprocess, temp_output_file):
    """Test provider manager partial fallback behavior."""
    # Mock first provider to fail, second to succeed
    with patch('src.llm_providers.GeminiProvider.generate_text', side_effect=APIError("Gemini failed", "TEST_ERROR")):
        manager = LLMProviderManager(error_handler)
        response = await manager.generate_text("Test prompt")
        
        assert response == '{"response": "Test response"}'
        mock_subprocess.assert_called_once()

def test_provider_names():
    """Test provider name getters."""
    assert GeminiProvider(None).get_provider_name() == "gemini"
    assert DeepSeekProvider(None).get_provider_name() == "deepseek"
    assert OpenAIProvider(None).get_provider_name() == "openai"
    assert LocalHuggingFaceProvider(None).get_provider_name() == "local" 