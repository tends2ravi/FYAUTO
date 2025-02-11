"""
Pytest configuration and shared fixtures.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch
import asyncio
from typing import Generator, Dict
import os

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as async")

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Set up mock API keys
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mock_credentials.json"
    os.environ["TOGETHER_API_KEY"] = "test_api_key"
    os.environ["ELEVENLABS_API_KEY"] = "test_api_key"
    
    # Create mock Google credentials file
    mock_creds = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key": "mock_key",
        "client_email": "test@test.com"
    }
    with open("mock_credentials.json", "w") as f:
        json.dump(mock_creds, f)
    
    yield
    
    # Cleanup
    if os.path.exists("mock_credentials.json"):
        os.remove("mock_credentials.json")

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def sample_script_data() -> Dict:
    """Sample script data for testing."""
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
            "estimated_duration_seconds": 120,
            "key_points": ["point 1", "point 2"],
            "target_keywords": ["test", "video"]
        }
    }

@pytest.fixture(scope="session")
def mock_together_client() -> MagicMock:
    """Mock Together AI client."""
    mock = MagicMock()
    # Mock successful image generation
    mock.images.generate.return_value.data = [
        MagicMock(b64_json="fake_base64_data")
    ]
    return mock

@pytest.fixture(scope="session")
def mock_elevenlabs_client() -> MagicMock:
    """Mock ElevenLabs client."""
    mock = MagicMock()
    # Mock successful audio generation
    mock.generate.return_value = b"fake_audio_data"
    return mock

@pytest.fixture(scope="session")
def mock_google_client() -> MagicMock:
    """Mock Google Cloud client."""
    mock = MagicMock()
    # Mock successful text-to-speech
    mock.synthesize_speech.return_value.audio_content = b"fake_audio_data"
    return mock

@pytest.fixture(scope="session")
def sample_image(test_dir: Path) -> Generator[Path, None, None]:
    """Create a sample test image."""
    from PIL import Image
    import numpy as np
    
    # Create a simple gradient image
    arr = np.linspace(0, 255, 100*100).reshape(100, 100).astype('uint8')
    img = Image.fromarray(arr)
    
    image_path = test_dir / "test_image.png"
    img.save(image_path)
    
    yield image_path
    
    if image_path.exists():
        image_path.unlink()

@pytest.fixture(scope="session")
def sample_audio(test_dir: Path) -> Generator[Path, None, None]:
    """Create a sample test audio file."""
    from scipy.io import wavfile
    import numpy as np
    
    # Create a simple sine wave
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    audio_path = test_dir / "test_audio.wav"
    wavfile.write(audio_path, sample_rate, audio_data)
    
    yield audio_path
    
    if audio_path.exists():
        audio_path.unlink()

@pytest.fixture(scope="function")
def mock_cache_dir(test_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary cache directory."""
    cache_dir = test_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Create cache subdirectories
    (cache_dir / "images").mkdir(exist_ok=True)
    (cache_dir / "prompts").mkdir(exist_ok=True)
    (cache_dir / "concepts").mkdir(exist_ok=True)
    (cache_dir / "metadata").mkdir(exist_ok=True)
    
    yield cache_dir
    
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

@pytest.fixture(scope="function")
def mock_config(test_dir: Path, monkeypatch):
    """Mock configuration for testing."""
    config_data = {
        "OUTPUT_DIR": str(test_dir / "output"),
        "TEMP_DIR": str(test_dir / "temp"),
        "LOG_LEVEL": "DEBUG",
        "TOGETHER_API_KEY": "fake_api_key",
        "ELEVENLABS_API_KEY": "fake_api_key",
        "GOOGLE_API_KEY": "fake_api_key"
    }
    
    config_path = test_dir / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    
    def mock_load_config():
        return config_data
    
    # Patch configuration loading
    import src.config
    monkeypatch.setattr(src.config, "load_config", mock_load_config)
    
    return config_data

@pytest.fixture
def mock_error_handler():
    """Provide a mock error handler for testing."""
    class MockErrorHandler:
        def __init__(self):
            self.errors = []
            
        def log_error(self, error, context=None):
            self.errors.append((error, context))
            
        def with_retry(self, max_retries=3, retry_delay=1):
            def decorator(func):
                async def wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)
                return wrapper
            return decorator
            
        def cleanup_resources(self):
            pass
            
    return MockErrorHandler()

@pytest.fixture(scope="function")
def mock_video_clip(test_dir: Path) -> Generator[Path, None, None]:
    """Create a mock video clip for testing."""
    from moviepy.editor import ColorClip
    import numpy as np
    
    # Create a simple color clip
    color_clip = ColorClip(size=(320, 240), color=[0, 0, 0], duration=1)
    video_path = test_dir / "test_clip.mp4"
    color_clip.write_videofile(str(video_path), fps=24)
    
    yield video_path
    
    # Cleanup
    if video_path.exists():
        try:
            video_path.unlink()
        except PermissionError:
            pass  # Ignore permission errors during cleanup

@pytest.fixture(scope="function")
def mock_error_handler():
    """Provide an error handler for testing."""
    class TestErrorHandler:
        def __init__(self):
            self.max_retries = 3
            self.retry_delay = 1
            self.errors = []
        
        def log_error(self, error, context=None):
            self.errors.append((error, context))
        
        async def run_with_retry(self, func, *args, max_retries=None, **kwargs):
            attempts = 0
            max_attempts = max_retries or self.max_retries
            
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    await asyncio.sleep(self.retry_delay)
        
        async def run_with_timeout(self, func, timeout=30):
            return await asyncio.wait_for(func(), timeout)
    
    return TestErrorHandler() 