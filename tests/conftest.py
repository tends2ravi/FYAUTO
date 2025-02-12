"""
Pytest configuration and shared fixtures.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch, Mock, AsyncMock
import asyncio
from typing import Generator, Dict, AsyncGenerator
import os
import redis

from src.error_handler import ErrorHandler
from src.caption_system import CaptionSystem
from src.visual_generator import VisualGenerator
from src.core.errors import ErrorHandler
from src.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB
from src.utils.logging_config import setup_logging
from src.config.settings import settings

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

# Set up logging for tests
setup_logging(log_level="DEBUG")

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
def temp_dir():
    """Create a temporary directory for test files.
    
    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def test_assets_dir(temp_dir):
    """Create a directory for test assets.
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        Path to test assets directory.
    """
    assets_dir = temp_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    return assets_dir

@pytest.fixture(scope="session")
def test_output_dir(temp_dir):
    """Create a directory for test output.
    
    Args:
        temp_dir: Temporary directory fixture.
        
    Returns:
        Path to test output directory.
    """
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

@pytest.fixture
def error_handler() -> ErrorHandler:
    """Create an error handler instance for testing."""
    return ErrorHandler(max_retries=2, retry_delay=0.1)

@pytest.fixture
def assets_dir(temp_dir: Path) -> Path:
    """Create a temporary assets directory with test files."""
    assets_dir = temp_dir / "assets"
    assets_dir.mkdir(parents=True)
    
    # Create fonts directory
    fonts_dir = assets_dir / "fonts"
    fonts_dir.mkdir()
    
    # Copy test font
    test_font = Path("tests/test_data/OpenSans-Regular.ttf")
    if test_font.exists():
        shutil.copy(test_font, fonts_dir / "OpenSans-Regular.ttf")
    
    return assets_dir

@pytest.fixture
def output_dir(temp_dir: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
async def caption_system(
    error_handler: ErrorHandler,
    assets_dir: Path
) -> AsyncGenerator[CaptionSystem, None]:
    """Create a caption system instance for testing."""
    system = CaptionSystem(error_handler)
    yield system
    await system.cleanup()

@pytest.fixture
async def visual_generator(
    error_handler: ErrorHandler,
    output_dir: Path
) -> AsyncGenerator[VisualGenerator, None]:
    """Create a visual generator instance for testing."""
    generator = VisualGenerator(
        error_handler=error_handler,
        output_dir=output_dir,
        model_version="test-model"  # Use test model version
    )
    yield generator
    await generator.cleanup()

@pytest.fixture
def test_video(temp_dir: Path) -> Path:
    """Create a test video file."""
    from moviepy.editor import ColorClip
    
    # Create a simple color clip
    clip = ColorClip(size=(640, 480), color=(0, 0, 0), duration=2)
    video_path = temp_dir / "test_video.mp4"
    clip.write_videofile(str(video_path), fps=30)
    
    return video_path

@pytest.fixture
def test_image(temp_dir: Path) -> Path:
    """Create a test image file."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new("RGB", (640, 480), color="black")
    image_path = temp_dir / "test_image.png"
    img.save(image_path)
    
    return image_path

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

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
def sample_image(test_assets_dir):
    """Create a sample image for testing.
    
    Args:
        test_assets_dir: Test assets directory fixture.
        
    Returns:
        Path to sample image file.
    """
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    img = Image.fromarray(
        (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    )
    
    image_path = test_assets_dir / "sample.png"
    img.save(image_path)
    
    return image_path

@pytest.fixture(scope="session")
def sample_audio(test_assets_dir):
    """Create a sample audio file for testing.
    
    Args:
        test_assets_dir: Test assets directory fixture.
        
    Returns:
        Path to sample audio file.
    """
    import numpy as np
    import soundfile as sf
    
    # Create a simple test audio signal
    sample_rate = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    audio_path = test_assets_dir / "sample.wav"
    sf.write(audio_path, signal, sample_rate)
    
    return audio_path

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

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    mock = Mock(spec=redis.Redis)
    mock.get.return_value = None
    return mock

@pytest.fixture
def mock_process():
    """Create a mock asyncio subprocess."""
    mock = AsyncMock()
    mock.communicate = AsyncMock(return_value=(b"Success", b""))
    mock.returncode = 0
    return mock

@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""
    mock = AsyncMock()
    mock.post = AsyncMock()
    mock.get = AsyncMock()
    mock.close = AsyncMock()
    return mock

@pytest.fixture
def mock_image():
    """Create a mock PIL Image."""
    mock = Mock()
    mock.save = Mock()
    return mock

@pytest.fixture
def mock_torch_device():
    """Create a mock torch device."""
    return Mock(return_value="cpu")

@pytest.fixture
def redis_client():
    """Create a real Redis client for integration tests."""
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB
    )
    yield client
    client.flushdb()  # Clean up after tests

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Mock settings for testing.
    
    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    # Override settings for testing
    test_settings = {
        "REDIS_URL": "redis://localhost:6379/1",
        "CACHE_TTL": 60,
        "MAX_WORKERS": 2,
        "BATCH_SIZE": 5
    }
    
    for key, value in test_settings.items():
        monkeypatch.setattr(settings, key, value)

@pytest.fixture
def mock_provider():
    """Create a mock provider for testing.
    
    Returns:
        Mock provider instance.
    """
    class MockProvider:
        def __init__(self):
            self.api_key = "test_key"
            self.provider_name = "mock_provider"
            
        def validate_credentials(self):
            return True
            
        def make_request(self, **kwargs):
            return {"status": "success", "data": kwargs}
    
    return MockProvider()

@pytest.fixture
def sample_text():
    """Provide sample text for testing.
    
    Returns:
        Sample text string.
    """
    return "This is a sample text for testing purposes." 