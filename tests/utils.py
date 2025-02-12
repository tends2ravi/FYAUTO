"""
Test utilities and base classes for testing.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import asyncio
from typing import Optional, Generator, AsyncGenerator, Any, Dict
from unittest.mock import Mock, AsyncMock

from src.core.errors import ErrorHandler

class BaseTest:
    """Base class for all test cases."""
    
    @pytest.fixture(autouse=True)
    def setup_base(self, tmp_path: Path):
        """Set up base test environment."""
        self.temp_dir = tmp_path
        self.output_dir = tmp_path / "output"
        self.cache_dir = tmp_path / "cache"
        self.logs_dir = tmp_path / "logs"
        
        # Create directories
        self.output_dir.mkdir()
        self.cache_dir.mkdir()
        self.logs_dir.mkdir()
        
        # Set up error handler
        self.error_handler = ErrorHandler()
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir)

class BaseAsyncTest(BaseTest):
    """Base class for async test cases."""
    
    @pytest.fixture(autouse=True)
    async def setup_async(self):
        """Set up async test environment."""
        # Set up any async resources
        yield
        # Clean up any async resources
        await asyncio.sleep(0)  # Let any pending tasks complete

class BaseGenerationTest(BaseAsyncTest):
    """Base class for generation module tests."""
    
    @pytest.fixture(autouse=True)
    def setup_generation(self):
        """Set up generation test environment."""
        self.test_prompt = "Test prompt"
        self.test_style = "standard"
        self.test_duration = 5.0
        
        # Create test assets directory
        self.assets_dir = self.temp_dir / "assets"
        self.assets_dir.mkdir()
        
        yield

class BaseProviderTest(BaseAsyncTest):
    """Base class for provider tests."""
    
    @pytest.fixture(autouse=True)
    def setup_provider(self):
        """Set up provider test environment."""
        # Mock API credentials
        self.mock_credentials = {
            "api_key": "test_key",
            "api_secret": "test_secret"
        }
        
        yield

class BaseFeaturesTest(BaseAsyncTest):
    """Base class for feature tests."""
    
    @pytest.fixture(autouse=True)
    def setup_features(self):
        """Set up features test environment."""
        # Create test media files directory
        self.media_dir = self.temp_dir / "media"
        self.media_dir.mkdir()
        
        yield

def create_test_file(
    directory: Path,
    filename: str,
    content: bytes = b"test content"
) -> Path:
    """Create a test file with given content."""
    file_path = directory / filename
    file_path.write_bytes(content)
    return file_path

def create_test_image(directory: Path, filename: str = "test.png") -> Path:
    """Create a test image file."""
    from PIL import Image
    import numpy as np
    
    # Create a simple gradient image
    arr = np.linspace(0, 255, 100*100).reshape(100, 100).astype('uint8')
    img = Image.fromarray(arr)
    
    file_path = directory / filename
    img.save(file_path)
    
    return file_path

def create_test_audio(directory: Path, filename: str = "test.wav") -> Path:
    """Create a test audio file."""
    from scipy.io import wavfile
    import numpy as np
    
    # Create a simple sine wave
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    file_path = directory / filename
    wavfile.write(file_path, sample_rate, audio_data)
    
    return file_path

def create_test_video(directory: Path, filename: str = "test.mp4") -> Path:
    """Create a test video file."""
    from moviepy.editor import ColorClip
    
    # Create a simple color clip
    clip = ColorClip(size=(640, 480), color=(0, 0, 0), duration=2)
    file_path = directory / filename
    clip.write_videofile(str(file_path), fps=30)
    
    return file_path

class MockResponse:
    """Mock aiohttp response."""
    
    def __init__(
        self,
        status: int = 200,
        json_data: Any = None,
        text: str = "",
        error: Optional[Exception] = None
    ):
        self.status = status
        self._json_data = json_data
        self._text = text
        self._error = error
    
    async def json(self):
        """Return mock JSON data."""
        if self._error:
            raise self._error
        return self._json_data
    
    async def text(self):
        """Return mock text data."""
        if self._error:
            raise self._error
        return self._text
    
    async def __aenter__(self):
        """Enter async context."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        pass

class MockClient:
    """Mock aiohttp client session."""
    
    def __init__(self, responses: Optional[dict] = None):
        self.responses = responses or {}
        self.requests = []
        self.closed = False
    
    async def get(self, url: str, **kwargs):
        """Mock GET request."""
        self.requests.append(("GET", url, kwargs))
        return self.responses.get(url, MockResponse())
    
    async def post(self, url: str, **kwargs):
        """Mock POST request."""
        self.requests.append(("POST", url, kwargs))
        return self.responses.get(url, MockResponse())
    
    async def close(self):
        """Mock close session."""
        self.closed = True 

@pytest.fixture
def mock_redis():
    """Mock Redis client fixture."""
    mock = Mock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    return mock

@pytest.fixture
def mock_llm_client():
    """Mock LLM client fixture."""
    mock = AsyncMock()
    mock.generate.return_value = {
        "text": "Generated text",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    }
    return mock

@pytest.fixture
def mock_tts_client():
    """Mock TTS client fixture."""
    mock = AsyncMock()
    mock.synthesize.return_value = b"fake_audio_data"
    return mock

@pytest.fixture
def mock_image_client():
    """Mock image generation client fixture."""
    mock = AsyncMock()
    mock.generate.return_value = b"fake_image_data"
    return mock

def create_test_script() -> Dict[str, Any]:
    """Create a test script object."""
    return {
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

def create_test_format_settings() -> Dict[str, Any]:
    """Create test format settings."""
    return {
        "duration_range": {
            "min": 180.0,
            "max": 1200.0,
            "recommended": 600.0
        },
        "resolution": {
            "width": 1920,
            "height": 1080
        },
        "fps": 30,
        "bitrate": "6M",
        "codec": "h264"
    }

def create_test_preferences() -> Dict[str, Any]:
    """Create test preferences."""
    return {
        "style": "informative",
        "voice": {
            "id": "test_voice",
            "language": "en",
            "gender": "female"
        },
        "music": {
            "genre": "ambient",
            "mood": "calm"
        },
        "visuals": {
            "style": "minimalist",
            "color_scheme": "light"
        }
    }

class MockVideoProcessor:
    """Mock video processor for testing."""
    
    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path or Path("test_output.mp4")
        self.processed_files = []
    
    async def process(self, input_path: Path) -> Path:
        """Mock video processing."""
        self.processed_files.append(input_path)
        return self.output_path
    
    async def cleanup(self):
        """Mock cleanup."""
        self.processed_files = []

class MockValidator:
    """Mock validator for testing."""
    
    def __init__(self, valid: bool = True, error_message: str = ""):
        self.valid = valid
        self.error_message = error_message
        self.validated_items = []
    
    async def validate(self, item: Any) -> bool:
        """Mock validation."""
        self.validated_items.append(item)
        if not self.valid:
            raise ValidationError(self.error_message)
        return True 