"""
Configuration module for the video generation system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"

# Create necessary directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Model configuration
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", str(BASE_DIR / "models" / "local_model"))

# Cache configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours in seconds
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1073741824"))  # 1GB in bytes

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Resource limits
MIN_MEMORY_MB = float(os.getenv("MIN_MEMORY_MB", "1000"))  # 1GB
MIN_DISK_MB = float(os.getenv("MIN_DISK_MB", "5000"))  # 5GB
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))  # seconds

# Video preferences
DEFAULT_VIDEO_SETTINGS = {
    "youtube": {
        "duration_range": {
            "min": 180.0,  # 3 minutes
            "max": 1200.0,  # 20 minutes
            "recommended": 600.0  # 10 minutes
        },
        "resolution": {
            "width": 1920,
            "height": 1080
        },
        "fps": 30,
        "bitrate": "6M",
        "codec": "h264"
    },
    "shorts": {
        "duration_range": {
            "min": 15.0,
            "max": 60.0,
            "recommended": 30.0
        },
        "resolution": {
            "width": 1080,
            "height": 1920
        },
        "fps": 30,
        "bitrate": "4M",
        "codec": "h264"
    }
}

# Content guidelines
CONTENT_GUIDELINES = {
    "default": {
        "script_guidelines": {
            "tone": "informative",
            "language": "simple",
            "structure": [
                "hook",
                "introduction",
                "main_content",
                "call_to_action"
            ]
        },
        "visual_guidelines": {
            "style": "modern",
            "color_scheme": "vibrant",
            "negative_prompts": [
                "blurry",
                "low quality",
                "distorted",
                "watermark"
            ]
        },
        "audio_guidelines": {
            "voice_style": "natural",
            "background_music": "subtle",
            "sound_effects": "minimal"
        }
    }
}

def load_config():
    """Load and validate configuration."""
    # Force reload of environment variables
    load_dotenv(override=True)
    
    # API Keys
    global DEEPSEEK_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, TOGETHER_API_KEY
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    
    # Google API Configuration
    global GOOGLE_API_KEY, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    
    # Application Settings
    global BASE_DIR, OUTPUT_DIR, TEMP_DIR, LOG_LEVEL
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "./temp"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # YouTube Settings
    global YOUTUBE_CHANNEL_ID, DEFAULT_VIDEO_PRIVACY
    YOUTUBE_CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")
    DEFAULT_VIDEO_PRIVACY = os.getenv("DEFAULT_VIDEO_PRIVACY", "private")
    
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Initialize configuration
load_config()

def validate_config():
    """Validate that all required configuration variables are set."""
    required_vars = [
        "DEEPSEEK_API_KEY",
        "ELEVENLABS_API_KEY",
        "TOGETHER_API_KEY",
    ]
    
    missing_vars = [var for var in required_vars if not globals().get(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        ) 