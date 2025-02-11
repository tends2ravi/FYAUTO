"""
Configuration management for the video production system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

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