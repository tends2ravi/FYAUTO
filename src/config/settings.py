"""Application settings and configuration management."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
SRC_DIR = BASE_DIR / "src"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

# Ensure required directories exist
for directory in [ASSETS_DIR, OUTPUT_DIR, TEMP_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class Settings:
    """Application settings management."""
    
    # API Keys and Credentials
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    
    # Model Settings
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")
    DEFAULT_TTS_PROVIDER: str = os.getenv("DEFAULT_TTS_PROVIDER", "google")
    DEFAULT_IMAGE_PROVIDER: str = os.getenv("DEFAULT_IMAGE_PROVIDER", "flux")
    
    # Cache Settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Performance Settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key for a specific provider.
        
        Args:
            provider: Name of the provider.
            
        Returns:
            API key if available, None otherwise.
        """
        key_map = {
            "openai": cls.OPENAI_API_KEY,
            "google": cls.GOOGLE_API_KEY,
            "anthropic": cls.ANTHROPIC_API_KEY,
            "deepseek": cls.DEEPSEEK_API_KEY
        }
        return key_map.get(provider.lower())
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary.
        
        Returns:
            Dictionary of all settings.
        """
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith("_") and key.isupper()
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required settings are configured.
        
        Returns:
            True if all required settings are valid, False otherwise.
        """
        required_settings = [
            "DEFAULT_LLM_PROVIDER",
            "DEFAULT_TTS_PROVIDER",
            "DEFAULT_IMAGE_PROVIDER"
        ]
        
        for setting in required_settings:
            if not getattr(cls, setting, None):
                return False
        return True

# Create settings instance
settings = Settings() 