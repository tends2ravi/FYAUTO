"""
Base provider classes for the video generation system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from pathlib import Path

from ..core.errors import ErrorHandler

class BaseProvider(ABC):
    """Abstract base class for all providers."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate provider credentials."""
        pass

class BaseImageProvider(BaseProvider):
    """Base class for image generation providers."""
    
    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate image."""
        pass
    
    @abstractmethod
    async def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """Validate generated image."""
        pass

class BaseAudioProvider(BaseProvider):
    """Base class for audio generation providers."""
    
    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        language: str = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate speech."""
        pass
    
    @abstractmethod
    async def validate_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Validate generated audio."""
        pass
    
    @abstractmethod
    async def get_available_voices(self, language: str = "en") -> List[Dict[str, Any]]:
        """Get available voices."""
        pass

class BaseLLMProvider(BaseProvider):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate text."""
        pass 