"""
Video Generation System
======================

A comprehensive system for automated video production with advanced features:
- Script generation with multiple LLM providers
- Audio generation with multiple TTS providers
- Visual generation with state-of-the-art models
- Video assembly and enhancement features
- YouTube upload capabilities

Example usage:
-------------
```python
from video_gen import VideoGenerator, VideoPreferences

# Initialize video generator
generator = VideoGenerator()

# Generate a video
video = await generator.create_video(
    topic="Python Programming Tips",
    style="educational",
    duration_minutes=5.0
)

# Upload to YouTube
await video.upload_to_youtube(
    title="Top Python Tips for 2024",
    description="Learn essential Python programming tips...",
    privacy="private"
)
```
"""

__version__ = "1.0.0"

# Core components
from .core.config import *
from .core.errors import ErrorHandler, VideoProductionError
from .core.cache import CacheManager

# Main generators
from .generation import AudioGenerator, VisualGenerator, ScriptGenerator

# Feature components
from .features import (
    CaptionSystem,
    VideoPreferences,
    WorkflowManager,
    BackgroundMusicManager,
    VideoAssembler,
    VideoSynchronizer,
    ConceptExtractor,
    YouTubeUploader
)

# Provider bases
from .providers import (
    BaseProvider,
    BaseImageProvider,
    BaseAudioProvider,
    BaseLLMProvider
)

__all__ = [
    # Version
    '__version__',
    
    # Core
    'ErrorHandler',
    'VideoProductionError',
    'CacheManager',
    
    # Generators
    'AudioGenerator',
    'VisualGenerator',
    'ScriptGenerator',
    
    # Features
    'CaptionSystem',
    'VideoPreferences',
    'WorkflowManager',
    'BackgroundMusicManager',
    'VideoAssembler',
    'VideoSynchronizer',
    'ConceptExtractor',
    'YouTubeUploader',
    
    # Provider bases
    'BaseProvider',
    'BaseImageProvider',
    'BaseAudioProvider',
    'BaseLLMProvider',
] 