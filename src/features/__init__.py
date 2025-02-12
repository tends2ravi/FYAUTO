"""
High-level features for video production system.
Includes workflow management, video assembly, and various enhancement features.
"""

from .captions import CaptionSystem
from .preferences import VideoPreferences
from .workflow import WorkflowManager
from .music import BackgroundMusicManager
from .assembler import VideoAssembler
from .synchronizer import VideoSynchronizer
from .concepts import ConceptExtractor
from .uploader import YouTubeUploader

__all__ = [
    'CaptionSystem',
    'VideoPreferences',
    'WorkflowManager',
    'BackgroundMusicManager',
    'VideoAssembler',
    'VideoSynchronizer',
    'ConceptExtractor',
    'YouTubeUploader',
] 