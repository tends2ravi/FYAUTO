"""
Video preferences and guidelines management module.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Union
from loguru import logger

from . import config

class VideoPreferences:
    """Handles video format preferences and content guidelines."""
    
    def __init__(self):
        self.guidelines_path = config.BASE_DIR / "configs" / "video_guidelines.json"
        self.guidelines = self._load_guidelines()
        
    def _load_guidelines(self) -> Dict:
        """Load video guidelines from JSON file."""
        try:
            with open(self.guidelines_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Guidelines file not found: {self.guidelines_path}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in guidelines file: {self.guidelines_path}")
            return {}
    
    def get_format_settings(
        self,
        format_type: str = "youtube",
        niche: Optional[str] = None
    ) -> Dict:
        """
        Get format settings for the specified video type and niche.
        
        Args:
            format_type: Type of video format (youtube/shorts)
            niche: Specific content niche (e.g., educational_tech, dodstory)
            
        Returns:
            Dictionary containing format settings
        """
        # Get niche guidelines, fallback to default if not found
        niche_guidelines = self.guidelines.get(niche or "default", self.guidelines["default"])
        
        # Get format settings
        format_settings = niche_guidelines["formats"].get(
            format_type,
            self.guidelines["default"]["formats"][format_type]
        )
        
        return format_settings
    
    def get_content_guidelines(
        self,
        niche: Optional[str] = None
    ) -> Dict:
        """
        Get content guidelines for the specified niche.
        
        Args:
            niche: Specific content niche
            
        Returns:
            Dictionary containing content guidelines
        """
        # Get niche guidelines, fallback to default if not found
        niche_guidelines = self.guidelines.get(niche or "default", self.guidelines["default"])
        
        return {
            "script": niche_guidelines["script_guidelines"],
            "visual": niche_guidelines["visual_guidelines"],
            "audio": niche_guidelines["audio_guidelines"]
        }
    
    def validate_duration(
        self,
        duration: float,
        format_type: str = "youtube",
        niche: Optional[str] = None
    ) -> bool:
        """
        Validate if the specified duration is within acceptable range.
        
        Args:
            duration: Video duration in seconds
            format_type: Type of video format
            niche: Specific content niche
            
        Returns:
            True if duration is valid, False otherwise
        """
        format_settings = self.get_format_settings(format_type, niche)
        duration_range = format_settings["duration_range"]
        
        return duration_range["min"] <= duration <= duration_range["max"]
    
    def get_resolution(
        self,
        format_type: str = "youtube",
        niche: Optional[str] = None
    ) -> tuple:
        """
        Get recommended resolution for the specified format.
        
        Args:
            format_type: Type of video format
            niche: Specific content niche
            
        Returns:
            Tuple of (width, height)
        """
        format_settings = self.get_format_settings(format_type, niche)
        resolution = format_settings["resolution"]
        
        return (resolution["width"], resolution["height"])
    
    def get_negative_prompts(
        self,
        niche: Optional[str] = None
    ) -> list:
        """
        Get negative prompts for visual generation.
        
        Args:
            niche: Specific content niche
            
        Returns:
            List of negative prompts
        """
        guidelines = self.get_content_guidelines(niche)
        return guidelines["visual"]["negative_prompts"] 