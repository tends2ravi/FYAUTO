"""
Video preferences and guidelines management module.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Union, NamedTuple
from loguru import logger

from . import config

class DurationRange(NamedTuple):
    """Duration range settings."""
    min: float
    max: float
    recommended: float

class Resolution(NamedTuple):
    """Resolution settings."""
    width: int
    height: int

class FormatSettings(NamedTuple):
    """Format settings."""
    duration_range: DurationRange
    resolution: Resolution
    fps: int = 30  # Default to 30 fps
    bitrate: str = "6M"  # Default to 6Mbps
    codec: str = "h264"  # Default to h264

class VideoPreferences:
    """Handles video format preferences and content guidelines."""
    
    def __init__(self):
        self.guidelines_path = config.BASE_DIR / "configs" / "video_guidelines.json"
        self.guidelines = self._load_guidelines()
        
        # Default format settings
        self.default_settings = {
            "youtube": {
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
        
        # Ensure all format settings have required fields
        for format_type in self.default_settings:
            settings = self.default_settings[format_type]
            if "fps" not in settings:
                settings["fps"] = 30
            if "bitrate" not in settings:
                settings["bitrate"] = "4M"
            if "codec" not in settings:
                settings["codec"] = "h264"
    
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
    
    def _create_format_settings(self, settings_dict: Dict) -> FormatSettings:
        """Create FormatSettings from dictionary."""
        try:
            # Ensure all required fields are present
            required_fields = ["duration_range", "resolution", "fps", "bitrate", "codec"]
            missing_fields = [field for field in required_fields if field not in settings_dict]
            if missing_fields:
                logger.warning(f"Missing fields in format settings: {missing_fields}")
                settings_dict = self.default_settings["youtube"]
            
            duration_range = DurationRange(
                min=float(settings_dict["duration_range"]["min"]),
                max=float(settings_dict["duration_range"]["max"]),
                recommended=float(settings_dict["duration_range"]["recommended"])
            )
            
            resolution = Resolution(
                width=int(settings_dict["resolution"]["width"]),
                height=int(settings_dict["resolution"]["height"])
            )
            
            return FormatSettings(
                duration_range=duration_range,
                resolution=resolution,
                fps=int(settings_dict["fps"]),
                bitrate=str(settings_dict["bitrate"]),
                codec=str(settings_dict["codec"])
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Error creating format settings: {e}")
            # Return default settings for YouTube
            return self._create_format_settings(self.default_settings["youtube"])
    
    def get_format_settings(
        self,
        format_type: str = "youtube",
        niche: Optional[str] = None
    ) -> FormatSettings:
        """
        Get format settings for the specified video type and niche.
        
        Args:
            format_type: Type of video format (youtube/shorts)
            niche: Specific content niche (e.g., educational_tech, dodstory)
            
        Returns:
            FormatSettings object containing format settings
        """
        try:
            # Get default settings for the format type
            default_format_settings = self.default_settings.get(format_type, self.default_settings["youtube"])
            
            # Get niche guidelines, fallback to default if not found
            niche_guidelines = self.guidelines.get(niche or "default", self.guidelines.get("default", {}))
            
            # Get format settings from guidelines, fallback to default
            format_settings_dict = niche_guidelines.get("formats", {}).get(
                format_type,
                default_format_settings
            )
            
            return self._create_format_settings(format_settings_dict)
            
        except Exception as e:
            logger.error(f"Error getting format settings: {e}")
            # Return default settings for the format type
            return self._create_format_settings(self.default_settings[format_type])
    
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
        niche_guidelines = self.guidelines.get(niche or "default", self.guidelines.get("default", {}))
        
        return {
            "script": niche_guidelines.get("script_guidelines", {}),
            "visual": niche_guidelines.get("visual_guidelines", {}),
            "audio": niche_guidelines.get("audio_guidelines", {})
        }
    
    def validate_duration(
        self,
        duration: float,
        format_settings: FormatSettings
    ) -> bool:
        """
        Validate if the specified duration is within acceptable range.
        
        Args:
            duration: Video duration in seconds
            format_settings: Format settings object
            
        Returns:
            True if duration is valid, False otherwise
        """
        return format_settings.duration_range.min <= duration <= format_settings.duration_range.max
    
    def get_resolution(
        self,
        format_type: str = "youtube",
        niche: Optional[str] = None
    ) -> Resolution:
        """
        Get recommended resolution for the specified format.
        
        Args:
            format_type: Type of video format
            niche: Specific content niche
            
        Returns:
            Resolution object
        """
        format_settings = self.get_format_settings(format_type, niche)
        return format_settings.resolution
    
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
        return guidelines.get("visual", {}).get("negative_prompts", []) 