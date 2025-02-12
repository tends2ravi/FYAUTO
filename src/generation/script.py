"""
Script generation module with caching and fallback support.
"""
# Standard library imports
from typing import Dict, Optional, Any, List, TypeVar, cast
from pathlib import Path
import json
import asyncio
import tempfile
import os
import hashlib
import pickle
from datetime import timedelta

# Third-party imports
import aiohttp
import redis
from loguru import logger

# Local imports
from ..core.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    OUTPUT_DIR
)
from ..core.errors import ErrorHandler, APIError, ValidationError
from ..features.preferences import FormatSettings
from ..providers.llm import LLMProviderManager

# Type variables
T = TypeVar('T', bound=Dict[str, Any])

class ScriptGenerator:
    """Handles script generation with caching and fallback support."""
    
    def __init__(
        self,
        error_handler: Optional[ErrorHandler] = None,
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the script generator.
        
        Args:
            error_handler: Optional error handler for managing errors
            output_dir: Optional custom output directory
        """
        self.error_handler = error_handler or ErrorHandler()
        self.llm_manager = LLMProviderManager(self.error_handler)
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False  # Keep binary for pickle
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
            self.redis_client = None
    
    def _get_cache_key(self, prompt: str) -> str:
        """
        Generate cache key from prompt.
        
        Args:
            prompt: Input prompt to hash
            
        Returns:
            Cache key string
        """
        return f"script_cache:{hashlib.sha256(prompt.encode()).hexdigest()}"
    
    async def _get_from_cache(self, prompt: str) -> Optional[T]:
        """
        Try to get script from cache.
        
        Args:
            prompt: Original generation prompt
            
        Returns:
            Cached script data if available, None otherwise
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(prompt)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                logger.info("Found script in cache")
                return cast(T, pickle.loads(cached_data))
            
            return None
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None
    
    async def _save_to_cache(self, prompt: str, script_data: T) -> None:
        """
        Save script to cache.
        
        Args:
            prompt: Original generation prompt
            script_data: Script data to cache
        """
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(prompt)
            self.redis_client.setex(
                cache_key,
                timedelta(hours=24),  # Cache for 24 hours
                pickle.dumps(script_data)
            )
            logger.info("Saved script to cache")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    async def generate_script(
        self,
        topic: str,
        style: str = "informative",
        duration_minutes: float = 5.0,
        target_audience: str = "general",
        additional_context: Optional[Dict[str, Any]] = None,
        format_settings: Optional[FormatSettings] = None
    ) -> T:
        """
        Generate a video script with caching and fallback support.
        
        Args:
            topic: Main topic of the video
            style: Writing style (informative, entertaining, educational)
            duration_minutes: Target video duration in minutes
            target_audience: Target audience for the video
            additional_context: Additional context or requirements
            format_settings: Video format settings
            
        Returns:
            Dictionary containing:
                - script: The generated script text
                - sections: List of script sections
                - metadata: Additional metadata about the script
                
        Raises:
            ValidationError: If the generated script is invalid
            APIError: If all LLM providers fail
        """
        prompt = self._create_prompt(
            topic, style, duration_minutes, target_audience, additional_context, format_settings
        )
        
        # Try to get from cache first
        cached_script = await self._get_from_cache(prompt)
        if cached_script:
            return cached_script
        
        try:
            # Generate script using LLM provider manager
            response = await self.llm_manager.generate_text(prompt)
            script_data = self._process_response(response)
            
            # Save to cache
            await self._save_to_cache(prompt, script_data)
            
            # Save the script to a file for inspection
            script_file = self.output_dir / "generated_script.json"
            with open(script_file, "w", encoding="utf-8") as f:
                json.dump(script_data, f, indent=2)
            logger.info(f"Saved generated script to: {script_file}")
            
            return script_data
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            raise
    
    def _create_prompt(
        self,
        topic: str,
        style: str,
        duration_minutes: float,
        target_audience: str,
        additional_context: Optional[Dict[str, Any]],
        format_settings: Optional[FormatSettings]
    ) -> str:
        """Create a detailed prompt for the AI."""
        context = additional_context or {}
        
        # Add format settings to context if provided
        if format_settings:
            context.update({
                "format_settings": {
                    "duration_range": {
                        "min": format_settings.duration_range.min,
                        "max": format_settings.duration_range.max,
                        "recommended": format_settings.duration_range.recommended
                    },
                    "resolution": {
                        "width": format_settings.resolution.width,
                        "height": format_settings.resolution.height
                    },
                    "fps": format_settings.fps,
                    "bitrate": format_settings.bitrate,
                    "codec": format_settings.codec
                }
            })
        
        prompt = f"""You are a professional YouTube script writer. Create a script for a video about {topic}.
        
        Style: {style}
        Target Duration: {duration_minutes} minutes
        Target Audience: {target_audience}
        
        Additional Requirements:
        - Include an engaging hook at the beginning
        - Break down complex topics into simple explanations
        - Include clear section transitions
        - End with a call to action
        
        Additional Context:
        {json.dumps(context, indent=2)}
        
        Format your response EXACTLY as a JSON object with the following structure:
        {{
            "title": "Video title",
            "hook": "Opening hook",
            "scenes": [
                {{
                    "id": "scene_1",
                    "title": "Scene title",
                    "content": "Scene content",
                    "duration": 60.0,
                    "visuals": [
                        {{
                            "description": "Visual description",
                            "duration": 20.0
                        }}
                    ]
                }}
            ],
            "call_to_action": "Call to action text",
            "metadata": {{
                "estimated_duration": {duration_minutes * 60},
                "key_points": ["point 1", "point 2"],
                "target_keywords": ["keyword1", "keyword2"]
            }}
        }}
        
        Make sure to format the response as valid JSON with all fields present. Each scene should have appropriate duration based on the total video length of {duration_minutes} minutes.
        """
        return prompt
    
    def _process_response(self, response: str) -> Dict:
        """Process and validate the API response."""
        try:
            # Extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            script_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["title", "hook", "scenes", "call_to_action", "metadata"]
            missing_fields = [field for field in required_fields if field not in script_data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in response: {missing_fields}")
            
            # Validate scene structure
            for scene in script_data["scenes"]:
                required_scene_fields = ["id", "title", "content", "duration", "visuals"]
                missing_scene_fields = [field for field in required_scene_fields if field not in scene]
                
                if missing_scene_fields:
                    raise ValueError(f"Missing required fields in scene: {missing_scene_fields}")
                
                # Ensure duration is a float
                scene["duration"] = float(scene["duration"])
                
                # Validate visuals
                for visual in scene["visuals"]:
                    required_visual_fields = ["description", "duration"]
                    missing_visual_fields = [field for field in required_visual_fields if field not in visual]
                    
                    if missing_visual_fields:
                        raise ValueError(f"Missing required fields in visual: {missing_visual_fields}")
                    
                    # Ensure duration is a float
                    visual["duration"] = float(visual["duration"])
            
            return script_data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing response: {str(e)}")
            raise ValueError("Invalid response format") 