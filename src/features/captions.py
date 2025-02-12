"""
Unified caption system for video production.
Handles text overlays, subtitles, and animated captions with style customization.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from loguru import logger
import asyncio
import tempfile
import json

from .error_handler import ErrorHandler, VideoProductionError

class CaptionSystem:
    """Unified system for video captions, subtitles, and text overlays."""
    
    # Default style settings
    DEFAULT_FONT_SIZE = 36
    DEFAULT_FONT_COLOR = (255, 255, 255)  # White
    DEFAULT_STROKE_COLOR = (0, 0, 0)  # Black
    DEFAULT_STROKE_WIDTH = 2
    DEFAULT_FONT_PATH = "assets/fonts/OpenSans-Regular.ttf"
    
    def __init__(self, error_handler: ErrorHandler):
        """Initialize the caption system.
        
        Args:
            error_handler: Error handler instance for retries and error management
        """
        self.error_handler = error_handler
        self.logger = logger.bind(context=self.__class__.__name__)
        self._temp_files: List[Path] = []
        
        # Load font
        try:
            font_path = Path(self.DEFAULT_FONT_PATH)
            self.font = ImageFont.truetype(str(font_path), self.DEFAULT_FONT_SIZE)
            logger.info(f"Loaded font from {font_path}")
        except Exception as e:
            logger.warning(f"Failed to load font: {e}, using default")
            self.font = ImageFont.load_default()
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="CAPTION_ERROR")
    async def add_caption(
        self,
        clip: VideoFileClip,
        text: str,
        position: Tuple[str, str] = ("center", "bottom"),
        duration: Optional[float] = None,
        start_time: float = 0.0,
        style: Optional[Dict] = None,
        animation: Optional[str] = None
    ) -> VideoFileClip:
        """Add caption to video clip with optional animation.
        
        Args:
            clip: Video clip to add caption to
            text: Caption text
            position: Tuple of (horizontal, vertical) position
            duration: Duration of caption (None for full clip duration)
            start_time: Start time of caption
            style: Dictionary of style parameters
            animation: Animation type ("fade", "slide", "scale", None)
            
        Returns:
            Video clip with caption
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Set default duration if not specified
            if duration is None:
                duration = clip.duration - start_time
            
            # Merge default and custom styles
            caption_style = self._get_default_style()
            if style:
                caption_style.update(style)
            
            # Create text clip in thread pool
            text_clip = await loop.run_in_executor(
                None,
                lambda: self._create_text_clip(
                    text,
                    caption_style,
                    clip.size,
                    duration
                )
            )
            
            # Apply animation if specified
            if animation:
                text_clip = await self._apply_animation(
                    text_clip,
                    animation,
                    duration,
                    clip.size
                )
            
            # Set position
            position_func = lambda t: self._calculate_position(
                position,
                text_clip.size,
                clip.size
            )
            text_clip = text_clip.set_position(position_func)
            
            # Set timing
            text_clip = text_clip.set_start(start_time)
            
            # Composite clips in thread pool
            final_clip = await loop.run_in_executor(
                None,
                lambda: CompositeVideoClip([clip, text_clip])
            )
            
            return final_clip
            
        except Exception as e:
            self.logger.error(f"Error adding caption: {str(e)}")
            raise VideoProductionError(
                "Caption addition failed",
                "CAPTION_ERROR",
                {"error": str(e)}
            )
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="SUBTITLE_ERROR")
    async def add_subtitles(
        self,
        clip: VideoFileClip,
        subtitles: List[Dict[str, Union[str, float]]],
        style: Optional[Dict] = None,
        animation: Optional[str] = None
    ) -> VideoFileClip:
        """Add subtitles to video clip with optional animation.
        
        Args:
            clip: Video clip to add subtitles to
            subtitles: List of subtitle dictionaries with text, start_time, and duration
            style: Dictionary of style parameters
            animation: Animation type for subtitles
            
        Returns:
            Video clip with subtitles
        """
        try:
            # Create text clips for each subtitle
            text_clips = []
            for subtitle in subtitles:
                text = subtitle["text"]
                start_time = subtitle["start_time"]
                duration = subtitle.get("duration", 2.0)
                
                # Add individual caption
                clip = await self.add_caption(
                    clip=clip,
                    text=text,
                    position=("center", "bottom"),
                    duration=duration,
                    start_time=start_time,
                    style=style,
                    animation=animation
                )
            
            return clip
            
        except Exception as e:
            self.logger.error(f"Error adding subtitles: {str(e)}")
            raise VideoProductionError(
                "Subtitle addition failed",
                "SUBTITLE_ERROR",
                {"error": str(e)}
            )
    
    async def _apply_animation(
        self,
        clip: TextClip,
        animation_type: str,
        duration: float,
        video_size: Tuple[int, int]
    ) -> TextClip:
        """Apply animation to text clip."""
        if animation_type == "fade":
            # Fade in and out
            clip = clip.fadein(0.5).fadeout(0.5)
            
        elif animation_type == "slide":
            # Slide from right to center
            w, h = video_size
            pos_func = lambda t: (
                w * (1 - t/duration) if t < duration/2
                else w/2 - clip.size[0]/2,
                h * 0.8
            )
            clip = clip.set_position(pos_func)
            
        elif animation_type == "scale":
            # Scale up from center
            def scale_transform(get_frame, t):
                frame = get_frame(t)
                progress = min(1, 2 * t/duration)
                scale = 0.5 + 0.5 * progress
                
                h, w = frame.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
                return cv2.warpAffine(frame, M, (w, h))
            
            clip = clip.fl(scale_transform)
        
        return clip
    
    def _get_default_style(self) -> Dict:
        """Get default style settings."""
        return {
            "font_size": self.DEFAULT_FONT_SIZE,
            "font_color": self.DEFAULT_FONT_COLOR,
            "stroke_color": self.DEFAULT_STROKE_COLOR,
            "stroke_width": self.DEFAULT_STROKE_WIDTH,
            "font": self.font
        }
    
    def _create_text_clip(
        self,
        text: str,
        style: Dict,
        clip_size: Tuple[int, int],
        duration: float
    ) -> TextClip:
        """Create text clip with style."""
        # Create transparent image for text
        img = Image.new("RGBA", clip_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), str(text), font=style["font"])
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center text
        x = (clip_size[0] - text_width) // 2
        y = (clip_size[1] - text_height) // 2
        
        # Draw text with stroke
        for dx in range(-style["stroke_width"], style["stroke_width"] + 1):
            for dy in range(-style["stroke_width"], style["stroke_width"] + 1):
                draw.text(
                    (x + dx, y + dy),
                    str(text),
                    font=style["font"],
                    fill=style["stroke_color"]
                )
        
        # Draw main text
        draw.text(
            (x, y),
            str(text),
            font=style["font"],
            fill=style["font_color"]
        )
        
        # Convert to TextClip
        text_clip = TextClip(
            img,
            duration=duration,
            transparent=True
        )
        
        return text_clip
    
    def _calculate_position(
        self,
        position: Tuple[str, str],
        text_size: Tuple[int, int],
        clip_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate text position."""
        h_pos, v_pos = position
        x, y = 0, 0
        
        # Horizontal position
        if h_pos == "left":
            x = 10
        elif h_pos == "center":
            x = (clip_size[0] - text_size[0]) // 2
        elif h_pos == "right":
            x = clip_size[0] - text_size[0] - 10
        
        # Vertical position
        if v_pos == "top":
            y = 10
        elif v_pos == "center":
            y = (clip_size[1] - text_size[1]) // 2
        elif v_pos == "bottom":
            y = clip_size[1] - text_size[1] - 10
        
        return (x, y)
    
    async def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                temp_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        self._temp_files.clear() 