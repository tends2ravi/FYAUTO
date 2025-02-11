"""
Caption generation and management module.
"""
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import asyncio

from .error_handler import ErrorHandler, VideoProductionError

class CaptionGenerator:
    """Handles caption generation and overlay for videos."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.font_path = str(Path(__file__).parent.parent / "assets" / "fonts" / "OpenSans-Regular.ttf")
        self.default_font_size = 32
        self.default_color = (255, 255, 255)  # White
        self.default_stroke_color = (0, 0, 0)  # Black
        self.default_stroke_width = 2
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="CAPTION_ERROR")
    async def generate_captions(
        self,
        text: str,
        frame_size: tuple,
        font_size: Optional[int] = None,
        color: Optional[tuple] = None,
        stroke_color: Optional[tuple] = None,
        stroke_width: Optional[int] = None,
        position: str = "bottom"
    ) -> Image.Image:
        """
        Generate captions for a video frame.
        
        Args:
            text: Caption text
            frame_size: Size of the video frame (width, height)
            font_size: Font size (defaults to self.default_font_size)
            color: Text color (RGB tuple, defaults to white)
            stroke_color: Stroke color (RGB tuple, defaults to black)
            stroke_width: Stroke width (defaults to 2)
            position: Caption position ("top", "bottom", "center")
            
        Returns:
            PIL Image with the caption
        """
        try:
            # Set defaults
            font_size = font_size or self.default_font_size
            color = color or self.default_color
            stroke_color = stroke_color or self.default_stroke_color
            stroke_width = stroke_width or self.default_stroke_width
            
            # Run image operations in thread pool
            loop = asyncio.get_event_loop()
            
            # Create transparent image
            caption_img = await loop.run_in_executor(
                None,
                lambda: Image.new("RGBA", frame_size, (0, 0, 0, 0))
            )
            draw = ImageDraw.Draw(caption_img)
            
            # Load font
            try:
                font = await loop.run_in_executor(
                    None,
                    lambda: ImageFont.truetype(self.font_path, font_size)
                )
            except Exception as e:
                logger.warning(f"Failed to load custom font: {str(e)}")
                font = ImageFont.load_default()
            
            # Calculate text size and position
            text_bbox = await loop.run_in_executor(
                None,
                lambda: draw.textbbox((0, 0), text, font=font)
            )
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Calculate position
            x = (frame_size[0] - text_width) // 2
            if position == "top":
                y = text_height
            elif position == "bottom":
                y = frame_size[1] - text_height * 2
            else:  # center
                y = (frame_size[1] - text_height) // 2
            
            # Draw text with stroke
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    await loop.run_in_executor(
                        None,
                        lambda: draw.text(
                            (x + dx, y + dy),
                            text,
                            font=font,
                            fill=stroke_color
                        )
                    )
            
            # Draw main text
            await loop.run_in_executor(
                None,
                lambda: draw.text(
                    (x, y),
                    text,
                    font=font,
                    fill=color
                )
            )
            
            return caption_img
            
        except Exception as e:
            logger.error(f"Error generating captions: {str(e)}")
            raise VideoProductionError(
                "Caption generation failed",
                "CAPTION_ERROR",
                {"error": str(e)}
            )
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="CAPTION_ERROR")
    async def overlay_captions(
        self,
        frame: np.ndarray,
        captions: Image.Image
    ) -> np.ndarray:
        """
        Overlay captions on a video frame.
        
        Args:
            frame: Video frame as numpy array
            captions: PIL Image with captions
            
        Returns:
            Frame with overlaid captions
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Convert PIL image to numpy array in thread pool
            caption_array = await loop.run_in_executor(
                None,
                lambda: np.array(captions)
            )
            
            # Extract alpha channel
            alpha = caption_array[:, :, 3] / 255.0
            
            # Remove alpha channel from caption array
            caption_array = caption_array[:, :, :3]
            
            # Blend captions with frame using alpha channel
            for c in range(3):
                frame[:, :, c] = frame[:, :, c] * (1 - alpha) + caption_array[:, :, c] * alpha
            
            return frame.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error overlaying captions: {str(e)}")
            raise VideoProductionError(
                "Caption overlay failed",
                "CAPTION_ERROR",
                {"error": str(e)}
            ) 