"""
Tests for the unified caption system.
"""
import pytest
from pathlib import Path
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.caption_system import CaptionSystem
from src.error_handler import VideoProductionError

@pytest.mark.asyncio
class TestCaptionSystem:
    """Test suite for CaptionSystem."""
    
    async def test_add_caption_basic(
        self,
        caption_system: CaptionSystem,
        test_video: Path
    ):
        """Test basic caption addition."""
        clip = VideoFileClip(str(test_video))
        text = "Test Caption"
        
        # Add caption
        result = await caption_system.add_caption(
            clip=clip,
            text=text,
            duration=1.0
        )
        
        assert result is not None
        assert result.duration == clip.duration
        
        # Clean up
        clip.close()
        result.close()
    
    async def test_add_caption_with_style(
        self,
        caption_system: CaptionSystem,
        test_video: Path
    ):
        """Test caption addition with custom style."""
        clip = VideoFileClip(str(test_video))
        text = "Styled Caption"
        
        style = {
            "font_size": 48,
            "font_color": (255, 0, 0),  # Red
            "stroke_color": (255, 255, 255),  # White
            "stroke_width": 3
        }
        
        # Add styled caption
        result = await caption_system.add_caption(
            clip=clip,
            text=text,
            style=style,
            duration=1.0
        )
        
        assert result is not None
        assert result.duration == clip.duration
        
        # Clean up
        clip.close()
        result.close()
    
    async def test_add_caption_with_animation(
        self,
        caption_system: CaptionSystem,
        test_video: Path
    ):
        """Test caption addition with animations."""
        clip = VideoFileClip(str(test_video))
        text = "Animated Caption"
        
        for animation in ["fade", "slide", "scale"]:
            # Add animated caption
            result = await caption_system.add_caption(
                clip=clip,
                text=text,
                animation=animation,
                duration=1.0
            )
            
            assert result is not None
            assert result.duration == clip.duration
            
            # Clean up
            result.close()
        
        clip.close()
    
    async def test_add_subtitles(
        self,
        caption_system: CaptionSystem,
        test_video: Path
    ):
        """Test subtitle addition."""
        clip = VideoFileClip(str(test_video))
        
        subtitles = [
            {"text": "First subtitle", "start_time": 0.0, "duration": 1.0},
            {"text": "Second subtitle", "start_time": 1.0, "duration": 1.0}
        ]
        
        # Add subtitles
        result = await caption_system.add_subtitles(
            clip=clip,
            subtitles=subtitles
        )
        
        assert result is not None
        assert result.duration == clip.duration
        
        # Clean up
        clip.close()
        result.close()
    
    async def test_add_subtitles_with_animation(
        self,
        caption_system: CaptionSystem,
        test_video: Path
    ):
        """Test subtitle addition with animation."""
        clip = VideoFileClip(str(test_video))
        
        subtitles = [
            {"text": "Animated subtitle 1", "start_time": 0.0, "duration": 1.0},
            {"text": "Animated subtitle 2", "start_time": 1.0, "duration": 1.0}
        ]
        
        # Add animated subtitles
        result = await caption_system.add_subtitles(
            clip=clip,
            subtitles=subtitles,
            animation="fade"
        )
        
        assert result is not None
        assert result.duration == clip.duration
        
        # Clean up
        clip.close()
        result.close()
    
    async def test_error_handling(
        self,
        caption_system: CaptionSystem,
        test_video: Path
    ):
        """Test error handling in caption system."""
        clip = VideoFileClip(str(test_video))
        
        # Test with invalid text
        with pytest.raises(VideoProductionError):
            await caption_system.add_caption(
                clip=clip,
                text=None,  # type: ignore
                duration=1.0
            )
        
        # Test with invalid style
        with pytest.raises(VideoProductionError):
            await caption_system.add_caption(
                clip=clip,
                text="Test",
                style={"font_size": "invalid"},  # type: ignore
                duration=1.0
            )
        
        # Test with invalid animation
        with pytest.raises(VideoProductionError):
            await caption_system.add_caption(
                clip=clip,
                text="Test",
                animation="invalid_animation",
                duration=1.0
            )
        
        # Clean up
        clip.close()
    
    async def test_position_calculation(
        self,
        caption_system: CaptionSystem
    ):
        """Test caption position calculation."""
        text_size = (100, 50)
        clip_size = (640, 480)
        
        # Test center position
        pos = caption_system._calculate_position(
            ("center", "center"),
            text_size,
            clip_size
        )
        assert pos == (270, 215)  # (640-100)/2, (480-50)/2
        
        # Test bottom-right position
        pos = caption_system._calculate_position(
            ("right", "bottom"),
            text_size,
            clip_size
        )
        assert pos == (530, 420)  # 640-100-10, 480-50-10
        
        # Test top-left position
        pos = caption_system._calculate_position(
            ("left", "top"),
            text_size,
            clip_size
        )
        assert pos == (10, 10)
    
    async def test_cleanup(
        self,
        caption_system: CaptionSystem,
        temp_dir: Path
    ):
        """Test cleanup of temporary files."""
        # Create some temp files
        temp_files = [
            temp_dir / f"temp_{i}.txt"
            for i in range(3)
        ]
        
        for temp_file in temp_files:
            temp_file.touch()
            caption_system._temp_files.append(temp_file)
        
        # Run cleanup
        await caption_system.cleanup()
        
        # Verify files are deleted
        for temp_file in temp_files:
            assert not temp_file.exists()
        
        assert len(caption_system._temp_files) == 0 