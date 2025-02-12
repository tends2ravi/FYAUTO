"""
Test suite for video production system.
"""
import pytest
import asyncio
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import shutil

from src.video_synchronizer import VideoSynchronizer
from src.visual_effects import VisualEffects
from src.background_music import BackgroundMusic
from src.caption_system import CaptionSystem
from src.error_handler import ErrorHandler

@pytest.fixture
def error_handler():
    """Create error handler instance."""
    return ErrorHandler()

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_video(temp_dir):
    """Create sample video for testing."""
    duration = 5.0
    fps = 30
    size = (640, 480)
    
    # Create random frames
    frames = []
    for _ in range(int(duration * fps)):
        frame = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        frames.append(frame)
    
    # Save as video file
    video_path = temp_dir / "sample.mp4"
    clip = VideoFileClip(frames, fps=fps)
    clip.write_videofile(str(video_path))
    
    return video_path

@pytest.fixture
def sample_audio(temp_dir):
    """Create sample audio for testing."""
    duration = 5.0
    sr = 44100
    
    # Create simple sine wave
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    # Save as audio file
    audio_path = temp_dir / "sample.wav"
    with open(audio_path, "wb") as f:
        f.write(audio.tobytes())
    
    return audio_path

class TestVideoSynchronizer:
    """Test video synchronizer functionality."""
    
    @pytest.mark.asyncio
    async def test_synchronize_content(self, error_handler, sample_video, sample_audio):
        """Test basic content synchronization."""
        synchronizer = VideoSynchronizer(error_handler)
        
        # Load test files
        video_clip = VideoFileClip(str(sample_video))
        audio_files = {"main": sample_audio}
        timing_data = {"main": 5.0}
        
        # Synchronize content
        result = await synchronizer.synchronize_content(
            [video_clip],
            audio_files,
            timing_data,
            5.0
        )
        
        # Verify result
        assert result is not None
        assert abs(result.duration - 5.0) < 0.1
        
        await synchronizer.cleanup()
    
    @pytest.mark.asyncio
    async def test_audio_analysis(self, error_handler, sample_audio):
        """Test audio analysis functionality."""
        synchronizer = VideoSynchronizer(error_handler)
        
        # Analyze audio
        analysis = await synchronizer._analyze_audio({"main": sample_audio})
        
        # Verify analysis results
        assert "main" in analysis
        assert "tempo" in analysis["main"]
        assert "beat_times" in analysis["main"]
        assert "energy" in analysis["main"]
        
        await synchronizer.cleanup()

class TestVisualEffects:
    """Test visual effects functionality."""
    
    @pytest.mark.asyncio
    async def test_ken_burns_effect(self, error_handler, sample_video):
        """Test Ken Burns effect."""
        effects = VisualEffects(error_handler)
        
        # Load test video
        clip = VideoFileClip(str(sample_video))
        
        # Apply effect
        result = await effects.apply_ken_burns(
            clip,
            zoom_range=(1.0, 1.5),
            direction="in"
        )
        
        # Verify result
        assert result is not None
        assert result.duration == clip.duration
        
        await effects.cleanup()
    
    @pytest.mark.asyncio
    async def test_motion_effect(self, error_handler, sample_video):
        """Test motion effect."""
        effects = VisualEffects(error_handler)
        
        # Load test video
        clip = VideoFileClip(str(sample_video))
        
        # Apply effect
        result = await effects.apply_motion_effect(
            clip,
            effect_type="shake",
            intensity=1.0
        )
        
        # Verify result
        assert result is not None
        assert result.duration == clip.duration
        
        await effects.cleanup()

class TestBackgroundMusic:
    """Test background music functionality."""
    
    @pytest.mark.asyncio
    async def test_select_music(self, error_handler, temp_dir):
        """Test music selection."""
        music = BackgroundMusic(error_handler)
        
        # Create test music library
        music_dir = temp_dir / "music"
        music_dir.mkdir()
        shutil.copy(str(sample_audio), str(music_dir / "test.mp3"))
        
        # Select music
        result = await music.select_music(
            mood="upbeat",
            duration=5.0,
            music_library_path=music_dir
        )
        
        # Verify result
        assert result is not None
        assert abs(result.duration - 5.0) < 0.1
        
        await music.cleanup()
    
    @pytest.mark.asyncio
    async def test_mix_audio(self, error_handler, sample_audio):
        """Test audio mixing."""
        music = BackgroundMusic(error_handler)
        
        # Load test audio
        main_audio = AudioFileClip(str(sample_audio))
        bg_music = AudioFileClip(str(sample_audio))
        
        # Mix audio
        result = await music.mix_audio(
            main_audio,
            bg_music,
            music_volume=0.3
        )
        
        # Verify result
        assert result is not None
        assert result.duration == main_audio.duration
        
        await music.cleanup()

class TestCaptionSystem:
    """Test caption system functionality."""
    
    @pytest.mark.asyncio
    async def test_add_caption(self, error_handler, sample_video):
        """Test adding caption."""
        captions = CaptionSystem(error_handler)
        
        # Load test video
        clip = VideoFileClip(str(sample_video))
        
        # Add caption
        result = await captions.add_caption(
            clip,
            "Test Caption",
            position=("center", "bottom"),
            duration=2.0
        )
        
        # Verify result
        assert result is not None
        assert result.duration == clip.duration
        
        await captions.cleanup()
    
    @pytest.mark.asyncio
    async def test_add_animated_caption(self, error_handler, sample_video):
        """Test adding animated caption."""
        captions = CaptionSystem(error_handler)
        
        # Load test video
        clip = VideoFileClip(str(sample_video))
        
        # Add animated caption
        result = await captions.add_animated_caption(
            clip,
            "Test Caption",
            animation_type="fade",
            duration=2.0
        )
        
        # Verify result
        assert result is not None
        assert result.duration == clip.duration
        
        await captions.cleanup()

@pytest.mark.asyncio
async def test_full_pipeline(
    error_handler,
    sample_video,
    sample_audio,
    temp_dir
):
    """Test full video production pipeline."""
    # Initialize components
    synchronizer = VideoSynchronizer(error_handler)
    effects = VisualEffects(error_handler)
    music = BackgroundMusic(error_handler)
    captions = CaptionSystem(error_handler)
    
    try:
        # Load test files
        video_clip = VideoFileClip(str(sample_video))
        audio_files = {"main": sample_audio}
        timing_data = {"main": 5.0}
        
        # Step 1: Synchronize content
        result = await synchronizer.synchronize_content(
            [video_clip],
            audio_files,
            timing_data,
            5.0
        )
        assert result is not None
        
        # Step 2: Apply visual effects
        result = await effects.apply_ken_burns(result)
        assert result is not None
        
        # Step 3: Add background music
        bg_music = await music.select_music(
            duration=result.duration,
            music_library_path=temp_dir / "music"
        )
        if bg_music is not None:
            result = result.set_audio(
                await music.mix_audio(result.audio, bg_music)
            )
        
        # Step 4: Add captions
        result = await captions.add_caption(
            result,
            "Test Video",
            duration=2.0
        )
        assert result is not None
        
        # Save final video
        output_path = temp_dir / "output.mp4"
        result.write_videofile(str(output_path))
        
        # Verify output file exists and has correct duration
        assert output_path.exists()
        final_clip = VideoFileClip(str(output_path))
        assert abs(final_clip.duration - 5.0) < 0.1
        
    finally:
        # Cleanup
        await synchronizer.cleanup()
        await effects.cleanup()
        await music.cleanup()
        await captions.cleanup() 