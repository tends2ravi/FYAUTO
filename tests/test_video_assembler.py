"""
Tests for the video assembly system.
"""
import pytest
from pathlib import Path
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip
from src.video_assembler import VideoAssembler
from src.error_handler import ErrorHandler

@pytest.mark.asyncio
class TestVideoAssembler:
    """Test suite for VideoAssembler."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler()
    
    @pytest.fixture
    def assembler(self, error_handler):
        """Create video assembler instance."""
        return VideoAssembler()
    
    @pytest.fixture
    def sample_video_clip(self, test_dir):
        """Create a sample video clip for testing."""
        # Create a simple color clip
        clip = ImageClip(np.zeros((100, 100, 3), dtype=np.uint8))
        clip = clip.set_duration(5)
        
        # Save to file
        video_path = test_dir / "test_clip.mp4"
        clip.write_videofile(str(video_path), fps=30)
        
        yield video_path
        
        # Cleanup
        if video_path.exists():
            video_path.unlink()
    
    @pytest.fixture
    def sample_audio_files(self, test_dir):
        """Create sample audio files for testing."""
        from scipy.io import wavfile
        
        audio_files = {}
        sections = ["hook", "section1", "section2", "call_to_action"]
        
        for section in sections:
            # Create simple audio data
            sample_rate = 44100
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            # Save to file
            audio_path = test_dir / f"{section}.wav"
            wavfile.write(str(audio_path), sample_rate, audio_data)
            audio_files[section] = audio_path
        
        yield audio_files
        
        # Cleanup
        for path in audio_files.values():
            if path.exists():
                path.unlink()
    
    async def test_create_video(
        self,
        assembler,
        sample_video_clip,
        sample_audio_files,
        test_dir
    ):
        """Test complete video creation process."""
        script_data = {
            "title": "Test Video",
            "sections": [
                {"title": "Section 1", "content": "Content 1"},
                {"title": "Section 2", "content": "Content 2"}
            ]
        }
        
        visual_files = {
            "title": [sample_video_clip],
            "Section 1": [sample_video_clip],
            "Section 2": [sample_video_clip]
        }
        
        output_path = test_dir / "final_video.mp4"
        
        result = await assembler.create_video(
            script_data=script_data,
            audio_files=sample_audio_files,
            visual_files=visual_files,
            output_path=output_path
        )
        
        assert result.exists()
        assert result.suffix == ".mp4"
        
        # Verify video properties
        video = VideoFileClip(str(result))
        assert video.duration > 0
        assert video.size == (1920, 1080)  # Default resolution
        video.close()
    
    async def test_process_sections(
        self,
        assembler,
        sample_video_clip,
        sample_audio_files
    ):
        """Test section processing."""
        script_data = {
            "sections": [
                {"title": "Section 1", "content": "Content 1"},
                {"title": "Section 2", "content": "Content 2"}
            ]
        }
        
        visual_files = {
            "Section 1": [sample_video_clip],
            "Section 2": [sample_video_clip]
        }
        
        clips = await assembler._process_sections(
            script_data=script_data,
            audio_files=sample_audio_files,
            visual_files=visual_files,
            resolution=(1920, 1080),
            fps=30
        )
        
        assert len(clips) == len(script_data["sections"])
        assert all(isinstance(clip, VideoFileClip) for clip in clips)
    
    async def test_create_section_clip(
        self,
        assembler,
        sample_video_clip,
        sample_audio_files
    ):
        """Test creation of a single section clip."""
        clip = await assembler._create_section_clip(
            audio_path=sample_audio_files["section1"],
            image_paths=[sample_video_clip],
            resolution=(1920, 1080),
            section_title="Test Section",
            fps=30
        )
        
        assert isinstance(clip, VideoFileClip)
        assert clip.duration > 0
        assert clip.size == (1920, 1080)
    
    async def test_load_audio(self, assembler, sample_audio_files):
        """Test audio loading."""
        audio = await assembler._load_audio(sample_audio_files["section1"])
        
        assert isinstance(audio, AudioFileClip)
        assert audio.duration > 0
    
    async def test_process_images(
        self,
        assembler,
        sample_video_clip,
        test_dir
    ):
        """Test image processing."""
        # Create test images
        image_paths = []
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            path = test_dir / f"test_image_{i}.png"
            img.save(path)
            image_paths.append(path)
        
        clips = await assembler._process_images(
            image_paths=image_paths,
            resolution=(1920, 1080),
            total_duration=10.0,
            fps=30
        )
        
        assert len(clips) == len(image_paths)
        assert all(isinstance(clip, ImageClip) for clip in clips)
        
        # Cleanup
        for path in image_paths:
            path.unlink()
    
    async def test_enhance_image(self, assembler, test_dir):
        """Test image enhancement."""
        # Create test image
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        path = test_dir / "test_image.png"
        img.save(path)
        
        enhanced = await assembler._enhance_image(
            image_path=path,
            resolution=(1920, 1080)
        )
        
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape[:2] == (1080, 1920)
        
        path.unlink()
    
    def test_add_motion_effect(self, assembler):
        """Test motion effect addition."""
        clip = ImageClip(np.zeros((100, 100, 3), dtype=np.uint8))
        
        with_motion = assembler._add_motion_effect(clip)
        assert isinstance(with_motion, ImageClip)
        assert hasattr(with_motion, 'fx')
    
    async def test_write_video_with_progress(
        self,
        assembler,
        sample_video_clip,
        test_dir
    ):
        """Test video writing with progress tracking."""
        clip = VideoFileClip(str(sample_video_clip))
        output_path = test_dir / "output_video.mp4"
        
        await assembler._write_video_with_progress(
            video=clip,
            output_path=output_path,
            fps=30
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        clip.close()
        output_path.unlink()
    
    async def test_cleanup_resources(
        self,
        assembler,
        sample_video_clip
    ):
        """Test resource cleanup."""
        clips = [VideoFileClip(str(sample_video_clip))]
        
        await assembler._cleanup_resources(clips)
        
        # Verify clips are closed
        assert all(not clip.running for clip in clips) 