"""
Tests for the video synchronization module.
"""
import pytest
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, CompositeVideoClip
from src.video_synchronizer import VideoSynchronizer
from src.error_handler import ErrorHandler

@pytest.mark.asyncio
class TestVideoSynchronizer:
    """Test suite for VideoSynchronizer."""
    
    @pytest.fixture
    def error_handler(self):
        """Provide error handler fixture."""
        return ErrorHandler()
    
    @pytest.fixture
    def synchronizer(self, error_handler):
        """Provide synchronizer fixture."""
        return VideoSynchronizer(error_handler)
    
    @pytest.fixture
    def sample_video_clip(self, test_dir):
        """Create a sample video clip."""
        # Create a simple color clip
        duration = 5.0
        fps = 30
        
        # Create a simple gradient clip
        clip = ColorClip(
            size=(100, 100),
            color=(255, 0, 0),
            duration=duration
        ).set_fps(fps)
        
        # Save to file
        output_path = Path(test_dir) / "test_clip.mp4"
        clip.write_videofile(str(output_path))
        
        video_clip = VideoFileClip(str(output_path))
        yield video_clip
        try:
            video_clip.close()
        except Exception as e:
            print(f"Error closing video clip: {e}")
    
    @pytest.fixture
    def sample_audio_files(self, test_dir):
        """Create sample audio files."""
        audio_files = {}
        
        # Create simple audio data
        sample_rate = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio for each section
        for section in ["intro", "main", "outro"]:
            freq = 440 if section == "main" else 220
            audio_data = np.sin(2 * np.pi * freq * t)
            
            # Save to file
            audio_path = test_dir / f"{section}_audio.wav"
            import soundfile as sf
            sf.write(str(audio_path), audio_data, sample_rate)
            
            audio_files[section] = audio_path
        
        return audio_files
    
    @pytest.mark.asyncio
    async def test_synchronize_content(
        self,
        synchronizer,
        sample_video_clip,
        sample_audio_files,
        test_dir
    ):
        """Test full content synchronization."""
        # Prepare test data
        clips = [sample_video_clip] * 3
        timing_data = {
            "intro": 5.0,
            "main": 5.0,
            "outro": 5.0
        }
        target_duration = 15.0
        
        # Run synchronization
        result = await synchronizer.synchronize_content(
            clips,
            sample_audio_files,
            timing_data,
            target_duration
        )
        
        assert isinstance(result, (VideoFileClip, CompositeVideoClip))
        assert abs(result.duration - target_duration) < 0.1
    
    @pytest.mark.asyncio
    async def test_audio_analysis(self, synchronizer, sample_audio_files):
        """Test audio analysis."""
        analysis = await synchronizer._analyze_audio(sample_audio_files)
        
        assert isinstance(analysis, dict)
        for section, section_data in analysis.items():
            assert isinstance(section_data, dict)
            assert "beat_times" in section_data
            assert "energy" in section_data
            assert "duration" in section_data
    
    @pytest.mark.asyncio
    async def test_optimize_clip_timing(
        self,
        synchronizer,
        sample_video_clip,
        sample_audio_files
    ):
        """Test clip timing optimization."""
        # Get audio analysis first
        audio_analysis = await synchronizer._analyze_audio(sample_audio_files)
        
        # Prepare test data
        clips = [sample_video_clip] * 3
        timing_data = {
            "intro": 4.0,
            "main": 5.0,
            "outro": 3.0
        }
        
        # Optimize timing
        optimized = await synchronizer._optimize_clip_timing(
            clips,
            audio_analysis,
            timing_data
        )
        
        assert len(optimized) == len(clips)
        for clip, target in zip(optimized, timing_data.values()):
            assert abs(clip.duration - target) < 0.1
    
    @pytest.mark.asyncio
    async def test_synchronize_clips(
        self,
        synchronizer,
        sample_video_clip,
        sample_audio_files
    ):
        """Test clip synchronization."""
        # Get audio analysis first
        audio_analysis = await synchronizer._analyze_audio(sample_audio_files)
        
        # Prepare clips
        clips = [sample_video_clip] * 3
        
        # Synchronize clips
        synchronized = await synchronizer._synchronize_clips(clips, audio_analysis)
        
        assert len(synchronized) == len(clips)
        assert all(isinstance(clip, VideoFileClip) for clip in synchronized)
    
    @pytest.mark.asyncio
    async def test_audio_based_transitions(
        self,
        synchronizer,
        sample_video_clip,
        sample_audio_files
    ):
        """Test audio-based transitions."""
        # Get audio analysis first
        audio_analysis = await synchronizer._analyze_audio(sample_audio_files)
        
        # Prepare clips
        clips = [sample_video_clip] * 3
        
        # Apply transitions
        result = await synchronizer._apply_audio_based_transitions(
            clips,
            audio_analysis
        )
        
        assert isinstance(result, (VideoFileClip, CompositeVideoClip))
        assert result.duration > 0
    
    @pytest.mark.asyncio
    async def test_duration_adjustment(self, synchronizer, sample_video_clip):
        """Test video duration adjustment."""
        target_duration = 3.0
        
        # Adjust duration
        adjusted = await synchronizer._adjust_duration(
            sample_video_clip,
            target_duration
        )
        
        assert isinstance(adjusted, VideoFileClip)
        assert abs(adjusted.duration - target_duration) < 0.1
    
    @pytest.mark.asyncio
    async def test_clip_speed_adjustment(self, synchronizer, sample_video_clip):
        """Test clip speed adjustment."""
        # Test with target duration
        adjusted = synchronizer._adjust_clip_speed(
            sample_video_clip,
            target_duration=2.5
        )
        assert abs(adjusted.duration - 2.5) < 0.1
        
        # Test with explicit speed factor
        adjusted = synchronizer._adjust_clip_speed(
            sample_video_clip,
            target_duration=None,
            speed_factor=2.0
        )
        assert abs(adjusted.duration - sample_video_clip.duration / 2.0) < 0.1
    
    @pytest.mark.asyncio
    async def test_optimal_start_time(self, synchronizer, sample_video_clip):
        """Test optimal start time calculation."""
        # Create sample beat times
        beat_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        # Find optimal start time
        start_time = synchronizer._find_optimal_start_time(
            sample_video_clip,
            beat_times
        )
        
        assert isinstance(start_time, float)
        assert 0.0 <= start_time <= sample_video_clip.duration
    
    @pytest.mark.asyncio
    async def test_optimal_duration_calculation(self, synchronizer):
        """Test optimal duration calculation."""
        analysis = {
            "tempo": 120,
            "beat_times": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            "energy": np.array([0.5] * 100),
            "duration": 2.0
        }
        target = 3.0
        
        duration = synchronizer._calculate_optimal_duration(analysis, target)
        
        assert isinstance(duration, float)
        assert duration > 0
    
    @pytest.mark.asyncio
    async def test_transition_duration_calculation(self, synchronizer):
        """Test transition duration calculation."""
        current = {"energy": np.array([0.5, 0.6, 0.7])}
        next_section = {"energy": np.array([0.6, 0.7, 0.8])}
        
        duration = synchronizer._calculate_transition_duration(
            current,
            next_section
        )
        
        assert isinstance(duration, float)
        assert duration > 0
    
    @pytest.mark.asyncio
    async def test_transition_type_selection(self, synchronizer):
        """Test transition type selection."""
        current = {
            "energy": np.array([0.5, 0.6, 0.7]),
            "tempo": 120
        }
        next_section = {
            "energy": np.array([0.6, 0.7, 0.8]),
            "tempo": 140
        }
        
        transition_type = synchronizer._select_transition_type(
            current,
            next_section
        )
        
        assert isinstance(transition_type, str)
        assert transition_type in synchronizer.TRANSITIONS 

    @pytest.fixture(autouse=True)
    def cleanup_clips(self):
        yield
        # Clean up any remaining clips
        try:
            for clip in self.clips if hasattr(self, 'clips') else []:
                try:
                    clip.close()
                except Exception as e:
                    print(f"Error closing clip: {e}")
        except Exception as e:
            print(f"Error in cleanup: {e}") 