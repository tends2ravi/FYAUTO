"""
Tests for the visual effects system.
"""
import pytest
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip, ImageClip, ColorClip, VideoClip
from src.visual_effects import VisualEffects

class TestVisualEffects:
    """Test suite for VisualEffects."""
    
    @pytest.fixture
    def effects(self):
        """Create effects instance."""
        return VisualEffects()
    
    @pytest.fixture
    def sample_clip(self, test_dir):
        """Create a sample video clip."""
        clip = ColorClip(size=(100, 100), color=(255, 0, 0), duration=2)
        clip_path = test_dir / "test_clip.mp4"
        clip.write_videofile(str(clip_path), fps=30)
        
        video = VideoFileClip(str(clip_path))
        yield video
        
        video.close()
        if clip_path.exists():
            clip_path.unlink()
    
    @pytest.fixture
    def sample_image(self, test_dir):
        """Create a sample image."""
        from PIL import Image
        
        # Create gradient image
        arr = np.linspace(0, 255, 100*100).reshape(100, 100).astype('uint8')
        rgb = np.stack([arr] * 3, axis=2)
        img = Image.fromarray(rgb)
        
        image_path = test_dir / "test_image.png"
        img.save(image_path)
        
        yield image_path
        
        if image_path.exists():
            image_path.unlink()
    
    async def test_ken_burns_effect(self, effects, sample_image):
        """Test Ken Burns effect."""
        clip = await effects.apply_ken_burns_effect(sample_image, duration=2)
        assert isinstance(clip, VideoClip)
        assert clip.duration == 2
    
    async def test_crossfade_transition(self, effects, sample_clip):
        """Test crossfade transition."""
        clip = await effects.apply_crossfade(sample_clip, sample_clip, transition_duration=1)
        assert isinstance(clip, VideoClip)
        assert abs(clip.duration - (sample_clip.duration * 2 - 1)) < 0.1
    
    async def test_dynamic_zoom(self, effects, sample_image):
        """Test dynamic zoom effect."""
        clip = await effects.apply_dynamic_zoom(sample_image, duration=2, zoom_range=(1, 1.5))
        assert isinstance(clip, VideoClip)
        assert clip.duration == 2
    
    async def test_pan_effect(self, effects, sample_image):
        """Test pan effect."""
        result = await effects.apply_pan(
            sample_image,
            direction="left",
            pan_speed=0.5,
            duration=2
        )
        
        assert isinstance(result, VideoClip)
        assert result.duration == 2
    
    async def test_parallax_effect(self, effects, sample_image):
        """Test parallax effect."""
        result = await effects.apply_parallax(
            sample_image,
            direction="right",
            depth_speed=0.3,
            duration=3.0
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 3.0) < 0.1
    
    async def test_floating_effect(self, effects, sample_image):
        """Test floating effect."""
        result = await effects.apply_floating(
            sample_image,
            amplitude=10.0,
            float_speed=1.0,
            duration=3.0
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 3.0) < 0.1
    
    async def test_pulse_effect(self, effects, sample_image):
        """Test pulse effect."""
        result = await effects.apply_pulse(
            sample_image,
            scale_range=(0.95, 1.05),
            pulse_speed=2.0,
            duration=3.0
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 3.0) < 0.1
    
    async def test_rotate_effect(self, effects, sample_image):
        """Test rotate effect."""
        result = await effects.apply_rotate(
            sample_image,
            angle_range=(-5, 5),
            rotation_speed=1.0,
            duration=3.0
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 3.0) < 0.1
    
    async def test_transition_sequence(self, effects, sample_clip):
        """Test transition sequence creation."""
        clips = [sample_clip] * 3
        
        result = await effects.create_transition_sequence(
            clips,
            transition_type="crossfade",
            transition_duration=1.0
        )
        
        assert isinstance(result, VideoClip)
        # Duration should be: sum of clip durations - (n-1) * transition_duration
        expected_duration = len(clips) * sample_clip.duration - (len(clips) - 1)
        assert abs(result.duration - expected_duration) < 0.1
    
    async def test_slide_transition(self, effects, sample_clip):
        """Test slide transition."""
        clip1 = sample_clip
        clip2 = sample_clip
        
        result = await effects.apply_slide_transition(
            clip1,
            clip2,
            transition_duration=1.0,
            direction="left"
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 1.0) < 0.1
    
    async def test_push_transition(self, effects, sample_clip):
        """Test push transition."""
        clip1 = sample_clip
        clip2 = sample_clip
        
        result = await effects.apply_push_transition(
            clip1,
            clip2,
            transition_duration=1.0,
            direction="left"
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 1.0) < 0.1
    
    async def test_zoom_transition(self, effects, sample_clip):
        """Test zoom transition."""
        clip1 = sample_clip
        clip2 = sample_clip
        
        result = await effects.apply_zoom_transition(
            clip1,
            clip2,
            transition_duration=1.0,
            zoom_type="in"
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 1.0) < 0.1
    
    async def test_wipe_transition(self, effects, sample_clip):
        """Test wipe transition."""
        clip1 = sample_clip
        clip2 = sample_clip
        
        result = await effects.apply_wipe_transition(
            clip1,
            clip2,
            transition_duration=1.0,
            direction="left"
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 1.0) < 0.1
    
    async def test_blur_transition(self, effects, sample_clip):
        """Test blur transition."""
        clip1 = sample_clip
        clip2 = sample_clip
        
        result = await effects.apply_blur_transition(
            clip1,
            clip2,
            transition_duration=1.0,
            blur_amount=20
        )
        
        assert isinstance(result, VideoClip)
        assert abs(result.duration - 1.0) < 0.1
    
    def test_available_transitions(self, effects):
        """Test available transitions list."""
        assert isinstance(effects.TRANSITIONS, dict)
        assert len(effects.TRANSITIONS) > 0
        assert all(isinstance(k, str) for k in effects.TRANSITIONS.keys())
        assert all(isinstance(v, str) for v in effects.TRANSITIONS.values())
        
        # Check specific transitions
        expected_transitions = {
            'crossfade', 'slide', 'push', 'zoom', 'wipe', 'blur'
        }
        assert set(effects.TRANSITIONS.keys()) == expected_transitions
    
    def test_available_effects(self, effects):
        """Test available effects list."""
        assert isinstance(effects.EFFECTS, dict)
        assert len(effects.EFFECTS) > 0
        assert all(isinstance(k, str) for k in effects.EFFECTS.keys())
        assert all(isinstance(v, str) for v in effects.EFFECTS.values())
        
        # Check specific effects
        expected_effects = {
            'ken_burns', 'dynamic_zoom', 'pan', 'parallax',
            'floating', 'pulse', 'rotate'
        }
        assert set(effects.EFFECTS.keys()) == expected_effects 