"""
Visual effects and transitions module.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    vfx,
    ColorClip,
    VideoClip
)
import numpy as np
from PIL import Image
import math
import cv2

class VisualEffects:
    """Handles visual effects and transitions for video clips."""
    
    # Available transition types
    TRANSITIONS = {
        'crossfade': 'apply_crossfade',
        'slide': 'apply_slide_transition',
        'push': 'apply_push_transition',
        'zoom': 'apply_zoom_transition',
        'wipe': 'apply_wipe_transition',
        'blur': 'apply_blur_transition'
    }
    
    # Available effect types
    EFFECTS = {
        'ken_burns': 'apply_ken_burns_effect',
        'dynamic_zoom': 'apply_dynamic_zoom',
        'pan': 'apply_pan',
        'parallax': 'apply_parallax',
        'floating': 'apply_floating',
        'pulse': 'apply_pulse',
        'rotate': 'apply_rotate'
    }
    
    @classmethod
    async def create(cls):
        """Create a new instance of VisualEffects."""
        return cls()

    async def apply_ken_burns_effect(self, image_path: Union[Path, str], duration: float = 5.0) -> VideoClip:
        """Apply Ken Burns effect to an image."""
        image = ImageClip(str(image_path)) if isinstance(image_path, (str, Path)) else image_path
        image = image.set_duration(duration)

        w, h = image.size
        screen = CompositeVideoClip([image], size=(w, h))

        def make_frame(t):
            progress = t / duration
            scale = 1 + (0.3 * progress)  # Zoom from 1x to 1.3x
            pos_x = 50 * progress  # Pan right
            pos_y = 30 * progress  # Pan down slightly
            
            frame = image.get_frame(t)
            M = cv2.getAffineTransform(
                np.float32([[0,0], [w,0], [0,h]]),
                np.float32([[pos_x,pos_y], [w*scale+pos_x,pos_y], [pos_x,h*scale+pos_y]])
            )
            return cv2.warpAffine(frame, M, (w, h))

        return VideoClip(make_frame, duration=duration)

    async def apply_crossfade(self, clip1: VideoClip, clip2: VideoClip, transition_duration: float = 1.0) -> VideoClip:
        """Apply crossfade transition between two clips."""
        clip1 = clip1.crossfadeout(transition_duration)
        clip2 = clip2.crossfadein(transition_duration)
        return CompositeVideoClip([
            clip1,
            clip2.set_start(clip1.duration - transition_duration)
        ])

    async def apply_dynamic_zoom(self, image: Union[Path, str], duration: float = 5.0, zoom_range: tuple = (1.0, 1.5)) -> VideoClip:
        """Apply dynamic zoom effect to an image."""
        image = ImageClip(str(image)) if isinstance(image, (str, Path)) else image
        image = image.set_duration(duration)
        start_zoom, end_zoom = zoom_range

        w, h = image.size
        screen = CompositeVideoClip([image], size=(w, h))

        def make_frame(t):
            progress = t / duration
            scale = start_zoom + (end_zoom - start_zoom) * progress
            frame = image.get_frame(t)
            M = cv2.getAffineTransform(
                np.float32([[0,0], [w,0], [0,h]]),
                np.float32([[0,0], [w*scale,0], [0,h*scale]])
            )
            return cv2.warpAffine(frame, M, (w, h))

        return VideoClip(make_frame, duration=duration)

    async def apply_pan(self, image: Union[Path, str], direction: str = 'right', pan_speed: float = 0.5, duration: float = 5.0) -> VideoClip:
        """Apply panning effect to an image."""
        image = ImageClip(str(image)) if isinstance(image, (str, Path)) else image
        image = image.set_duration(duration)

        w, h = image.size
        screen = CompositeVideoClip([image], size=(w, h))

        def make_frame(t):
            progress = t / duration
            frame = image.get_frame(t)
            
            if direction == 'right':
                x = -w * progress * pan_speed
                y = 0
            elif direction == 'left':
                x = w * progress * pan_speed
                y = 0
            elif direction == 'up':
                x = 0
                y = h * progress * pan_speed
            else:  # down
                x = 0
                y = -h * progress * pan_speed
                
            M = np.float32([[1, 0, x], [0, 1, y]])
            return cv2.warpAffine(frame, M, (w, h))

        return VideoClip(make_frame, duration=duration)

    async def apply_parallax(self, image: Union[Path, str], direction: str = 'right', depth_speed: float = 0.3, duration: float = 5.0) -> VideoClip:
        """Apply parallax effect to an image."""
        image = ImageClip(str(image)) if isinstance(image, (str, Path)) else image
        image = image.set_duration(duration)

        w, h = image.size
        screen = CompositeVideoClip([image], size=(w, h))

        def make_frame(t):
            progress = t / duration
            frame = image.get_frame(t)
            
            offset_x = w * depth_speed * math.sin(2 * math.pi * progress)
            offset_y = h * depth_speed * math.cos(2 * math.pi * progress)
            
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            return cv2.warpAffine(frame, M, (w, h))

        return VideoClip(make_frame, duration=duration)

    async def apply_floating(self, image: Union[Path, str], amplitude: float = 20.0, float_speed: float = 1.0, duration: float = 5.0) -> VideoClip:
        """Apply floating effect to an image."""
        image = ImageClip(str(image)) if isinstance(image, (str, Path)) else image
        image = image.set_duration(duration)

        w, h = image.size
        screen = CompositeVideoClip([image], size=(w, h))

        def make_frame(t):
            progress = t / duration
            frame = image.get_frame(t)
            
            offset_y = amplitude * math.sin(2 * math.pi * progress * float_speed)
            
            M = np.float32([[1, 0, 0], [0, 1, offset_y]])
            return cv2.warpAffine(frame, M, (w, h))

        return VideoClip(make_frame, duration=duration)

    async def apply_pulse(self, image: Union[Path, str], scale_range: tuple = (0.95, 1.05), pulse_speed: float = 2.0, duration: float = 5.0) -> VideoClip:
        """Apply pulsing effect to an image."""
        image = ImageClip(str(image)) if isinstance(image, (str, Path)) else image
        image = image.set_duration(duration)
        min_scale, max_scale = scale_range

        w, h = image.size
        screen = CompositeVideoClip([image], size=(w, h))

        def make_frame(t):
            progress = t / duration
            frame = image.get_frame(t)
            
            scale = min_scale + (max_scale - min_scale) * (math.sin(pulse_speed * math.pi * progress) + 1) / 2
            
            M = cv2.getAffineTransform(
                np.float32([[0,0], [w,0], [0,h]]),
                np.float32([[0,0], [w*scale,0], [0,h*scale]])
            )
            return cv2.warpAffine(frame, M, (w, h))

        return VideoClip(make_frame, duration=duration)

    async def apply_rotate(self, image: Union[Path, str], angle_range: tuple = (-5, 5), rotation_speed: float = 1.0, duration: float = 5.0) -> VideoClip:
        """Apply rotation effect to an image."""
        image = ImageClip(str(image)) if isinstance(image, (str, Path)) else image
        image = image.set_duration(duration)
        min_angle, max_angle = angle_range

        w, h = image.size
        screen = CompositeVideoClip([image], size=(w, h))

        def make_frame(t):
            progress = t / duration
            frame = image.get_frame(t)
            
            angle = min_angle + (max_angle - min_angle) * (math.sin(rotation_speed * math.pi * progress) + 1) / 2
            
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            return cv2.warpAffine(frame, M, (w, h))

        return VideoClip(make_frame, duration=duration)

    async def create_transition_sequence(
        self,
        clips: List[VideoClip],
        transition_type: str = 'crossfade',
        transition_duration: float = 1.0
    ) -> VideoClip:
        """Create a sequence of clips with transitions."""
        if transition_type not in self.TRANSITIONS:
            raise ValueError(f"Unknown transition type: {transition_type}")

        transition_method = getattr(self, self.TRANSITIONS[transition_type])
        final_clips = []
        
        for i in range(len(clips)):
            if i == 0:
                final_clips.append(clips[i])
            else:
                # Create transition between current and previous clip
                transition = await transition_method(
                    clips[i-1],
                    clips[i],
                    transition_duration=transition_duration
                )
                # Add transition and current clip
                final_clips.append(transition)
                if i < len(clips) - 1:
                    final_clips.append(clips[i])

        # Calculate total duration
        total_duration = sum(clip.duration for clip in clips)
        # Subtract overlap duration for each transition
        total_duration -= transition_duration * (len(clips) - 1)
        
        # Concatenate all clips
        result = concatenate_videoclips(final_clips)
        # Set the correct duration
        result = result.set_duration(total_duration)
        
        return result

    async def apply_slide_transition(
        self,
        clip1: VideoClip,
        clip2: VideoClip,
        transition_duration: float = 1.0,
        direction: str = 'left'
    ) -> VideoClip:
        """Create a sliding transition between clips."""
        w, h = clip1.size

        def make_frame(t):
            progress = t / transition_duration
            frame1 = clip1.get_frame(t)
            frame2 = clip2.get_frame(t)
            
            if direction == 'left':
                x = int(w * progress)
                frame1 = np.roll(frame1, x, axis=1)
                frame2 = np.roll(frame2, x - w, axis=1)
            elif direction == 'right':
                x = int(-w * progress)
                frame1 = np.roll(frame1, x, axis=1)
                frame2 = np.roll(frame2, x + w, axis=1)
            elif direction == 'up':
                y = int(h * progress)
                frame1 = np.roll(frame1, y, axis=0)
                frame2 = np.roll(frame2, y - h, axis=0)
            else:  # down
                y = int(-h * progress)
                frame1 = np.roll(frame1, y, axis=0)
                frame2 = np.roll(frame2, y + h, axis=0)
            
            mask = np.zeros_like(frame1)
            if direction in ['left', 'right']:
                mask[:, :x] = 1
            else:
                mask[:y] = 1
            
            return frame1 * (1 - mask) + frame2 * mask

        return VideoClip(make_frame, duration=transition_duration)

    async def apply_push_transition(
        self,
        clip1: VideoClip,
        clip2: VideoClip,
        transition_duration: float = 1.0,
        direction: str = 'left'
    ) -> VideoClip:
        """Create a push transition between clips."""
        w, h = clip1.size

        def make_frame(t):
            progress = t / transition_duration
            frame1 = clip1.get_frame(t)
            frame2 = clip2.get_frame(t)
            
            if direction == 'left':
                x = int(w * progress)
                frame = np.concatenate((frame1[:, x:], frame2[:, :x]), axis=1)
            elif direction == 'right':
                x = int(w * (1 - progress))
                frame = np.concatenate((frame2[:, x:], frame1[:, :x]), axis=1)
            elif direction == 'up':
                y = int(h * progress)
                frame = np.concatenate((frame1[y:], frame2[:y]), axis=0)
            else:  # down
                y = int(h * (1 - progress))
                frame = np.concatenate((frame2[y:], frame1[:y]), axis=0)
            
            return frame

        return VideoClip(make_frame, duration=transition_duration)

    async def apply_zoom_transition(
        self,
        clip1: VideoClip,
        clip2: VideoClip,
        transition_duration: float = 1.0,
        zoom_type: str = 'in'
    ) -> VideoClip:
        """Create a zoom transition between clips."""
        w, h = clip1.size

        def make_frame(t):
            progress = t / transition_duration
            frame1 = clip1.get_frame(t)
            frame2 = clip2.get_frame(t)
            
            if zoom_type == 'in':
                scale1 = 1 + progress
                scale2 = 2 - progress
            else:  # zoom out
                scale1 = 2 - progress
                scale2 = 1 + progress
            
            M1 = cv2.getAffineTransform(
                np.float32([[0,0], [w,0], [0,h]]),
                np.float32([[0,0], [w*scale1,0], [0,h*scale1]])
            )
            M2 = cv2.getAffineTransform(
                np.float32([[0,0], [w,0], [0,h]]),
                np.float32([[0,0], [w*scale2,0], [0,h*scale2]])
            )
            
            warped1 = cv2.warpAffine(frame1, M1, (w, h))
            warped2 = cv2.warpAffine(frame2, M2, (w, h))
            
            return warped1 * (1 - progress) + warped2 * progress

        return VideoClip(make_frame, duration=transition_duration)

    async def apply_wipe_transition(
        self,
        clip1: VideoClip,
        clip2: VideoClip,
        transition_duration: float = 1.0,
        direction: str = 'left'
    ) -> VideoClip:
        """Create a wipe transition between clips."""
        w, h = clip1.size

        def make_frame(t):
            progress = t / transition_duration
            frame1 = clip1.get_frame(t)
            frame2 = clip2.get_frame(t)
            
            if direction == 'left':
                x = int(w * progress)
                mask = np.zeros((h, w))
                mask[:, :x] = 1
            elif direction == 'right':
                x = int(w * (1 - progress))
                mask = np.zeros((h, w))
                mask[:, x:] = 1
            elif direction == 'up':
                y = int(h * progress)
                mask = np.zeros((h, w))
                mask[:y, :] = 1
            else:  # down
                y = int(h * (1 - progress))
                mask = np.zeros((h, w))
                mask[y:, :] = 1
            
            mask = np.stack([mask] * 3, axis=2)
            return frame1 * (1 - mask) + frame2 * mask

        return VideoClip(make_frame, duration=transition_duration)

    async def apply_blur_transition(
        self,
        clip1: VideoClip,
        clip2: VideoClip,
        transition_duration: float = 1.0,
        blur_amount: int = 20
    ) -> VideoClip:
        """Create a blur transition between clips."""
        w, h = clip1.size

        def make_frame(t):
            progress = t / transition_duration
            frame1 = clip1.get_frame(t)
            frame2 = clip2.get_frame(t)
            
            # Apply increasing blur to clip1 and decreasing blur to clip2
            kernel_size1 = int(blur_amount * progress)
            kernel_size2 = int(blur_amount * (1 - progress))
            
            if kernel_size1 > 0:
                frame1 = cv2.GaussianBlur(frame1, (kernel_size1 * 2 + 1, kernel_size1 * 2 + 1), 0)
            if kernel_size2 > 0:
                frame2 = cv2.GaussianBlur(frame2, (kernel_size2 * 2 + 1, kernel_size2 * 2 + 1), 0)
            
            return frame1 * (1 - progress) + frame2 * progress

        return VideoClip(make_frame, duration=transition_duration) 