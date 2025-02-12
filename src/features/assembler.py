"""
Video assembly module.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger
from moviepy.editor import VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
import numpy as np
from PIL import Image
import tempfile
import shutil
import asyncio
import time

from . import config
from .error_handler import ErrorHandler, VideoProductionError
from .video_preferences import FormatSettings, Resolution, DurationRange

class VideoAssembler:
    """Handles video assembly and post-processing."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        self.temp_dir = config.OUTPUT_DIR / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_video(
        self,
        script_data: Dict,
        audio_files: Dict[str, Path],
        visuals: Dict[str, List[Path]],
        format_settings: FormatSettings
    ) -> Path:
        """
        Create a video from audio and visual components.
        
        Args:
            script_data: Script data with timing information
            audio_files: Dictionary of audio file paths
            visuals: Dictionary of visual file paths
            format_settings: Video format settings
            
        Returns:
            Path to the assembled video file
        """
        try:
            # Create temporary directory for intermediate files
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Load audio clips
                audio_clips = self._load_audio_clips(audio_files)
                
                # Load and resize visuals
                visual_clips = self._load_visual_clips(
                    visuals,
                    format_settings.resolution,
                    format_settings.fps
                )
                
                # Combine audio and visuals
                video = await self._combine_clips(
                    audio_clips,
                    visual_clips,
                    script_data,
                    format_settings
                )
                
                # Apply post-processing
                final_video = await self._post_process(
                    video,
                    format_settings
                )
                
                # Save to output directory
                output_path = config.OUTPUT_DIR / f"video_{int(time.time())}.mp4"
                final_video.write_videofile(
                    str(output_path),
                    fps=format_settings.fps,
                    codec=format_settings.codec,
                    bitrate=format_settings.bitrate,
                    threads=4,
                    logger=None
                )
                
                return output_path
                
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise VideoProductionError(
                "Video creation failed",
                "ASSEMBLY_ERROR",
                {"error": str(e)}
            )
    
    def _load_audio_clips(self, audio_files: Dict[str, Path]) -> Dict[str, AudioFileClip]:
        """Load audio clips from files."""
        clips = {}
        for name, path in audio_files.items():
            try:
                # Load audio clip with moviepy
                clips[name] = AudioFileClip(str(path))
                logger.debug(f"Loaded audio clip: {name} ({path})")
            except Exception as e:
                logger.error(f"Error loading audio clip {name}: {str(e)}")
                raise VideoProductionError(
                    f"Failed to load audio clip {name}",
                    "AUDIO_LOAD_ERROR",
                    {"error": str(e)}
                )
        return clips
    
    def _load_visual_clips(
        self,
        visuals: Dict[str, List[Path]],
        resolution: Resolution,
        fps: int
    ) -> Dict[str, List[ImageClip]]:
        """Load and resize visual clips."""
        clips = {}
        for name, paths in visuals.items():
            try:
                scene_clips = []
                for path in paths:
                    # Load and resize image
                    img = Image.open(path)
                    img = img.resize((resolution.width, resolution.height))
                    
                    # Convert to numpy array and create clip
                    img_array = np.array(img)
                    clip = ImageClip(img_array).set_fps(fps)
                    
                    scene_clips.append(clip)
                
                clips[name] = scene_clips
                
            except Exception as e:
                logger.error(f"Error loading visual clip {name}: {str(e)}")
                raise VideoProductionError(
                    f"Failed to load visual clip {name}",
                    "VISUAL_LOAD_ERROR",
                    {"error": str(e)}
                )
        
        return clips
    
    async def _combine_clips(
        self,
        audio_clips: Dict[str, AudioFileClip],
        visual_clips: Dict[str, List[ImageClip]],
        script_data: Dict,
        format_settings: FormatSettings
    ) -> Union[VideoFileClip, CompositeVideoClip]:
        """Combine audio and visual clips according to script timing."""
        try:
            # Create list of clips in order
            final_clips = []
            current_time = 0
            
            # Add hook if available
            if "hook" in audio_clips and "hook" in visual_clips:
                hook_audio = audio_clips["hook"]
                hook_visuals = visual_clips["hook"]
                hook_clip = self._create_scene_clip(
                    hook_visuals,
                    hook_audio,
                    hook_audio.duration,
                    format_settings
                )
                final_clips.append(hook_clip.set_start(current_time))
                current_time += hook_audio.duration
            
            # Add scenes
            for scene in script_data["scenes"]:
                scene_id = scene["id"]
                if scene_id in audio_clips and scene_id in visual_clips:
                    scene_audio = audio_clips[scene_id]
                    scene_visuals = visual_clips[scene_id]
                    
                    scene_clip = self._create_scene_clip(
                        scene_visuals,
                        scene_audio,
                        scene_audio.duration,
                        format_settings
                    )
                    final_clips.append(scene_clip.set_start(current_time))
                    current_time += scene_audio.duration
            
            # Add call to action if available
            if "call_to_action" in audio_clips and "call_to_action" in visual_clips:
                cta_audio = audio_clips["call_to_action"]
                cta_visuals = visual_clips["call_to_action"]
                cta_clip = self._create_scene_clip(
                    cta_visuals,
                    cta_audio,
                    cta_audio.duration,
                    format_settings
                )
                final_clips.append(cta_clip.set_start(current_time))
            
            # Concatenate all clips
            final_video = CompositeVideoClip(final_clips)
            
            return final_video
            
        except Exception as e:
            logger.error(f"Error combining clips: {str(e)}")
            raise VideoProductionError(
                "Failed to combine clips",
                "COMBINE_ERROR",
                {"error": str(e)}
            )
    
    def _create_scene_clip(
        self,
        visuals: List[ImageClip],
        audio: AudioFileClip,
        duration: float,
        format_settings: FormatSettings
    ) -> Union[VideoFileClip, CompositeVideoClip]:
        """Create a clip for a single scene."""
        try:
            # Calculate duration for each visual
            visual_duration = duration / len(visuals)
            
            # Set duration for each visual
            timed_visuals = [
                visual.set_duration(visual_duration)
                for visual in visuals
            ]
            
            # Concatenate visuals
            scene_video = concatenate_videoclips(timed_visuals)
            
            # Add audio
            scene_video = scene_video.set_audio(audio)
            
            return scene_video
            
        except Exception as e:
            logger.error(f"Error creating scene clip: {str(e)}")
            raise VideoProductionError(
                "Failed to create scene clip",
                "SCENE_ERROR",
                {"error": str(e)}
            )
    
    async def _post_process(
        self,
        video: Union[VideoFileClip, CompositeVideoClip],
        format_settings: FormatSettings
    ) -> Union[VideoFileClip, CompositeVideoClip]:
        """Apply post-processing effects."""
        try:
            # Ensure correct resolution
            if video.size != (format_settings.resolution.width, format_settings.resolution.height):
                video = video.resize(
                    (format_settings.resolution.width, format_settings.resolution.height)
                )
            
            # Add any additional post-processing here
            
            return video
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            raise VideoProductionError(
                "Post-processing failed",
                "POST_PROCESS_ERROR",
                {"error": str(e)}
            )
    
    async def _write_video_with_progress(
        self,
        video: VideoFileClip,
        output_path: Path,
        fps: int
    ) -> None:
        """Write video file with progress tracking."""
        loop = asyncio.get_event_loop()
        total_frames = int(video.duration * fps)
        
        # Create progress callback
        def progress_callback(frame_num):
            progress = frame_num / total_frames * 100
            logger.info(f"Writing video: {progress:.1f}% complete")
        
        # Write video in thread pool
        await loop.run_in_executor(
            None,
            lambda: video.write_videofile(
                str(output_path),
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                progress_bar=False,
                logger=None,
                callback=progress_callback
            )
        )
    
    async def _cleanup_resources(self, clips: List[VideoFileClip]) -> None:
        """Clean up video clips and temporary files."""
        loop = asyncio.get_event_loop()
        
        for clip in clips:
            if clip is not None:
                await loop.run_in_executor(
                    None,
                    lambda: clip.close()
                )
            
            # Clean up temporary files
            for temp_file in self.temp_dir.glob("*"):
                try:
                    temp_file.unlink()
                except:
                    pass 