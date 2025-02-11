"""
Video assembly module using MoviePy.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    TextClip,
    ColorClip,
    VideoFileClip
)
from PIL import Image
import numpy as np
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor

from . import config
from .error_handler import ErrorHandler, VideoProductionError
from .video_synchronizer import VideoSynchronizer

class VideoAssembler:
    """Handles assembly of audio and visual components into a final video."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.output_dir = config.OUTPUT_DIR
        self.temp_dir = config.TEMP_DIR
        
        # Initialize error handler
        self.error_handler = error_handler or ErrorHandler()
        
        # Initialize synchronizer
        self.synchronizer = VideoSynchronizer(self.error_handler)
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="ASSEMBLY_ERROR")
    async def create_video(
        self,
        script_data: Dict,
        audio_files: Dict[str, Path],
        visual_files: Dict[str, List[Path]],
        output_path: Optional[Path] = None,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30
    ) -> Path:
        """
        Create the final video by combining audio and visuals.
        
        Args:
            script_data: The original script data
            audio_files: Dictionary mapping sections to audio file paths
            visual_files: Dictionary mapping sections to lists of image file paths
            output_path: Path for the output video file
            resolution: Video resolution (width, height)
            fps: Frames per second
            
        Returns:
            Path to the generated video file
        """
        output_path = output_path or self.output_dir / "final_video.mp4"
        width, height = resolution
        
        try:
            # Process sections in parallel
            section_clips = await self._process_sections(
                script_data,
                audio_files,
                visual_files,
                resolution,
                fps
            )
            
            # Extract timing data
            timing_data = {
                section["title"]: section.get("duration_seconds", 60)
                for section in script_data["sections"]
            }
            
            # Synchronize content
            synchronized_video = await self.synchronizer.synchronize_content(
                section_clips,
                audio_files,
                timing_data,
                sum(timing_data.values())
            )
            
            # Write final video with progress tracking
            await self._write_video_with_progress(
                synchronized_video,
                output_path,
                fps
            )
            
            logger.info(f"Generated final video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error assembling video: {str(e)}")
            raise VideoProductionError(
                "Video assembly failed",
                "ASSEMBLY_ERROR",
                {"error": str(e)}
            )
        finally:
            # Clean up resources
            await self._cleanup_resources(section_clips)
    
    async def _process_sections(
        self,
        script_data: Dict,
        audio_files: Dict[str, Path],
        visual_files: Dict[str, List[Path]],
        resolution: Tuple[int, int],
        fps: int
    ) -> List[VideoFileClip]:
        """Process video sections in parallel."""
        tasks = []
        width, height = resolution
        
        # Create tasks for each section
        for section in script_data["sections"]:
            title = section["title"]
            if title in audio_files and title in visual_files:
                task = self._create_section_clip(
                    audio_path=audio_files[title],
                    image_paths=visual_files[title],
                    resolution=resolution,
                    section_title=title,
                    fps=fps
                )
                tasks.append(task)
        
        # Process tasks in parallel
        section_clips = await asyncio.gather(*tasks)
        return [clip for clip in section_clips if clip is not None]
    
    async def _create_section_clip(
        self,
        audio_path: Path,
        image_paths: List[Path],
        resolution: Tuple[int, int],
        section_title: Optional[str] = None,
        fps: int = 30,
        transition_duration: float = 1.0
    ) -> VideoFileClip:
        """Create a video clip for a section with enhanced quality."""
        try:
            width, height = resolution
            loop = asyncio.get_event_loop()
            
            # Load and process audio
            audio = await self._load_audio(audio_path)
            total_duration = audio.duration
            
            # Process images in parallel
            image_clips = await self._process_images(
                image_paths,
                resolution,
                total_duration,
                fps
            )
            
            # Create composite clip in thread pool
            video = await loop.run_in_executor(
                None,
                lambda: CompositeVideoClip(
                    image_clips,
                    size=resolution
                ).set_duration(total_duration)
            )
            
            # Add audio in thread pool
            video = await loop.run_in_executor(
                None,
                lambda: video.set_audio(audio)
            )
            
            return video
            
        except Exception as e:
            logger.error(f"Error creating section clip: {str(e)}")
            return None
    
    async def _load_audio(self, audio_path: Path) -> AudioFileClip:
        """Load audio file with error handling."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: AudioFileClip(str(audio_path))
            )
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise
    
    async def _process_images(
        self,
        image_paths: List[Path],
        resolution: Tuple[int, int],
        total_duration: float,
        fps: int
    ) -> List[ImageClip]:
        """Process images in parallel with enhanced quality."""
        width, height = resolution
        image_duration = total_duration / len(image_paths)
        loop = asyncio.get_event_loop()
        
        async def process_image(img_path: Path, index: int) -> ImageClip:
            try:
                # Open and enhance image
                img = await self._enhance_image(img_path, resolution)
                
                # Create clip with motion in thread pool
                clip = await loop.run_in_executor(
                    None,
                    lambda: ImageClip(img)
                        .set_duration(image_duration)
                        .set_start(index * image_duration)
                )
                
                # Add subtle motion in thread pool
                clip = await loop.run_in_executor(
                    None,
                    lambda: self._add_motion_effect(clip)
                )
                
                return clip
                
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
                return None
        
        # Process images in parallel
        tasks = [
            process_image(path, i)
            for i, path in enumerate(image_paths)
        ]
        
        clips = await asyncio.gather(*tasks)
        return [clip for clip in clips if clip is not None]
    
    async def _enhance_image(
        self,
        image_path: Path,
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Enhance image quality."""
        width, height = resolution
        loop = asyncio.get_event_loop()
        
        async def process():
            # Open image with PIL in thread pool
            img = await loop.run_in_executor(
                None,
                lambda: Image.open(str(image_path))
            )
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = await loop.run_in_executor(
                    None,
                    lambda: img.convert("RGB")
                )
            
            # Resize to target resolution
            img = await loop.run_in_executor(
                None,
                lambda: img.resize((width, height), Image.Resampling.LANCZOS)
            )
            
            # Convert to numpy array
            return np.array(img)
        
        return await process()
    
    def _add_motion_effect(self, clip: ImageClip) -> ImageClip:
        """Add subtle motion effect to image clip."""
        # Add a slow zoom effect
        zoom_factor = 1.1
        duration = clip.duration
        
        def zoom(t):
            scale = 1 + (zoom_factor - 1) * t / duration
            return scale
        
        return clip.resize(zoom)
    
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