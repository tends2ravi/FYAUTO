"""
Main workflow orchestrator for the video production system.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from . import config
from .script_generator import ScriptGenerator
from .audio_generator import AudioGenerator
from .visual_generator import VisualGenerator
from .video_assembler import VideoAssembler
from .caption_generator import CaptionGenerator
from .background_music import BackgroundMusicManager
from .video_preferences import VideoPreferences
from .youtube_uploader import YouTubeUploader
from .error_handler import ErrorHandler, VideoProductionError
from .caching import ContentCache

class VideoProductionWorkflow:
    """Orchestrates the entire video production process."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        # Initialize error handler with default settings
        self.error_handler = ErrorHandler(max_retries=max_retries, retry_delay=retry_delay)
        self.max_retries = max_retries  # Store max_retries at class level
        self.retry_delay = retry_delay  # Store retry_delay at class level
        
        # Initialize caching
        self.cache = ContentCache()
        
        # Initialize components with error handler
        self.script_generator = ScriptGenerator(error_handler=self.error_handler)
        self.audio_generator = AudioGenerator(error_handler=self.error_handler)
        self.visual_generator = VisualGenerator(error_handler=self.error_handler)
        self.video_assembler = VideoAssembler(error_handler=self.error_handler)
        self.caption_generator = CaptionGenerator(error_handler=self.error_handler)
        self.music_manager = BackgroundMusicManager(error_handler=self.error_handler)
        
        # Initialize preferences
        self.preferences = VideoPreferences()
        
        # Initialize YouTube uploader
        self.youtube_uploader = YouTubeUploader()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Track active tasks
        self.active_tasks: Set[asyncio.Task] = set()
        
        # Set up logging
        logger.add(
            config.OUTPUT_DIR / "workflow.log",
            rotation="1 day",
            retention="7 days",
            level=config.LOG_LEVEL
        )
        
        logger.info("Initialized VideoProductionWorkflow")
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="WORKFLOW_ERROR")
    async def create_video(
        self,
        topic: str,
        style: str = "informative",
        format: str = "standard",
        niche: str = "general",
        caption_style: str = "standard",
        music_style: str = "ambient",
        duration: Optional[float] = None,
        additional_context: Optional[Dict] = None,
        youtube_settings: Optional[Dict] = None,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Create a video from start to finish.
        
        Args:
            topic: Main topic or title for the video
            style: Content style (informative, entertaining, etc.)
            format: Video format (youtube/shorts)
            niche: Content niche
            caption_style: Style for captions
            music_style: Style for background music
            duration: Target duration in seconds
            additional_context: Additional context for generation
            youtube_settings: Settings for YouTube upload
            output_path: Custom output path for the video
        
        Returns:
            Dictionary containing video path and optional YouTube URL
        """
        try:
            logger.info(f"Starting video production workflow for topic: {topic}")
            
            # Get format settings
            format_settings = self.preferences.get_format_settings(format)
            
            # Validate and set duration
            duration = self._validate_duration(duration, format_settings)
            
            # Generate script
            script_data = await self.script_generator.generate_script(
                topic=topic,
                style=style,
                duration_minutes=duration / 60,
                target_audience=niche,
                additional_context=additional_context
            )
            
            # Create tasks for parallel processing
            tasks = {
                "audio": asyncio.create_task(self.audio_generator.generate_audio(script_data)),
                "visuals": asyncio.create_task(self.visual_generator.generate_visuals(script_data)),
                "captions": asyncio.create_task(self.caption_generator.generate_captions(script_data)),
                "music": asyncio.create_task(self.music_manager.generate_background_music(
                    duration=duration,
                    style=music_style
                ))
            }
            
            # Wait for all tasks to complete
            results = await self._wait_for_tasks(tasks)
            
            # Assemble video
            video_path = await self._assemble_video(
                script_data=script_data,
                audio_files=results["audio"],
                visuals=results["visuals"],
                captions=results["captions"],
                music_style=music_style,
                format_settings=format_settings,
                output_path=output_path
            )
            
            # Upload to YouTube if settings provided
            if youtube_settings:
                video_url = await self._upload_to_youtube(
                    video_path,
                    script_data,
                    youtube_settings
                )
                return {"video_path": video_path, "video_url": video_url}
            
            return {"video_path": video_path}
            
        except Exception as e:
            logger.error(f"Error in video production workflow: {str(e)}")
            raise VideoProductionError(
                "Video production failed",
                "WORKFLOW_ERROR",
                {"error": str(e)}
            )
            
        finally:
            await self._cleanup()
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="SCRIPT_ERROR")
    async def _generate_script(
        self,
        topic: str,
        style: str,
        duration: float,
        guidelines: Dict
    ) -> Dict:
        """Generate script with error handling and caching."""
        cache_key = f"script_{topic}_{style}_{duration}"
        cached_script = self.cache.get_script(cache_key)
        
        if cached_script:
            return cached_script
        
        script_data = await self.error_handler.run_with_timeout(
            self.script_generator.generate_script,
            timeout=60,
            topic=topic,
            style=style,
            target_duration=duration,
            guidelines=guidelines
        )
        
        self.cache.cache_script(cache_key, script_data)
        return script_data
    
    async def _prepare_resources(self) -> None:
        """Prepare necessary resources in parallel."""
        tasks = [
            self._ensure_directories(),
            self._load_models(),
            self._verify_api_keys()
        ]
        await asyncio.gather(*tasks)
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="AUDIO_ERROR")
    async def _generate_audio(self, script_data: Dict, guidelines: Dict) -> Dict[str, Path]:
        """Generate audio with error handling."""
        return await self.error_handler.run_with_timeout(
            self.audio_generator.generate_audio,
            timeout=300,
            script_data=script_data,
            guidelines=guidelines
        )
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="VISUAL_ERROR")
    async def _generate_visuals(
        self,
        script_data: Dict,
        style: str,
        resolution: tuple
    ) -> Dict[str, List[Path]]:
        """Generate visuals with error handling."""
        return await self.error_handler.run_with_timeout(
            self.visual_generator.generate_visuals_for_script,
            timeout=600,
            script_data=script_data,
            style=style,
            resolution=resolution
        )
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="CAPTION_ERROR")
    async def _generate_captions(
        self,
        script_data: Dict,
        style: str
    ) -> Dict[str, List[Dict]]:
        """Generate captions with error handling."""
        return await self.error_handler.run_with_timeout(
            self.caption_generator.generate_captions_from_script,
            timeout=120,
            script_data=script_data,
            style=style
        )
    
    async def _assemble_video(
        self,
        script_data: Dict,
        audio_files: Dict[str, Path],
        visuals: Dict[str, List[Path]],
        captions: Dict[str, List[Dict]],
        music_style: Optional[str],
        format_settings: Dict,
        output_path: Optional[Path]
    ) -> Path:
        """Assemble video with progress tracking."""
        try:
            # Create base video
            base_video = await self.error_handler.run_with_timeout(
                self.video_assembler.create_video,
                timeout=1800,
                script_data=script_data,
                audio_files=audio_files,
                visuals=visuals
            )
            
            # Add captions
            captioned_video = await self.error_handler.run_with_timeout(
                self.caption_generator.apply_captions_to_video,
                timeout=300,
                video_path=base_video,
                captions=captions
            )
            
            # Add background music if specified
            if music_style:
                final_video = await self.error_handler.run_with_timeout(
                    self.music_manager.add_background_music,
                    timeout=300,
                    video_path=captioned_video,
                    style=music_style
                )
            else:
                final_video = captioned_video
            
            # Move to final output path if specified
            if output_path:
                from shutil import move
                move(str(final_video), str(output_path))
                return output_path
            
            return final_video
            
        except Exception as e:
            logger.error(f"Error assembling video: {str(e)}")
            raise VideoProductionError(
                "Video assembly failed",
                "ASSEMBLY_ERROR",
                {"error": str(e)}
            )
    
    async def _upload_to_youtube(
        self,
        video_path: Path,
        script_data: Dict,
        youtube_settings: Optional[Dict]
    ) -> str:
        """Upload video to YouTube with error handling."""
        try:
            # Prepare upload settings
            settings = {
                "title": script_data["title"],
                "description": self._generate_video_description(script_data),
                "tags": self._generate_tags(script_data),
                "privacy_status": "private",
                "language": "en"
            }
            
            if youtube_settings:
                settings.update(youtube_settings)
            
            # Upload video
            return await self.error_handler.run_with_timeout(
                self.youtube_uploader.upload_video,
                timeout=3600,
                video_path=video_path,
                **settings
            )
            
        except Exception as e:
            logger.error(f"Error uploading to YouTube: {str(e)}")
            raise VideoProductionError(
                "YouTube upload failed",
                "UPLOAD_ERROR",
                {"error": str(e)}
            )
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="TASK_ERROR")
    async def _wait_for_tasks(self, tasks: Dict[str, asyncio.Task]) -> Dict:
        """Wait for tasks to complete with error handling."""
        results = {}
        
        try:
            for name, task in tasks.items():
                self.active_tasks.add(task)
                try:
                    results[name] = await task
                except Exception as e:
                    logger.error(f"Task {name} failed: {str(e)}")
                    raise
                finally:
                    self.active_tasks.remove(task)
            
            return results
            
        except Exception as e:
            # Cancel remaining tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            
            raise VideoProductionError(
                "Task execution failed",
                "TASK_ERROR",
                {"error": str(e)}
            )
    
    def _validate_duration(
        self,
        duration: Optional[float],
        format_settings: Dict
    ) -> float:
        """Validate and set video duration."""
        if duration:
            if not self.preferences.validate_duration(duration, format_settings):
                logger.warning(
                    f"Requested duration {duration}s is outside recommended range"
                )
                duration = format_settings["duration_range"]["recommended"]
        else:
            duration = format_settings["duration_range"]["recommended"]
        
        return duration
    
    async def _cleanup(self) -> None:
        """Clean up resources and temporary files."""
        try:
            # Cancel any remaining tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to finish
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
            
            # Clean up components
            await self.error_handler.cleanup_resources()
            self.executor.shutdown(wait=True)
            
            # Clear caches
            self.cache.clear_expired()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def create_video(*args, **kwargs) -> Dict:
    """
    Convenience function to create a video without instantiating the workflow class.
    Uses default retry settings.
    """
    workflow = VideoProductionWorkflow(max_retries=3, retry_delay=1.0)
    return workflow.create_video(*args, **kwargs)  # Return the coroutine directly

async def create_video_with_retries(max_retries: int = 3, retry_delay: float = 1.0, *args, **kwargs) -> Dict:
    """
    Convenience function to create a video with custom retry settings.
    """
    workflow = VideoProductionWorkflow(max_retries=max_retries, retry_delay=retry_delay)
    return await workflow.create_video(*args, **kwargs) 