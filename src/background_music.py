"""
Background music management and integration module.
"""
from pathlib import Path
import json
from typing import Dict, Optional
import numpy as np
from loguru import logger
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import librosa
import soundfile as sf
import asyncio

from . import config
from .error_handler import ErrorHandler, VideoProductionError

class BackgroundMusicManager:
    """Handles background music selection and integration."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.categories = {
            "informative": {
                "volume": 0.3,
                "crossfade": 1.0,
                "tags": ["ambient", "calm", "instrumental"]
            },
            "entertaining": {
                "volume": 0.4,
                "crossfade": 0.8,
                "tags": ["upbeat", "energetic", "fun"]
            },
            "dramatic": {
                "volume": 0.5,
                "crossfade": 1.2,
                "tags": ["intense", "dramatic", "cinematic"]
            }
        }
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="MUSIC_ERROR")
    async def add_background_music(
        self,
        video: VideoFileClip,
        style: str,
        music_path: Optional[Path] = None
    ) -> VideoFileClip:
        """
        Add background music to video.
        
        Args:
            video: Input video clip
            style: Video style category
            music_path: Optional path to music file (if None, will select based on style)
            
        Returns:
            Video with background music
        """
        try:
            if music_path is None:
                music_path = await self._select_music(style)
            
            # Prepare music
            music = await self._prepare_music(
                music_path,
                video.duration,
                style
            )
            
            # Combine video and music
            final_video = await self._combine_video_and_music(video, music)
            
            # Save metadata
            await self._save_music_metadata(
                Path(video.filename),
                {
                    "music_file": str(music_path),
                    "style": style,
                    "duration": video.duration
                }
            )
            
            return final_video
            
        except Exception as e:
            logger.error(f"Error adding background music: {str(e)}")
            raise VideoProductionError(
                "Background music integration failed",
                "MUSIC_ERROR",
                {"error": str(e)}
            )
    
    async def _select_music(self, style: str) -> Path:
        """Select appropriate music file based on style."""
        style_settings = self.categories.get(style)
        if not style_settings:
            raise ValueError(f"Invalid style category: {style}")
        
        music_dir = config.ASSETS_DIR / "music" / style
        if not music_dir.exists():
            raise FileNotFoundError(f"Music directory not found: {music_dir}")
        
        # Find music files matching style tags
        music_files = []
        for tag in style_settings["tags"]:
            music_files.extend(music_dir.glob(f"*{tag}*.mp3"))
            music_files.extend(music_dir.glob(f"*{tag}*.wav"))
        
        if not music_files:
            raise FileNotFoundError(f"No music files found for style: {style}")
        
        # Select random file
        return np.random.choice(music_files)
    
    async def _prepare_music(
        self,
        music_path: Path,
        target_duration: float,
        style: str,
        volume: Optional[float] = None,
        crossfade: Optional[float] = None
    ) -> AudioFileClip:
        """Prepare music file for the video."""
        loop = asyncio.get_event_loop()
        
        # Load audio file using librosa in thread pool
        y, sr = await loop.run_in_executor(
            None,
            lambda: librosa.load(str(music_path))
        )
        
        # Get style settings
        style_settings = self.categories[style]
        volume = volume or style_settings["volume"]
        crossfade = crossfade or style_settings["crossfade"]
        
        # Adjust volume
        y = y * volume
        
        # Loop if necessary
        if len(y) / sr < target_duration:
            loops_needed = int(np.ceil(target_duration * sr / len(y)))
            crossfade_samples = int(crossfade * sr)
            
            # Create looped audio with crossfade
            looped_y = np.zeros(int(target_duration * sr))
            current_pos = 0
            
            for i in range(loops_needed):
                end_pos = min(current_pos + len(y), len(looped_y))
                chunk_len = end_pos - current_pos
                
                if i > 0 and crossfade_samples > 0:
                    # Apply crossfade
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    fade_out = np.linspace(1, 0, crossfade_samples)
                    
                    # Fade out previous chunk
                    looped_y[current_pos:current_pos + crossfade_samples] *= fade_out
                    # Fade in new chunk
                    y_chunk = y[:chunk_len].copy()
                    y_chunk[:crossfade_samples] *= fade_in
                    looped_y[current_pos:end_pos] += y_chunk
                else:
                    looped_y[current_pos:end_pos] = y[:chunk_len]
                
                current_pos = end_pos - crossfade_samples
                if current_pos >= len(looped_y):
                    break
            
            y = looped_y
        
        # Trim to target duration
        y = y[:int(target_duration * sr)]
        
        # Save temporary file in thread pool
        temp_path = config.TEMP_DIR / f"temp_music_{int(np.random.random() * 1000000)}.wav"
        await loop.run_in_executor(
            None,
            lambda: sf.write(str(temp_path), y, sr)
        )
        
        # Create MoviePy audio clip in thread pool
        audio_clip = await loop.run_in_executor(
            None,
            lambda: AudioFileClip(str(temp_path))
        )
        
        return audio_clip
    
    async def _combine_video_and_music(
        self,
        video: VideoFileClip,
        music: AudioFileClip
    ) -> CompositeVideoClip:
        """Combine video and background music."""
        loop = asyncio.get_event_loop()
        
        # Get original audio if it exists
        original_audio = video.audio if video.audio is not None else None
        
        if original_audio is not None:
            # Mix original audio with music in thread pool
            final_audio = await loop.run_in_executor(
                None,
                lambda: CompositeVideoClip([
                    video.set_audio(original_audio.volumex(1.0)),
                    video.set_audio(music)
                ]).audio
            )
        else:
            final_audio = music
        
        # Create final video with mixed audio in thread pool
        final_video = await loop.run_in_executor(
            None,
            lambda: video.set_audio(final_audio)
        )
        
        return final_video
    
    async def _save_music_metadata(self, video_path: Path, metadata: Dict):
        """Save music metadata for future reference."""
        metadata_path = video_path.with_suffix(".music.json")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._write_metadata(metadata_path, metadata)
        )
        
        logger.info(f"Saved music metadata to: {metadata_path}")
    
    def _write_metadata(self, path: Path, metadata: Dict):
        """Helper method to write metadata to file."""
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2) 