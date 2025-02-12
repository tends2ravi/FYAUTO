"""
Background music system for video production.
Handles music selection, volume adjustment, and audio mixing.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
from moviepy.editor import AudioFileClip, CompositeAudioClip
from loguru import logger
import asyncio
import tempfile
import os

from .error_handler import ErrorHandler, VideoProductionError
from . import config

class BackgroundMusic:
    """Handles background music processing and mixing."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the background music system.
        
        Args:
            error_handler: Error handler instance for retries and error management
        """
        self.error_handler = error_handler or ErrorHandler()
        self.logger = logger.bind(context=self.__class__.__name__)
        self._temp_files: List[Path] = []
        self.music_dir = config.BASE_DIR / "assets" / "music"
        self.music_dir.mkdir(parents=True, exist_ok=True)
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="MUSIC_ERROR")
    async def select_music(
        self,
        mood: str = "upbeat",
        duration: float = 60.0,
        music_library_path: Optional[Path] = None
    ) -> AudioFileClip:
        """Select appropriate background music.
        
        Args:
            mood: Desired mood of the music
            duration: Target duration in seconds
            music_library_path: Path to music library directory
            
        Returns:
            Selected audio clip
        """
        try:
            if music_library_path is None:
                music_library_path = Path("assets/music")
            
            # Get list of available music files
            music_files = list(music_library_path.glob("*.mp3"))
            if not music_files:
                raise VideoProductionError("No music files found in library")
            
            # Analyze each music file
            analysis_results = []
            loop = asyncio.get_event_loop()
            
            for music_file in music_files:
                try:
                    # Load and analyze audio in thread pool
                    y, sr = await loop.run_in_executor(
                        None,
                        lambda: librosa.load(str(music_file))
                    )
                    
                    # Calculate tempo and energy
                    tempo, _ = await loop.run_in_executor(
                        None,
                        lambda: librosa.beat.beat_track(y=y, sr=sr)
                    )
                    
                    energy = np.mean(np.abs(librosa.stft(y)))
                    
                    analysis_results.append({
                        "path": music_file,
                        "tempo": tempo,
                        "energy": energy,
                        "duration": len(y) / sr
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing music file {music_file}: {str(e)}")
            
            if not analysis_results:
                raise VideoProductionError("No valid music files found")
            
            # Select appropriate music based on mood
            selected_file = await self._select_by_mood(analysis_results, mood)
            
            # Load selected music
            audio_clip = await loop.run_in_executor(
                None,
                lambda: AudioFileClip(str(selected_file))
            )
            
            # Adjust duration
            if audio_clip.duration < duration:
                # Loop the audio if needed
                num_loops = int(np.ceil(duration / audio_clip.duration))
                audio_clip = audio_clip.loop(num_loops)
            
            # Trim to exact duration
            audio_clip = audio_clip.subclip(0, duration)
            
            return audio_clip
            
        except Exception as e:
            self.logger.error(f"Error selecting music: {str(e)}")
            raise VideoProductionError("Music selection failed")
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="AUDIO_ERROR")
    async def adjust_volume(
        self,
        audio_clip: AudioFileClip,
        target_db: float = -20.0
    ) -> AudioFileClip:
        """Adjust audio volume to target level.
        
        Args:
            audio_clip: Audio clip to adjust
            target_db: Target volume level in dB
            
        Returns:
            Volume-adjusted audio clip
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Get audio data
            audio_array = audio_clip.to_soundarray()
            
            # Calculate current RMS level
            current_rms = np.sqrt(np.mean(audio_array**2))
            current_db = 20 * np.log10(current_rms)
            
            # Calculate volume adjustment
            db_change = target_db - current_db
            volume_factor = 10 ** (db_change / 20)
            
            # Apply volume adjustment in thread pool
            adjusted_clip = await loop.run_in_executor(
                None,
                lambda: audio_clip.volumex(volume_factor)
            )
            
            return adjusted_clip
            
        except Exception as e:
            self.logger.error(f"Error adjusting volume: {str(e)}")
            return audio_clip
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="AUDIO_ERROR")
    async def mix_audio(
        self,
        main_audio: AudioFileClip,
        background_music: AudioFileClip,
        music_volume: float = 0.3
    ) -> AudioFileClip:
        """Mix main audio with background music.
        
        Args:
            main_audio: Main audio clip
            background_music: Background music clip
            music_volume: Volume factor for background music
            
        Returns:
            Mixed audio clip
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Adjust background music volume
            background_music = await loop.run_in_executor(
                None,
                lambda: background_music.volumex(music_volume)
            )
            
            # Create composite audio in thread pool
            mixed_audio = await loop.run_in_executor(
                None,
                lambda: CompositeAudioClip([main_audio, background_music])
            )
            
            return mixed_audio
            
        except Exception as e:
            self.logger.error(f"Error mixing audio: {str(e)}")
            return main_audio
    
    async def _select_by_mood(
        self,
        analysis_results: List[Dict],
        mood: str
    ) -> Path:
        """Select music file based on mood."""
        try:
            # Define mood characteristics
            mood_map = {
                "upbeat": {"min_tempo": 120, "min_energy": 0.6},
                "calm": {"max_tempo": 100, "max_energy": 0.4},
                "energetic": {"min_tempo": 140, "min_energy": 0.8},
                "relaxed": {"max_tempo": 90, "max_energy": 0.3}
            }
            
            mood_params = mood_map.get(mood, mood_map["upbeat"])
            
            # Filter music files by mood parameters
            suitable_tracks = []
            for track in analysis_results:
                matches_mood = True
                
                if "min_tempo" in mood_params and track["tempo"] < mood_params["min_tempo"]:
                    matches_mood = False
                if "max_tempo" in mood_params and track["tempo"] > mood_params["max_tempo"]:
                    matches_mood = False
                if "min_energy" in mood_params and track["energy"] < mood_params["min_energy"]:
                    matches_mood = False
                if "max_energy" in mood_params and track["energy"] > mood_params["max_energy"]:
                    matches_mood = False
                
                if matches_mood:
                    suitable_tracks.append(track)
            
            if not suitable_tracks:
                self.logger.warning(f"No tracks match mood {mood}, using first available track")
                return analysis_results[0]["path"]
            
            # Select random track from suitable ones
            selected_track = np.random.choice(suitable_tracks)
            return selected_track["path"]
            
        except Exception as e:
            self.logger.error(f"Error selecting by mood: {str(e)}")
            return analysis_results[0]["path"]
    
    async def cleanup(self):
        """Clean up temporary files."""
        try:
            for temp_file in self._temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            self._temp_files.clear()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    async def generate_background_music(
        self,
        duration: float,
        style: str = "ambient",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate background music for the video.
        
        Args:
            duration: Target duration in seconds
            style: Music style (ambient, upbeat, etc.)
            output_dir: Directory to save music file
            
        Returns:
            Path to the generated music file
        """
        try:
            output_dir = output_dir or config.OUTPUT_DIR / "music"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # For now, use a pre-made music track based on style
            source_track = self._get_music_track(style)
            if not source_track:
                raise VideoProductionError(
                    f"No music track found for style: {style}",
                    "MUSIC_NOT_FOUND"
                )
            
            # Load the source track
            music = AudioFileClip(str(source_track))
            
            # Adjust duration
            if music.duration < duration:
                # Loop the track if it's too short
                loops_needed = int(np.ceil(duration / music.duration))
                music = CompositeAudioClip([music] * loops_needed)
            
            # Trim to exact duration
            music = music.subclip(0, duration)
            
            # Add fade in/out
            fade_duration = min(3, duration / 10)  # 3 seconds or 10% of duration
            music = music.audio_fadein(fade_duration).audio_fadeout(fade_duration)
            
            # Lower volume for background
            music = music.volumex(0.3)  # 30% volume
            
            # Save the final track
            output_path = output_dir / f"background_{style}_{int(duration)}s.mp3"
            music.write_audiofile(str(output_path), fps=44100, bitrate="192k", logger=None)
            
            # Clean up
            music.close()
            
            logger.info(f"Generated background music: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating background music: {str(e)}")
            raise VideoProductionError(
                "Background music generation failed",
                "MUSIC_ERROR",
                {"error": str(e)}
            )
    
    def _get_music_track(self, style: str) -> Optional[Path]:
        """Get a pre-made music track for the given style."""
        # Map styles to filenames
        style_map = {
            "ambient": "ambient_background.mp3",
            "upbeat": "upbeat_background.mp3",
            "suspenseful": "suspenseful_background.mp3",
            "tech": "tech_background.mp3",
            "superhero": "superhero_background.mp3"
        }
        
        # Get filename for style, fallback to ambient
        filename = style_map.get(style, style_map["ambient"])
        track_path = self.music_dir / filename
        
        # Check if file exists
        if track_path.exists():
            return track_path
        
        # If not found, try to create a default track
        return self._create_default_track(track_path)
    
    def _create_default_track(self, output_path: Path) -> Optional[Path]:
        """Create a default music track if no pre-made track is found."""
        try:
            # Create a simple sine wave as default background
            from scipy.io import wavfile
            import numpy as np
            
            # Generate 10 seconds of sine wave at 440 Hz
            sample_rate = 44100
            duration = 10
            t = np.linspace(0, duration, int(sample_rate * duration))
            data = np.sin(2 * np.pi * 440 * t) * 0.3  # 30% amplitude
            
            # Convert to 16-bit PCM
            data = (data * 32767).astype(np.int16)
            
            # Save as WAV
            temp_wav = output_path.with_suffix('.wav')
            wavfile.write(str(temp_wav), sample_rate, data)
            
            # Convert to MP3 using moviepy
            audio = AudioFileClip(str(temp_wav))
            audio.write_audiofile(str(output_path), fps=44100, bitrate="192k", logger=None)
            
            # Clean up WAV file
            temp_wav.unlink()
            audio.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create default music track: {e}")
            return None 