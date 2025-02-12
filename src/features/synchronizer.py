"""
Video synchronization and timing optimization module.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, vfx
import librosa
import soundfile as sf
from scipy.signal import correlate
import cv2
import asyncio
import tempfile

from .error_handler import ErrorHandler, VideoProductionError

class VideoSynchronizer:
    """Handles audio-visual synchronization and timing optimization."""
    
    # Available transition types
    TRANSITIONS = {
        "crossfade": "_create_crossfade_transition",
        "fade_through_black": "_create_fade_through_black_transition",
        "slide": "_create_slide_transition",
        "zoom": "_create_zoom_transition",
        "wipe": "_create_wipe_transition"
    }
    
    def __init__(self, error_handler: ErrorHandler):
        """Initialize the synchronizer.
        
        Args:
            error_handler: Error handler instance for retries and error management
        """
        self.error_handler = error_handler
        self.logger = logger.bind(context=self.__class__.__name__)
        self.fps = 30  # Default frame rate
        self.crossfade_duration = 0.5  # Default crossfade duration
        self.max_retries = 3  # Maximum number of retries for operations
        self.retry_delay = 1.0  # Delay between retries in seconds
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="SYNC_ERROR")
    async def synchronize_content(
        self,
        video_clips: List[VideoFileClip],
        audio_files: Dict[str, Path],
        timing_data: Dict[str, float],
        target_duration: float
    ) -> VideoFileClip:
        """
        Synchronize video clips with audio and optimize timing.
        
        Args:
            video_clips: List of video clips to synchronize
            audio_files: Dictionary of audio file paths
            timing_data: Dictionary of timing information
            target_duration: Target video duration
            
        Returns:
            Synchronized video clip
        """
        try:
            # Analyze audio beats and energy
            audio_analysis = await self._analyze_audio(audio_files)
            
            # Optimize clip timing based on audio
            optimized_clips = await self._optimize_clip_timing(
                video_clips,
                audio_analysis,
                timing_data
            )
            
            # Synchronize clips with audio
            synchronized_clips = await self._synchronize_clips(
                optimized_clips,
                audio_analysis
            )
            
            # Apply transitions based on audio energy
            final_video = await self._apply_audio_based_transitions(
                synchronized_clips,
                audio_analysis
            )
            
            # Adjust to target duration
            final_video = await self._adjust_duration(
                final_video,
                target_duration
            )
            
            return final_video
            
        except Exception as e:
            self.logger.error(f"Error synchronizing content: {str(e)}")
            raise VideoProductionError(
                "Content synchronization failed",
                "SYNC_ERROR",
                {"error": str(e)}
            )
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="AUDIO_ERROR")
    async def _analyze_audio(
        self,
        audio_files: Dict[str, Path]
    ) -> Dict[str, Dict]:
        """
        Analyze audio files for beats and energy.
        
        Args:
            audio_files: Dictionary mapping section names to audio file paths
            
        Returns:
            Dictionary of audio analysis results
        """
        try:
            analysis_results = {}
            loop = asyncio.get_event_loop()
            
            for section, audio_path in audio_files.items():
                try:
                    # Load audio file in thread pool
                    y, sr = await loop.run_in_executor(
                        None,
                        lambda: librosa.load(str(audio_path))
                    )
                    
                    # Analyze tempo and beats in thread pool
                    tempo, beat_frames = await loop.run_in_executor(
                        None,
                        lambda: librosa.beat.beat_track(y=y, sr=sr)
                    )
                    
                    # Convert beat frames to times
                    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                    
                    # Calculate onset strength in thread pool
                    onset_env = await loop.run_in_executor(
                        None,
                        lambda: librosa.onset.onset_strength(y=y, sr=sr)
                    )
                    
                    # Calculate energy in thread pool
                    energy = await loop.run_in_executor(
                        None,
                        lambda: np.abs(librosa.stft(y))
                    )
                    energy = np.mean(energy, axis=0)
                    
                    analysis_results[section] = {
                        "tempo": float(tempo),
                        "onset_strength": onset_env.tolist(),
                        "energy": energy.tolist(),
                        "beat_times": beat_times.tolist(),
                        "duration": float(len(y) / sr)
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing audio for section {section}: {str(e)}")
                    # Generate fallback analysis
                    analysis_results[section] = await self._generate_fallback_analysis_for_section(audio_path)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio: {str(e)}")
            # Fallback to basic analysis
            return await self._generate_fallback_analysis(audio_files)
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="TIMING_ERROR")
    async def _optimize_clip_timing(
        self,
        clips: List[VideoFileClip],
        audio_analysis: Dict[str, Dict],
        timing_data: Dict[str, float]
    ) -> List[VideoFileClip]:
        """Optimize clip timing based on audio analysis and target durations."""
        try:
            optimized_clips = []
            loop = asyncio.get_event_loop()
            
            for i, clip in enumerate(clips):
                section = list(timing_data.keys())[i % len(timing_data)]
                target_duration = timing_data[section]
                
                # Calculate speed adjustment
                current_duration = clip.duration
                speed_factor = current_duration / target_duration
                
                # Adjust clip speed to match target duration in thread pool
                adjusted_clip = await loop.run_in_executor(
                    None,
                    lambda: clip.fx(vfx.speedx, 1.0 / speed_factor)
                )
                
                # Verify duration
                if not abs(adjusted_clip.duration - target_duration) < 0.1:
                    self.logger.warning(f"Duration mismatch after speed adjustment: {adjusted_clip.duration} vs {target_duration}")
                    # Fine-tune duration if needed
                    if adjusted_clip.duration > target_duration:
                        adjusted_clip = await loop.run_in_executor(
                            None,
                            lambda: adjusted_clip.subclip(0, target_duration)
                        )
                    else:
                        # Slow down slightly to match target duration
                        fine_tune_factor = adjusted_clip.duration / target_duration
                        adjusted_clip = await loop.run_in_executor(
                            None,
                            lambda: adjusted_clip.fx(vfx.speedx, fine_tune_factor)
                        )
                
                optimized_clips.append(adjusted_clip)
            
            return optimized_clips
            
        except Exception as e:
            self.logger.error(f"Error optimizing clip timing: {str(e)}")
            raise VideoProductionError("Timing optimization failed")
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="SYNC_ERROR")
    async def _synchronize_clips(
        self,
        clips: List[VideoFileClip],
        audio_analysis: Dict[str, Dict]
    ) -> List[VideoFileClip]:
        """
        Synchronize clips with audio analysis.
        
        Args:
            clips: List of video clips to synchronize
            audio_analysis: Dictionary of audio analysis results
            
        Returns:
            List of synchronized video clips
        """
        try:
            synchronized_clips = []
            current_time = 0.0
            loop = asyncio.get_event_loop()
            
            for i, clip in enumerate(clips):
                section = list(audio_analysis.keys())[i % len(audio_analysis)]
                section_analysis = audio_analysis[section]
                
                # Find the nearest beat time for clip start
                beat_times = section_analysis["beat_times"]
                if not beat_times:
                    self.logger.warning(f"No beat times available for section {section}, using default timing")
                    clip_start = current_time
                else:
                    # Find the nearest beat time that's after current_time
                    next_beats = [t for t in beat_times if t > current_time]
                    clip_start = next_beats[0] if next_beats else current_time
                
                # Set clip timing in thread pool
                clip = await loop.run_in_executor(
                    None,
                    lambda: clip.set_start(clip_start)
                )
                synchronized_clips.append(clip)
                current_time = clip_start + clip.duration
            
            return synchronized_clips
            
        except Exception as e:
            self.logger.error(f"Error synchronizing clips: {str(e)}")
            raise VideoProductionError("Clip synchronization failed")
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="TRANSITION_ERROR")
    async def _apply_audio_based_transitions(
        self,
        clips: List[VideoFileClip],
        audio_analysis: Dict[str, Dict]
    ) -> VideoFileClip:
        """
        Apply transitions based on audio energy.
        
        Args:
            clips: List of video clips to transition between
            audio_analysis: Dictionary of audio analysis results
            
        Returns:
            Final video with transitions
        """
        try:
            # Store clips for cleanup
            self._clips = clips.copy()
            
            # Process transitions in parallel
            transitions = await self._process_transitions_parallel(clips, audio_analysis)
            
            # Build final video
            final_clips = []
            for i, clip in enumerate(clips):
                final_clips.append(clip)
                if i < len(transitions):
                    final_clips.append(transitions[i])
            
            # Concatenate all clips in thread pool
            loop = asyncio.get_event_loop()
            final_video = await loop.run_in_executor(
                None,
                lambda: concatenate_videoclips(final_clips)
            )
            
            return final_video
            
        except Exception as e:
            self.logger.error(f"Error applying transitions: {str(e)}")
            raise VideoProductionError("Transition application failed")
    
    async def _adjust_duration(
        self,
        video: VideoFileClip,
        target_duration: float
    ) -> VideoFileClip:
        """Adjust video duration to match target."""
        loop = asyncio.get_event_loop()
        
        try:
            current_duration = video.duration
            if abs(current_duration - target_duration) < 0.1:
                return video
            
            # Adjust speed to match target duration
            speed_factor = current_duration / target_duration
            adjusted_video = await loop.run_in_executor(
                None,
                lambda: video.fx(vfx.speedx, 1.0 / speed_factor)
            )
            
            return adjusted_video
            
        except Exception as e:
            self.logger.error(f"Error adjusting duration: {str(e)}")
            return video
    
    def _calculate_optimal_duration(
        self,
        analysis: Dict,
        target_duration: float
    ) -> float:
        """Calculate optimal duration based on audio analysis."""
        try:
            # Get beat times
            beat_times = analysis.get("beat_times", [])
            if not beat_times:
                return target_duration
            
            # Find the nearest beat time to target duration
            nearest_beat = min(beat_times, key=lambda x: abs(x - target_duration))
            
            # If nearest beat is within 10% of target, use it
            if abs(nearest_beat - target_duration) <= target_duration * 0.1:
                return nearest_beat
            
            return target_duration
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal duration: {str(e)}")
            return target_duration
    
    async def _adjust_clip_speed(
        self,
        clip: VideoFileClip,
        target_duration: float,
        speed_factor: Optional[float] = None
    ) -> VideoFileClip:
        """Adjust clip speed to match target duration."""
        try:
            loop = asyncio.get_event_loop()
            
            if speed_factor is None:
                speed_factor = clip.duration / target_duration
            
            # Apply speed change in thread pool
            adjusted_clip = await loop.run_in_executor(
                None,
                lambda: clip.fx(vfx.speedx, speed_factor)
            )
            
            return adjusted_clip
            
        except Exception as e:
            self.logger.error(f"Error adjusting clip speed: {str(e)}")
            return clip
    
    async def _find_optimal_start_time(
        self,
        clip: VideoFileClip,
        beat_times: np.ndarray
    ) -> float:
        """Find optimal start time aligned with beats."""
        try:
            if len(beat_times) == 0:
                return 0.0
            
            # Find the nearest beat time that's after current_time
            next_beats = beat_times[beat_times > 0]
            if len(next_beats) == 0:
                return 0.0
            
            return float(next_beats[0])
            
        except Exception as e:
            self.logger.error(f"Error finding optimal start time: {str(e)}")
            return 0.0
    
    async def _apply_energy_based_effects(
        self,
        clip: VideoFileClip,
        energy: np.ndarray
    ) -> VideoFileClip:
        """Apply effects based on audio energy."""
        try:
            loop = asyncio.get_event_loop()
            
            # Normalize energy
            energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
            
            # Create effect transform
            def effect_transform(get_frame, t):
                # Get base frame
                frame = get_frame(t)
                
                # Get energy at current time
                energy_idx = int(t * len(energy) / clip.duration)
                current_energy = energy[min(energy_idx, len(energy) - 1)]
                
                # Apply zoom based on energy
                zoom_factor = 1.0 + current_energy * 0.1
                h, w = frame.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom_factor)
                frame = cv2.warpAffine(frame, M, (w, h))
                
                return frame
            
            # Apply transform in thread pool
            transformed_clip = await loop.run_in_executor(
                None,
                lambda: clip.fl(effect_transform)
            )
            
            return transformed_clip
            
        except Exception as e:
            self.logger.error(f"Error applying energy-based effects: {str(e)}")
            return clip
    
    def _calculate_transition_duration(
        self,
        current_analysis: Dict,
        next_analysis: Dict
    ) -> float:
        """Calculate optimal transition duration based on audio analysis."""
        try:
            # Get average energy levels
            current_energy = np.mean(current_analysis.get("energy", [0.5]))
            next_energy = np.mean(next_analysis.get("energy", [0.5]))
            
            # Calculate transition duration based on energy
            base_duration = 1.0  # Base transition duration
            energy_factor = (current_energy + next_energy) / 2
            
            # Scale duration between 0.5 and 1.5 seconds based on energy
            duration = base_duration * (0.5 + energy_factor)
            
            return min(max(duration, 0.5), 1.5)  # Clamp between 0.5 and 1.5 seconds
            
        except Exception as e:
            self.logger.error(f"Error calculating transition duration: {str(e)}")
            return 1.0  # Default duration
    
    def _select_transition_type(
        self,
        current_analysis: Dict,
        next_analysis: Dict
    ) -> str:
        """Select transition type based on audio characteristics."""
        try:
            # Get average energy levels
            current_energy = np.mean(current_analysis.get("energy", [0.5]))
            next_energy = np.mean(next_analysis.get("energy", [0.5]))
            
            # Calculate energy change
            energy_change = next_energy - current_energy
            
            # Select transition based on energy change
            if abs(energy_change) > 0.5:
                return "zoom"  # Big energy change
            elif abs(energy_change) > 0.3:
                return "slide"  # Moderate change
            elif abs(energy_change) > 0.1:
                return "crossfade"  # Small change
            else:
                return "fade_through_black"  # Minimal change
                
        except Exception as e:
            self.logger.error(f"Error selecting transition type: {str(e)}")
            return "crossfade"  # Default transition
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="TRANSITION_ERROR")
    async def _create_crossfade_transition(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip,
        duration: float = 1.0
    ) -> VideoFileClip:
        """Create a crossfade transition between clips."""
        try:
            loop = asyncio.get_event_loop()
            
            # Create crossfade in thread pool
            transition = await loop.run_in_executor(
                None,
                lambda: CompositeVideoClip([
                    clip1.crossfadeout(duration),
                    clip2.crossfadein(duration).set_start(0)
                ]).set_duration(duration)
            )
            
            return transition
            
        except Exception as e:
            self.logger.error(f"Error creating crossfade transition: {str(e)}")
            return None

    @ErrorHandler.with_retry(retry_on=Exception, error_code="TRANSITION_ERROR")
    async def _create_fade_through_black_transition(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip,
        duration: float = 1.0
    ) -> VideoFileClip:
        """Create a fade through black transition."""
        try:
            loop = asyncio.get_event_loop()
            
            # Create fade out and fade in clips in thread pool
            fade_out = await loop.run_in_executor(
                None,
                lambda: clip1.crossfadeout(duration/2)
            )
            fade_in = await loop.run_in_executor(
                None,
                lambda: clip2.crossfadein(duration/2)
            )
            
            # Concatenate the clips
            transition = await loop.run_in_executor(
                None,
                lambda: concatenate_videoclips([fade_out, fade_in])
            )
            
            return transition
            
        except Exception as e:
            self.logger.error(f"Error creating fade through black transition: {str(e)}")
            return None

    @ErrorHandler.with_retry(retry_on=Exception, error_code="TRANSITION_ERROR")
    async def _create_slide_transition(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip,
        duration: float = 1.0,
        direction: str = "left"
    ) -> VideoFileClip:
        """Create a slide transition between clips."""
        try:
            loop = asyncio.get_event_loop()
            
            # Determine slide direction based on energy change
            if direction == "left":
                pos1 = lambda t: (-(t/duration) * clip1.w, 0)
                pos2 = lambda t: (clip1.w - (t/duration) * clip1.w, 0)
            else:  # right
                pos1 = lambda t: ((t/duration) * clip1.w, 0)
                pos2 = lambda t: (-clip1.w + (t/duration) * clip1.w, 0)
            
            # Create slide transition in thread pool
            transition = await loop.run_in_executor(
                None,
                lambda: CompositeVideoClip([
                    clip1.set_position(pos1),
                    clip2.set_position(pos2)
                ], size=clip1.size).set_duration(duration)
            )
            
            return transition
            
        except Exception as e:
            self.logger.error(f"Error creating slide transition: {str(e)}")
            return None

    @ErrorHandler.with_retry(retry_on=Exception, error_code="TRANSITION_ERROR")
    async def _create_zoom_transition(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip,
        duration: float = 1.0
    ) -> VideoFileClip:
        """Create a zoom transition between clips."""
        try:
            loop = asyncio.get_event_loop()
            
            def zoom_transform(get_frame, t):
                progress = t / duration
                if progress < 0.5:
                    # Zoom out first clip
                    frame = get_frame(t)
                    zoom = 1.0 + progress
                    h, w = frame.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom)
                    return cv2.warpAffine(frame, M, (w, h))
                else:
                    # Zoom in second clip
                    frame = get_frame(t)
                    zoom = 2.0 - progress
                    h, w = frame.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom)
                    return cv2.warpAffine(frame, M, (w, h))
            
            # Create zoom transition in thread pool
            transition = await loop.run_in_executor(
                None,
                lambda: concatenate_videoclips([
                    clip1.subclip(0, duration/2).fl(zoom_transform),
                    clip2.subclip(-duration/2).fl(zoom_transform)
                ])
            )
            
            return transition
            
        except Exception as e:
            self.logger.error(f"Error creating zoom transition: {str(e)}")
            return None

    @ErrorHandler.with_retry(retry_on=Exception, error_code="TRANSITION_ERROR")
    async def _create_wipe_transition(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip,
        duration: float = 1.0,
        direction: str = "left"
    ) -> VideoFileClip:
        """Create a wipe transition between clips."""
        try:
            loop = asyncio.get_event_loop()
            
            def wipe_transform(get_frame, t):
                progress = t / duration
                frame1 = clip1.get_frame(t)
                frame2 = clip2.get_frame(t)
                
                h, w = frame1.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                
                if direction == "left":
                    wipe_pos = int(w * progress)
                    mask[:, :wipe_pos] = 255
                else:  # right
                    wipe_pos = int(w * (1 - progress))
                    mask[:, wipe_pos:] = 255
                
                # Create composite frame
                result = frame1.copy()
                result[mask == 255] = frame2[mask == 255]
                
                return result
            
            # Create wipe transition in thread pool
            transition = await loop.run_in_executor(
                None,
                lambda: VideoFileClip(wipe_transform, duration=duration)
            )
            
            return transition
            
        except Exception as e:
            self.logger.error(f"Error creating wipe transition: {str(e)}")
            return None
    
    async def _generate_fallback_analysis(
        self,
        audio_files: Dict[str, Path]
    ) -> Dict[str, Dict]:
        """Generate fallback audio analysis when librosa is not available."""
        analysis_results = {}
        for section, audio_path in audio_files.items():
            analysis_results[section] = await self._generate_fallback_analysis_for_section(audio_path)
        return analysis_results
    
    async def _generate_fallback_analysis_for_section(
        self,
        audio_path: Path
    ) -> Dict:
        """Generate fallback analysis for a single audio file."""
        loop = asyncio.get_event_loop()
        
        # Get basic duration using soundfile
        duration = await loop.run_in_executor(
            None,
            lambda: sf.info(str(audio_path)).duration
        )
        
        # Generate synthetic beat times
        beat_interval = 0.5  # 120 BPM
        beat_times = np.arange(0, duration, beat_interval).tolist()
        
        return {
            "tempo": 120.0,
            "onset_strength": [0.5] * int(duration * 2),  # 2 values per second
            "energy": [0.5] * int(duration * 2),
            "beat_times": beat_times,
            "duration": float(duration)
        }

    async def cleanup(self):
        """Clean up resources and temporary files."""
        try:
            loop = asyncio.get_event_loop()
            
            # Close all video clips
            for clip in getattr(self, '_clips', []):
                if hasattr(clip, 'close'):
                    await loop.run_in_executor(None, clip.close)
            
            # Clear stored clips
            self._clips = []
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    async def _process_transitions_parallel(
        self,
        clips: List[VideoFileClip],
        audio_analysis: Dict[str, Dict]
    ) -> List[VideoFileClip]:
        """Process transitions in parallel for better performance."""
        try:
            tasks = []
            loop = asyncio.get_event_loop()
            
            for i in range(len(clips) - 1):
                clip = clips[i]
                next_clip = clips[i + 1]
                section = list(audio_analysis.keys())[i % len(audio_analysis)]
                next_section = list(audio_analysis.keys())[(i + 1) % len(audio_analysis)]
                
                # Calculate transition parameters
                duration = self._calculate_transition_duration(
                    audio_analysis[section],
                    audio_analysis[next_section]
                )
                transition_type = self._select_transition_type(
                    audio_analysis[section],
                    audio_analysis[next_section]
                )
                
                # Create transition task
                task = asyncio.create_task(self._create_transition_with_fallback(
                    clip, next_clip, transition_type, duration
                ))
                tasks.append(task)
            
            # Wait for all transitions to complete
            transitions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any errors
            final_transitions = []
            for i, transition in enumerate(transitions):
                if isinstance(transition, Exception):
                    self.logger.error(f"Error in transition {i}: {str(transition)}")
                    # Create fallback transition
                    transition = await self._create_crossfade_transition(
                        clips[i], clips[i + 1], 1.0
                    )
                final_transitions.append(transition)
            
            return final_transitions
            
        except Exception as e:
            self.logger.error(f"Error processing parallel transitions: {str(e)}")
            raise VideoProductionError("Parallel transition processing failed")

    async def _create_transition_with_fallback(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip,
        transition_type: str,
        duration: float
    ) -> VideoFileClip:
        """Create a transition with automatic fallback to simpler transitions."""
        try:
            # Optimize memory usage for large clips
            clip1 = await self._optimize_memory_usage(clip1)
            clip2 = await self._optimize_memory_usage(clip2)
            
            # Validate clips before attempting transition
            if not self._validate_clips_for_transition(clip1, clip2):
                self.logger.warning("Clips incompatible for complex transition, using crossfade")
                return await self._create_crossfade_transition(clip1, clip2, duration)
            
            # Determine transition direction for applicable types
            if transition_type in ["slide", "wipe"]:
                direction = await self._determine_transition_direction(clip1, clip2)
                if direction == "fade":
                    transition_type = "fade_through_black"
            
            # Get transition method
            transition_method_name = self.TRANSITIONS.get(transition_type, "crossfade")
            transition_method = getattr(self, transition_method_name)
            
            # Try creating the transition
            try:
                transition = await transition_method(clip1, clip2, duration)
                if transition is None:
                    raise ValueError(f"Transition {transition_type} returned None")
                return transition
            except Exception as e:
                # Attempt recovery with fallback transitions
                return await self._recover_failed_transition(
                    clip1, clip2, e, transition_type
                )
            
        except Exception as e:
            self.logger.error(f"Error creating transition with fallback: {str(e)}")
            return await self._create_crossfade_transition(clip1, clip2, duration)

    def _validate_clips_for_transition(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip
    ) -> bool:
        """Validate that two clips are compatible for complex transitions."""
        try:
            # Check if clips exist and have valid durations
            if not clip1 or not clip2:
                return False
            if clip1.duration <= 0 or clip2.duration <= 0:
                return False
            
            # Check if clips have same dimensions
            if clip1.size != clip2.size:
                return False
            
            # Check if clips have valid frame rates
            if not hasattr(clip1, 'fps') or not hasattr(clip2, 'fps'):
                return False
            if clip1.fps <= 0 or clip2.fps <= 0:
                return False
            
            # Check if clips have valid frames
            try:
                _ = clip1.get_frame(0)
                _ = clip2.get_frame(0)
            except Exception:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating clips: {str(e)}")
            return False

    async def _optimize_memory_usage(self, clip: VideoFileClip) -> VideoFileClip:
        """Optimize memory usage for large video clips."""
        try:
            loop = asyncio.get_event_loop()
            
            # Get clip properties
            fps = clip.fps
            duration = clip.duration
            size = clip.size
            
            # Calculate target bitrate based on resolution
            target_bitrate = size[0] * size[1] * fps * 0.1  # Rough estimate
            
            # Create a temporary file for the optimized clip
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            # Write clip to temporary file with optimized settings
            await loop.run_in_executor(
                None,
                lambda: clip.write_videofile(
                    str(temp_path),
                    fps=fps,
                    bitrate=str(int(target_bitrate)),
                    preset='ultrafast',  # Fast encoding for temp file
                    audio=False  # Audio will be handled separately
                )
            )
            
            # Load optimized clip
            optimized_clip = await loop.run_in_executor(
                None,
                lambda: VideoFileClip(str(temp_path))
            )
            
            # Clean up
            temp_path.unlink()
            
            return optimized_clip
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {str(e)}")
            return clip

    async def _determine_transition_direction(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip
    ) -> str:
        """Determine optimal transition direction based on scene analysis."""
        try:
            loop = asyncio.get_event_loop()
            
            # Get last frame of first clip and first frame of second clip
            last_frame = await loop.run_in_executor(None, clip1.get_frame, clip1.duration - 0.1)
            first_frame = await loop.run_in_executor(None, clip2.get_frame, 0)
            
            # Convert frames to grayscale for analysis
            last_gray = cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY)
            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                last_gray, first_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Analyze dominant motion direction
            avg_flow_x = np.mean(flow[..., 0])
            
            # Determine direction based on flow
            if abs(avg_flow_x) < 0.5:
                return "fade"  # Little motion, use fade
            elif avg_flow_x > 0:
                return "right"
            else:
                return "left"
            
        except Exception as e:
            self.logger.error(f"Error determining transition direction: {str(e)}")
            return "left"  # Default direction

    async def _recover_failed_transition(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip,
        error: Exception,
        attempted_type: str
    ) -> VideoFileClip:
        """Recover from failed transition with graceful fallback."""
        try:
            self.logger.warning(f"Recovering from failed {attempted_type} transition: {str(error)}")
            
            # Try simpler transitions in order of complexity
            fallback_types = [
                ("slide", self._create_slide_transition),
                ("fade_through_black", self._create_fade_through_black_transition),
                ("crossfade", self._create_crossfade_transition)
            ]
            
            for transition_type, transition_method in fallback_types:
                if transition_type != attempted_type:
                    try:
                        self.logger.info(f"Attempting {transition_type} as fallback")
                        transition = await transition_method(clip1, clip2, 1.0)
                        if transition is not None:
                            return transition
                    except Exception as e:
                        self.logger.warning(f"Fallback {transition_type} failed: {str(e)}")
            
            # If all fallbacks fail, create a simple cut
            self.logger.warning("All fallbacks failed, using simple cut")
            return await self._create_simple_cut(clip1, clip2)
            
        except Exception as e:
            self.logger.error(f"Error in transition recovery: {str(e)}")
            return None

    async def _create_simple_cut(
        self,
        clip1: VideoFileClip,
        clip2: VideoFileClip
    ) -> VideoFileClip:
        """Create a simple cut between clips as last resort."""
        try:
            loop = asyncio.get_event_loop()
            
            # Create a brief overlap
            overlap = 0.1
            clip1 = clip1.subclip(0, clip1.duration - overlap/2)
            clip2 = clip2.subclip(overlap/2)
            
            # Concatenate with minimal processing
            cut = await loop.run_in_executor(
                None,
                lambda: concatenate_videoclips([clip1, clip2])
            )
            
            return cut
            
        except Exception as e:
            self.logger.error(f"Error creating simple cut: {str(e)}")
            return None 