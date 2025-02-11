"""
Audio generation module using Google Cloud Text-to-Speech.
"""
from pathlib import Path
import time
from typing import Dict, Optional
from loguru import logger
import numpy as np
from google.cloud import texttospeech
import librosa
import soundfile as sf
import os
import asyncio

from . import config
from .error_handler import ErrorHandler, VideoProductionError

class AudioGenerator:
    """Handles audio generation using Google Cloud Text-to-Speech."""
    
    def __init__(self, api_key=None, error_handler=None):
        """Initialize the AudioGenerator.
        
        Args:
            api_key (str, optional): The API key for Google Cloud Text-to-Speech.
                If not provided, will try to get from environment variable.
            error_handler (ErrorHandler, optional): Error handler instance.
                If not provided, will create a new one.
        """
        from src.error_handler import ErrorHandler
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google Cloud API key not found")
            
        self.error_handler = error_handler or ErrorHandler()
        self.client = texttospeech.TextToSpeechClient()
    
    @ErrorHandler.with_retry(retry_on=Exception, error_code="AUDIO_ERROR")
    async def generate_audio_for_script(
        self,
        script_data: Dict,
        voice_name: str = "en-US-Neural2-D",  # Default to a male voice
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Generate audio files for each section of the script.
        
        Args:
            script_data: Dictionary containing script sections
            voice_name: Google TTS voice name to use
            output_dir: Directory to save audio files (defaults to config.OUTPUT_DIR)
            
        Returns:
            Dictionary mapping section titles to audio file paths
        """
        output_dir = output_dir or config.OUTPUT_DIR / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            audio_files = {}
            
            # Generate audio for the hook
            hook_path = await self._generate_audio_segment(
                text=script_data["hook"],
                output_path=output_dir / "00_hook.wav",
                voice_name=voice_name
            )
            audio_files["hook"] = hook_path
            await asyncio.sleep(1)  # Small delay between requests
            
            # Generate audio for each section
            for i, section in enumerate(script_data["sections"], 1):
                section_path = await self._generate_audio_segment(
                    text=section["content"],
                    output_path=output_dir / f"{i:02d}_{self._sanitize_filename(section['title'])}.wav",
                    voice_name=voice_name
                )
                audio_files[section["title"]] = section_path
                await asyncio.sleep(1)  # Small delay between requests
            
            # Generate audio for call to action
            cta_path = await self._generate_audio_segment(
                text=script_data["call_to_action"],
                output_path=output_dir / "99_call_to_action.wav",
                voice_name=voice_name
            )
            audio_files["call_to_action"] = cta_path
            
            return audio_files
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise VideoProductionError(
                "Audio generation failed",
                "AUDIO_ERROR",
                {"error": str(e)}
            )
    
    async def _generate_audio_segment(
        self,
        text: str,
        voice_name: str,
        output_path: Path
    ) -> Path:
        """Generate audio for a segment of text."""
        temp_files = []
        try:
            # Configure the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name
            )
            
            # Configure audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # Use WAV format
                speaking_rate=1.0,
                pitch=0.0,
                volume_gain_db=0.0
            )
            
            # Split text into chunks if needed (Google TTS has a 5000 character limit)
            max_chars = 4800  # Leave some margin
            text_chunks = []
            
            if len(text) > max_chars:
                # Split at sentence boundaries
                sentences = text.split('.')
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip() + "."
                    if len(current_chunk) + len(sentence) <= max_chars:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            text_chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    text_chunks.append(current_chunk)
            else:
                text_chunks = [text]
            
            logger.info(f"Generating audio for text with {len(text)} characters (split into {len(text_chunks)} chunks)")
            
            # Generate audio for each chunk
            for i, chunk in enumerate(text_chunks):
                temp_path = output_path.parent / f"temp_{int(time.time())}_{i}_{output_path.name}"
                
                # Create synthesis input
                synthesis_input = texttospeech.SynthesisInput(text=chunk)
                
                # Perform text-to-speech request in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.synthesize_speech(
                        input=synthesis_input,
                        voice=voice,
                        audio_config=audio_config
                    )
                )
                
                # Save chunk audio
                with open(temp_path, "wb") as f:
                    f.write(response.audio_content)
                
                temp_files.append(temp_path)
                logger.info(f"Generated audio chunk {i+1}/{len(text_chunks)}")
                
                # Add a small delay between requests
                if i < len(text_chunks) - 1:
                    await asyncio.sleep(1)
            
            # Combine audio chunks if needed
            if len(temp_files) > 1:
                # Load and combine audio chunks using librosa
                combined_y = None
                sr = None
                
                for temp_file in temp_files:
                    # Run librosa.load in a thread pool to avoid blocking
                    y, this_sr = await loop.run_in_executor(
                        None,
                        lambda: librosa.load(str(temp_file))
                    )
                    if combined_y is None:
                        combined_y = y
                        sr = this_sr
                    else:
                        combined_y = np.concatenate([combined_y, y])
                
                # Save to a temporary file first
                temp_output = output_path.parent / f"temp_final_{int(time.time())}_{output_path.name}"
                await loop.run_in_executor(
                    None,
                    lambda: sf.write(str(temp_output), combined_y, sr)
                )
                
                # Clean up chunk files
                for temp_file in temp_files:
                    try:
                        temp_file.unlink()
                    except:
                        pass
                
                # Remove existing output file if it exists
                if output_path.exists():
                    output_path.unlink()
                
                # Move the temporary file to the final location
                temp_output.rename(output_path)
            else:
                # Remove existing output file if it exists
                if output_path.exists():
                    output_path.unlink()
                
                # Move the single temp file
                temp_files[0].rename(output_path)
            
            logger.info(f"Generated audio file: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            # Clean up any temp files
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except:
                    pass
            raise
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Convert a string into a valid filename."""
        # Remove invalid characters
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        filename = "".join(c for c in filename if c in valid_chars)
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        return filename.lower() 