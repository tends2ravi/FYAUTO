"""
Audio generation providers for the video production system.
Combines TTS and audio processing functionality.
"""
from typing import Dict, Optional, Any, List
from pathlib import Path
import asyncio
import aiohttp
import json
import soundfile as sf
import numpy as np
import librosa
from google.cloud import texttospeech
from elevenlabs import generate, set_api_key, voices
import torch
from TTS.api import TTS

from .base import BaseAudioProvider
from ..core.errors import AudioGenerationError, APIError
from ..core.config import (
    ELEVENLABS_API_KEY,
    GOOGLE_CLOUD_CREDENTIALS,
    CACHE_DIR
)

class GoogleTTSProvider(BaseAudioProvider):
    """Google Cloud Text-to-Speech provider."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = texttospeech.TextToSpeechClient()
        
    def get_provider_name(self) -> str:
        return "google_tts"
    
    async def validate_credentials(self) -> bool:
        try:
            # Test API access
            voices = self.client.list_voices()
            return len(voices.voices) > 0
        except Exception as e:
            self.error_handler.handle_api_error(e, {
                'provider': self.get_provider_name()
            })
            return False
    
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        language: str = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        try:
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language,
                name=voice_id
            )
            
            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=speed,
                pitch=pitch
            )
            
            # Perform the text-to-speech request
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.synthesize_speech,
                synthesis_input,
                voice,
                audio_config
            )
            
            # Write the response to the output file
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
            
            # Validate the generated audio
            validation_result = await self.validate_audio(output_path)
            
            return {
                'provider': self.get_provider_name(),
                'output_path': str(output_path),
                'validation': validation_result
            }
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to generate speech with Google TTS: {str(e)}",
                "GOOGLE_TTS_ERROR",
                {
                    'text': text,
                    'voice_id': voice_id,
                    'language': language
                }
            )
    
    async def validate_audio(self, audio_path: Path) -> Dict[str, Any]:
        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path))
            
            # Calculate audio metrics
            duration = librosa.get_duration(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)[0].mean()
            zero_crossings = librosa.zero_crossings(y).sum()
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'rms_level': float(rms),
                'zero_crossings': int(zero_crossings)
            }
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to validate audio: {str(e)}",
                "AUDIO_VALIDATION_ERROR",
                {'audio_path': str(audio_path)}
            )
    
    async def get_available_voices(self, language: str = "en") -> List[Dict[str, Any]]:
        try:
            voices = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.list_voices
            )
            
            return [{
                'id': voice.name,
                'name': voice.name,
                'language': voice.language_codes[0],
                'gender': voice.ssml_gender.name.lower()
            } for voice in voices.voices if language in voice.language_codes]
            
        except Exception as e:
            raise APIError(
                f"Failed to get Google TTS voices: {str(e)}",
                "GOOGLE_TTS_API_ERROR"
            )

class ElevenLabsProvider(BaseAudioProvider):
    """ElevenLabs Text-to-Speech provider."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_api_key(ELEVENLABS_API_KEY)
    
    def get_provider_name(self) -> str:
        return "elevenlabs"
    
    async def validate_credentials(self) -> bool:
        try:
            # Test API access by getting available voices
            voice_list = voices()
            return len(voice_list) > 0
        except Exception as e:
            self.error_handler.handle_api_error(e, {
                'provider': self.get_provider_name()
            })
            return False
    
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        language: str = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        try:
            # Generate audio using ElevenLabs API
            audio = await asyncio.get_event_loop().run_in_executor(
                None,
                generate,
                text,
                voice_id,
                model="eleven_multilingual_v2"
            )
            
            # Save audio to file
            with open(output_path, "wb") as out:
                out.write(audio)
            
            # Validate the generated audio
            validation_result = await self.validate_audio(output_path)
            
            return {
                'provider': self.get_provider_name(),
                'output_path': str(output_path),
                'validation': validation_result
            }
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to generate speech with ElevenLabs: {str(e)}",
                "ELEVENLABS_ERROR",
                {
                    'text': text,
                    'voice_id': voice_id,
                    'language': language
                }
            )
    
    async def validate_audio(self, audio_path: Path) -> Dict[str, Any]:
        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path))
            
            # Calculate audio metrics
            duration = librosa.get_duration(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)[0].mean()
            zero_crossings = librosa.zero_crossings(y).sum()
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'rms_level': float(rms),
                'zero_crossings': int(zero_crossings)
            }
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to validate audio: {str(e)}",
                "AUDIO_VALIDATION_ERROR",
                {'audio_path': str(audio_path)}
            )
    
    async def get_available_voices(self, language: str = "en") -> List[Dict[str, Any]]:
        try:
            voice_list = await asyncio.get_event_loop().run_in_executor(
                None,
                voices
            )
            
            return [{
                'id': voice.voice_id,
                'name': voice.name,
                'language': language,  # ElevenLabs doesn't provide language info
                'gender': 'neutral'  # ElevenLabs doesn't provide gender info
            } for voice in voice_list]
            
        except Exception as e:
            raise APIError(
                f"Failed to get ElevenLabs voices: {str(e)}",
                "ELEVENLABS_API_ERROR"
            )

class CoquiTTSProvider(BaseAudioProvider):
    """Coqui TTS provider for offline text-to-speech generation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.model_path = CACHE_DIR / "coqui_model"
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def get_provider_name(self) -> str:
        return "coqui_tts"
    
    async def validate_credentials(self) -> bool:
        try:
            if self.model is None:
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            return True
        except Exception as e:
            self.error_handler.handle_api_error(e, {
                'provider': self.get_provider_name()
            })
            return False
    
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        language: str = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        **kwargs: Any
    ) -> Dict[str, Any]:
        try:
            if self.model is None:
                await self.validate_credentials()
            
            # Generate audio using Coqui TTS
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.model.tts_to_file,
                text,
                speaker_wav=voice_id,
                language=language,
                file_path=str(output_path)
            )
            
            # Validate the generated audio
            validation_result = await self.validate_audio(output_path)
            
            return {
                'provider': self.get_provider_name(),
                'output_path': str(output_path),
                'validation': validation_result
            }
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to generate speech with Coqui TTS: {str(e)}",
                "COQUI_TTS_ERROR",
                {
                    'text': text,
                    'voice_id': voice_id,
                    'language': language
                }
            )
    
    async def validate_audio(self, audio_path: Path) -> Dict[str, Any]:
        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path))
            
            # Calculate audio metrics
            duration = librosa.get_duration(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)[0].mean()
            zero_crossings = librosa.zero_crossings(y).sum()
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'rms_level': float(rms),
                'zero_crossings': int(zero_crossings)
            }
            
        except Exception as e:
            raise AudioGenerationError(
                f"Failed to validate audio: {str(e)}",
                "AUDIO_VALIDATION_ERROR",
                {'audio_path': str(audio_path)}
            )
    
    async def get_available_voices(self, language: str = "en") -> List[Dict[str, Any]]:
        # Coqui TTS uses reference audio files as voices
        voice_dir = self.model_path / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        voices = []
        for voice_file in voice_dir.glob("*.wav"):
            voices.append({
                'id': str(voice_file),
                'name': voice_file.stem,
                'language': language,
                'gender': 'neutral'  # No gender info available
            })
        
        return voices 