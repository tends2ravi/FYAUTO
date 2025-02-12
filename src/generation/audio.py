"""
Unified audio generation system with caching and quality validation.
"""
from typing import Dict, Optional, Any, List, Type
from pathlib import Path
import asyncio
import json
import hashlib
import redis
from loguru import logger

from ..providers.audio import (
    BaseAudioProvider,
    GoogleTTSProvider,
    ElevenLabsProvider,
    CoquiTTSProvider
)
from ..core.errors import (
    AudioGenerationError,
    ValidationError,
    ResourceError
)
from ..core.config import (
    CACHE_TTL,
    AUDIO_OUTPUT_DIR,
    CACHE_DIR,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD
)

class AudioGenerator:
    """Unified audio generation system with caching and fallback support."""
    
    def __init__(
        self,
        providers: Optional[List[Type[BaseAudioProvider]]] = None,
        cache_ttl: int = CACHE_TTL,
        output_dir: Path = AUDIO_OUTPUT_DIR
    ):
        self.providers = providers or [
            GoogleTTSProvider,
            ElevenLabsProvider,
            CoquiTTSProvider
        ]
        self.provider_instances = []
        self.cache_ttl = cache_ttl
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
    
    async def initialize(self):
        """Initialize providers and validate credentials."""
        for provider_class in self.providers:
            provider = provider_class()
            if await provider.validate_credentials():
                self.provider_instances.append(provider)
                logger.info(f"Initialized {provider.get_provider_name()} provider")
            else:
                logger.warning(f"Failed to initialize {provider_class.__name__}")
    
    def _get_cache_key(self, text: str, voice_id: str, **kwargs) -> str:
        """Generate cache key for audio generation request."""
        params = {
            'text': text,
            'voice_id': voice_id,
            **kwargs
        }
        params_str = json.dumps(params, sort_keys=True)
        return f"audio:{hashlib.sha256(params_str.encode()).hexdigest()}"
    
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        validate: bool = True,
        use_cache: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate speech using available providers with caching."""
        if not self.provider_instances:
            await self.initialize()
        
        if not self.provider_instances:
            raise AudioGenerationError(
                "No audio providers available",
                "NO_PROVIDERS_ERROR"
            )
        
        # Check cache if enabled
        cache_key = self._get_cache_key(text, voice_id, language=language, speed=speed, pitch=pitch, **kwargs)
        if use_cache:
            cached_path = self.redis.get(cache_key)
            if cached_path:
                cached_path = Path(cached_path)
                if cached_path.exists():
                    logger.info(f"Using cached audio: {cached_path}")
                    if validate:
                        validation_result = await self.provider_instances[0].validate_audio(cached_path)
                        return {
                            'output_path': str(cached_path),
                            'validation': validation_result,
                            'cached': True
                        }
                    return {
                        'output_path': str(cached_path),
                        'cached': True
                    }
        
        # Try each provider in sequence
        last_error = None
        for provider in self.provider_instances:
            try:
                output_path = self.output_dir / f"{provider.get_provider_name()}_{cache_key[-8:]}.wav"
                
                result = await provider.generate_speech(
                    text=text,
                    voice_id=voice_id,
                    output_path=output_path,
                    language=language,
                    speed=speed,
                    pitch=pitch,
                    **kwargs
                )
                
                # Cache the result
                if use_cache:
                    self.redis.set(cache_key, str(output_path), ex=self.cache_ttl)
                
                return {
                    **result,
                    'cached': False
                }
                
            except Exception as e:
                logger.warning(f"Provider {provider.get_provider_name()} failed: {str(e)}")
                last_error = e
                continue
        
        # If all providers failed
        raise AudioGenerationError(
            "All providers failed to generate speech",
            "ALL_PROVIDERS_FAILED",
            {'last_error': str(last_error)}
        )
    
    async def generate_speech_for_script(
        self,
        script: List[Dict[str, str]],
        voice_id: str,
        language: str = "en",
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Generate speech for multiple scenes in a script."""
        results = []
        for scene in script:
            try:
                result = await self.generate_speech(
                    text=scene['text'],
                    voice_id=voice_id,
                    language=language,
                    **kwargs
                )
                results.append({
                    'scene_id': scene.get('id'),
                    'text': scene['text'],
                    **result
                })
            except Exception as e:
                logger.error(f"Failed to generate speech for scene: {str(e)}")
                results.append({
                    'scene_id': scene.get('id'),
                    'text': scene['text'],
                    'error': str(e)
                })
        return results
    
    async def get_available_voices(
        self,
        language: str = "en",
        provider_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get available voices from all providers or a specific provider."""
        if not self.provider_instances:
            await self.initialize()
        
        voices = {}
        for provider in self.provider_instances:
            if provider_name and provider.get_provider_name() != provider_name:
                continue
            
            try:
                provider_voices = await provider.get_available_voices(language)
                voices[provider.get_provider_name()] = provider_voices
            except Exception as e:
                logger.warning(f"Failed to get voices from {provider.get_provider_name()}: {str(e)}")
                voices[provider.get_provider_name()] = []
        
        return voices
    
    async def cleanup(self):
        """Clean up resources and expired cache entries."""
        # Clean up provider resources
        for provider in self.provider_instances:
            if hasattr(provider, 'cleanup'):
                await provider.cleanup()
        
        # Clean up old cache files
        cache_dir = Path(CACHE_DIR)
        if cache_dir.exists():
            for file in cache_dir.glob("*.wav"):
                if not self.redis.exists(f"audio:{file.stem}"):
                    try:
                        file.unlink()
                        logger.info(f"Removed expired cache file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {file}: {str(e)}") 