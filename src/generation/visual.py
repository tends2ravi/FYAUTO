"""
Unified visual generation system with support for multiple models and styles.
Incorporates FLUX and other visual generation capabilities with fallbacks.
"""
# Standard library imports
from typing import Dict, Optional, Any, List, TypeVar, cast, Sequence
from pathlib import Path
import asyncio
import json
import hashlib
import pickle
from datetime import timedelta

# Third-party imports
import numpy as np
from PIL import Image
import torch
import redis
from loguru import logger
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

# Local imports
from ..core.errors import ErrorHandler, APIError, ResourceError, ValidationError
from ..core.config import (
    FLUX_API_KEY,
    FLUX_API_URL,
    FLUX_MODEL_VERSION,
    FLUX_DEV_MODEL_VERSION,
    IMAGE_OUTPUT_DIR,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    CACHE_TTL
)
from ..providers.image import ImageProviderManager, ImageQualityValidator

# Type variables
T = TypeVar('T', bound=Dict[str, Any])

# Constants
DEFAULT_STYLES = {
    "standard": "high quality, detailed, photorealistic",
    "minimal": "minimalist, clean, simple lines",
    "dramatic": "cinematic, dramatic lighting, intense atmosphere",
    "artistic": "creative, artistic, stylized interpretation",
    "modern": "contemporary, sleek, professional look",
    "vintage": "retro, nostalgic, classic aesthetic"
}

class VisualGenerator:
    """Unified visual generation system with fallback options."""
    
    def __init__(
        self,
        error_handler: Optional[ErrorHandler] = None,
        model_version: str = FLUX_MODEL_VERSION,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the visual generator.
        
        Args:
            error_handler: Optional error handler for managing errors
            model_version: FLUX model version to use
            output_dir: Optional custom output directory
            device: Optional device to use (cuda/cpu)
        """
        self.error_handler = error_handler or ErrorHandler()
        self.model_version = model_version
        self.output_dir = output_dir or Path(IMAGE_OUTPUT_DIR)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API session
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {FLUX_API_KEY}"}
        )
        
        # Initialize image provider manager and validator
        self.image_manager = ImageProviderManager(self.error_handler)
        self.validator = ImageQualityValidator()
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False  # Keep binary for pickle
            )
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
            self.redis_client = None
    
    async def __aenter__(self) -> 'VisualGenerator':
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=lambda e: isinstance(e, (APIError, ResourceError))
    )
    async def generate_image(
        self,
        prompt: str,
        style: str = "standard",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        output_format: str = "png"
    ) -> Path:
        """
        Generate an image with fallback options.
        
        Args:
            prompt: Text description of the desired image
            style: Visual style to apply
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            negative_prompt: What to avoid in the image
            seed: Random seed for reproducibility
            output_format: Output image format
            
        Returns:
            Path to the generated image
            
        Raises:
            APIError: If all generation attempts fail
            ResourceError: If required resources are unavailable
        """
        try:
            # Try primary FLUX model
            return await self._generate_with_flux(
                prompt=prompt,
                style=style,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                seed=seed,
                output_format=output_format
            )
        except Exception as e:
            logger.warning(f"Primary FLUX model failed: {str(e)}, trying fallback...")
            
            # Try FLUX.1-dev model as fallback
            try:
                return await self._generate_with_flux_dev(
                    prompt=prompt,
                    style=style,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    output_format=output_format
                )
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {str(e2)}")
                raise APIError(
                    "All visual generation attempts failed",
                    "VISUAL_GENERATION_ERROR",
                    {
                        "primary_error": str(e),
                        "fallback_error": str(e2),
                        "prompt": prompt,
                        "style": style
                    }
                )
    
    async def _generate_with_flux(
        self,
        prompt: str,
        style: str,
        **kwargs
    ) -> Path:
        """Generate image using primary FLUX model."""
        payload = {
            "prompt": self._enhance_prompt(prompt, style),
            "model_version": self.model_version,
            **kwargs
        }
        
        async with self.session.post(
            f"{FLUX_API_URL}/generate",
            json=payload
        ) as response:
            if response.status != 200:
                raise APIError(
                    f"FLUX API error: {response.status}",
                    "FLUX_API_ERROR",
                    {"response": await response.text()}
                )
            
            data = await response.json()
            image_data = data["image"]
            
            # Save image
            output_path = self.output_dir / f"flux_{hash(prompt)}_{kwargs.get('seed', 0)}.{kwargs['output_format']}"
            image = Image.open(image_data)
            image.save(output_path)
            return output_path
    
    async def _generate_with_flux_dev(
        self,
        prompt: str,
        style: str,
        **kwargs
    ) -> Path:
        """Generate image using FLUX.1-dev model."""
        payload = {
            "prompt": self._enhance_prompt(prompt, style),
            "model_version": FLUX_DEV_MODEL_VERSION,
            "quality_preference": "speed",  # Faster generation for fallback
            **kwargs
        }
        
        async with self.session.post(
            f"{FLUX_API_URL}/generate",
            json=payload
        ) as response:
            if response.status != 200:
                raise APIError(
                    f"FLUX.1-dev API error: {response.status}",
                    "FLUX_DEV_API_ERROR",
                    {"response": await response.text()}
                )
            
            data = await response.json()
            image_data = data["image"]
            
            # Save image
            output_path = self.output_dir / f"flux_dev_{hash(prompt)}_{kwargs.get('seed', 0)}.{kwargs['output_format']}"
            image = Image.open(image_data)
            image.save(output_path)
            return output_path
    
    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """Enhance prompt with style-specific modifiers."""
        style_modifiers = {
            "standard": "high quality, detailed, photorealistic",
            "minimal": "minimalist, clean, simple lines",
            "dramatic": "cinematic, dramatic lighting, intense atmosphere"
        }
        
        return f"{prompt}, {style_modifiers.get(style, style_modifiers['standard'])}"
    
    async def generate_sequence(
        self,
        prompts: List[str],
        style: str = "standard",
        **kwargs
    ) -> List[Path]:
        """Generate a sequence of related images."""
        image_paths = []
        base_seed = kwargs.pop("seed", None) or np.random.randint(0, 1000000)
        
        for i, prompt in enumerate(prompts):
            # Use consistent but varying seeds for sequence coherence
            seed = base_seed + i
            
            image_path = await self.generate_image(
                prompt=prompt,
                style=style,
                seed=seed,
                **kwargs
            )
            image_paths.append(image_path)
        
        return image_paths
    
    async def apply_style_transfer(
        self,
        image_path: Path,
        style: str,
        strength: float = 1.0
    ) -> Path:
        """Apply style transfer to an existing image."""
        # Implementation of style transfer
        # This is a placeholder for future implementation
        pass

    def _get_cache_key(self, prompt: str, style: Optional[str], dimensions: tuple) -> str:
        """Generate cache key from parameters."""
        params = f"{prompt}:{style or 'default'}:{dimensions[0]}x{dimensions[1]}"
        return f"visual_cache:{hashlib.sha256(params.encode()).hexdigest()}"
    
    async def _get_from_cache(
        self,
        prompt: str,
        style: Optional[str],
        dimensions: tuple
    ) -> Optional[Dict[str, Any]]:
        """Try to get visual from cache."""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(prompt, style, dimensions)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = pickle.loads(cached_data)
                path = Path(data["path"])
                if path.exists():
                    logger.info("Found visual in cache")
                    return data
            
            return None
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None
    
    async def _save_to_cache(
        self,
        prompt: str,
        style: Optional[str],
        dimensions: tuple,
        result: Dict[str, Any]
    ) -> None:
        """Save visual data to cache."""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(prompt, style, dimensions)
            self.redis_client.setex(
                cache_key,
                timedelta(hours=24),  # Cache for 24 hours
                pickle.dumps(result)
            )
            logger.info("Saved visual data to cache")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    async def generate_visual(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        validate: bool = True,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate visual with caching and quality validation.
        
        Args:
            prompt: Text prompt for visual generation
            output_path: Path to save the generated visual
            width: Visual width in pixels
            height: Visual height in pixels
            style: Optional style guidance
            negative_prompt: Optional negative prompt
            validate: Whether to validate visual quality
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary containing:
                - path: Path to the generated visual
                - metadata: Generation metadata
                - provider: Name of the provider used
                - cached: Whether the visual was from cache
        """
        dimensions = (width, height)
        
        # Try to get from cache first
        cached_result = await self._get_from_cache(prompt, style, dimensions)
        if cached_result:
            if validate:
                try:
                    metrics = await self.validator.validate_image(Path(cached_result["path"]))
                    cached_result["metadata"]["quality_metrics"] = metrics
                except Exception as e:
                    logger.warning(f"Cached visual failed validation: {e}. Generating new one.")
                    cached_result = None
            
            if cached_result:
                return {**cached_result, "cached": True}
        
        try:
            # Generate visual using image provider manager
            result = await self.image_manager.generate_image(
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                style=style,
                negative_prompt=negative_prompt,
                **kwargs
            )
            
            # Save to cache
            await self._save_to_cache(prompt, style, dimensions, result)
            
            return {**result, "cached": False}
            
        except Exception as e:
            logger.error(f"Error generating visual: {str(e)}")
            raise
    
    async def generate_visuals_for_script(
        self,
        script_data: Dict[str, Any],
        output_dir: Path,
        style: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Path]:
        """
        Generate visuals for all scenes in a script.
        
        Args:
            script_data: Script data containing scenes
            output_dir: Directory to save generated visuals
            style: Optional style guidance
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping scene IDs to visual paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        visual_paths = {}
        
        for scene in script_data["scenes"]:
            scene_id = scene["id"]
            
            for i, visual in enumerate(scene["visuals"]):
                try:
                    output_path = output_dir / f"{scene_id}_visual_{i}.png"
                    
                    result = await self.generate_visual(
                        prompt=visual["description"],
                        output_path=output_path,
                        style=style,
                        **kwargs
                    )
                    
                    visual_paths[f"{scene_id}_visual_{i}"] = result["path"]
                    
                except Exception as e:
                    logger.error(f"Error generating visual for scene {scene_id}: {str(e)}")
                    raise APIError(
                        f"Failed to generate visual for scene {scene_id}",
                        "SCENE_VISUAL_ERROR",
                        {"scene_id": scene_id, "error": str(e)}
                    )
        
        return visual_paths 