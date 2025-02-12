"""
Image generation providers with fallback support.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
import aiohttp
import numpy as np
from PIL import Image
import cv2
from loguru import logger

from .base_provider import BaseImageProvider
from .error_handler import ErrorHandler, APIError
from . import config

class ImageQualityValidator:
    """Validates image quality metrics."""
    
    @staticmethod
    async def validate_image(image_path: Path) -> Dict[str, Any]:
        """
        Validate image quality.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Failed to read image")
            
            # Calculate metrics
            metrics = {
                "resolution": image.shape[:2],
                "aspect_ratio": image.shape[1] / image.shape[0],
                "blur_score": cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
                "mean_brightness": image.mean(),
                "std_brightness": image.std()
            }
            
            # Validate metrics
            if metrics["blur_score"] < 100:
                raise ValueError("Image too blurry")
            
            if metrics["mean_brightness"] < 20 or metrics["mean_brightness"] > 235:
                raise ValueError("Image brightness out of acceptable range")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            raise APIError(
                f"Image validation failed: {str(e)}",
                "VALIDATION_ERROR"
            )

class FluxDevProvider(BaseImageProvider):
    """FLUX.1-dev image generation provider."""
    
    def get_provider_name(self) -> str:
        return "flux-dev"
    
    async def validate_credentials(self) -> bool:
        """Validate FLUX.1-dev API credentials."""
        if not config.FLUX_API_KEY:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.flux-dev.ai/v1/validate",
                    headers={"Authorization": f"Bearer {config.FLUX_API_KEY}"}
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate image using FLUX.1-dev."""
        try:
            # Prepare request
            url = "https://api.flux-dev.ai/v1/generate/image"
            headers = {"Authorization": f"Bearer {config.FLUX_API_KEY}"}
            
            data = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "negative_prompt": negative_prompt or "",
                "style": style or "realistic",
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(
                            f"FLUX.1-dev API error: {error_text}",
                            "FLUX_API_ERROR"
                        )
                    
                    image_data = await response.read()
            
            # Save image
            output_path.write_bytes(image_data)
            
            # Validate image
            metrics = await self.validate_image(output_path)
            
            return {
                "path": output_path,
                "metadata": {
                    "provider": self.get_provider_name(),
                    "prompt": prompt,
                    "style": style,
                    "dimensions": (width, height),
                    "quality_metrics": metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating image with FLUX.1-dev: {str(e)}")
            raise APIError(
                f"FLUX.1-dev generation failed: {str(e)}",
                "FLUX_GENERATION_ERROR"
            )
    
    async def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """Validate generated image."""
        validator = ImageQualityValidator()
        return await validator.validate_image(image_path)

class StableDiffusionProvider(BaseImageProvider):
    """Local Stable Diffusion provider."""
    
    def get_provider_name(self) -> str:
        return "stable-diffusion"
    
    async def validate_credentials(self) -> bool:
        """Check if local model is available."""
        model_path = Path(config.LOCAL_MODEL_PATH) / "stable-diffusion"
        return model_path.exists()
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate image using local Stable Diffusion."""
        try:
            cmd = [
                str(Path(config.VENV_PATH) / "bin" / "python"),
                str(Path("tools") / "stable_diffusion.py"),
                "--prompt", prompt,
                "--output", str(output_path),
                "--width", str(width),
                "--height", str(height)
            ]
            
            if negative_prompt:
                cmd.extend(["--negative-prompt", negative_prompt])
            
            if style:
                cmd.extend(["--style", style])
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise APIError(
                    f"Stable Diffusion error: {stderr.decode()}",
                    "SD_ERROR"
                )
            
            # Validate image
            metrics = await self.validate_image(output_path)
            
            return {
                "path": output_path,
                "metadata": {
                    "provider": self.get_provider_name(),
                    "prompt": prompt,
                    "style": style,
                    "dimensions": (width, height),
                    "quality_metrics": metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating image with Stable Diffusion: {str(e)}")
            raise APIError(
                f"Stable Diffusion generation failed: {str(e)}",
                "SD_GENERATION_ERROR"
            )
    
    async def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """Validate generated image."""
        validator = ImageQualityValidator()
        return await validator.validate_image(image_path)

class DallEProvider(BaseImageProvider):
    """DALL·E provider."""
    
    def get_provider_name(self) -> str:
        return "dalle"
    
    async def validate_credentials(self) -> bool:
        """Validate OpenAI API credentials."""
        if not config.OPENAI_API_KEY:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate image using DALL·E."""
        try:
            # Prepare request
            url = "https://api.openai.com/v1/images/generations"
            headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
            
            # DALL·E requires specific dimensions
            size = f"{width}x{height}"
            if size not in ["256x256", "512x512", "1024x1024"]:
                size = "1024x1024"  # Default to largest size
            
            data = {
                "prompt": prompt,
                "n": 1,
                "size": size,
                "response_format": "b64_json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(
                            f"DALL·E API error: {error_text}",
                            "DALLE_API_ERROR"
                        )
                    
                    result = await response.json()
            
            # Save image
            import base64
            image_data = base64.b64decode(result["data"][0]["b64_json"])
            output_path.write_bytes(image_data)
            
            # Validate image
            metrics = await self.validate_image(output_path)
            
            return {
                "path": output_path,
                "metadata": {
                    "provider": self.get_provider_name(),
                    "prompt": prompt,
                    "style": style,
                    "dimensions": (width, height),
                    "quality_metrics": metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating image with DALL·E: {str(e)}")
            raise APIError(
                f"DALL·E generation failed: {str(e)}",
                "DALLE_GENERATION_ERROR"
            )
    
    async def validate_image(self, image_path: Path) -> Dict[str, Any]:
        """Validate generated image."""
        validator = ImageQualityValidator()
        return await validator.validate_image(image_path)

class ImageProviderManager:
    """Manages multiple image providers with fallback support."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        
        # Initialize providers in order of preference
        self.providers = [
            FluxDevProvider(self.error_handler),
            StableDiffusionProvider(self.error_handler),
            DallEProvider(self.error_handler)
        ]
    
    async def generate_image(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1024,
        height: int = 1024,
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate image using available providers with fallback.
        
        Args:
            prompt: Text prompt for image generation
            output_path: Path to save the generated image
            width: Image width in pixels
            height: Image height in pixels
            style: Optional style guidance
            negative_prompt: Optional negative prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary containing generation results and metadata
            
        Raises:
            APIError: If all providers fail
        """
        errors = []
        
        for provider in self.providers:
            try:
                # Check if provider is available
                if not await provider.validate_credentials():
                    logger.warning(f"Provider {provider.get_provider_name()} not available")
                    continue
                
                logger.info(f"Trying provider: {provider.get_provider_name()}")
                result = await provider.generate_image(
                    prompt=prompt,
                    output_path=output_path,
                    width=width,
                    height=height,
                    style=style,
                    negative_prompt=negative_prompt,
                    **kwargs
                )
                logger.info(f"Successfully generated image using {provider.get_provider_name()}")
                return result
            except Exception as e:
                logger.warning(f"Provider {provider.get_provider_name()} failed: {str(e)}")
                errors.append({
                    "provider": provider.get_provider_name(),
                    "error": str(e)
                })
        
        # If all providers failed, raise error with details
        raise APIError(
            "All image providers failed",
            "ALL_PROVIDERS_FAILED",
            {"errors": errors}
        ) 