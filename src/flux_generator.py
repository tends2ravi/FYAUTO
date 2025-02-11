"""
Optimized FLUX AI image generation module.
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from loguru import logger
import base64
from PIL import Image
import io
import asyncio
import concurrent.futures
from together import Together
import json

from .caching import ContentCache

class FluxOptimizer:
    """Handles prompt optimization for FLUX AI."""
    
    # Style modifiers for different types of content
    STYLE_MODIFIERS = {
        "realistic": "highly detailed, photorealistic, 8k uhd, professional photography, sharp focus",
        "artistic": "digital art, highly detailed, intricate, elegant composition, vibrant colors",
        "minimal": "minimalist design, clean lines, simple composition, elegant, modern",
        "technical": "technical illustration, detailed diagram, precise, professional, informative"
    }
    
    # Quality boosters to enhance image quality
    QUALITY_BOOSTERS = [
        "masterpiece",
        "best quality",
        "highly detailed",
        "sharp focus",
        "professional",
        "8k uhd"
    ]
    
    # Negative prompts to avoid common issues
    NEGATIVE_PROMPTS = [
        "blurry",
        "low quality",
        "pixelated",
        "watermark",
        "signature",
        "out of frame",
        "poorly drawn",
        "distorted",
        "deformed",
        "amateur"
    ]
    
    @classmethod
    def optimize_prompt(
        cls,
        base_prompt: str,
        style: str = "realistic",
        emphasis: float = 1.0
    ) -> Tuple[str, str]:
        """
        Optimize a prompt for FLUX AI.
        
        Args:
            base_prompt: Base prompt to optimize
            style: Style modifier to use
            emphasis: Emphasis level for style (0.0-2.0)
            
        Returns:
            Tuple of (optimized prompt, negative prompt)
        """
        # Clean and normalize base prompt
        base_prompt = base_prompt.strip()
        if not base_prompt.endswith((".", ",")):
            base_prompt += ","
        
        # Add style modifiers
        style_text = cls.STYLE_MODIFIERS.get(style, cls.STYLE_MODIFIERS["realistic"])
        
        # Add quality boosters with emphasis
        quality_text = ", ".join(cls.QUALITY_BOOSTERS[:int(len(cls.QUALITY_BOOSTERS) * emphasis)])
        
        # Combine all parts
        optimized_prompt = f"{base_prompt} {style_text}, {quality_text}"
        
        # Create negative prompt
        negative_prompt = ", ".join(cls.NEGATIVE_PROMPTS)
        
        return optimized_prompt, negative_prompt

class OptimizedFluxGenerator:
    """Optimized image generation using FLUX AI."""
    
    def __init__(self, api_key: str, max_workers: int = 4):
        self.client = Together(api_key=api_key)
        self.model = "black-forest-labs/FLUX.1-schnell-Free"
        self.max_workers = max_workers
        self.optimizer = FluxOptimizer()
        self.cache = ContentCache()
    
    async def generate_batch(
        self,
        prompts: List[str],
        size: Tuple[int, int] = (1024, 1024),
        style: str = "realistic",
        batch_size: int = 4
    ) -> List[Image.Image]:
        """
        Generate multiple images in parallel.
        
        Args:
            prompts: List of prompts
            size: Image size (width, height)
            style: Style to apply
            batch_size: Number of simultaneous generations
            
        Returns:
            List of generated images
        """
        width, height = size
        images = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_futures = []
            
            # Create thread pool for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit generation tasks
                for prompt in batch_prompts:
                    # Check cache first
                    cached_prompt = self.cache.get_optimized_prompt(
                        prompt,
                        style=style,
                        emphasis=1.0
                    )
                    
                    if cached_prompt:
                        optimized_prompt = cached_prompt["optimized_prompt"]
                        negative_prompt = cached_prompt["negative_prompt"]
                    else:
                        optimized_prompt, negative_prompt = FluxOptimizer.optimize_prompt(
                            prompt,
                            style=style
                        )
                        # Cache the optimized prompt
                        self.cache.cache_optimized_prompt(
                            optimized_prompt,
                            negative_prompt,
                            prompt,
                            style=style,
                            emphasis=1.0
                        )
                    
                    # Check image cache
                    cached_image_path = self.cache.get_image(
                        optimized_prompt,
                        model=self.model,
                        size=size,
                        style=style
                    )
                    
                    if cached_image_path:
                        # Load cached image
                        try:
                            image = Image.open(cached_image_path)
                            images.append(image)
                            continue
                        except Exception as e:
                            logger.warning(f"Failed to load cached image: {e}")
                    
                    # Generate new image if not cached
                    future = executor.submit(
                        self._generate_single,
                        optimized_prompt,
                        negative_prompt,
                        width,
                        height
                    )
                    batch_futures.append((future, optimized_prompt))
                
                # Collect results
                for future, prompt in batch_futures:
                    try:
                        image = future.result()
                        if image:
                            # Cache the generated image
                            self.cache.cache_image(
                                image,
                                prompt=prompt,
                                model=self.model,
                                size=size,
                                style=style
                            )
                            images.append(image)
                    except Exception as e:
                        logger.error(f"Error generating image: {str(e)}")
                        continue
        
        return images
    
    def _generate_single(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int
    ) -> Optional[Image.Image]:
        """Generate a single image."""
        try:
            # Log generation attempt
            logger.debug(f"Generating image with prompt: {prompt[:100]}...")
            
            # Make API call
            response = self.client.images.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=self.model,
                width=width,
                height=height,
                steps=30,  # Increased for better quality
                n=1,
                response_format="b64_json"
            )
            
            # Process response
            image_bytes = base64.b64decode(response.data[0].b64_json)
            image = Image.open(io.BytesIO(image_bytes))
            
            logger.debug("Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error in single image generation: {str(e)}")
            return None
    
    def save_image(
        self,
        image: Image.Image,
        output_path: Path,
        format: str = "PNG",
        optimize: bool = True
    ) -> Path:
        """
        Save an image with optimization.
        
        Args:
            image: PIL Image to save
            output_path: Path to save to
            format: Image format
            optimize: Whether to optimize the image
            
        Returns:
            Path to saved image
        """
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with optimization if supported
            image.save(
                output_path,
                format=format,
                optimize=optimize,
                quality=95  # High quality
            )
            
            logger.debug(f"Saved image to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise 