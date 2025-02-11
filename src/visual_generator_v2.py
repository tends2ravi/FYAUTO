"""
Enhanced visual generation module with multi-model support and consistency checks.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
import json
import base64
from PIL import Image
import numpy as np
from together import Together
import requests
from moviepy.editor import VideoClip, concatenate_videoclips

from . import config
from .visual_effects import VisualEffects
from .concept_extractor import ConceptExtractor
from .flux_generator import OptimizedFluxGenerator, FluxOptimizer
from .visual_consistency import VisualConsistencyChecker
from .caching import ContentCache

class VisualStyle:
    """Represents a consistent visual style for a video."""
    
    def __init__(
        self,
        style_name: str,
        color_scheme: List[str],
        composition: str,
        lighting: str,
        texture: str
    ):
        self.style_name = style_name
        self.color_scheme = color_scheme
        self.composition = composition
        self.lighting = lighting
        self.texture = texture
    
    def to_prompt_suffix(self) -> str:
        """Convert style to a prompt suffix."""
        return f", {self.composition}, {self.lighting}, {self.texture}, color scheme: {', '.join(self.color_scheme)}"

class ImageGenerator:
    """Base class for image generation models."""
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        size: Tuple[int, int],
        **kwargs
    ) -> Image.Image:
        """Generate an image."""
        raise NotImplementedError

class FluxGenerator(ImageGenerator):
    """Together AI's FLUX model."""
    
    def __init__(self, api_key: str):
        self.client = Together(api_key=api_key)
        self.model = "black-forest-labs/FLUX.1-schnell-Free"
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        size: Tuple[int, int],
        **kwargs
    ) -> Image.Image:
        try:
            response = self.client.images.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=self.model,
                width=size[0],
                height=size[1],
                steps=kwargs.get("steps", 30),
                n=1,
                response_format="b64_json"
            )
            
            image_bytes = base64.b64decode(response.data[0].b64_json)
            return Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            logger.error(f"Error generating image with FLUX: {str(e)}")
            raise

class StableDiffusionGenerator(ImageGenerator):
    """Stable Diffusion via API."""
    
    def __init__(self, api_key: str, api_url: str):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        size: Tuple[int, int],
        **kwargs
    ) -> Image.Image:
        try:
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": size[0],
                "height": size[1],
                "num_inference_steps": kwargs.get("steps", 30),
                "guidance_scale": kwargs.get("guidance_scale", 7.5)
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            image_bytes = base64.b64decode(response.json()["images"][0])
            return Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            logger.error(f"Error generating image with Stable Diffusion: {str(e)}")
            raise

class VisualGeneratorV2:
    """Enhanced visual content generation with multi-model support."""
    
    def __init__(self):
        # Load API keys
        self.flux_key = config.TOGETHER_API_KEY
        self.sd_key = config.STABLE_DIFFUSION_API_KEY
        self.sd_url = config.STABLE_DIFFUSION_API_URL
        
        # Initialize generators
        self.generators = {
            "flux": OptimizedFluxGenerator(self.flux_key),
            "stable_diffusion": StableDiffusionGenerator(self.sd_key, self.sd_url)
        }
        
        # Define visual styles with enhanced prompting
        self.styles = {
            "modern_minimal": VisualStyle(
                "modern_minimal",
                ["#FFFFFF", "#000000", "#E0E0E0"],
                "minimalist composition with strong visual hierarchy, clean lines, ample negative space",
                "soft diffused lighting with subtle shadows",
                "smooth, matte surfaces with minimal texture"
            ),
            "tech_dynamic": VisualStyle(
                "tech_dynamic",
                ["#0A192F", "#64FFDA", "#112240"],
                "dynamic angles and grid-based layout with futuristic elements",
                "neon accents and high contrast lighting with dramatic shadows",
                "glossy, metallic surfaces with subtle reflections"
            ),
            "nature_organic": VisualStyle(
                "nature_organic",
                ["#2D5A27", "#8FC1B5", "#FAF3DD"],
                "organic shapes and flowing composition with natural elements",
                "natural sunlight with soft, dappled shadows",
                "organic, textured surfaces with rich detail"
            )
        }
        
        # Output directory
        self.output_dir = config.OUTPUT_DIR / "visuals"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.concept_extractor = ConceptExtractor()
        self.consistency_checker = VisualConsistencyChecker(tolerance=0.3)
        self.cache = ContentCache()
    
    async def generate_visuals_for_script(
        self,
        script_data: Dict,
        style: str = "modern_minimal",
        resolution: Tuple[int, int] = (1024, 1024),
        model: str = "flux",
        consistency_check: bool = True
    ) -> Dict[str, List[Path]]:
        """
        Generate visuals for each section of the script.
        
        Args:
            script_data: The script data containing sections
            style: Visual style to use
            resolution: Image resolution
            model: Model to use for generation
            consistency_check: Whether to check visual consistency
            
        Returns:
            Dictionary mapping section titles to lists of image paths
        """
        try:
            # Get style settings
            visual_style = self.styles.get(style, self.styles["modern_minimal"])
            
            # Get generator
            generator = self.generators.get(model, self.generators["flux"])
            
            # Generate images for each section
            visuals = {}
            
            # Generate title card
            title_prompt = self._create_title_prompt(
                script_data["title"],
                visual_style
            )
            title_images = await self._generate_images_batch(
                [title_prompt],
                generator=generator,
                prefix="00_title",
                resolution=resolution,
                style=style
            )
            visuals["title"] = title_images
            
            # Generate section visuals
            for i, section in enumerate(script_data["sections"], 1):
                # Extract key concepts
                concepts = self._extract_key_concepts(section["content"])
                
                # Create prompts for each concept
                section_prompts = [
                    self._create_section_prompt(concept, visual_style, section["title"])
                    for concept in concepts
                ]
                
                # Generate images in parallel
                section_images = await self._generate_images_batch(
                    section_prompts,
                    generator=generator,
                    prefix=f"{i:02d}_{self._sanitize_filename(section['title'])}",
                    resolution=resolution,
                    style=style
                )
                
                # Check visual consistency if enabled
                if consistency_check:
                    section_images = self._ensure_visual_consistency(
                        section_images,
                        visual_style
                    )
                
                visuals[section["title"]] = section_images
            
            return visuals
            
        except Exception as e:
            logger.error(f"Error generating visuals: {str(e)}")
            raise
    
    async def _generate_images_batch(
        self,
        prompts: List[str],
        generator: Union[OptimizedFluxGenerator, 'StableDiffusionGenerator'],
        prefix: str,
        resolution: Tuple[int, int],
        style: str = "realistic"
    ) -> List[Path]:
        """Generate a batch of images in parallel."""
        try:
            # Generate images
            images = await generator.generate_batch(
                prompts,
                size=resolution,
                style=style
            )
            
            # Save images
            image_paths = []
            for i, image in enumerate(images):
                if image:
                    output_path = self.output_dir / f"{prefix}_{i:02d}.png"
                    saved_path = generator.save_image(image, output_path)
                    image_paths.append(saved_path)
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error in batch image generation: {str(e)}")
            return []
    
    def _create_title_prompt(self, title: str, style: VisualStyle) -> str:
        """Create an optimized prompt for the title card."""
        base_prompt = f"""
        Create a visually striking and professional title card for "{title}".
        {style.composition} with a focus on visual impact and professionalism.
        {style.lighting} to create depth and atmosphere.
        {style.texture} to maintain visual interest.
        Designed for maximum viewer engagement and brand consistency.
        No text or words in the image. Suitable for video thumbnail.
        """
        return base_prompt.strip()
    
    def _create_section_prompt(
        self,
        concept: str,
        style: VisualStyle,
        section_title: str
    ) -> str:
        """Create an optimized prompt for a section visual."""
        base_prompt = f"""
        Create a compelling visual representation of {concept} from "{section_title}".
        {style.composition} to effectively communicate the concept.
        {style.lighting} to enhance visual appeal and clarity.
        {style.texture} to add depth and sophistication.
        Professional and polished appearance with clear focus on the main subject.
        No text or words. Suitable for educational content.
        """
        return base_prompt.strip()
    
    def _ensure_visual_consistency(
        self,
        image_paths: List[Path],
        style: VisualStyle
    ) -> List[Path]:
        """
        Check and ensure visual consistency across images.
        
        Args:
            image_paths: List of image paths to check
            style: Visual style being used
            
        Returns:
            List of paths to consistent images
        """
        try:
            if not image_paths:
                return image_paths
            
            # Check consistency scores
            scores = self.consistency_checker.check_consistency(image_paths)
            
            # Log consistency scores
            logger.debug(f"Visual consistency scores: {scores}")
            
            # Check if adjustment is needed
            needs_adjustment = False
            for metric, values in scores.items():
                if any(score < self.consistency_checker.tolerance for score in values):
                    needs_adjustment = True
                    logger.info(f"Inconsistent {metric} detected, will adjust images")
                    break
            
            if needs_adjustment:
                # Create output directory for consistent images
                consistent_dir = self.output_dir / "consistent"
                consistent_dir.mkdir(exist_ok=True)
                
                # Adjust images for consistency
                adjusted_paths = self.consistency_checker.ensure_consistency(
                    image_paths,
                    output_dir=consistent_dir
                )
                
                # Verify adjustment improved consistency
                new_scores = self.consistency_checker.check_consistency(adjusted_paths)
                logger.debug(f"Post-adjustment consistency scores: {new_scores}")
                
                return adjusted_paths
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error ensuring visual consistency: {str(e)}")
            # Return original images if consistency check fails
            return image_paths
    
    def _extract_key_concepts(self, text: str, max_concepts: int = 3) -> List[str]:
        """Extract key concepts that would make good visuals."""
        # Check cache first
        cached_concepts = self.cache.get_concepts(
            text,
            max_concepts=max_concepts,
            min_relevance=0.3
        )
        
        if cached_concepts:
            logger.debug("Using cached concepts")
            return [self._concept_to_prompt(concept) for concept in cached_concepts]
        
        # Extract new concepts
        concepts = self.concept_extractor.extract_concepts(
            text,
            max_concepts=max_concepts,
            min_relevance=0.3
        )
        
        # Cache the concepts
        self.cache.cache_concepts(
            concepts,
            text,
            max_concepts=max_concepts,
            min_relevance=0.3
        )
        
        # Convert concept dictionaries to prompts
        return [self._concept_to_prompt(concept) for concept in concepts]
    
    def _concept_to_prompt(self, concept: Dict) -> str:
        """Convert a concept dictionary to a prompt."""
        prompt = concept['text']
        
        # Add type-specific modifiers
        if concept['type'] == 'entity':
            prompt = f"detailed view of {prompt}"
        elif concept['type'] == 'noun_phrase':
            prompt = f"visual representation of {prompt}"
        
        # Add sentiment-based modifiers
        if concept['sentiment'] > 0.3:
            prompt = f"positive and uplifting {prompt}"
        elif concept['sentiment'] < -0.3:
            prompt = f"dramatic and intense {prompt}"
        
        return prompt
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Convert a string into a valid filename."""
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        filename = "".join(c for c in filename if c in valid_chars)
        filename = filename.replace(" ", "_")
        return filename.lower()
    
    def apply_effects_to_section(
        self,
        image_paths: List[Path],
        duration_per_image: float = 5.0,
        transition_duration: float = 1.0,
        effect_type: str = "ken_burns"
    ) -> VideoClip:
        """
        Apply visual effects to a section's images.
        
        Args:
            image_paths: List of image paths
            duration_per_image: Duration for each image
            transition_duration: Duration of transitions
            effect_type: Type of effect to apply
            
        Returns:
            VideoClip with effects applied
        """
        if not image_paths:
            raise ValueError("No images provided")
        
        # Create clips with effects
        clips = []
        for i, image_path in enumerate(image_paths):
            if effect_type == "ken_burns":
                # Randomize direction for variety
                direction = np.random.choice(["in", "out", "left", "right"])
                clip = VisualEffects.apply_ken_burns(
                    image_path,
                    duration_per_image,
                    direction=direction
                )
            elif effect_type == "parallax":
                direction = np.random.choice(["left", "right"])
                clip = VisualEffects.apply_parallax(
                    image_path,
                    duration_per_image,
                    depth_factor=0.15,
                    direction=direction
                )
            elif effect_type == "floating":
                clip = VisualEffects.apply_floating(
                    image_path,
                    duration_per_image,
                    amplitude=15.0,
                    frequency=0.5
                )
            elif effect_type == "pulse":
                clip = VisualEffects.apply_pulse(
                    image_path,
                    duration_per_image,
                    scale_range=(0.95, 1.05),
                    frequency=0.3
                )
            elif effect_type == "rotate":
                clip = VisualEffects.apply_rotate(
                    image_path,
                    duration_per_image,
                    angle_range=(-3, 3),
                    frequency=0.2
                )
            else:
                # Default to Ken Burns effect
                clip = VisualEffects.apply_ken_burns(
                    image_path,
                    duration_per_image
                )
            
            clips.append(clip)
        
        # Create sequence with transitions
        return VisualEffects.create_transition_sequence(
            clips,
            transition_type=self._get_transition_type(effect_type),
            transition_duration=transition_duration
        )
    
    def generate_video_sequence(
        self,
        visuals: Dict[str, List[Path]],
        duration_per_image: float = 5.0,
        transition_duration: float = 1.0,
        effect_type: str = "ken_burns"
    ) -> VideoClip:
        """
        Generate a complete video sequence from visuals.
        
        Args:
            visuals: Dictionary mapping section titles to image paths
            duration_per_image: Duration for each image
            transition_duration: Duration of transitions
            effect_type: Type of effect to apply
            
        Returns:
            Final video sequence
        """
        if not visuals:
            raise ValueError("No visuals provided")
        
        # Process each section
        section_clips = []
        
        for section_title, image_paths in visuals.items():
            # Vary effects for visual interest
            section_effect = self._get_section_effect(effect_type, section_title)
            
            # Apply effects to section
            section_clip = self.apply_effects_to_section(
                image_paths,
                duration_per_image=duration_per_image,
                transition_duration=transition_duration,
                effect_type=section_effect
            )
            
            section_clips.append(section_clip)
        
        # Create final sequence with varied transitions
        return VisualEffects.create_transition_sequence(
            section_clips,
            transition_type=self._get_section_transition(),
            transition_duration=transition_duration * 1.5  # Slightly longer transitions between sections
        )
    
    def _get_transition_type(self, effect_type: str) -> str:
        """Get appropriate transition type based on effect."""
        effect_transition_map = {
            "ken_burns": "crossfade",
            "parallax": "slide_left",
            "floating": "fade_through_black",
            "pulse": "zoom_in",
            "rotate": "blur"
        }
        return effect_transition_map.get(effect_type, "crossfade")
    
    def _get_section_effect(self, base_effect: str, section_title: str) -> str:
        """Get varied effect for section based on content."""
        # Use different effects for different types of content
        if any(word in section_title.lower() for word in ["intro", "overview", "summary"]):
            return "ken_burns"
        elif any(word in section_title.lower() for word in ["process", "flow", "steps"]):
            return "parallax"
        elif any(word in section_title.lower() for word in ["highlight", "feature"]):
            return "pulse"
        elif any(word in section_title.lower() for word in ["dynamic", "motion"]):
            return "floating"
        elif any(word in section_title.lower() for word in ["transition", "change"]):
            return "rotate"
        else:
            return base_effect
    
    def _get_section_transition(self) -> str:
        """Get a random transition type for variety."""
        transitions = [
            "fade_through_black",
            "zoom_in",
            "slide_left",
            "push_right",
            "blur"
        ]
        return np.random.choice(transitions) 