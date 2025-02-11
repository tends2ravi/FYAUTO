"""
Visual generation module using Together AI.
"""
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from together import Together
import os
import asyncio

from . import config

class VisualGenerator:
    """Handles visual content generation using Together AI."""
    
    def __init__(self, api_key=None, error_handler=None):
        """Initialize the VisualGenerator.
        
        Args:
            api_key (str, optional): The API key for Together AI.
                If not provided, will try to get from environment variable.
            error_handler (ErrorHandler, optional): Error handler instance.
                If not provided, will create a new one.
        """
        from src.error_handler import ErrorHandler
        
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together AI API key not found")
            
        self.error_handler = error_handler or ErrorHandler()
        self.client = Together(api_key=self.api_key)
        self.logger = logger.bind(context=self.__class__.__name__)
        self.logger.info(f"Initialized VisualGenerator with API key: {self.api_key[:8]}...")
    
    async def generate_visuals_for_script(
        self,
        script_data: Dict,
        style: str = "modern minimalist",
        resolution: Tuple[int, int] = (1024, 1024),
        output_dir: Optional[Path] = None
    ) -> Dict[str, List[Path]]:
        """
        Generate visuals for each section of the script.
        
        Args:
            script_data: The script data containing title, sections, etc.
            style: Visual style to use
            resolution: Image resolution (width, height)
            output_dir: Directory to save generated images
            
        Returns:
            Dictionary mapping section titles to lists of image paths
        """
        output_dir = output_dir or config.OUTPUT_DIR / "visuals"
        output_dir.mkdir(parents=True, exist_ok=True)
        visuals = {}
        
        # Generate title card
        title_prompt = self._create_title_prompt(script_data["title"], style)
        title_images = await self._generate_images(
            prompt=title_prompt,
            output_dir=output_dir,
            prefix="00_title",
            count=1,
            resolution=resolution
        )
        visuals["title"] = title_images
        
        # Generate visuals for each section
        for i, section in enumerate(script_data["sections"], 1):
            # Extract key concepts from section content
            concepts = self._extract_key_concepts(section["content"])
            
            # Generate multiple images for each section
            section_images = []
            for j, concept in enumerate(concepts):
                prompt = self._create_section_prompt(concept, style, section["title"])
                images = await self._generate_images(
                    prompt=prompt,
                    output_dir=output_dir,
                    prefix=f"{i:02d}_{j:02d}_{self._sanitize_filename(section['title'])}",
                    count=1,
                    resolution=resolution
                )
                section_images.extend(images)
                await asyncio.sleep(1)  # Add delay between requests
            
            visuals[section["title"]] = section_images
        
        # Generate end card
        end_prompt = self._create_end_prompt(script_data["call_to_action"], style)
        end_images = await self._generate_images(
            prompt=end_prompt,
            output_dir=output_dir,
            prefix="99_end",
            count=1,
            resolution=resolution
        )
        visuals["end"] = end_images
        
        return visuals
    
    async def _generate_images(
        self,
        prompt: str,
        output_dir: Optional[Path] = None,
        prefix: str = "image",
        count: int = 1,
        resolution: Tuple[int, int] = (1024, 1024)
    ) -> List[Path]:
        """Generate images using Together AI's API."""
        output_dir = output_dir or config.OUTPUT_DIR / "visuals"
        output_dir.mkdir(parents=True, exist_ok=True)
        width, height = resolution
        image_paths = []
        
        try:
            # Log request details for debugging
            logger.debug(f"Generating image with prompt: {prompt}")
            
            # Generate image using Together AI client in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.images.generate(
                    prompt=prompt,
                    model="black-forest-labs/FLUX.1-schnell-Free",
                    width=width,
                    height=height,
                    steps=1,
                    n=count,
                    response_format="b64_json"
                )
            )
            
            # Handle response
            for i, image_data in enumerate(response.data):
                image_bytes = base64.b64decode(image_data.b64_json)
                output_path = output_dir / f"{prefix}_{i:02d}.png"
                
                # Write file in thread pool to avoid blocking
                await loop.run_in_executor(
                    None,
                    lambda: output_path.write_bytes(image_bytes)
                )
                
                logger.info(f"Generated image: {output_path}")
                image_paths.append(output_path)
            
            if not image_paths:
                logger.error("No images were generated")
                raise ValueError("No images were generated")
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            raise
    
    def _create_title_prompt(self, title: str, style: str) -> str:
        """Create a prompt for the title card."""
        return f"""
        A visually striking title card for a YouTube video titled "{title}".
        Style: {style}. Clean, professional layout with modern typography.
        No text in the image. Suitable for YouTube thumbnail.
        Focus on aviation themes with a modern, minimalist aesthetic.
        Show an airplane silhouette or wing design in a clean, abstract way.
        """
    
    def _create_section_prompt(self, concept: str, style: str, section_title: str) -> str:
        """Create a prompt for a section visual."""
        return f"""
        A visual representation of {concept} from the section "{section_title}".
        Style: {style}. Clean, minimalist composition.
        No text or words in the image. Suitable for educational content.
        Focus on clear, technical illustrations with a modern aesthetic.
        Use simple shapes and strong visual hierarchy.
        """
    
    def _create_end_prompt(self, call_to_action: str, style: str) -> str:
        """Create a prompt for the end card."""
        return f"""
        An engaging end card visual that represents: "{call_to_action}".
        Style: {style}. Professional and clean layout.
        No text in the image. Suitable for YouTube end screen.
        Focus on aviation themes with an inspiring and forward-looking feel.
        Use dynamic composition that suggests motion and progress.
        """
    
    def _extract_key_concepts(self, text: str, max_concepts: int = 3) -> List[str]:
        """
        Extract key concepts from text that would make good visuals.
        In a production environment, this would use NLP, but for now we'll use a simple approach.
        """
        # Split into sentences and take the first few as concepts
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        concepts = []
        
        for sentence in sentences[:max_concepts]:
            # Remove common words and keep the essence
            concept = sentence.replace("is", "").replace("are", "").replace("the", "")
            concept = concept.strip()
            if concept:
                concepts.append(concept)
        
        return concepts
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Convert a string into a valid filename."""
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        filename = "".join(c for c in filename if c in valid_chars)
        filename = filename.replace(" ", "_")
        return filename.lower() 