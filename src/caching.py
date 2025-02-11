"""
Caching module for storing and retrieving generated content.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from loguru import logger
import json
import hashlib
import pickle
from PIL import Image
import io
import time

class ContentCache:
    """Handles caching of generated content and intermediate results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory for cache storage
        """
        from . import config
        self.cache_dir = cache_dir or config.BASE_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of content
        self.image_cache = self.cache_dir / "images"
        self.prompt_cache = self.cache_dir / "prompts"
        self.concept_cache = self.cache_dir / "concepts"
        self.metadata_cache = self.cache_dir / "metadata"
        
        for directory in [self.image_cache, self.prompt_cache, 
                         self.concept_cache, self.metadata_cache]:
            directory.mkdir(exist_ok=True)
        
        # Load cache metadata
        self.metadata = self._load_metadata()
    
    def get_image(
        self,
        prompt: str,
        model: str,
        size: tuple,
        style: str
    ) -> Optional[Path]:
        """
        Get a cached image if available.
        
        Args:
            prompt: Image generation prompt
            model: Model used for generation
            size: Image size
            style: Visual style
            
        Returns:
            Path to cached image if exists, None otherwise
        """
        cache_key = self._generate_key({
            "prompt": prompt,
            "model": model,
            "size": size,
            "style": style
        })
        
        image_path = self.image_cache / f"{cache_key}.png"
        if image_path.exists():
            # Check if cache is still valid
            if self._is_cache_valid(cache_key):
                logger.debug(f"Cache hit for image: {cache_key}")
                return image_path
            
            # Remove expired cache
            image_path.unlink()
            self._remove_metadata(cache_key)
        
        return None
    
    def cache_image(
        self,
        image: Image.Image,
        prompt: str,
        model: str,
        size: tuple,
        style: str,
        ttl: int = 86400  # 24 hours
    ) -> Path:
        """
        Cache an image.
        
        Args:
            image: PIL Image to cache
            prompt: Generation prompt
            model: Model used
            size: Image size
            style: Visual style
            ttl: Time to live in seconds
            
        Returns:
            Path to cached image
        """
        cache_key = self._generate_key({
            "prompt": prompt,
            "model": model,
            "size": size,
            "style": style
        })
        
        # Save image
        image_path = self.image_cache / f"{cache_key}.png"
        image.save(image_path, "PNG", optimize=True)
        
        # Update metadata
        self._update_metadata(cache_key, {
            "type": "image",
            "prompt": prompt,
            "model": model,
            "size": size,
            "style": style,
            "created": time.time(),
            "ttl": ttl
        })
        
        logger.debug(f"Cached image: {cache_key}")
        return image_path
    
    def get_concepts(
        self,
        text: str,
        max_concepts: int,
        min_relevance: float
    ) -> Optional[List[Dict]]:
        """Get cached concepts if available."""
        cache_key = self._generate_key({
            "text": text,
            "max_concepts": max_concepts,
            "min_relevance": min_relevance
        })
        
        concept_path = self.concept_cache / f"{cache_key}.pkl"
        if concept_path.exists():
            if self._is_cache_valid(cache_key):
                with open(concept_path, "rb") as f:
                    return pickle.load(f)
            
            concept_path.unlink()
            self._remove_metadata(cache_key)
        
        return None
    
    def cache_concepts(
        self,
        concepts: List[Dict],
        text: str,
        max_concepts: int,
        min_relevance: float,
        ttl: int = 604800  # 1 week
    ) -> None:
        """Cache extracted concepts."""
        cache_key = self._generate_key({
            "text": text,
            "max_concepts": max_concepts,
            "min_relevance": min_relevance
        })
        
        # Save concepts
        concept_path = self.concept_cache / f"{cache_key}.pkl"
        with open(concept_path, "wb") as f:
            pickle.dump(concepts, f)
        
        # Update metadata
        self._update_metadata(cache_key, {
            "type": "concepts",
            "text_hash": hashlib.md5(text.encode()).hexdigest(),
            "max_concepts": max_concepts,
            "min_relevance": min_relevance,
            "created": time.time(),
            "ttl": ttl
        })
        
        logger.debug(f"Cached concepts: {cache_key}")
    
    def get_optimized_prompt(
        self,
        base_prompt: str,
        style: str,
        emphasis: float
    ) -> Optional[Dict[str, str]]:
        """Get cached optimized prompt if available."""
        cache_key = self._generate_key({
            "base_prompt": base_prompt,
            "style": style,
            "emphasis": emphasis
        })
        
        prompt_path = self.prompt_cache / f"{cache_key}.json"
        if prompt_path.exists():
            if self._is_cache_valid(cache_key):
                with open(prompt_path, "r") as f:
                    return json.load(f)
            
            prompt_path.unlink()
            self._remove_metadata(cache_key)
        
        return None
    
    def cache_optimized_prompt(
        self,
        optimized_prompt: str,
        negative_prompt: str,
        base_prompt: str,
        style: str,
        emphasis: float,
        ttl: int = 604800  # 1 week
    ) -> None:
        """Cache an optimized prompt."""
        cache_key = self._generate_key({
            "base_prompt": base_prompt,
            "style": style,
            "emphasis": emphasis
        })
        
        # Save prompt
        prompt_path = self.prompt_cache / f"{cache_key}.json"
        with open(prompt_path, "w") as f:
            json.dump({
                "optimized_prompt": optimized_prompt,
                "negative_prompt": negative_prompt
            }, f)
        
        # Update metadata
        self._update_metadata(cache_key, {
            "type": "prompt",
            "base_prompt_hash": hashlib.md5(base_prompt.encode()).hexdigest(),
            "style": style,
            "emphasis": emphasis,
            "created": time.time(),
            "ttl": ttl
        })
        
        logger.debug(f"Cached optimized prompt: {cache_key}")
    
    def clear_expired(self) -> None:
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in self.metadata.items():
            if current_time > metadata["created"] + metadata["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_cache_entry(key)
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def _generate_key(self, data: Dict[str, Any]) -> str:
        """Generate a cache key from data."""
        # Convert data to string and hash
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        metadata_path = self.metadata_cache / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        metadata_path = self.metadata_cache / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def _update_metadata(self, key: str, metadata: Dict) -> None:
        """Update metadata for a cache entry."""
        self.metadata[key] = metadata
        self._save_metadata()
    
    def _remove_metadata(self, key: str) -> None:
        """Remove metadata for a cache entry."""
        if key in self.metadata:
            del self.metadata[key]
            self._save_metadata()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid."""
        if key not in self.metadata:
            return False
        
        metadata = self.metadata[key]
        current_time = time.time()
        
        return current_time <= metadata["created"] + metadata["ttl"]
    
    def _remove_cache_entry(self, key: str) -> None:
        """Remove a cache entry and its metadata."""
        # Remove cached file
        for cache_dir in [self.image_cache, self.prompt_cache, self.concept_cache]:
            for ext in [".png", ".json", ".pkl"]:
                cache_path = cache_dir / f"{key}{ext}"
                if cache_path.exists():
                    cache_path.unlink()
        
        # Remove metadata
        self._remove_metadata(key) 