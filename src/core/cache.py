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
    
    def get_script(self, key: str) -> Optional[Dict]:
        """Get cached script if available."""
        cache_key = self._generate_key({"key": key})
        script_path = self.concept_cache / f"{cache_key}.json"
        
        if script_path.exists() and self._is_cache_valid(cache_key):
            try:
                with open(script_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached script: {e}")
                self._remove_cache_entry(cache_key)
        
        return None
    
    def cache_script(self, key: str, script_data: Dict, ttl: int = 86400) -> None:
        """Cache a generated script."""
        cache_key = self._generate_key({"key": key})
        script_path = self.concept_cache / f"{cache_key}.json"
        
        try:
            with open(script_path, "w") as f:
                json.dump(script_data, f)
            
            self._update_metadata(cache_key, {
                "type": "script",
                "key": key,
                "created": time.time(),
                "ttl": ttl
            })
            
            logger.debug(f"Cached script: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to cache script: {e}")
            if script_path.exists():
                script_path.unlink()
    
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
        # Create a stable dictionary for key generation
        key_data = {
            "prompt": prompt,
            "model": model,
            "size_width": size[0],
            "size_height": size[1],
            "style": style
        }
        
        cache_key = self._generate_key(key_data)
        image_path = self.image_cache / f"{cache_key}.png"
        
        if image_path.exists() and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for image: {cache_key}")
            return image_path
        
        if image_path.exists():
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
        """Cache an image."""
        # Create a stable dictionary for key generation
        key_data = {
            "prompt": prompt,
            "model": model,
            "size_width": size[0],
            "size_height": size[1],
            "style": style
        }
        
        cache_key = self._generate_key(key_data)
        image_path = self.image_cache / f"{cache_key}.png"
        
        try:
            image.save(image_path, "PNG", optimize=True)
            
            self._update_metadata(cache_key, {
                "type": "image",
                "data": key_data,
                "created": time.time(),
                "ttl": ttl
            })
            
            logger.debug(f"Cached image: {cache_key}")
            return image_path
            
        except Exception as e:
            logger.error(f"Failed to cache image: {e}")
            if image_path.exists():
                image_path.unlink()
            raise
    
    def get_optimized_prompt(
        self,
        base_prompt: str,
        style: str,
        emphasis: float
    ) -> Optional[Dict[str, str]]:
        """Get cached optimized prompt if available."""
        key_data = {
            "base_prompt": base_prompt,
            "style": style,
            "emphasis": emphasis
        }
        
        cache_key = self._generate_key(key_data)
        prompt_path = self.prompt_cache / f"{cache_key}.json"
        
        if prompt_path.exists() and self._is_cache_valid(cache_key):
            try:
                with open(prompt_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached prompt: {e}")
                self._remove_cache_entry(cache_key)
        
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
        key_data = {
            "base_prompt": base_prompt,
            "style": style,
            "emphasis": emphasis
        }
        
        cache_key = self._generate_key(key_data)
        prompt_path = self.prompt_cache / f"{cache_key}.json"
        
        try:
            with open(prompt_path, "w") as f:
                json.dump({
                    "optimized_prompt": optimized_prompt,
                    "negative_prompt": negative_prompt
                }, f)
            
            self._update_metadata(cache_key, {
                "type": "prompt",
                "data": key_data,
                "created": time.time(),
                "ttl": ttl
            })
            
            logger.debug(f"Cached optimized prompt: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to cache prompt: {e}")
            if prompt_path.exists():
                prompt_path.unlink()
    
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
        """Generate a stable cache key from input data."""
        try:
            # Convert data to a stable string representation
            key_string = json.dumps(data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            # Fallback to a simple string hash if JSON serialization fails
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        metadata_file = self.metadata_cache / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        metadata_file = self.metadata_cache / "metadata.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _update_metadata(self, key: str, metadata: Dict) -> None:
        """Update metadata for a cache entry."""
        try:
            self.metadata[key] = metadata
            self._save_metadata()
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
    
    def _remove_metadata(self, key: str) -> None:
        """Remove metadata for a cache entry."""
        try:
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
        except Exception as e:
            logger.error(f"Failed to remove metadata: {e}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid."""
        try:
            if key not in self.metadata:
                return False
            
            metadata = self.metadata[key]
            current_time = time.time()
            return current_time <= metadata["created"] + metadata["ttl"]
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def _remove_cache_entry(self, key: str) -> None:
        """Remove a cache entry and its metadata."""
        try:
            # Remove metadata
            self._remove_metadata(key)
            
            # Remove cached files
            for cache_dir in [self.image_cache, self.prompt_cache, 
                            self.concept_cache]:
                for ext in [".png", ".json", ".pkl"]:
                    cache_file = cache_dir / f"{key}{ext}"
                    if cache_file.exists():
                        cache_file.unlink()
                        
        except Exception as e:
            logger.error(f"Failed to remove cache entry: {e}") 