"""
Tests for the caching system.
"""
import pytest
from pathlib import Path
import time
from PIL import Image
import numpy as np
from src.caching import ContentCache

class TestContentCache:
    """Test suite for ContentCache."""
    
    async def test_cache_initialization(self, mock_cache_dir):
        """Test cache initialization creates required directories."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        assert cache.image_cache.exists()
        assert cache.prompt_cache.exists()
        assert cache.concept_cache.exists()
    
    async def test_image_caching(self, mock_cache_dir, sample_image):
        """Test image caching functionality."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        # Test parameters
        prompt = "test prompt"
        model = "test_model"
        size = (100, 100)
        style = "test_style"
        
        # Create test image
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = Image.fromarray(arr)
        
        # Cache the image
        cached_path = await cache.cache_image(
            img, prompt, model, size, style
        )
        
        assert cached_path.exists()
        
        # Retrieve from cache
        retrieved_path = await cache.get_image(prompt, model, size, style)
        assert retrieved_path == cached_path
        
        # Verify image content
        retrieved_img = Image.open(retrieved_path)
        assert retrieved_img.size == size
    
    async def test_prompt_caching(self, mock_cache_dir):
        """Test prompt caching functionality."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        # Test data
        base_prompt = "test base prompt"
        style = "test_style"
        emphasis = 1.0
        optimized = {
            "prompt": "optimized test prompt",
            "negative_prompt": "test negative prompt"
        }
        
        # Cache the prompt
        await cache.cache_optimized_prompt(
            optimized["prompt"],
            optimized["negative_prompt"],
            base_prompt,
            style,
            emphasis
        )
        
        # Retrieve from cache
        retrieved = await cache.get_optimized_prompt(base_prompt, style, emphasis)
        assert retrieved is not None
        assert retrieved["prompt"] == optimized["prompt"]
        assert retrieved["negative_prompt"] == optimized["negative_prompt"]
    
    async def test_concept_caching(self, mock_cache_dir):
        """Test concept caching functionality."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        # Test data
        text = "test text for concept extraction"
        max_concepts = 3
        min_relevance = 0.3
        concepts = [
            {"text": "test", "relevance": 0.8},
            {"text": "concept", "relevance": 0.7}
        ]
        
        # Cache the concepts
        await cache.cache_concepts(
            concepts,
            text,
            max_concepts,
            min_relevance
        )
        
        # Retrieve from cache
        retrieved = await cache.get_concepts(text, max_concepts, min_relevance)
        assert retrieved == concepts
    
    async def test_cache_expiration(self, mock_cache_dir, sample_image):
        """Test cache entry expiration."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        # Create test image
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = Image.fromarray(arr)
        
        # Cache with short TTL
        cached_path = await cache.cache_image(
            img,
            "test prompt",
            "test_model",
            (100, 100),
            "test_style",
            ttl=1  # 1 second TTL
        )
        
        assert cached_path.exists()
        
        # Wait for expiration
        time.sleep(2)
        
        # Try to retrieve expired entry
        retrieved_path = await cache.get_image(
            "test prompt",
            "test_model",
            (100, 100),
            "test_style"
        )
        assert retrieved_path is None
    
    async def test_cache_cleanup(self, mock_cache_dir, sample_image):
        """Test cleanup of expired cache entries."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        # Create test images with different TTLs
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = Image.fromarray(arr)
        
        # Cache multiple entries
        paths = []
        for i in range(3):
            path = await cache.cache_image(
                img,
                f"test prompt {i}",
                "test_model",
                (100, 100),
                "test_style",
                ttl=1 if i == 0 else 3600  # First entry expires quickly
            )
            paths.append(path)
        
        # Wait for first entry to expire
        time.sleep(2)
        
        # Run cleanup
        await cache.clear_expired()
        
        # Verify results
        assert not paths[0].exists()  # Expired entry should be removed
        assert paths[1].exists()  # Other entries should remain
        assert paths[2].exists()
    
    async def test_cache_key_generation(self, mock_cache_dir):
        """Test cache key generation."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        # Test data
        data1 = {"a": 1, "b": "test"}
        data2 = {"b": "test", "a": 1}  # Same content, different order
        
        # Generate keys
        key1 = cache._generate_key(data1)
        key2 = cache._generate_key(data2)
        
        # Keys should be the same for same content
        assert key1 == key2
        
        # Different content should have different keys
        key3 = cache._generate_key({"a": 2, "b": "test"})
        assert key1 != key3
    
    async def test_metadata_persistence(self, mock_cache_dir):
        """Test metadata persistence."""
        cache = ContentCache(cache_dir=mock_cache_dir)
        
        # Add test metadata
        key = "test_key"
        metadata = {
            "timestamp": time.time(),
            "ttl": 3600,
            "type": "test"
        }
        
        # Update metadata
        cache._update_metadata(key, metadata)
        
        # Save and reload
        cache._save_metadata()
        loaded_metadata = cache._load_metadata()
        
        # Verify persistence
        assert key in loaded_metadata
        assert loaded_metadata[key] == metadata 