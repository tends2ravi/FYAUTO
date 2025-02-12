"""
Tests for the visual consistency checker.
"""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from src.visual_consistency import VisualConsistencyChecker

@pytest.mark.asyncio
class TestVisualConsistencyChecker:
    """Test suite for VisualConsistencyChecker."""
    
    @pytest.fixture
    def checker(self):
        return VisualConsistencyChecker()

    async def test_initialization(self):
        """Test checker initialization."""
        checker = VisualConsistencyChecker(tolerance=0.3)
        assert checker.tolerance == 0.3
    
    @pytest.mark.asyncio
    async def test_check_consistency(self, checker, tmp_path):
        """Test consistency checking."""
        # Create test images
        image_paths = []
        for i in range(3):
            img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
            path = tmp_path / f"test_{i}.png"
            cv2.imwrite(str(path), img)
            image_paths.append(path)
        
        scores = await checker.check_consistency(image_paths)
        assert isinstance(scores, dict)
        assert len(scores) == len(image_paths)
        assert all(isinstance(v, float) for v in scores.values())
    
    @pytest.mark.asyncio
    async def test_ensure_consistency(self, checker, tmp_path):
        """Test consistency enforcement."""
        # Create test images
        image_paths = []
        for i in range(3):
            img = np.ones((100, 100, 3), dtype=np.uint8) * (i * 50)
            path = tmp_path / f"test_{i}.png"
            cv2.imwrite(str(path), img)
            image_paths.append(path)
        
        adjusted_paths = await checker.ensure_consistency(image_paths)
        assert isinstance(adjusted_paths, list)
        assert len(adjusted_paths) == len(image_paths)
        assert all(isinstance(p, (str, Path)) for p in adjusted_paths)
    
    async def test_color_histogram(self, test_dir):
        """Test color histogram calculation."""
        checker = VisualConsistencyChecker()
        
        # Create test image
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        
        # Calculate histogram
        hist = checker._calculate_color_histogram(img)
        
        assert isinstance(hist, np.ndarray)
        assert len(hist) == 8  # 8 clusters
        assert np.isclose(hist.sum(), 1.0)  # Normalized
    
    async def test_edge_detection(self, test_dir):
        """Test edge detection."""
        checker = VisualConsistencyChecker()
        
        # Create test image with clear edges
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[40:60, 40:60] = 255  # White square
        img = Image.fromarray(arr)
        
        # Detect edges
        edges = checker._detect_edges(img)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape == (100, 100)
        assert edges.sum() > 0  # Should detect some edges
    
    async def test_brightness_calculation(self, test_dir):
        """Test brightness calculation."""
        checker = VisualConsistencyChecker()
        
        # Create test images with different brightness
        test_cases = [
            (np.zeros((100, 100, 3), dtype=np.uint8), 0),  # Black
            (np.ones((100, 100, 3), dtype=np.uint8) * 127, 127),  # Gray
            (np.ones((100, 100, 3), dtype=np.uint8) * 255, 255)  # White
        ]
        
        for arr, expected in test_cases:
            img = Image.fromarray(arr)
            brightness = checker._calculate_brightness(img)
            assert np.isclose(brightness, expected, atol=1)
    
    def test_color_adjustment(self, checker):
        """Test color distribution adjustment."""
        # Create source and reference images
        source_arr = np.ones((100, 100, 3), dtype=np.uint8) * 100
        ref_arr = np.ones((100, 100, 3), dtype=np.uint8) * 150
        
        adjusted_arr = checker._adjust_colors(source_arr, ref_arr)
        assert isinstance(adjusted_arr, np.ndarray)
        assert adjusted_arr.shape == source_arr.shape
        assert adjusted_arr.dtype == source_arr.dtype
        assert np.mean(adjusted_arr) > np.mean(source_arr)
    
    async def test_brightness_adjustment(self, test_dir):
        """Test brightness adjustment."""
        checker = VisualConsistencyChecker()
        
        # Create source and reference images
        source_arr = np.ones((100, 100, 3), dtype=np.uint8) * 50
        ref_arr = np.ones((100, 100, 3), dtype=np.uint8) * 150
        
        source_img = Image.fromarray(source_arr)
        ref_img = Image.fromarray(ref_arr)
        
        # Adjust brightness
        adjusted = checker._adjust_brightness(source_img, ref_img)
        
        assert isinstance(adjusted, Image.Image)
        adjusted_arr = np.array(adjusted)
        assert adjusted_arr.mean() > source_arr.mean()
    
    async def test_edge_comparison(self, test_dir):
        """Test edge map comparison."""
        checker = VisualConsistencyChecker()
        
        # Create two similar edge maps
        edges1 = np.zeros((100, 100), dtype=np.uint8)
        edges2 = np.zeros((100, 100), dtype=np.uint8)
        
        # Add similar patterns
        edges1[40:60, 40:60] = 255
        edges2[41:61, 41:61] = 255  # Slightly shifted
        
        similarity = checker._compare_edge_maps(edges1, edges2)
        assert 0 < similarity < 1  # Should be similar but not identical
    
    @pytest.mark.asyncio
    async def test_consistency_with_reference(self, checker, tmp_path):
        """Test consistency checking with reference image."""
        # Create test and reference images
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        ref_img = np.ones((100, 100, 3), dtype=np.uint8) * 150

        test_path = tmp_path / "test.png"
        ref_path = tmp_path / "ref.png"
        cv2.imwrite(str(test_path), test_img)
        cv2.imwrite(str(ref_path), ref_img)
        
        scores = await checker.check_consistency([test_path], reference_path=ref_path)
        
        assert isinstance(scores, dict)
        assert len(scores) == 1
        assert all(isinstance(v, float) for v in scores.values()) 