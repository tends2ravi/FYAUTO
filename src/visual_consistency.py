"""
Visual consistency checking and enforcement module.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger
import numpy as np
from PIL import Image
import cv2
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

class VisualConsistencyChecker:
    """Checks and ensures visual consistency across images."""
    
    def __init__(self, tolerance: float = 0.2):
        """
        Initialize the consistency checker.
        
        Args:
            tolerance: Tolerance for consistency checks (0-1)
        """
        self.tolerance = tolerance
    
    async def check_consistency(
        self,
        image_paths: List[Path],
        reference_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Check visual consistency across images.
        
        Args:
            image_paths: List of image paths to check
            reference_path: Optional reference image path
            
        Returns:
            Dictionary of consistency scores
        """
        scores = {}
        images = [cv2.imread(str(path)) for path in image_paths]
        
        if reference_path:
            reference = cv2.imread(str(reference_path))
        else:
            reference = images[0]
            
        for i, img in enumerate(images):
            scores[str(image_paths[i])] = self._calculate_consistency_score(img, reference)
            
        return scores
    
    async def ensure_consistency(
        self,
        image_paths: List[Path],
        reference_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Ensure visual consistency across images by adjusting them.
        
        Args:
            image_paths: List of image paths to process
            reference_path: Optional reference image path
            output_dir: Directory for adjusted images
            
        Returns:
            List of paths to consistent images
        """
        try:
            images = [cv2.imread(str(path)) for path in image_paths]
            
            if reference_path:
                reference = cv2.imread(str(reference_path))
            else:
                reference = images[0]
            
            # Create output directory if needed
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = image_paths[0].parent
            
            # Adjust images for consistency
            adjusted_paths = []
            
            for i, img in enumerate(images):
                adjusted = self._adjust_colors(img, reference)
                output_path = output_dir / f"adjusted_{image_paths[i].name}"
                cv2.imwrite(str(output_path), adjusted)
                adjusted_paths.append(output_path)
            
            return adjusted_paths
            
        except Exception as e:
            logger.error(f"Error ensuring visual consistency: {str(e)}")
            raise
    
    def _calculate_consistency_score(self, img1, img2):
        """Calculate consistency score between two images."""
        # Convert to LAB color space
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        mean1 = np.mean(img1_lab, axis=(0,1))
        mean2 = np.mean(img2_lab, axis=(0,1))
        std1 = np.std(img1_lab, axis=(0,1))
        std2 = np.std(img2_lab, axis=(0,1))
        
        # Calculate consistency score
        color_diff = np.mean(np.abs(mean1 - mean2))
        std_diff = np.mean(np.abs(std1 - std2))
        
        score = 1.0 - (color_diff + std_diff) / 255.0
        return max(0.0, min(1.0, score))
    
    def _adjust_colors(self, source, reference):
        """Adjust colors of source image to match reference."""
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(float)
        
        # Calculate statistics
        src_mean = np.mean(source_lab, axis=(0,1))
        ref_mean = np.mean(reference_lab, axis=(0,1))
        src_std = np.std(source_lab, axis=(0,1))
        ref_std = np.std(reference_lab, axis=(0,1))
        
        # Adjust each channel
        for i in range(3):
            source_lab[:,:,i] = ((source_lab[:,:,i] - src_mean[i]) * 
                                (ref_std[i] / src_std[i]) + ref_mean[i])
                                
        # Clip values and convert back to BGR
        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR) 