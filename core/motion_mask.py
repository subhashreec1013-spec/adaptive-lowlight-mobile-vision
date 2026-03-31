
"""
motion_mask.py
Generates motion masks from optical flow
"""

import cv2
import numpy as np

class MotionMaskGenerator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def compute_motion_magnitude(self, flow):
        """Compute motion magnitude"""
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return magnitude
    
    def generate_mask(self, flow, adaptive=True):
        """Generate binary motion mask"""
        magnitude = self.compute_motion_magnitude(flow)
        
        if adaptive:
            threshold = np.median(magnitude) + 2 * np.std(magnitude)
        else:
            threshold = self.threshold
        
        mask = (magnitude > threshold).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def generate_soft_mask(self, flow, sigma=10.0):
        """Generate soft motion mask"""
        magnitude = self.compute_motion_magnitude(flow)
        soft_mask = np.exp(- (magnitude ** 2) / (2 * sigma ** 2))
        return soft_mask

def create_motion_masks(flows):
    """Create motion masks for flows"""
    generator = MotionMaskGenerator()
    masks = []
    
    for flow in flows:
        hard_mask = generator.generate_mask(flow, adaptive=True)
        soft_mask = generator.generate_soft_mask(flow)
        
        masks.append({
            'hard': hard_mask,
            'soft': soft_mask,
            'magnitude': generator.compute_motion_magnitude(flow)
        })
    
    return masks