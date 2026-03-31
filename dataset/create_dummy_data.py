"""
create_dummy_data.py
Creates sample low-light images for testing
"""

import cv2
import numpy as np
import os
from config import config

def create_low_light_image(save_path="data/input"):
    """Create synthetic low-light images for testing"""
    os.makedirs(save_path, exist_ok=True)
    
    # Create test pattern
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        img[i, :] = [i//2, i//3, i//4]
    cv2.circle(img, (256, 256), 100, (100, 150, 200), -1)
    cv2.rectangle(img, (50, 50), (150, 150), (150, 100, 100), -1)
    
    low_light_images = []
    
    for i in range(config.data.num_frames):
        # Reduce brightness
        gamma = np.random.uniform(0.3, 0.7)
        low_light = np.power(img / 255.0, 1/gamma) * 255
        low_light = low_light.astype(np.uint8)
        
        # Add noise
        noise = np.random.normal(0, 10, low_light.shape)
        low_light = np.clip(low_light + noise, 0, 255).astype(np.uint8)
        
        low_light_images.append(low_light)
        
        # Save frames
        frame_path = os.path.join(save_path, f"frame_{i:03d}.png")
        cv2.imwrite(frame_path, low_light)
        print(f"Saved: {frame_path}")
    
    print(f"\n✅ Created {len(low_light_images)} low-light test frames")
    return low_light_images

if __name__ == "__main__":
    print("Creating dummy low-light data...")
    create_low_light_image()