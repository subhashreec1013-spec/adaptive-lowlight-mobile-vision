
"""
dataset_preparation.py
Prepares and loads dataset
"""

import cv2
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from config import config

class LowLightDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.sequences = self._load_sequences()
    
    def _load_sequences(self):
        """Load image sequences"""
        sequences = []
        frame_pattern = os.path.join(self.input_dir, "frame_*.png")
        frames = sorted(glob.glob(frame_pattern))
        
        if len(frames) >= config.data.num_frames:
            sequences.append(frames[:config.data.num_frames])
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        
        frames = []
        for path in frame_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, config.data.image_size)
            frames.append(img)
        
        frames_np = np.stack(frames, axis=0)
        frames_tensor = np.transpose(frames_np, (0, 3, 1, 2)).astype(np.float32) / 255.0
        
        return {
            'frames': frames_tensor,
            'paths': frame_paths,
            'idx': idx
        }

def create_dataloaders():
    """Create dataloaders"""
    dataset = LowLightDataset(input_dir=config.data.input_dir)
    
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Created dataloader with {len(dataset)} sequences")
    return loader