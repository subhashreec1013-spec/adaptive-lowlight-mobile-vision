"""
config.py
Configuration parameters for the pipeline
"""

import os
from dataclasses import dataclass

@dataclass
class DataConfig:
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    image_size: tuple = (512, 512)
    num_frames: int = 3
    batch_size: int = 4

@dataclass
class ModelConfig:
    learning_rate: float = 1e-4
    num_epochs: int = 100
    device: str = "cuda"

@dataclass
class OpticalFlowConfig:
    method: str = "farneback"
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2

@dataclass
class EnhancementConfig:
    gamma_correction: float = 0.8
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    denoise_strength: float = 0.1

class Config:
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.optical_flow = OpticalFlowConfig()
        self.enhancement = EnhancementConfig()
        
    def display(self):
        print("\n" + "="*50)
        print("CONFIGURATION")
        print("="*50)
        print(f"Input Directory: {self.data.input_dir}")
        print(f"Output Directory: {self.data.output_dir}")
        print(f"Image Size: {self.data.image_size}")
        print(f"Number of Frames: {self.data.num_frames}")
        print(f"Device: {self.model.device}")
        print("="*50 + "\n")

config = Config()