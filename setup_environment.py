"""
setup_environment.py
Sets up the environment and installs required dependencies
"""

import subprocess
import sys
import os

def setup_environment():
    """Install required packages"""
    print("Setting up environment for Low-Light Image Enhancement...")
    
    packages = [
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "scikit-image>=0.21.0"
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Create directories
    directories = ['data/input', 'data/output', 'models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\n✅ Environment setup complete!")

if __name__ == "__main__":
    setup_environment()