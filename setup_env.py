#!/usr/bin/env python3

import os
import sys
import subprocess
import pkg_resources

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'torch': 'torch>=2.0.0',
        'numpy': 'numpy>=1.21.0',
        'pybullet': 'pybullet>=3.2.5',
        'pybullet_data': 'pybullet-data>=1.0.0',
        'gym': 'gym>=0.26.0',
        'matplotlib': 'matplotlib>=3.4.0',
        'pillow': 'pillow>=8.3.0',
        'scipy': 'scipy>=1.7.0',
        'tqdm': 'tqdm>=4.65.0',
        'tensorboard': 'tensorboard>=2.12.0'
    }
    
    missing_packages = []
    
    print("Checking Python packages...")
    for package, version in required_packages.items():
        try:
            pkg_resources.require(version)
            print(f"✓ {package} - OK")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"✗ {package} - Missing or wrong version")
            missing_packages.append(version)
    
    if missing_packages:
        print("\nInstalling missing packages...")
        for package in missing_packages:
            print(f"\nInstalling {package}")
            if install_package(package):
                print(f"Successfully installed {package}")
            else:
                print(f"Failed to install {package}")
                return False
    
    # Verify PyTorch installation and CUDA availability
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("Failed to import PyTorch after installation!")
        return False
    
    print("\nAll dependencies installed successfully!")
    return True

if __name__ == "__main__":
    try:
        print("Setting up environment for Orange Harvesting Robot Training")
        print("======================================================")
        
        success = check_dependencies()
        
        if success:
            print("\nEnvironment setup complete! You can now run the training.")
            sys.exit(0)
        else:
            print("\nEnvironment setup failed! Please check the errors above.")
            sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        sys.exit(1)