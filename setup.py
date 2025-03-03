#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="orange-harvester",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pybullet>=3.2.5',
        'numpy>=1.21.0',
        'torch>=2.0.0',
        'gym>=0.26.0',
        'matplotlib>=3.4.0',
        'pillow>=8.3.0',
        'scipy>=1.7.0',
        'tqdm>=4.65.0',
        'tensorboard>=2.12.0',
        'pytest>=7.0.0'
    ],
    entry_points={
        'console_scripts': [
            'train-harvester=train_harvester:main',
            'evaluate-harvester=evaluate_harvester:main',
        ],
    },
)