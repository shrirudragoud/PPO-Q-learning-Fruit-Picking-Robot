#!/usr/bin/env python3

import numpy as np

class Space:
    """Base class for observation and action spaces"""
    def __init__(self, shape, low, high, dtype=np.float32):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dtype = dtype
        
    def sample(self):
        """Sample a random value from the space"""
        return np.random.uniform(
            low=self.low,
            high=self.high,
            size=self.shape
        ).astype(self.dtype)

class Box(Space):
    """Continuous space with bounds"""
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if shape is None:
            shape = low.shape if isinstance(low, np.ndarray) else (low,)
        super().__init__(shape, low, high, dtype)

class Env:
    """Base class for environments"""
    def __init__(self):
        self.observation_space = None
        self.action_space = None
    
    def reset(self):
        """Reset environment to initial state"""
        raise NotImplementedError
    
    def step(self, action):
        """Execute action and return (observation, reward, done, info)"""
        raise NotImplementedError
    
    def close(self):
        """Clean up environment"""
        pass