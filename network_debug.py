#!/usr/bin/env python3

import numpy as np
import torch
from orange_harvest_env import OrangeHarvestEnv
from ppo_training import Actor, Critic

def debug_network_dimensions():
    print("Debugging Network Dimensions")
    print("===========================")
    
    try:
        # Create environment and get dimensions
        env = OrangeHarvestEnv(gui=False)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"\nEnvironment Dimensions:")
        print(f"State dim: {state_dim}")
        print(f"Action dim: {action_dim}")
        
        # Create networks
        print("\nCreating networks...")
        actor = Actor(state_dim, action_dim)
        critic = Critic(state_dim)
        
        # Get sample state
        print("\nTesting with sample state...")
        state = env.reset()
        print(f"Sample state shape: {state.shape}")
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        print(f"State tensor shape: {state_tensor.shape}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        mean, std = actor(state_tensor)
        value = critic(state_tensor)
        
        print(f"Actor output shapes:")
        print(f"- Mean shape: {mean.shape}")
        print(f"- Std shape: {std.shape}")
        print(f"Critic output shape: {value.shape}")
        
        print("\nAll dimensions match correctly!")
        
    except Exception as e:
        print(f"\nError during debug: {e}")
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    debug_network_dimensions()