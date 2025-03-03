#!/usr/bin/env python3

import numpy as np
from orange_harvest_env import OrangeHarvestEnv

def analyze_dimensions():
    print("Analyzing Environment Dimensions")
    print("===============================")
    
    env = None
    state_dim = None
    action_dim = None
    
    try:
        print("Creating environment...")
        env = OrangeHarvestEnv(gui=False)
        
        # Get space dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"\nSpace Dimensions:")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        
        # Get sample observation
        print("\nGetting sample observation...")
        obs = env.reset()
        
        # Get state components
        robot_state = env._get_robot_state()
        env_state = env._get_environment_state()
        lidar_state = env._get_lidar_readings()
        
        print("\nComponent Dimensions:")
        print(f"Robot state: {len(robot_state)} dimensions")
        print(f"Environment state: {len(env_state)} dimensions")
        print(f"Lidar readings: {len(lidar_state)} dimensions")
        total_size = len(robot_state) + len(env_state) + len(lidar_state)
        print(f"Total state size: {total_size}")
        
        # Verify dimensions match
        print("\nDimension Verification:")
        print(f"Observation shape: {obs.shape}")
        print(f"Expected total: {state_dim}")
        
        if obs.shape[0] != state_dim:
            print("WARNING: Observation shape doesn't match space dimension!")
        else:
            print("All dimensions match correctly!")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
    finally:
        if env is not None:
            env.close()
            
    return state_dim, action_dim

def print_network_recommendations(state_dim, action_dim):
    if state_dim is not None and action_dim is not None:
        print("\nRecommended network dimensions:")
        print("==============================")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print("\nRecommended network architecture:")
        print("Actor Network:")
        print(f"  Input Layer:  {state_dim} neurons")
        print(f"  Hidden Layer: {state_dim * 2} neurons")
        print(f"  Output Layer: {action_dim} neurons")
        print("\nCritic Network:")
        print(f"  Input Layer:  {state_dim} neurons")
        print(f"  Hidden Layer: {state_dim * 2} neurons")
        print("  Output Layer: 1 neuron")
    else:
        print("\nCould not determine network dimensions due to errors.")

if __name__ == "__main__":
    state_dim, action_dim = analyze_dimensions()
    print_network_recommendations(state_dim, action_dim)