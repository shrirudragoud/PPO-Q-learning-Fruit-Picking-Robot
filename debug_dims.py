#!/usr/bin/env python3

from orange_harvest_env import OrangeHarvestEnv

def print_dimensions():
    """Print environment dimensions and sample observations"""
    print("Analyzing Environment Dimensions")
    print("===============================")
    
    try:
        env = OrangeHarvestEnv(gui=False)
        
        # Print space dimensions
        print("\nSpace Dimensions:")
        print(f"Observation Space Shape: {env.observation_space.shape}")
        print(f"Action Space Shape: {env.action_space.shape}")
        
        # Get sample observation
        obs = env.reset()
        print("\nSample Observation:")
        print(f"Type: {type(obs)}")
        print(f"Shape: {obs.shape}")
        print(f"Min Value: {obs.min()}")
        print(f"Max Value: {obs.max()}")
        
        # Print component dimensions
        robot_state = env._get_robot_state()
        env_state = env._get_environment_state()
        lidar_state = env._get_lidar_readings()
        
        print("\nComponent Dimensions:")
        print(f"Robot State Shape: {robot_state.shape}")
        print(f"Environment State Shape: {env_state.shape}")
        print(f"Lidar Readings Shape: {lidar_state.shape}")
        
        print("\nVerifying concatenation:")
        total_dim = robot_state.shape[0] + env_state.shape[0] + lidar_state.shape[0]
        print(f"Sum of component dimensions: {total_dim}")
        print(f"Total observation dimension: {obs.shape[0]}")
        
        if total_dim != obs.shape[0]:
            print("\nWARNING: Dimension mismatch between components and total observation!")
        else:
            print("\nAll dimensions match correctly.")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    print_dimensions()