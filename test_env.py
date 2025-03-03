#!/usr/bin/env python3

import time
from orange_harvest_env import OrangeHarvestEnv

def test_environment():
    print("Testing Orange Harvesting Environment")
    print("=====================================")
    
    try:
        # Create environment
        print("\n1. Creating environment...")
        env = OrangeHarvestEnv(gui=True)
        print("Environment created successfully")
        
        # Test reset
        print("\n2. Testing reset...")
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Test random actions
        print("\n3. Testing random actions...")
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            if i % 20 == 0:
                print(f"\nStep {i}:")
                print(f"Reward: {reward:.3f}")
                print(f"Fruits Collected: {info['fruits_collected']}")
                print(f"Distance to nearest fruit: {info.get('distance_reward', 0):.3f}")
            
            if done:
                print("\nEpisode finished after {} steps".format(i+1))
                break
            
            time.sleep(0.01)  # Slow down for visualization
        
        print("\nEnvironment test completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        raise
    
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    test_environment()