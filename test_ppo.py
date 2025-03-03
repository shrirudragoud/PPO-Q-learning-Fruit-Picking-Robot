#!/usr/bin/env python3

import numpy as np
import torch
from orange_harvest_env import OrangeHarvestEnv
from ppo_training import PPOTrainer
from config import SimConfig as cfg

def test_network_initialization():
    print("\nTesting Network Initialization")
    print("=============================")
    
    try:
        # Create environment
        env = OrangeHarvestEnv(gui=False)
        state = env.reset()
        
        print(f"\nState Information:")
        print(f"Shape: {state.shape}")
        print(f"Type: {state.dtype}")
        print(f"Range: [{state.min():.3f}, {state.max():.3f}]")
        
        # Create trainer
        trainer = PPOTrainer(env)
        
        print("\nNetwork Architecture:")
        print("Actor:", trainer.actor)
        print("\nCritic:", trainer.critic)
        
        # Test forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            action, log_prob, value = trainer.select_action(state)
        
        print(f"\nAction Information:")
        print(f"Shape: {action.shape}")
        print(f"Range: [{action.min():.3f}, {action.max():.3f}]")
        
        # Test policy update
        print("\nTesting policy update...")
        next_state, reward, done, _ = env.step(action)
        
        # Store transition
        trainer.memory.states.append(state)
        trainer.memory.actions.append(action)
        trainer.memory.rewards.append(reward)
        trainer.memory.values.append(value)
        trainer.memory.log_probs.append(log_prob)
        trainer.memory.dones.append(done)
        
        # Update policy
        trainer.update_policy()
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        raise
        
    finally:
        if 'env' in locals():
            env.close()

def test_training_loop():
    print("\nTesting Training Loop")
    print("====================")
    
    try:
        # Create environment and trainer
        env = OrangeHarvestEnv(gui=False)
        trainer = PPOTrainer(env)
        
        # Run mini training loop
        print("\nRunning 5 training episodes...")
        for episode in range(5):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                # Select and execute action
                action, log_prob, value = trainer.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                trainer.memory.states.append(state)
                trainer.memory.actions.append(action)
                trainer.memory.rewards.append(reward)
                trainer.memory.values.append(value)
                trainer.memory.log_probs.append(log_prob)
                trainer.memory.dones.append(done)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done or steps >= 100:  # Limit steps for testing
                    break
            
            # Update policy
            trainer.update_policy()
            trainer.memory.clear()
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        
        print("\nTraining loop test completed successfully!")
        
    except Exception as e:
        print(f"\nError during training test: {e}")
        raise
        
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    # Run tests
    print("Running PPO Implementation Tests")
    print("===============================")
    
    test_network_initialization()
    test_training_loop()
    
    print("\nAll tests completed successfully!")