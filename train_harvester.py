#!/usr/bin/env python3

import os
import torch
import numpy as np
from orange_harvest_env import OrangeHarvestEnv
from ppo_training import PPOTrainer
from training_utils import TrainingLogger
from config import SimConfig as cfg

def create_output_dirs():
    """Create directories for saving training outputs"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def train_phase(env, phase_config, start_episode, logger):
    """Train for a specific phase"""
    trainer = PPOTrainer(env)
    
    # Load previous weights if continuing training
    if start_episode > 0:
        trainer.load(f"models/phase_{phase_config['phase']}_episode_{start_episode}.pt")
    
    total_episodes = phase_config['episodes']
    print(f"\nStarting Phase {phase_config['phase']}: {phase_config['name']}")
    print(f"Episodes: {total_episodes}, Learning Rate: {phase_config['lr']}")
    print(f"Reward Weights: {phase_config['reward_weights']}")
    
    # Update environment reward weights
    env.set_reward_weights(phase_config['reward_weights'])
    
    # Training loop
    for episode in range(start_episode, total_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        fruits_collected = 0
        
        # Episode loop
        while True:
            # Select action
            action, log_prob, value = trainer.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            trainer.memory.states.append(state)
            trainer.memory.actions.append(action)
            trainer.memory.rewards.append(reward)
            trainer.memory.values.append(value)
            trainer.memory.log_probs.append(log_prob)
            trainer.memory.dones.append(done)
            
            episode_reward += reward
            episode_steps += 1
            fruits_collected = info['fruits_collected']
            
            if done:
                break
                
            state = next_state
        
        # Update policy
        trainer.update_policy()
        trainer.memory.clear()
        
        # Log progress
        metrics = {
            'fruits_collected': fruits_collected,
            'success_rate': fruits_collected / env.total_fruits,
            'efficiency': episode_reward / episode_steps if episode_steps > 0 else 0
        }
        
        logger.log_episode(
            episode=episode + start_episode,
            reward=episode_reward,
            steps=episode_steps,
            metrics=metrics
        )
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            save_path = f"models/phase_{phase_config['phase']}_episode_{episode + 1}.pt"
            trainer.save(save_path)
            
            # Plot progress
            logger.plot_training_progress(
                save_path=f"logs/phase_{phase_config['phase']}_progress_{episode + 1}.png"
            )

def main():
    """Main training function"""
    create_output_dirs()
    
    # Initialize environment
    env = OrangeHarvestEnv(gui=cfg.GUI_ENABLED)
    logger = TrainingLogger()
    
    # Define training phases
    training_phases = [
        {
            'phase': 1,
            'name': 'Basic Movement',
            'episodes': 500,
            'lr': 3e-4,
            'reward_weights': {
                'distance': 1.0,
                'progress': 0.5,
                'action': 0.1,
                'contact': 0.0,
                'success': 1.0
            }
        },
        {
            'phase': 2,
            'name': 'Precise Control',
            'episodes': 1000,
            'lr': 1e-4,
            'reward_weights': {
                'distance': 0.5,
                'progress': 1.0,
                'action': 0.2,
                'contact': 1.0,
                'success': 2.0
            }
        },
        {
            'phase': 3,
            'name': 'Full Task',
            'episodes': 1500,
            'lr': 5e-5,
            'reward_weights': {
                'distance': 0.3,
                'progress': 1.0,
                'action': 0.3,
                'contact': 2.0,
                'success': 5.0
            }
        }
    ]
    
    try:
        # Train through each phase
        current_episode = 0
        for phase_config in training_phases:
            train_phase(env, phase_config, current_episode, logger)
            current_episode += phase_config['episodes']
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Clean up
        env.close()
        logger.close()

if __name__ == "__main__":
    main()