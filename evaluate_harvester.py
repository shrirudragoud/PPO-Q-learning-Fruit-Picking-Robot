#!/usr/bin/env python3

import os
import numpy as np
import torch
from orange_harvest_env import OrangeHarvestEnv
from ppo_training import PPOTrainer
from config import SimConfig as cfg

def evaluate_episode(env, trainer):
    """Run single evaluation episode"""
    state = env.reset()
    total_reward = 0
    steps = 0
    fruits_collected = 0
    metrics = {
        'distances': [],
        'action_magnitudes': [],
        'gripper_positions': []
    }
    
    while True:
        # Select action without exploration noise
        with torch.no_grad():
            action, _, _ = trainer.select_action(state, eval_mode=True)
            
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Track metrics
        metrics['distances'].append(info.get('distance_to_fruit', 0))
        metrics['action_magnitudes'].append(np.mean(np.abs(action)))
        gripper_pos = env.robot.get_gripper_position()
        metrics['gripper_positions'].append(gripper_pos)
        
        total_reward += reward
        steps += 1
        fruits_collected = info['fruits_collected']
        
        if done:
            break
            
        state = next_state
    
    # Calculate episode metrics
    episode_metrics = {
        'total_reward': total_reward,
        'steps': steps,
        'fruits_collected': fruits_collected,
        'success_rate': fruits_collected / env.total_fruits,
        'avg_action_magnitude': np.mean(metrics['action_magnitudes']),
        'min_distance': min(metrics['distances']) if metrics['distances'] else float('inf'),
        'completion_time': steps * cfg.TIMESTEP
    }
    
    return episode_metrics

def evaluate(model_path, num_episodes=10, gui=True):
    """Evaluate trained model over multiple episodes"""
    # Initialize environment and trainer
    env = OrangeHarvestEnv(gui=gui)
    trainer = PPOTrainer(env)
    
    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    trainer.load(model_path)
    
    # Run evaluation episodes
    all_metrics = []
    for episode in range(num_episodes):
        print(f"\nEvaluating episode {episode + 1}/{num_episodes}")
        metrics = evaluate_episode(env, trainer)
        all_metrics.append(metrics)
        
        # Print episode results
        print(f"Episode Results:")
        print(f"  Fruits Collected: {metrics['fruits_collected']}/{env.total_fruits}")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Steps: {metrics['steps']}")
        print(f"  Total Reward: {metrics['total_reward']:.2f}")
        print(f"  Completion Time: {metrics['completion_time']:.2f}s")
    
    # Calculate aggregate metrics
    avg_metrics = {
        'success_rate': np.mean([m['success_rate'] for m in all_metrics]),
        'avg_steps': np.mean([m['steps'] for m in all_metrics]),
        'avg_reward': np.mean([m['total_reward'] for m in all_metrics]),
        'avg_completion_time': np.mean([m['completion_time'] for m in all_metrics]),
        'std_completion_time': np.std([m['completion_time'] for m in all_metrics]),
        'avg_action_magnitude': np.mean([m['avg_action_magnitude'] for m in all_metrics])
    }
    
    print("\nOverall Performance:")
    print(f"Average Success Rate: {avg_metrics['success_rate']*100:.1f}%")
    print(f"Average Episode Length: {avg_metrics['avg_steps']:.1f} steps")
    print(f"Average Reward: {avg_metrics['avg_reward']:.2f}")
    print(f"Average Completion Time: {avg_metrics['avg_completion_time']:.2f}s")
    print(f"Completion Time Std: {avg_metrics['std_completion_time']:.2f}s")
    print(f"Average Action Magnitude: {avg_metrics['avg_action_magnitude']:.3f}")
    
    env.close()
    return avg_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained harvester model")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    args = parser.parse_args()
    
    evaluate(args.model_path, num_episodes=args.episodes, gui=not args.no_gui)