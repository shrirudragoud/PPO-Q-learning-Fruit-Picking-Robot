#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from config import SimConfig as cfg

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        return self.net(state)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def sample(self):
        # Fix division by zero error
        if len(self.states) == 0:
            return []
        
        if self.batch_size >= len(self.states):
            # If batch size is larger than available data, return all indices
            return list(range(len(self.states)))
        
        batch_step = len(self.states) // self.batch_size
        if batch_step == 0:
            batch_step = 1
        
        indices = np.arange(0, len(self.states), batch_step)
        return indices

class PPOTrainer:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        
        # Fix config attribute access - determine dimensions dynamically
        try:
            # Try to get dimensions from config
            self.state_dim = (
                len(cfg.Training.Observation.ROBOT_STATE) +
                len(cfg.Training.Observation.GRIPPER_STATE) +
                len(cfg.Training.Observation.FRUIT_STATE)
            )
        except AttributeError:
            # Fallback: determine from environment
            print("Warning: Config attributes not found. Determining dimensions from environment...")
            sample_state = self.env.reset()
            if isinstance(sample_state, (list, np.ndarray)):
                self.state_dim = len(sample_state)
            else:
                # Default fallback
                self.state_dim = 9  # Based on your error showing 1x9 input
        
        try:
            self.action_dim = len(cfg.Training.Action.CONTINUOUS)
        except AttributeError:
            # Default action dimension
            self.action_dim = 3  # Common for robot control (x, y, z movement)
        
        print(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")
        
        # Initialize networks
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim).to(device)
        
        # Setup optimizers with default learning rates if config fails
        try:
            actor_lr = cfg.Training.ACTOR_LR
            critic_lr = cfg.Training.CRITIC_LR
        except AttributeError:
            actor_lr = 3e-4
            critic_lr = 1e-3
            print(f"Warning: Using default learning rates - Actor: {actor_lr}, Critic: {critic_lr}")
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize memory with default batch size if config fails
        try:
            batch_size = cfg.Training.BATCH_SIZE
        except AttributeError:
            batch_size = 64
            print(f"Warning: Using default batch size: {batch_size}")
        
        self.memory = PPOMemory(batch_size)
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        if len(rewards) <= 1:
            return torch.tensor([0.0])
        
        # Get hyperparameters with defaults
        try:
            gamma = cfg.Training.GAMMA
            gae_lambda = cfg.Training.GAE_LAMBDA
        except AttributeError:
            gamma = 0.99
            gae_lambda = 0.95
        
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1-dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32)
    
    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            # Ensure state is properly formatted
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=np.float32)
            elif not isinstance(state, np.ndarray):
                state = np.array([state], dtype=np.float32)
            
            # Ensure state has correct dimensions
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            mean, std = self.actor(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state_tensor)
            
        return action.cpu().numpy().squeeze(), log_prob.cpu().numpy().item(), value.cpu().numpy().item()
    
    def update_policy(self):
        """Update policy using PPO algorithm"""
        if len(self.memory.states) == 0:
            print("Warning: No data in memory to update policy")
            return
        
        # Convert memory to tensors more efficiently
        states = np.array(self.memory.states, dtype=np.float32)
        actions = np.array(self.memory.actions, dtype=np.float32)
        old_log_probs = np.array(self.memory.log_probs, dtype=np.float32)
        values = np.array(self.memory.values, dtype=np.float32)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Compute advantages
        advantages = self.compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones
        ).to(self.device)
        
        # Handle case where advantages is empty or has single value
        if len(advantages) == 0:
            print("Warning: No advantages computed")
            return
        
        # Normalize advantages only if we have multiple values
        if len(advantages) > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get hyperparameters with defaults
        try:
            epochs = cfg.Training.EPOCHS
            clip_epsilon = cfg.Training.CLIP_EPSILON
        except AttributeError:
            epochs = 4
            clip_epsilon = 0.2
        
        # PPO update
        for epoch in range(epochs):
            indices = self.memory.sample()
            if len(indices) == 0:
                continue
                
            for idx in indices:
                if idx >= len(states) or idx >= len(advantages):
                    continue
                    
                # Get batch (single sample for now)
                state = states[idx:idx+1]  # Keep batch dimension
                action = actions[idx:idx+1] if actions.ndim > 1 else actions[idx].unsqueeze(0)
                old_log_prob = old_log_probs[idx]
                advantage = advantages[idx]
                
                # Evaluate actions
                mean, std = self.actor(state)
                dist = Normal(mean, std)
                new_log_prob = dist.log_prob(action).sum(dim=-1)
                
                # PPO loss calculation
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic - fix tensor shape issue
                value = self.critic(state).squeeze()
                target_value = advantage + values[idx]  # Use stored value instead
                critic_loss = nn.MSELoss()(value, target_value)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
    
    def train(self, num_episodes=None):
        """Main training loop"""
        # Get hyperparameters with defaults
        try:
            if num_episodes is None:
                num_episodes = cfg.Training.EPISODES
            steps_per_episode = cfg.Training.STEPS_PER_EPISODE
        except AttributeError:
            if num_episodes is None:
                num_episodes = 2000  # Increased from 1000
            steps_per_episode = 500  # Increased from 200
            print(f"Warning: Using default training parameters - Episodes: {num_episodes}, Steps: {steps_per_episode}")
            
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Select action
                action, log_prob, value = self.select_action(state)
                
                # Execute action
                try:
                    next_state, reward, done, _ = self.env.step(action)
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    break
                
                # Store transition
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.rewards.append(reward)
                self.memory.values.append(value)
                self.memory.log_probs.append(log_prob)
                self.memory.dones.append(done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update policy only if we have enough data
            if len(self.memory.states) > 0:
                self.update_policy()
                self.memory.clear()
            
            # Track progress
            self.episode_rewards.append(episode_reward)
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
            
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
            
            # Early stopping if performance is improving
            if len(self.episode_rewards) >= 100 and avg_reward > -100:
                print(f"Early stopping - Good performance achieved!")
                break
            
            # Save checkpoint periodically
            if (episode + 1) % 500 == 0:  # Save every 500 episodes
                self.save(f'ppo_checkpoint_episode_{episode + 1}.pth')
    
    def save(self, path):
        """Save model checkpoints"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoints"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    import pybullet as p
    import pybullet_data
    from environment import FarmEnvironment

    # Start PyBullet in DIRECT mode (no GUI for training)
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    env = FarmEnvironment(client)

    # Improved environment wrapper
    class PPOEnvWrapper:
        def __init__(self, env):
            self.env = env
            self.robot_position = [0, 0, 0.1]
            self.gripper_position = [0, 0, 0.1]
            self.target_position = [1, 1, 0.5]  # Example target
            
        def reset(self):
            try:
                self.env.reset()
                # Reset positions
                self.robot_position = [0, 0, 0.1]
                self.gripper_position = [0, 0, 0.1]
                observation = self.env.get_observation(self.robot_position, self.gripper_position)
                
                # Ensure observation is a flat array
                if isinstance(observation, (list, tuple)):
                    observation = np.array(observation, dtype=np.float32).flatten()
                elif isinstance(observation, np.ndarray):
                    observation = observation.flatten().astype(np.float32)
                
                return observation
            except Exception as e:
                print(f"Error in reset: {e}")
                # Return default observation
                return np.zeros(9, dtype=np.float32)
        
        def step(self, action):
            try:
                # Apply action to update positions (simplified)
                if isinstance(action, np.ndarray) and len(action) >= 3:
                    self.robot_position[0] += action[0] * 0.1
                    self.robot_position[1] += action[1] * 0.1
                    self.robot_position[2] += action[2] * 0.1
                    
                    # Update gripper position to follow robot
                    self.gripper_position = self.robot_position.copy()
                
                # Get next observation
                next_state = self.env.get_observation(self.robot_position, self.gripper_position)
                if isinstance(next_state, (list, tuple)):
                    next_state = np.array(next_state, dtype=np.float32).flatten()
                elif isinstance(next_state, np.ndarray):
                    next_state = next_state.flatten().astype(np.float32)
                
                # Compute reward and done
                reward, done = self.env.compute_reward(self.gripper_position, self.target_position)
                
                return next_state, reward, done, {}
                
            except Exception as e:
                print(f"Error in step: {e}")
                # Return safe defaults
                return np.zeros(9, dtype=np.float32), 0.0, True, {}
    
    try:
        wrapped_env = PPOEnvWrapper(env)
        trainer = PPOTrainer(wrapped_env)
        
        # Start with more episodes for proper training
        print("Starting PPO training with proper parameters...")
        print("Note: This will take much longer but is necessary for learning")
        
        trainer.train(num_episodes=2000)  # Increased significantly
        print("Training complete.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        p.disconnect()