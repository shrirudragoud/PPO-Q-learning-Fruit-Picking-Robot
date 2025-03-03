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
        batch_step = len(self.states) // self.batch_size
        indices = np.arange(0, len(self.states), batch_step)
        return indices

class PPOTrainer:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        
        # Determine dimensions
        self.state_dim = len(cfg.Training.Observation.ROBOT_STATE) + \
                        len(cfg.Training.Observation.ENVIRONMENT_STATE) + \
                        cfg.Training.Observation.LIDAR_RAYS
        self.action_dim = len(cfg.Training.Action.CONTINUOUS)
        
        # Initialize networks
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim).to(device)
        
        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.Training.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.Training.CRITIC_LR)
        
        # Initialize memory
        self.memory = PPOMemory(cfg.Training.BATCH_SIZE)
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + cfg.Training.GAMMA * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + cfg.Training.GAMMA * cfg.Training.GAE_LAMBDA * (1-dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages)
    
    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state)
            
        return action.cpu().numpy().squeeze(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def update_policy(self):
        """Update policy using PPO algorithm"""
        # Convert memory to tensors
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.FloatTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        
        # Compute advantages
        advantages = self.compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones
        ).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(cfg.Training.EPOCHS):
            for idx in self.memory.sample():
                # Get batch
                state = states[idx]
                action = actions[idx]
                old_log_prob = old_log_probs[idx]
                advantage = advantages[idx]
                
                # Evaluate actions
                mean, std = self.actor(state)
                dist = Normal(mean, std)
                new_log_prob = dist.log_prob(action).sum(dim=-1)
                
                # PPO loss calculation
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-cfg.Training.CLIP_EPSILON, 1+cfg.Training.CLIP_EPSILON) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update critic
                value = self.critic(state)
                critic_loss = nn.MSELoss()(value, advantage + value.detach())
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
    
    def train(self, num_episodes=None):
        """Main training loop"""
        if num_episodes is None:
            num_episodes = cfg.Training.EPISODES
            
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(cfg.Training.STEPS_PER_EPISODE):
                # Select action
                action, log_prob, value = self.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                
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
            
            # Update policy
            self.update_policy()
            self.memory.clear()
            
            # Track progress
            self.episode_rewards.append(episode_reward)
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
    
    def save(self, path):
        """Save model checkpoints"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model checkpoints"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])