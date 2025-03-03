#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
from collections import deque
from config import SimConfig as cfg

# Setup PyTorch with error handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except ImportError as e:
    print("Failed to import PyTorch. Please install with: pip install torch")
    print(f"Error: {e}")
    sys.exit(1)

from training_logger import TrainingLogger


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        print(f"Creating Actor Network - Input dim: {state_dim}, Output dim: {action_dim}")
        
        # Calculate layer sizes
        hidden1_size = max(state_dim * 2, 64)  # Ensure minimum size
        hidden2_size = max(state_dim, 32)      # Ensure minimum size
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size, momentum=0.01)
        
        # Hidden layer
        self.hidden_layer = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size, momentum=0.01)
        
        # Output layer
        self.output_layer = nn.Linear(hidden2_size, action_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize log standard deviation and weights
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        # Ensure input is tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Forward pass through layers
        x = self.input_layer(state)
        x = self.relu(x)
        x = self.bn1(x)
        
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.bn2(x)
        
        mean = self.tanh(self.output_layer(x))
        std = torch.exp(self.log_std.expand(mean.size(0), -1)).clamp(-20, 2)
        
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        print(f"Creating Critic Network - Input dim: {state_dim}")
        
        # Calculate layer sizes
        hidden1_size = max(state_dim * 2, 64)  # Ensure minimum size
        hidden2_size = max(state_dim, 32)      # Ensure minimum size
        
        # Input layer
        self.input_layer = nn.Linear(state_dim, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size, momentum=0.01)
        
        # Hidden layer
        self.hidden_layer = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size, momentum=0.01)
        
        # Output layer
        self.output_layer = nn.Linear(hidden2_size, 1)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        # Ensure input is tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(next(self.parameters()).device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Handle single sample during evaluation
        if state.size(0) == 1:
            is_single = True
            state = state.repeat(2, 1)  # Create fake batch
        else:
            is_single = False
            
        # Forward pass through layers
        x = self.input_layer(state)
        x = self.relu(x)
        x = self.bn1(x)
        
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.bn2(x)
        
        value = self.output_layer(x)
        
        # Return only the first value if we created a fake batch
        if is_single:
            value = value[0].unsqueeze(0)
            
        return value

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
        
        # Get and verify dimensions from environment
        print("\nInitializing PPO Networks...")
        try:
            # Get initial state to verify dimensions
            initial_state = env.reset()
            actual_state_dim = len(initial_state)
            
            # Store dimensions
            self.state_dim = actual_state_dim
            self.action_dim = env.action_space.shape[0]
            
            print(f"\nNetwork Dimensions:")
            print(f"State: {self.state_dim}")
            print(f"Action: {self.action_dim}")
            
        except Exception as e:
            print(f"Error initializing networks: {e}")
            raise
        
        # Initialize networks
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim).to(device)
        
        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.Training.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.Training.CRITIC_LR)
        
        # Initialize memory
        self.memory = PPOMemory(cfg.Training.BATCH_SIZE)
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.actor_losses = deque(maxlen=100)
        self.critic_losses = deque(maxlen=100)
        self.value_estimates = deque(maxlen=100)
        self.total_steps = 0
        self.episodes_completed = 0
        
        # Setup logging
        self.logger = TrainingLogger()
    
    def _log_metrics(self, metrics, step):
        """Log training metrics"""
        try:
            self.logger.log_metrics(metrics, step)
        except Exception as e:
            print(f"Error logging metrics: {e}")
    
    def _validate_state(self, state):
        """Validate and preprocess state"""
        if len(state) != self.state_dim:
            print(f"\nWarning: State dimension mismatch!")
            print(f"Expected: {self.state_dim}, Got: {len(state)}")
            state = state[:self.state_dim] if len(state) > self.state_dim else \
                   np.pad(state, (0, self.state_dim - len(state)))
        return torch.FloatTensor(state).to(self.device)

    def select_action(self, state, eval_mode=False):
        """Select action using current policy"""
        try:
            if eval_mode:
                self.actor.eval()
                self.critic.eval()
            
            with torch.no_grad():
                # Process state
                state_tensor = self._validate_state(state)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                
                # Create fake batch for batch norm during evaluation
                if eval_mode and state_tensor.size(0) == 1:
                    state_tensor = state_tensor.repeat(2, 1)
                    use_fake_batch = True
                else:
                    use_fake_batch = False
                
                # Get action distribution parameters
                mean, std = self.actor(state_tensor)
                
                # Remove fake batch if used
                if use_fake_batch:
                    mean = mean[0].unsqueeze(0)
                    std = std[0].unsqueeze(0)
                
                if eval_mode:
                    action = mean
                else:
                    dist = Normal(mean, std)
                    action = dist.sample()
                
                # Get log probability and value
                dist = Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state_tensor)
                
                # Clip actions to valid range
                action = torch.clamp(action, -1.0, 1.0)
                
                # Convert to numpy
                action_np = action.cpu().numpy().squeeze()
                log_prob_np = log_prob.cpu().numpy()
                value_np = value.cpu().numpy().squeeze()
            
            # Reset to training mode
            if eval_mode:
                self.actor.train()
                self.critic.train()
            
            return action_np, log_prob_np, value_np
            
        except Exception as e:
            print(f"\nError in select_action:")
            print(f"State shape: {state.shape if hasattr(state, 'shape') else len(state)}")
            print(f"Error details: {str(e)}")
            raise

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + cfg.Training.GAMMA * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + cfg.Training.GAMMA * cfg.Training.GAE_LAMBDA * (1-dones[t]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages)

    def _process_batch(self, states, actions, old_log_probs, advantages, batch_indices):
        """Process a batch of data during policy update"""
        try:
            # Get batch data
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            
            # Ensure proper batch size for BatchNorm
            if batch_states.size(0) == 1:
                batch_states = batch_states.repeat(2, 1)
                batch_actions = batch_actions.repeat(2, 1)
                batch_old_log_probs = batch_old_log_probs.repeat(2)
                batch_advantages = batch_advantages.repeat(2)
                using_fake_batch = True
            else:
                using_fake_batch = False
            
            # Actor forward pass
            mean, std = self.actor(batch_states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            
            # Compute ratio and surrogate objectives
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1-cfg.Training.CLIP_EPSILON,
                              1+cfg.Training.CLIP_EPSILON) * batch_advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic forward pass
            values = self.critic(batch_states)
            critic_loss = nn.MSELoss()(values, batch_advantages.unsqueeze(-1) + values.detach())
            
            # If using fake batch, only use first element
            if using_fake_batch:
                actor_loss = actor_loss / 2
                critic_loss = critic_loss / 2
            
            return actor_loss, critic_loss
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            print(f"Batch states shape: {batch_states.shape}")
            print(f"Batch actions shape: {batch_actions.shape}")
            raise

    def update_policy(self):
        """Update policy using collected experience"""
        try:
            # Convert memory to tensors
            states = torch.FloatTensor(self.memory.states).to(self.device)
            actions = torch.FloatTensor(self.memory.actions).to(self.device)
            old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
            
            # Compute and normalize advantages
            advantages = self.compute_gae(
                self.memory.rewards,
                self.memory.values,
                self.memory.dones
            ).to(self.device)
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            total_actor_loss = 0
            total_critic_loss = 0
            num_updates = 0
            
            # Multiple training epochs
            for _ in range(cfg.Training.EPOCHS):
                for idx in self.memory.sample():
                    # Process batch
                    actor_loss, critic_loss = self._process_batch(
                        states, actions, old_log_probs, advantages, idx)
                    
                    # Update actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()
                    
                    # Update critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
                    
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    num_updates += 1
            
            # Log average losses
            if num_updates > 0:
                avg_actor_loss = total_actor_loss / num_updates
                avg_critic_loss = total_critic_loss / num_updates
                self.actor_losses.append(avg_actor_loss)
                self.critic_losses.append(avg_critic_loss)
                
                metrics = {
                    'actor_loss': avg_actor_loss,
                    'critic_loss': avg_critic_loss,
                    'advantage_mean': advantages.mean().item(),
                    'advantage_std': advantages.std().item()
                }
                self._log_metrics(metrics, self.total_steps)
                    
        except Exception as e:
            print(f"Error in policy update: {e}")
            raise
    
    def train(self, num_episodes=None):
        """Train the agent"""
        if num_episodes is None:
            num_episodes = cfg.Training.EPISODES
            
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            for step in range(cfg.Training.STEPS_PER_EPISODE):
                # Select and execute action
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.rewards.append(reward)
                self.memory.values.append(value)
                self.memory.log_probs.append(log_prob)
                self.memory.dones.append(done)
                
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                
                if done:
                    break
            
            # Update policy
            self.update_policy()
            self.memory.clear()
            
            # Track progress
            self.episode_rewards.append(episode_reward)
            self.episodes_completed += 1
            
            # Log episode results
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            metrics = {
                'episode_reward': episode_reward,
                'average_reward': avg_reward,
                'episode_length': step + 1,
                'fruits_collected': info.get('fruits_collected', 0)
            }
            self._log_metrics(metrics, self.total_steps)
            
            print(f"Episode {episode + 1}: "
                  f"Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Steps: {step + 1}, "
                  f"Fruits: {info.get('fruits_collected', 0)}")
            
            # Save checkpoint
            if (episode + 1) % cfg.Training.CHECKPOINT_INTERVAL == 0:
                self.save(f"models/checkpoint_{episode + 1}.pt")
    
    def save(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episodes_completed': self.episodes_completed
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at: {path}")
        
        try:
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.total_steps = checkpoint.get('total_steps', 0)
            self.episodes_completed = checkpoint.get('episodes_completed', 0)
            print(f"Successfully loaded checkpoint from {path}")
            print(f"Resumed from: Steps={self.total_steps}, Episodes={self.episodes_completed}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise