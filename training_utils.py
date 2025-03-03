#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import os

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        """Initialize training logger with TensorBoard support"""
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(self.log_dir)
        self.episode_rewards = []
        self.avg_rewards = []
        self.steps_history = []
        
    def log_episode(self, episode, reward, steps, losses=None, metrics=None):
        """Log episode results"""
        self.episode_rewards.append(reward)
        self.steps_history.append(steps)
        
        # Calculate running average
        avg_reward = np.mean(self.episode_rewards[-100:])
        self.avg_rewards.append(avg_reward)
        
        # Log to TensorBoard
        self.writer.add_scalar('Reward/episode', reward, episode)
        self.writer.add_scalar('Reward/average', avg_reward, episode)
        self.writer.add_scalar('Steps/episode', steps, episode)
        
        if losses:
            for name, value in losses.items():
                self.writer.add_scalar(f'Loss/{name}', value, episode)
        
        if metrics:
            for name, value in metrics.items():
                self.writer.add_scalar(f'Metrics/{name}', value, episode)
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot rewards
        episodes = range(len(self.episode_rewards))
        ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
        ax1.plot(episodes, self.avg_rewards, 'r-', label='Average Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # Plot steps
        ax2.plot(episodes, self.steps_history, 'g-', label='Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        """Initialize replay buffer for off-policy learning"""
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        
        # Allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices])
        )
    
    def clear(self):
        """Clear buffer"""
        self.pointer = 0
        self.size = 0

def create_checkpoint(model, optimizer, epoch, path):
    """Create a training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load a training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """Initialize early stopping"""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop