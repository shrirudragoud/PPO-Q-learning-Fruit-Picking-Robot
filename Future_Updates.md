To resolve the BatchNorm error during action selection in your PPO implementation, implement these changes:

**1. Network Architecture Modification**  
Replace all `BatchNorm1d` layers with `LayerNorm` in both actor and critic networks:

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Linear(64, 1)
        )
```

**2. Action Selection Handling**  
Ensure proper batch dimension handling in your `select_action` method:

```python
def select_action(self, state):
    try:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            action_mean = self.actor(state)
            # ... rest of your action selection logic
        return action.squeeze(0).cpu().numpy(), log_prob, value  # Remove batch dimension
    except Exception as e:
        logger.error(f"Action selection error: {e}")
        return np.zeros(self.action_dim), 0, 0  # Safe fallback
```

**3. Training Process Improvements**  
Add these enhancements to your training loop:

```python
# During training:
self.actor.train()
self.critic.train()

# During evaluation:
self.actor.eval()
self.critic.eval()

# Consider adding experience replay buffer
replay_buffer = ReplayBuffer(capacity=10000)
```

**4. Architectural Recommendations**  
Consider these additional improvements:

- Increase network capacity to 3-4 hidden layers (256-512 units)
- Add attention mechanisms for processing lidar data
- Implement separate network branches for different sensor modalities

By using LayerNorm instead of BatchNorm, you eliminate the batch size dependency while maintaining the benefits of normalization. This change addresses the root cause of the error while being more suitable for RL applications.
