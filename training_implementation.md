# Orange Harvesting Robot - Training Implementation Plan

## 1. Environment Wrapper (training_env.py)

```python
class OrangeHarvestEnv:
    def __init__(self):
        self.env = FarmEnvironment()
        self.robot = RobotController()

        # State/Action spaces
        self.observation_space = spaces.Box(...)  # Based on config
        self.action_space = spaces.Box(...)       # Based on config

        # Training parameters
        self.max_steps = cfg.Training.STEPS_PER_EPISODE
        self.current_step = 0
        self.total_fruits = cfg.Environment.FRUITS_PER_TREE * cfg.Environment.NUM_TREES
```

### Key Methods:

```python
def reset(self):
    """Reset environment and return initial state"""
    self.current_step = 0
    # Reset environment and robot
    # Return normalized observation

def step(self, action):
    """Execute action and return (state, reward, done, info)"""
    # Scale and apply action
    # Get gripper and fruit positions
    # Calculate reward
    # Return next state and info

def get_observation(self):
    """Get normalized observation vector"""
    # Combine robot state, gripper position, nearest fruit info

def calculate_reward(self, gripper_pos, fruit_pos, action):
    """Calculate reward based on multiple components"""
    # Distance reward
    # Contact reward
    # Efficiency penalty
    # Success reward
```

## 2. Training Script (train_harvester.py)

```python
def main():
    # Initialize environment and PPO trainer
    env = OrangeHarvestEnv()
    trainer = PPOTrainer(env)
    logger = TrainingLogger()

    # Training loop
    for episode in range(cfg.Training.EPISODES):
        state = env.reset()
        episode_reward = 0

        # Episode loop
        for step in range(cfg.Training.STEPS_PER_EPISODE):
            action = trainer.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store transition
            trainer.memory.store(state, action, reward, next_state, done)

            if done: break
            state = next_state

        # Update policy
        trainer.update()

        # Log progress
        logger.log_episode(...)
```

## 3. Evaluation Script (evaluate_harvester.py)

```python
def evaluate(model_path, num_episodes=10):
    # Load trained model
    env = OrangeHarvestEnv()
    trainer = PPOTrainer(env)
    trainer.load(model_path)

    # Evaluation metrics
    metrics = {
        'success_rate': [],
        'completion_time': [],
        'efficiency': []
    }

    # Run evaluation episodes
    for episode in range(num_episodes):
        # Run episode and collect metrics
```

## 4. Implementation Order

1. Environment Wrapper:

   - State/action space definition
   - Reset/step logic
   - Reward calculation
   - Observation normalization

2. Training Integration:

   - Environment instantiation
   - PPO parameter tuning
   - Training loop setup
   - Progress logging

3. Evaluation System:
   - Metrics tracking
   - Visualization
   - Performance analysis

## 5. Key Components

### State Space

- Robot state (9 dimensions)
- Environment state (5 dimensions)
- Lidar readings (16 dimensions)
  Total: 30 dimensions

### Action Space

- Robot movement (2 dimensions)
- Arm control (3 dimensions)
- Gripper control (1 dimension)
  Total: 6 dimensions

### Reward Components

1. Distance Reward:

   ```python
   distance_reward = -np.tanh(distance_to_fruit)
   ```

2. Contact Reward:

   ```python
   contact_reward = 10.0 if gripper_contact else 0.0
   ```

3. Efficiency Penalty:

   ```python
   efficiency_penalty = -0.01 * np.sum(np.abs(action))
   ```

4. Success Reward:
   ```python
   success_reward = 50.0 if fruit_picked else 0.0
   ```

## 6. Training Schedule

### Phase 1: Basic Movement (500 episodes)

- Focus on navigation
- Higher weight on distance reward
- Simplified arm control

### Phase 2: Precise Control (1000 episodes)

- Introduce arm movement
- Add contact rewards
- Start efficiency penalties

### Phase 3: Full Task (1500 episodes)

- All reward components
- Full state/action space
- Performance optimization

## 7. Hyperparameter Configuration

```python
{
    'learning_rate': 3e-4,
    'batch_size': 64,
    'epochs': 10,
    'clip_epsilon': 0.2,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5
}
```

## 8. Success Metrics

1. Episode Success Rate:

   - Number of fruits successfully picked / Total fruits

2. Completion Time:

   - Steps per successful pick
   - Total episode duration

3. Efficiency Metrics:

   - Average action magnitude
   - Path optimization
   - Energy usage

4. Learning Metrics:
   - Reward curve
   - Policy loss
   - Value loss
   - Success rate over time
