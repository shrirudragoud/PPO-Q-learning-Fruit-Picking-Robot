# Training Plan for Orange Harvesting Robot

## 1. Environment Wrapper (OrangeHarvestEnv)

### State Space

- Robot state: position, orientation, velocities
- Arm joint angles and gripper position
- Nearest fruit position relative to gripper
- Distance and direction to target fruit
- Additional observations: remaining fruits, etc.

### Action Space

- Continuous control for:
  - Robot movement (forward/turn)
  - Arm joints (shoulder, elbow, wrist)
  - Gripper control
- All actions normalized to [-1, 1]

### Reward Structure

1. **Distance-based Reward**

   - Negative reward proportional to distance from gripper to nearest fruit
   - Encourages efficient movement towards fruits

2. **Contact Reward**

   - Large positive reward when gripper touches fruit
   - Additional reward for successful grip and removal

3. **Efficiency Rewards**

   - Small negative reward per step to encourage speed
   - Penalties for excessive movement or harsh actions
   - Energy efficiency bonus for smooth movements

4. **Task Completion**
   - Bonus rewards for successful fruit collection
   - Episode completion bonus based on total fruits collected

### Episode Management

- **Reset Conditions**

  - Start with robot in random position
  - Randomize fruit positions on trees
  - Clear performance metrics

- **Step Function**

  - Process actions and update environment
  - Calculate rewards
  - Generate new state observations
  - Check terminal conditions

- **Done Conditions**
  - All fruits collected
  - Maximum steps reached
  - Robot collision with obstacles
  - Successful fruit pick

## 2. Training Parameters

### PPO Configuration

- Batch size: 64
- Learning rates:
  - Actor: 3e-4
  - Critic: 1e-3
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- PPO epochs: 10
- Clip epsilon: 0.2

### Neural Network Architecture

- Actor Network:

  - Input: State dimension
  - Hidden: [256, 256]
  - Output: Action dimension
  - Activation: ReLU, tanh

- Critic Network:
  - Input: State dimension
  - Hidden: [256, 256]
  - Output: 1 (Value)
  - Activation: ReLU

## 3. Implementation Steps

1. Create Environment Wrapper

   - Implement state/action space
   - Set up reward calculation
   - Define reset/step functions

2. Modify Existing Code

   - Update fruit detection for gripper position
   - Add contact detection between gripper and fruits
   - Implement smooth movement controls

3. Training Script

   - Set up data collection
   - Implement PPO training loop
   - Add logging and visualization

4. Evaluation

   - Create metrics for success rate
   - Monitor training progress
   - Save best models

5. Testing and Tuning
   - Adjust reward weights
   - Fine-tune hyperparameters
   - Test in different scenarios

## 4. Success Metrics

1. Training Metrics

   - Average episode reward
   - Success rate (fruits collected/total fruits)
   - Average steps per successful pick
   - Learning curve stability

2. Performance Metrics
   - Time to collect all fruits
   - Energy efficiency
   - Smoothness of motion
   - Collision avoidance success

## 5. Phases of Training

1. Basic Movement (Phase 1)

   - Learn to navigate to fruits
   - Basic arm positioning

2. Precise Control (Phase 2)

   - Fine-grained gripper control
   - Accurate fruit targeting

3. Efficiency Training (Phase 3)

   - Optimize movement patterns
   - Reduce energy usage
   - Increase speed

4. Full Task (Phase 4)
   - Complete harvesting sequence
   - Handle multiple fruits
   - Adapt to different scenarios
