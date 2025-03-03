# Orange Harvesting Robot Implementation Details

## Control System Architecture

### 1. Robot Control Structure

- **Base Movement**:

  - Forward/backward velocity (±1.0 m/s)
  - Turning velocity (±1.0 rad/s)
  - Uses differential drive with 4 wheels

- **Arm Control**:
  - Shoulder joint: ±1.57 rad range
  - Elbow joint: ±2.0 rad range
  - Wrist joint: ±1.57 rad range
  - Gripper with 0.03m maximum opening

### 2. Observation Space

- **Robot State** (9 dimensions):

  - Position (x, y)
  - Orientation (sine and cosine)
  - Linear and angular velocities
  - Joint angles (shoulder, elbow, wrist)

- **Environment State** (4 dimensions):

  - Distance to nearest fruit
  - Direction to nearest fruit
  - Progress metric
  - Fruits remaining

- **LIDAR Readings** (16 dimensions):
  - 16 rays for obstacle detection
  - 5.0m maximum ray length

### 3. Action Space (6 dimensions):

- Forward velocity
- Turning velocity
- Shoulder position
- Elbow position
- Wrist position
- Gripper position

## RL Network Architecture

### 1. Actor Network

- **Input Layer**: State dimension (29 features)
- **Hidden Layers**:
  - Layer 1: 2x state dim (58 neurons)
  - Batch Normalization
  - ReLU activation
  - Layer 2: 1x state dim (29 neurons)
  - Batch Normalization
  - ReLU activation
- **Output Layer**:
  - 6 neurons (action dimensions)
  - Tanh activation
  - Learnable standard deviation

### 2. Critic Network

- **Input Layer**: State dimension (29 features)
- **Hidden Layers**:
  - Similar structure to Actor
  - Batch normalization for stable training
- **Output Layer**:
  - 1 neuron (value estimate)
  - Linear activation

### 3. Training Parameters

- **PPO Specific**:
  - Clip epsilon: 0.2
  - GAE lambda: 0.95
  - Gamma (discount): 0.99
  - Learning rates:
    - Actor: 3e-4
    - Critic: 1e-3

### 4. Reward Structure

1. **Distance Reward**:

   - Exponential scaling for precision
   - Encourages moving closer to fruits

2. **Progress Reward**:

   - Based on improvement in position
   - Promotes efficient paths

3. **Action Efficiency**:

   - Penalties for unnecessary movements
   - Promotes smooth trajectories

4. **Success Rewards**:
   - Large rewards for successful grips
   - Completion bonuses for fruit collection

### 5. Training Implementation

- Batch size: 64
- Episodes: 1000 per phase
- Steps per episode: 1000
- Phase-based training:
  1. Basic Movement
  2. Precise Control
  3. Full Task

## Key Implementation Features

### Batch Normalization

- Used in both Actor and Critic networks
- Helps stabilize training
- Improves gradient flow
- Reduces internal covariate shift

### Gradient Clipping

- Maximum gradient norm: 0.5
- Prevents exploding gradients
- Enables stable learning

### Multi-Phase Training

1. **Phase 1: Basic Movement**

   - Focus on navigation
   - Higher weight on distance rewards
   - Simplified arm control

2. **Phase 2: Precise Control**

   - Introduces arm movement complexity
   - Adds contact rewards
   - Starts efficiency penalties

3. **Phase 3: Full Task**
   - Complete task with all reward components
   - Optimizes for speed and efficiency
   - Fine-tunes movement patterns

### Error Handling and Safety

- Input validation for state dimensions
- Action clipping to safe ranges
- Batch size adjustments for single samples
- Comprehensive logging and monitoring

### Performance Monitoring

- Tensorboard integration
- Custom metrics logging
- Episode reward tracking
- Success rate monitoring
- Network loss visualization
