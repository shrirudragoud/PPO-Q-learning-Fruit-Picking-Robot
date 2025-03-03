# Troubleshooting Guide for Orange Harvesting Robot

## Common Training Issues

### 1. Dimension Mismatch Errors

```
Problem: ValueError: mat1 and mat2 shapes cannot be multiplied
Solution:
- Run analyze_dimensions.py to verify state and action spaces
- Check observation space construction in orange_harvest_env.py
- Ensure network input/output dimensions match environment
```

### 2. BatchNorm Issues

```
Problem: Expected more than 1 value per channel when training
Solution:
- Use fake batch creation for single samples
- Ensure training mode is properly set/unset
- Add batch dimension for single observations
```

### 3. Training Instability

```
Problem: Exploding/vanishing gradients
Solutions:
- Verify learning rates (Actor: 3e-4, Critic: 1e-3)
- Check gradient clipping (max_norm=0.5)
- Ensure proper reward scaling
- Monitor advantage normalization
```

### 4. Poor Performance

```
Problem: Robot not learning effectively
Solutions:
1. Check reward components:
   - Distance reward scaling
   - Progress reward calculation
   - Action penalties
   - Success rewards

2. Verify training parameters:
   - GAE Lambda (0.95)
   - Discount factor (0.99)
   - PPO clip epsilon (0.2)

3. Analyze logs:
   - Monitor average rewards
   - Check action distributions
   - Verify value function estimates
```

### 5. Robot Control Issues

```
Problem: Erratic movement or poor control
Solutions:
1. Base movement:
   - Verify velocity limits
   - Check differential drive parameters
   - Monitor wheel encoders

2. Arm control:
   - Validate joint limits
   - Check inverse kinematics
   - Verify grip threshold
```

## Debugging Tools

### 1. Network Analysis

```bash
# Run dimension analysis
python analyze_dimensions.py

# Test network initialization
python test_ppo.py
```

### 2. Environment Testing

```bash
# Test environment setup
python test_environment.bat

# Verify robot control
python test_robot_control.py
```

### 3. Training Monitoring

```bash
# Start tensorboard
tensorboard --logdir logs

# Monitor metrics:
- Actor/Critic losses
- Reward components
- Success rates
- Value estimates
```

## Performance Optimization

### 1. Memory Usage

- Batch size adjustments
- Replay buffer management
- Gradient accumulation for large batches

### 2. Training Speed

- GPU utilization
- Parallel environment simulation
- Action sampling optimization

### 3. Learning Efficiency

- Curriculum learning progression
- Reward shaping adjustments
- Experience prioritization

## Common Error Messages

### 1. Environment Errors

```
Error: "Failed to load robot URDF"
- Check URDF file path
- Verify file permissions
- Validate URDF syntax
```

### 2. Training Errors

```
Error: "CUDA out of memory"
- Reduce batch size
- Monitor GPU memory usage
- Clear unused tensors

Error: "Loss is NaN"
- Check gradient clipping
- Verify reward scaling
- Monitor network initialization
```

### 3. Control Errors

```
Error: "Joint limit exceeded"
- Verify action clipping
- Check joint angle limits
- Monitor velocity constraints
```

## Logging and Monitoring

### 1. Essential Metrics

- Episode rewards
- Action distributions
- Value predictions
- Advantage estimates

### 2. Debug Information

- State dimensions
- Network architecture
- Gradient norms
- Memory usage

### 3. Performance Indicators

- Success rate
- Average episode length
- Reward components
- Learning curves

## Configuration Checklist

### 1. Environment Setup

- [ ] PyBullet installation
- [ ] URDF file paths
- [ ] Physics parameters
- [ ] Robot constraints

### 2. Network Configuration

- [ ] Layer dimensions
- [ ] Activation functions
- [ ] BatchNorm parameters
- [ ] Weight initialization

### 3. Training Parameters

- [ ] Learning rates
- [ ] Batch sizes
- [ ] Episode lengths
- [ ] Reward weights
