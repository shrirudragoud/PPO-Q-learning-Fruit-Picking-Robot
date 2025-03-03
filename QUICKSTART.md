# Orange Harvesting Robot - Training Guide

## Setup

1. Automated Setup (Recommended):

```bash
# Run the automated setup script
setup_and_verify.bat

# This will:
# - Create a Python virtual environment
# - Install PyTorch with CUDA support
# - Install all dependencies
# - Verify the environment
# - Test dimensions and robot initialization
```

2. Verify Installation:

After setup completes, check that:

- All dependencies were installed successfully
- PyTorch can detect your GPU (if available)
- Environment dimensions are correct
- Robot initializes without errors

3. Test Environment:

```bash
# Run the environment test script
test_environment.bat

# This will:
# - Test robot movement
# - Verify reward calculations
# - Check fruit collection mechanics
```

4. Troubleshooting:

If you encounter issues:

```bash
# Run dimension analysis
python debug_dims.py

# Check PyTorch/CUDA status
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Reinstall PyTorch with CUDA (if needed)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Training

The training process is divided into three phases to gradually build up the robot's capabilities:

### Phase 1: Basic Movement (500 episodes)

- Focuses on navigation and positioning
- Higher weight on distance-based rewards
- Simplified arm control

```bash
python train_harvester.py --phase 1
```

### Phase 2: Precise Control (1000 episodes)

- Introduces arm movement complexity
- Adds contact rewards
- Starts efficiency penalties

```bash
python train_harvester.py --phase 2
```

### Phase 3: Full Task (1500 episodes)

- Complete task with all reward components
- Optimizes for speed and efficiency
- Fine-tunes movement patterns

```bash
python train_harvester.py --phase 3
```

## Training Parameters

You can customize training parameters using command line arguments:

```bash
python train_harvester.py \
    --phase 1 \
    --episodes 500 \
    --learning-rate 3e-4 \
    --batch-size 64 \
    --no-gui  # Run without visualization
```

## Evaluation

To evaluate a trained model:

```bash
python evaluate_harvester.py \
    models/phase_3_final.pt \
    --episodes 10
```

The evaluation will show:

- Success rate (fruits collected)
- Average completion time
- Movement efficiency
- Overall performance metrics

## Reward Components

The training uses multiple reward components:

1. Distance Reward

   - Encourages moving closer to fruits
   - Exponential scaling for precision

2. Progress Reward

   - Rewards improvement in position
   - Encourages efficient paths

3. Action Efficiency

   - Penalties for unnecessary movements
   - Promotes smooth trajectories

4. Contact & Success Rewards
   - Large rewards for successful grips
   - Completion bonuses for fruit collection

## Training Tips

1. Monitor Progress

   - Watch tensorboard logs for learning curves
   - Check success rate trends
   - Monitor average episode length

2. Common Issues

   - If the robot moves erratically, lower the learning rate
   - If progress stalls, try adjusting reward weights
   - If arm control is imprecise, increase training episodes

3. Performance Optimization
   - Use --no-gui for faster training
   - Adjust batch size based on available memory
   - Consider lowering action frequency for smoother control

## Visualization

To visualize training progress:

```bash
tensorboard --logdir logs
```

Key metrics to watch:

- Average episode reward
- Success rate
- Action distributions
- Value function estimates

## Model Files

Trained models are saved in the `models/` directory:

- `phase_1_final.pt`: Basic movement model
- `phase_2_final.pt`: Precise control model
- `phase_3_final.pt`: Complete task model

Checkpoints are saved every 100 episodes during training.

## Next Steps

After successful training, you can:

1. Fine-tune the model for specific scenarios
2. Experiment with different reward weightings
3. Test in varied environments
4. Implement multi-robot coordination
