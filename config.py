#!/usr/bin/env python3

"""
Configuration file for the fruit harvesting robot simulation.
Includes parameters for simulation, robot control, and future RL training.
"""

class SimConfig:
    # Simulation parameters
    TIMESTEP = 1.0/240.0  # 240 Hz simulation
    MAX_STEPS = 1000      # Maximum steps per episode
    GUI_ENABLED = True    # Enable visualization
    
    # Physics parameters
    GRAVITY = -9.81
    LATERAL_FRICTION = 0.8
    SPINNING_FRICTION = 0.1
    ROLLING_FRICTION = 0.1
    
    # Robot parameters
    class Robot:
        # Movement limits
        MAX_VELOCITY = 1.0        # m/s
        MAX_TURNING_SPEED = 1.0   # rad/s
        
        # Arm joint limits
        SHOULDER_LIMITS = (-1.57, 1.57)  # radians
        ELBOW_LIMITS = (-2.0, 2.0)       # radians
        WRIST_LIMITS = (-1.57, 1.57)     # radians
        
        # Gripper parameters
        GRIPPER_FORCE = 10.0      # N
        GRIPPER_MAX_DIST = 0.03   # m
        GRIP_THRESHOLD = 0.01     # m
    
    # Environment parameters
    class Environment:
        # Farm layout
        GROUND_SIZE = 20.0        # m
        NUM_TREES = 5
        TREE_MIN_DIST = 2.0       # Minimum distance between trees
        
        # Tree parameters
        TRUNK_HEIGHT = 2.0
        TRUNK_RADIUS = 0.2
        CANOPY_RADIUS = 1.5
        
        # Fruit parameters
        FRUITS_PER_TREE = 5
        FRUIT_RADIUS = 0.05
        FRUIT_MASS = 0.1
        
        # Reward parameters
        REWARD_FRUIT_PICKED = 10.0
        REWARD_COLLISION = -5.0
        REWARD_STEP = -0.01
    
    # RL Training parameters (for future use)
    class Training:
        # PPO parameters
        EPISODES = 1000
        STEPS_PER_EPISODE = 1000
        BATCH_SIZE = 64
        EPOCHS = 10
        CHECKPOINT_INTERVAL = 100  # Save checkpoint every N episodes
        
        # Learning rates
        ACTOR_LR = 0.0003
        CRITIC_LR = 0.001
        
        # PPO specific
        GAMMA = 0.99              # Discount factor
        GAE_LAMBDA = 0.95         # GAE parameter
        CLIP_EPSILON = 0.2        # PPO clip parameter
        VALUE_CLIP = 0.2          # Value function clip parameter
        ENTROPY_COEF = 0.01       # Entropy coefficient
        
        # Observation space
        class Observation:
            ROBOT_STATE = [
                'position_x',
                'position_y',
                'orientation',
                'linear_velocity',
                'angular_velocity',
                'shoulder_angle',
                'elbow_angle',
                'wrist_angle',
                'gripper_position'
            ]
            
            ENVIRONMENT_STATE = [
                'nearest_fruit_distance',
                'nearest_fruit_direction',
                'nearest_tree_distance',
                'nearest_tree_direction',
                'fruits_remaining'
            ]
            
            LIDAR_RAYS = 16       # Number of lidar rays
            RAY_LENGTH = 5.0      # Maximum ray length
        
        # Action space
        class Action:
            CONTINUOUS = [
                'forward_velocity',
                'turning_velocity',
                'shoulder_position',
                'elbow_position',
                'wrist_position',
                'gripper_position'
            ]
            
            # Action bounds
            VELOCITY_RANGE = (-1.0, 1.0)
            JOINT_RANGE = (-1.0, 1.0)
            GRIPPER_RANGE = (0.0, 1.0)
    
    # Visualization parameters
    class Visualization:
        # Camera settings
        CAMERA_DISTANCE = 7.0
        CAMERA_YAW = 45.0
        CAMERA_PITCH = -30.0
        
        # Debug options
        SHOW_JOINT_INFO = True
        SHOW_TRAJECTORIES = True
        SHOW_GRID = True
        SHOW_TREE_IDS = True
        SHOW_FRUIT_IDS = True
        
        # Colors (RGBA)
        COLORS = {
            'grid': [0.5, 0.5, 0.5, 0.5],
            'trajectory': [0.0, 1.0, 0.0, 0.8],
            'highlight': [1.0, 1.0, 0.0, 1.0],
            'selected': [0.0, 1.0, 0.0, 1.0],
            'warning': [1.0, 0.0, 0.0, 1.0]
        }
        
        # UI parameters
        TEXT_SIZE = 1.2
        LINE_WIDTH = 2.0
        POINT_SIZE = 3.0