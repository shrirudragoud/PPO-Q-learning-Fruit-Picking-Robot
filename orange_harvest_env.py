#!/usr/bin/env python3

import os
import numpy as np
import pybullet as p
import pybullet_data
from environment import FarmEnvironment
from robot_control import RobotController
from env_base import Env, Box
from config import SimConfig as cfg

class OrangeHarvestEnv(Env):
    def __init__(self, gui=False):
        super().__init__()
        
        # Initialize simulation
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, cfg.GRAVITY)
        p.setTimeStep(cfg.TIMESTEP)

        # Setup resources path
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Create environment
        self.env = FarmEnvironment(self.client)
        
        # Load robot and create controller
        try:
            self.robot_id = self._load_robot()
            if self.robot_id is None:
                raise Exception("Failed to load robot URDF")
            self.robot = RobotController(self.robot_id)
        except Exception as e:
            print(f"Error initializing robot: {e}")
            p.disconnect(self.client)
            raise
        
        # Training parameters
        self.max_steps = cfg.Training.STEPS_PER_EPISODE
        self.current_step = 0
        self.total_fruits = cfg.Environment.FRUITS_PER_TREE * cfg.Environment.NUM_TREES
        self.fruits_collected = 0
        
        # Reward weights (for phased training)
        self.reward_weights = {
            'distance': 1.0,
            'progress': 1.0,
            'action': 1.0,
            'contact': 1.0,
            'success': 1.0
        }
        
        # Define action space
        action_dim = len(cfg.Training.Action.CONTINUOUS)
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=action_dim
        )
        
        # Calculate actual state dimensions
        robot_state = self._get_robot_state()
        env_state = self._get_environment_state()
        lidar_state = self._get_lidar_readings()
        
        # Total observation dimension
        obs_dim = len(robot_state) + len(env_state) + len(lidar_state)
        print(f"Actual state dimensions: Robot({len(robot_state)}), "
              f"Env({len(env_state)}), Lidar({len(lidar_state)})")
        
        # Define observation space with correct dimensions
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,)
        )
        
        # Initialize metrics
        self.episode_rewards = 0
        self.last_distance = None
    
    def _load_robot(self):
        """Load robot URDF and return its ID"""
        try:
            # Get absolute path to URDF file
            urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "fruit_harvesting_robot.urdf")
            
            # Verify file exists
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"Robot URDF file not found at: {urdf_path}")
            
            # Load URDF with error checking
            robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0.1],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=False,
                flags=p.URDF_USE_SELF_COLLISION
            )
            
            if robot_id is None:
                raise Exception("Failed to load robot URDF")
                
            print(f"Successfully loaded robot with ID: {robot_id}")
            return robot_id
            
        except Exception as e:
            print(f"Error loading robot URDF: {e}")
            raise
    
    def set_reward_weights(self, weights):
        """Update reward weights for different components"""
        for key, value in weights.items():
            if key in self.reward_weights:
                self.reward_weights[key] = value
    
    def reset(self):
        """Reset environment and return initial observation"""
        try:
            # Reset counters
            self.current_step = 0
            self.fruits_collected = 0
            self.episode_rewards = 0
            
            # Reset PyBullet simulation
            p.resetSimulation(self.client)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, cfg.GRAVITY)
            p.setTimeStep(cfg.TIMESTEP)
            
            # Reset environment and create new instance
            print("Resetting environment...")
            self.env = FarmEnvironment(self.client)
            
            # Load robot with error checking
            print("Reloading robot...")
            self.robot_id = self._load_robot()
            if self.robot_id is None:
                raise Exception("Failed to reload robot")
            
            # Create new robot controller
            self.robot = RobotController(self.robot_id)
            
            # Initialize robot state
            self.robot.stop()  # Reset all joint velocities
            
            # Get initial observation with error checking
            try:
                observation = self._get_observation()
                self.last_distance = self._get_distance_to_nearest_fruit()
                print("Environment reset successful")
                return observation
            except Exception as e:
                raise Exception(f"Failed to get initial observation: {e}")
            
        except Exception as e:
            print(f"Error during reset: {e}")
            # Try to clean up in case of error
            try:
                p.resetSimulation(self.client)
            except:
                pass
            raise
    
    def step(self, action):
        """Execute action and return (observation, reward, done, info)"""
        self.current_step += 1
        
        # Scale and apply action
        scaled_action = self._scale_action(action)
        self._apply_action(scaled_action)
        p.stepSimulation()
        
        # Get observation and calculate reward
        observation = self._get_observation()
        reward, reward_info = self._calculate_reward(action)
        self.episode_rewards += reward
        
        # Check if episode is done
        done = self._is_done()
        
        # Compile info
        info = {
            'fruits_collected': self.fruits_collected,
            'steps': self.current_step,
            'total_reward': self.episode_rewards,
            **reward_info
        }
        
        return observation, reward, done, info
    
    def _scale_action(self, action):
        """Scale actions from [-1, 1] to actual robot control values"""
        # Movement scaling
        forward_velocity = action[0] * cfg.Robot.MAX_VELOCITY
        turning_velocity = action[1] * cfg.Robot.MAX_TURNING_SPEED
        
        # Arm joint scaling
        shoulder = np.interp(action[2], [-1, 1], cfg.Robot.SHOULDER_LIMITS)
        elbow = np.interp(action[3], [-1, 1], cfg.Robot.ELBOW_LIMITS)
        wrist = np.interp(action[4], [-1, 1], cfg.Robot.WRIST_LIMITS)
        
        # Gripper scaling
        gripper = np.interp(action[5], [-1, 1], [0, cfg.Robot.GRIPPER_MAX_DIST])
        
        return [forward_velocity, turning_velocity, shoulder, elbow, wrist, gripper]
    
    def _apply_action(self, scaled_action):
        """Apply scaled action to robot"""
        forward_vel, turn_vel, shoulder, elbow, wrist, gripper = scaled_action
        
        # Apply movement
        if abs(forward_vel) > 0.05:  # Small threshold to prevent tiny movements
            self.robot.move_forward(forward_vel)
        if abs(turn_vel) > 0.05:
            self.robot.turn(turn_vel)
            
        # Apply arm control
        self.robot.control_arm(shoulder=shoulder, elbow=elbow, wrist=wrist)
        
        # Apply gripper control
        self.robot.control_gripper(position=gripper)
    
    def _get_observation(self):
        """Get current observation of environment"""
        # Get robot state
        robot_state = self._get_robot_state()
        
        # Get environment state
        env_state = self._get_environment_state()
        
        # Get lidar-like readings
        lidar_state = self._get_lidar_readings()
        
        # Combine all states
        observation = np.concatenate([robot_state, env_state, lidar_state])
        return observation.astype(np.float32)
    
    def _get_robot_state(self):
        """Get normalized robot state"""
        gripper_pos = self.robot.get_gripper_position()
        joint_states = self.robot.get_joint_states()
        
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        
        # Normalize positions to environment size
        norm_pos = np.array(robot_pos[:2]) / (cfg.Environment.GROUND_SIZE / 2)
        
        # Convert quaternion to euler angles
        euler = p.getEulerFromQuaternion(robot_orn)
        
        # Compile robot state
        state = [
            norm_pos[0],                    # x position
            norm_pos[1],                    # y position
            np.sin(euler[2]),              # yaw sine
            np.cos(euler[2]),              # yaw cosine
            linear_vel[0]/cfg.Robot.MAX_VELOCITY,    # normalized linear velocity
            angular_vel[2]/cfg.Robot.MAX_TURNING_SPEED,  # normalized angular velocity
            joint_states['shoulder_joint']['position']/cfg.Robot.SHOULDER_LIMITS[1],
            joint_states['elbow_joint']['position']/cfg.Robot.ELBOW_LIMITS[1],
            joint_states['wrist_pitch_joint']['position']/cfg.Robot.WRIST_LIMITS[1]
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _get_environment_state(self):
        """Get normalized environment state"""
        gripper_pos = self.robot.get_gripper_position()
        closest_fruit, distance, fruit_pos = self.env.get_closest_fruit(gripper_pos)
        
        if fruit_pos is not None:
            # Calculate relative position
            relative_pos = np.array(fruit_pos) - np.array(gripper_pos)
            direction = np.arctan2(relative_pos[1], relative_pos[0])
            
            # Normalize distance
            norm_distance = distance / (cfg.Environment.GROUND_SIZE / 2)
            
            state = [
                norm_distance,           # distance to nearest fruit
                np.sin(direction),      # direction sine
                np.cos(direction),      # direction cosine
                self.fruits_collected / self.total_fruits  # progress
            ]
        else:
            state = [1.0, 0.0, 0.0, 1.0]  # No fruits left
        
        return np.array(state, dtype=np.float32)
    
    def _get_lidar_readings(self):
        """Simulate lidar-like readings for obstacle detection"""
        num_rays = cfg.Training.Observation.LIDAR_RAYS
        max_distance = cfg.Training.Observation.RAY_LENGTH
        
        # Get robot position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        readings = []
        for i in range(num_rays):
            angle = (2 * np.pi * i) / num_rays
            # Calculate ray direction
            ray_dx = np.cos(angle)
            ray_dy = np.sin(angle)
            
            end_pos = [
                robot_pos[0] + ray_dx * max_distance,
                robot_pos[1] + ray_dy * max_distance,
                robot_pos[2]
            ]
            
            # Cast ray
            result = p.rayTest(robot_pos, end_pos)[0]
            distance = result[2] * max_distance
            
            # Normalize reading
            readings.append(distance / max_distance)
        
        return np.array(readings, dtype=np.float32)
    
    def _get_distance_to_nearest_fruit(self):
        """Get distance from gripper to nearest fruit"""
        gripper_pos = self.robot.get_gripper_position()
        _, distance, _ = self.env.get_closest_fruit(gripper_pos)
        return distance
    
    def _calculate_reward(self, action):
        """Calculate reward based on multiple components"""
        reward = 0
        reward_info = {}
        
        # Get current distances
        current_distance = self._get_distance_to_nearest_fruit()
        distance_improvement = self.last_distance - current_distance if self.last_distance else 0
        self.last_distance = current_distance
        
        # Distance reward
        distance_reward = np.exp(-2 * current_distance) * self.reward_weights['distance']
        reward += distance_reward
        reward_info['distance_reward'] = distance_reward
        
        # Progress reward
        progress_reward = distance_improvement * 2.0 * self.reward_weights['progress']
        reward += progress_reward
        reward_info['progress_reward'] = progress_reward
        
        # Action efficiency penalty
        action_penalty = -0.01 * np.sum(np.square(action)) * self.reward_weights['action']
        reward += action_penalty
        reward_info['action_penalty'] = action_penalty
        
        # Contact reward
        gripper_pos = self.robot.get_gripper_position()
        closest_fruit, distance, _ = self.env.get_closest_fruit(gripper_pos)
        
        if distance < cfg.Robot.GRIP_THRESHOLD:
            # Successfully gripped fruit
            contact_reward = 10.0 * self.reward_weights['contact']
            reward += contact_reward
            reward_info['contact_reward'] = contact_reward
            
            # Remove fruit and update counter
            if self.env.remove_fruit(closest_fruit):
                self.fruits_collected += 1
                success_reward = 50.0 * self.reward_weights['success']
                reward += success_reward
                reward_info['success_reward'] = success_reward
        
        return reward, reward_info
    
    def _is_done(self):
        """Check if episode should end"""
        # Episode ends if:
        # 1. All fruits collected
        # 2. Max steps reached
        # 3. Robot collision (future implementation)
        
        if self.fruits_collected == self.total_fruits:
            return True
            
        if self.current_step >= self.max_steps:
            return True
            
        return False
    
    def close(self):
        """Clean up simulation"""
        p.disconnect(self.client)