#!/usr/bin/env python3

import pybullet as p
import numpy as np
import math

class RobotController:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.joint_states = {}
        self.current_positions = {}
        self.target_positions = {}
        self.movement_speed = 0.0
        self.turn_speed = 0.0
        
        # Initialize joint control
        self.initialize_joints()
        print("Robot controller initialized")
    
    def initialize_joints(self):
        """Initialize joint information and control parameters"""
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Found {num_joints} joints")
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            print(f"Initializing joint: {joint_name}")
            
            # Store joint information
            self.joint_states[joint_name] = {
                'index': info[0],
                'type': info[2],
                'lower_limit': info[8] if info[8] != 0 else -math.pi,
                'upper_limit': info[9] if info[9] != 0 else math.pi,
                'max_force': info[10],
                'max_velocity': info[11],
            }
            
            # Initialize position tracking
            self.current_positions[joint_name] = 0.0
            self.target_positions[joint_name] = 0.0
            
            # Reset joint state
            p.resetJointState(self.robot_id, info[0], 0.0)
            
            # Enable motor control
            p.setJointMotorControl2(
                self.robot_id,
                info[0],
                p.VELOCITY_CONTROL,
                force=info[10]
            )
    
    def smooth_control(self, joint_name, target, max_velocity=1.0):
        """Apply smooth control to joint"""
        if joint_name not in self.joint_states:
            return
            
        joint_info = self.joint_states[joint_name]
        current_pos = self.current_positions[joint_name]
        
        # Limit target to joint limits
        target = np.clip(target, joint_info['lower_limit'], joint_info['upper_limit'])
        
        # Calculate smooth velocity
        diff = target - current_pos
        velocity = np.clip(diff * 5.0, -max_velocity, max_velocity)
        
        try:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_info['index'],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=joint_info['max_force']
            )
            self.target_positions[joint_name] = target
            
        except Exception as e:
            print(f"Error controlling joint {joint_name}: {e}")
    
    def move_forward(self, speed=1.0):
        """Move robot forward with smooth acceleration"""
        target_speed = speed * 10.0
        current_speed = self.movement_speed
        # Smooth acceleration
        new_speed = current_speed + np.clip(target_speed - current_speed, -0.5, 0.5)
        self.movement_speed = new_speed
        
        for wheel in ['front_left_wheel_joint', 'front_right_wheel_joint',
                     'rear_left_wheel_joint', 'rear_right_wheel_joint']:
            self.smooth_control(wheel, new_speed, max_velocity=20.0)
    
    def turn(self, direction=1.0):
        """Turn robot with smooth transition"""
        target_turn = direction * 5.0
        current_turn = self.turn_speed
        # Smooth turning
        new_turn = current_turn + np.clip(target_turn - current_turn, -0.3, 0.3)
        self.turn_speed = new_turn
        
        left_speed = -new_turn
        right_speed = new_turn
        
        # Apply to wheels with smooth control
        for wheel, speed in [
            ('front_left_wheel_joint', left_speed),
            ('rear_left_wheel_joint', left_speed),
            ('front_right_wheel_joint', right_speed),
            ('rear_right_wheel_joint', right_speed)
        ]:
            self.smooth_control(wheel, speed, max_velocity=10.0)
    
    def control_arm(self, shoulder=None, elbow=None, wrist=None):
        """Control arm joints with smooth interpolation"""
        if shoulder is not None:
            self.smooth_control('shoulder_joint', shoulder)
        if elbow is not None:
            self.smooth_control('elbow_joint', elbow)
        if wrist is not None:
            self.smooth_control('wrist_pitch_joint', wrist)
    
    def control_gripper(self, position=None, spin=None):
        """Control gripper with smooth motion"""
        if position is not None:
            self.smooth_control('left_finger_joint', position, max_velocity=0.1)
            self.smooth_control('right_finger_joint', position, max_velocity=0.1)
        if spin is not None:
            self.smooth_control('spin_joint', spin)
    
    def stop(self):
        """Smoothly stop all movement"""
        self.movement_speed = 0.0
        self.turn_speed = 0.0
        
        for joint_name in self.joint_states:
            if 'wheel' in joint_name.lower():
                self.smooth_control(joint_name, 0.0)
    
    def update(self):
        """Update joint states and apply smooth control"""
        for joint_name, info in self.joint_states.items():
            state = p.getJointState(self.robot_id, info['index'])
            self.current_positions[joint_name] = state[0]
            
            # Apply smooth control towards target positions
            if abs(self.target_positions[joint_name] - state[0]) > 0.01:
                self.smooth_control(joint_name, self.target_positions[joint_name])
    
    def get_joint_states(self):
        """Get current state of all joints"""
        states = {}
        for joint_name, info in self.joint_states.items():
            state = p.getJointState(self.robot_id, info['index'])
            states[joint_name] = {
                'position': state[0],
                'velocity': state[1],
                'reaction_forces': state[2],
                'applied_motor_torque': state[3]
            }
        return states

    def get_gripper_position(self):
        """Get the current position of the gripper (end effector)"""
        # Get link positions
        gripper_link_state = p.getLinkState(self.robot_id,
                                          self.joint_states['right_finger_joint']['index'])
        return gripper_link_state[0]  # Return world position coordinates
        
    def move_to_target(self, target_pos, max_arm_extension=1.0):
        """Move robot and arm towards target position with smooth approach"""
        # Get current positions
        gripper_pos = self.get_gripper_position()
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Calculate distances and angles
        robot_to_target = np.array(target_pos) - np.array(robot_pos)
        distance_to_target = np.linalg.norm(robot_to_target[:2])  # XY plane distance
        gripper_to_target = np.array(target_pos) - np.array(gripper_pos)
        gripper_distance = np.linalg.norm(gripper_to_target)
        
        # Calculate target angle with smoother transitions
        target_angle = math.atan2(robot_to_target[1], robot_to_target[0])
        euler = p.getEulerFromQuaternion(robot_orn)
        current_angle = euler[2]  # Yaw angle
        
        # Smooth angle difference calculation
        angle_diff = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
        
        # Determine movement phase
        if abs(angle_diff) > 0.1:
            # Turning phase - slower turn speed when closer to target
            turn_speed = min(1.0, abs(angle_diff))
            self.turn(np.sign(angle_diff) * turn_speed)
            return False
            
        # Movement phase
        if distance_to_target > max_arm_extension:
            # Forward movement with speed proportional to distance
            speed = min(1.0, distance_to_target / 5.0)  # Gradually slow down
            self.move_forward(speed)
            return False
            
        # Arm control phase
        height_diff = target_pos[2] - gripper_pos[2]
        dist_diff = distance_to_target - 0.3  # Closer desired distance
        
        # Advanced inverse kinematics for smoother arm movement
        shoulder_angle = math.atan2(height_diff, dist_diff) * 0.8
        elbow_angle = math.atan2(height_diff, dist_diff) * 0.6
        wrist_angle = -shoulder_angle * 0.3  # Compensate for shoulder movement
        
        # Smooth arm control
        self.control_arm(
            shoulder=shoulder_angle,
            elbow=elbow_angle,
            wrist=wrist_angle
        )
        
        # Prepare gripper when close
        if gripper_distance < 0.3:
            self.control_gripper(position=0.03)  # Open gripper
        
        # Return success if gripper is close enough
        return gripper_distance < 0.2

    def get_gripper_position(self):
        """Get the current position of the gripper (end effector)"""
        # Get link positions
        gripper_link_state = p.getLinkState(self.robot_id,
                                          self.joint_states['right_finger_joint']['index'])
        return gripper_link_state[0]  # Return world position coordinates
        
    def move_to_target(self, target_pos, max_arm_extension=1.0):
        """Move robot and arm towards target position"""
        # Get current positions
        gripper_pos = self.get_gripper_position()
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Calculate distances and angles
        robot_to_target = np.array(target_pos) - np.array(robot_pos)
        distance_to_target = np.linalg.norm(robot_to_target[:2])  # XY plane distance
        
        # Calculate target angle
        target_angle = math.atan2(robot_to_target[1], robot_to_target[0])
        current_angle = math.atan2(2 * (robot_orn[3] * robot_orn[2]),
                                 1 - 2 * (robot_orn[2] * robot_orn[2]))
        
        # Determine if we need to turn
        angle_diff = target_angle - current_angle
        if abs(angle_diff) > 0.1:
            self.turn(np.sign(angle_diff))
            return False
            
        # Move forward if we're too far
        if distance_to_target > max_arm_extension:
            self.move_forward(0.5)
            return False
            
        # Once in position, control arm
        height_diff = target_pos[2] - gripper_pos[2]
        dist_diff = distance_to_target - 0.5  # Desired distance from base
        
        # Simple inverse kinematics approximation
        shoulder_angle = math.atan2(height_diff, dist_diff)
        elbow_angle = math.atan2(height_diff, dist_diff) * 0.5
        
        self.control_arm(shoulder=shoulder_angle, elbow=elbow_angle)
        
        # Check if we're close enough to target
        gripper_to_target = np.array(target_pos) - np.array(gripper_pos)
        return np.linalg.norm(gripper_to_target) < 0.2  # Return True if close enough