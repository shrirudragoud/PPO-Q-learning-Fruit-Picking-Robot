#!/usr/bin/env python3

import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
from robot_control import RobotController
from environment import FarmEnvironment

class FarmSimulation:
    def __init__(self, gui=True):
        # Initialize PyBullet
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Configure debug visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        
        # Physics settings
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1./120.)  # Reduce physics update frequency
        
        # Step counter for visual updates
        self.step_counter = 0
        
        # Load environment and robot
        self.env = FarmEnvironment(self.client)
        self.robot_id = self.load_robot()
        self.robot_controller = RobotController(self.robot_id)
        
        # Control states
        self.controls_enabled = False
        self.demo_mode = True
        self.demo_start_time = time.time()
        self.demo_duration = 10.0  # seconds
        
        # Setup UI and camera
        self.setup_ui()
        print("Simulation initialized. Demo mode active. Press X to enable manual controls.")
    
    def load_robot(self):
        """Load the fruit harvesting robot URDF"""
        urdf_path = os.path.join(os.path.dirname(__file__), "fruit_harvesting_robot.urdf")
        robot_id = p.loadURDF(urdf_path, [0, 0, 0.1], useFixedBase=False)
        print(f"Robot loaded with ID: {robot_id}")
        return robot_id
    
    def setup_ui(self):
        """Setup UI elements"""
        self.status_item = p.addUserDebugText(
            text="Demo Mode Active",
            textPosition=[0, 0, 2],  # Raised higher for better visibility
            textColorRGB=[0, 1, 1],
            textSize=1.5
        )
        
        # Add helper text for closest fruit
        self.fruit_status = p.addUserDebugText(
            text="",
            textPosition=[0, 0, 1.5],
            textColorRGB=[1, 1, 0],
            textSize=1.2
        )
    
    def update_status(self, text, color=[1, 1, 1]):
        """Update status display"""
        p.removeUserDebugItem(self.status_item)
        self.status_item = p.addUserDebugText(
            text=text,
            textPosition=[0, 0, 2],
            textColorRGB=color,
            textSize=1.5
        )
        print(text)
    
    def update_fruit_status(self):
        """Update closest fruit information relative to gripper"""
        gripper_pos = self.robot_controller.get_gripper_position()
        closest_fruit, distance, fruit_pos = self.env.get_closest_fruit(gripper_pos)
        
        if closest_fruit is not None and distance < 5.0:  # Only show if fruit is within 5 meters
            # Update status text
            p.removeUserDebugItem(self.fruit_status)
            self.fruit_status = p.addUserDebugText(
                text=f"Nearest fruit: {distance:.2f}m",
                textPosition=[0, 0, 1.5],
                textColorRGB=[1, 1, 0] if distance > 0.5 else [0, 1, 0],
                textSize=1.2
            )
            
            # Draw line from gripper to closest fruit
            if fruit_pos is not None:
                p.addUserDebugLine(
                    gripper_pos,
                    fruit_pos,
                    [0, 1, 0] if distance < 0.5 else [1, 1, 0],
                    lifeTime=0.1
                )
                
                # If in control mode and close enough, attempt to move to fruit
                if self.controls_enabled and distance > 0.2:
                    self.robot_controller.move_to_target(fruit_pos)
    
    def run_demo_animation(self):
        """Run demo animation sequence"""
        t = time.time() - self.demo_start_time
        phase = (t % self.demo_duration) / self.demo_duration
        
        if phase < 0.25:  # Move arms
            shoulder = 0.5 * math.sin(phase * 8 * math.pi)
            elbow = 0.3 * math.sin(phase * 8 * math.pi)
            wrist = 0.2 * math.sin(phase * 8 * math.pi)
            self.robot_controller.control_arm(shoulder=shoulder, elbow=elbow, wrist=wrist)
            self.update_status("Demo: Moving Arms", [0, 1, 1])
            
        elif phase < 0.5:  # Drive forward and turn
            self.robot_controller.move_forward(math.sin(phase * 4 * math.pi))
            self.update_status("Demo: Driving", [0, 1, 1])
            
        elif phase < 0.75:  # Turn in place
            self.robot_controller.turn(math.sin(phase * 4 * math.pi))
            self.update_status("Demo: Turning", [0, 1, 1])
            
        else:  # Gripper demonstration
            gripper_pos = 0.03 * (1 + math.sin(phase * 8 * math.pi)) / 2
            self.robot_controller.control_gripper(position=gripper_pos)
            self.update_status("Demo: Gripper Test", [0, 1, 1])
    
    def handle_keyboard(self):
        """Process keyboard input"""
        keys = p.getKeyboardEvents()
        
        # Toggle controls with X key
        if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
            self.controls_enabled = not self.controls_enabled
            self.demo_mode = not self.controls_enabled
            if self.controls_enabled:
                self.update_status("Controls ENABLED", [0, 1, 0])
            else:
                self.update_status("Demo Mode Active", [0, 1, 1])
                self.demo_start_time = time.time()
            self.robot_controller.stop()
        
        if not self.controls_enabled:
            return True
        
        # Movement controls
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            self.robot_controller.move_forward(1.0)
            self.update_status("Moving Forward", [0, 1, 0])
        elif ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            self.robot_controller.move_forward(-1.0)
            self.update_status("Moving Backward", [0, 1, 0])
        elif ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            self.robot_controller.turn(-1.0)
            self.update_status("Turning Left", [0, 1, 0])
        elif ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            self.robot_controller.turn(1.0)
            self.update_status("Turning Right", [0, 1, 0])
        else:
            self.robot_controller.stop()
        
        # Arm controls
        if ord('i') in keys and keys[ord('i')] & p.KEY_IS_DOWN:
            self.robot_controller.control_arm(shoulder=0.5)
            self.update_status("Shoulder Up", [0, 1, 1])
        elif ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN:
            self.robot_controller.control_arm(shoulder=-0.5)
            self.update_status("Shoulder Down", [0, 1, 1])
            
        if ord('o') in keys and keys[ord('o')] & p.KEY_IS_DOWN:
            self.robot_controller.control_arm(elbow=0.5)
            self.update_status("Elbow Up", [0, 1, 1])
        elif ord('l') in keys and keys[ord('l')] & p.KEY_IS_DOWN:
            self.robot_controller.control_arm(elbow=-0.5)
            self.update_status("Elbow Down", [0, 1, 1])
            
        # Gripper control
        if ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN:
            self.robot_controller.control_gripper(position=0.03)
            self.update_status("Gripper Open", [1, 1, 0])
            
            # Check for fruit picking
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            closest_fruit, distance = self.env.get_closest_fruit(robot_pos)
            if closest_fruit and distance < 0.5:  # If fruit is within gripper range
                self.env.remove_fruit(closest_fruit)
                self.update_status("Fruit Picked!", [0, 1, 0])
        else:
            self.robot_controller.control_gripper(position=0.0)
        
        # Reset
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            self.reset()
            return True
        
        # Quit
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            return False
        
        return True
    
    def reset(self):
        """Reset simulation"""
        p.resetSimulation()
        self.__init__(gui=(self.client == p.GUI))
        self.update_status("Simulation Reset", [1, 1, 0])
    
    def step(self):
        """Step simulation"""
        if self.demo_mode:
            self.run_demo_animation()
        
        # Update physics and visuals at a reduced rate
        p.stepSimulation()
        self.step_counter += 1
        
        # Update visuals every 2 steps
        if self.step_counter % 2 == 0:
            self.update_fruit_status()
        
        self.robot_controller.update()
    
    def close(self):
        """Close simulation"""
        p.disconnect()
        print("Simulation closed.")

def main():
    """Main function"""
    sim = FarmSimulation(gui=True)
    
    print("\nOrange Farm Robot Simulation")
    print("===========================")
    print("Demo mode active. Shows robot capabilities.")
    print("\nControls (Press X to enable):")
    print("W/S - Forward/Backward")
    print("A/D - Turn Left/Right")
    print("I/K - Shoulder Up/Down")
    print("O/L - Elbow Up/Down")
    print("Space - Gripper (hold near fruit to pick)")
    print("R - Reset")
    print("Q - Quit")
    
    try:
        running = True
        while running:
            running = sim.handle_keyboard()
            sim.step()
            time.sleep(1./120.)  # Match physics timestep
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        sim.close()

if __name__ == "__main__":
    main()