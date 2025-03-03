#!/usr/bin/env python3

import pygame
import numpy as np
from config import SimConfig as cfg

class UserController:
    def __init__(self, use_gamepad=False):
        """Initialize user control interface with both keyboard and gamepad support"""
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Create a small hidden window to capture keyboard events
        self.screen = pygame.display.set_mode((1, 1), pygame.HIDDEN)
        
        self.use_gamepad = use_gamepad and pygame.joystick.get_count() > 0
        if self.use_gamepad:
            self.gamepad = pygame.joystick.Joystick(0)
            self.gamepad.init()
            print(f"Gamepad detected: {self.gamepad.get_name()}")
        
        # Control states
        self.reset_states()
        
        # Control sensitivity
        self.movement_speed = 1.0
        self.turn_speed = 1.0
        self.arm_speed = 0.05
        
        print("User controller initialized.")
    
    def reset_states(self):
        """Reset all control states"""
        self.movement = {'forward': 0.0, 'turn': 0.0}
        self.arm = {'shoulder': 0.0, 'elbow': 0.0, 'wrist': 0.0}
        self.gripper = {'position': 0.0, 'spin': 0.0}
        self.commands = {'reset': False, 'quit': False}
    
    def update(self):
        """Update control states based on user input"""
        # Reset commands but keep other states
        self.commands = {'reset': False, 'quit': False}
        
        # Handle all queued events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.commands['quit'] = True
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Movement controls
        self.movement['forward'] = 0.0
        self.movement['turn'] = 0.0
        
        if keys[pygame.K_w]:
            self.movement['forward'] = self.movement_speed
            print("Forward pressed")
        elif keys[pygame.K_s]:
            self.movement['forward'] = -self.movement_speed
            print("Backward pressed")
            
        if keys[pygame.K_a]:
            self.movement['turn'] = -self.turn_speed
            print("Left pressed")
        elif keys[pygame.K_d]:
            self.movement['turn'] = self.turn_speed
            print("Right pressed")
        
        # Arm controls
        if keys[pygame.K_i]:
            self.arm['shoulder'] += self.arm_speed
            print("Shoulder up")
        elif keys[pygame.K_k]:
            self.arm['shoulder'] -= self.arm_speed
            print("Shoulder down")
            
        if keys[pygame.K_o]:
            self.arm['elbow'] += self.arm_speed
            print("Elbow up")
        elif keys[pygame.K_l]:
            self.arm['elbow'] -= self.arm_speed
            print("Elbow down")
            
        if keys[pygame.K_p]:
            self.arm['wrist'] += self.arm_speed
            print("Wrist up")
        elif keys[pygame.K_SEMICOLON]:
            self.arm['wrist'] -= self.arm_speed
            print("Wrist down")
        
        # Gripper controls
        self.gripper['position'] = 1.0 if keys[pygame.K_SPACE] else 0.0
        self.gripper['spin'] = 1.0 if keys[pygame.K_r] else 0.0
        
        # System controls
        if keys[pygame.K_BACKSPACE]:
            self.commands['reset'] = True
        if keys[pygame.K_ESCAPE]:
            self.commands['quit'] = True
        
        # Clamp arm values
        self.arm['shoulder'] = np.clip(self.arm['shoulder'], -1.57, 1.57)
        self.arm['elbow'] = np.clip(self.arm['elbow'], -2.0, 2.0)
        self.arm['wrist'] = np.clip(self.arm['wrist'], -1.57, 1.57)
        
        # Get gamepad input if enabled
        if self.use_gamepad:
            self._update_gamepad()
        
        # Get current state
        state = self._get_control_state()
        
        # Debug output
        if any(state['movement'].values()) or any(state['arm'].values()) or any(state['gripper'].values()):
            print(f"Control state: {state}")
        
        return state
    
    def _update_gamepad(self):
        """Update controls from gamepad input"""
        # Left stick for movement
        self.movement['forward'] = -self.gamepad.get_axis(1) * self.movement_speed
        self.movement['turn'] = -self.gamepad.get_axis(0) * self.turn_speed
        
        # Right stick for arm control
        self.arm['shoulder'] += self.gamepad.get_axis(3) * self.arm_speed
        self.arm['elbow'] += self.gamepad.get_axis(4) * self.arm_speed
        
        # Triggers for gripper
        self.gripper['position'] = (self.gamepad.get_axis(2) + 1) / 2
        self.gripper['spin'] = self.gamepad.get_axis(5)
        
        # Buttons
        if self.gamepad.get_button(0):  # A button
            self.commands['reset'] = True
        if self.gamepad.get_button(1):  # B button
            self.commands['quit'] = True
    
    def _get_control_state(self):
        """Get current control state"""
        return {
            'movement': self.movement.copy(),
            'arm': self.arm.copy(),
            'gripper': self.gripper.copy(),
            'commands': self.commands.copy()
        }
    
    def close(self):
        """Clean up pygame resources"""
        if self.use_gamepad:
            self.gamepad.quit()
        pygame.quit()
        print("User controller closed.")