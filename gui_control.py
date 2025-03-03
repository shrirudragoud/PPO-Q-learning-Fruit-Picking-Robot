#!/usr/bin/env python3

import pygame
import numpy as np
import threading
import queue
import time

class ControlGUI:
    def __init__(self, command_queue):
        # Initialize Pygame
        pygame.init()
        
        # Window setup
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Robot Control Interface")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # Control states
        self.controls_enabled = False
        self.movement = {'forward': 0.0, 'turn': 0.0}
        self.arm = {'shoulder': 0.0, 'elbow': 0.0, 'wrist': 0.0}
        self.gripper = {'position': 0.0, 'spin': 0.0}
        
        # Command queue for simulation
        self.command_queue = command_queue
        
        # Font setup
        self.font = pygame.font.Font(None, 36)
        
        # Active keys
        self.active_keys = set()
        
        # Control sensitivity
        self.movement_speed = 1.0
        self.turn_speed = 1.0
        self.arm_speed = 0.05
        
        print("GUI Control Interface initialized")
    
    def draw_control_status(self):
        """Draw control status and active keys"""
        # Draw background
        self.screen.fill(self.BLACK)
        
        # Draw control status
        status_text = "Controls: ENABLED" if self.controls_enabled else "Controls: DISABLED (Press X)"
        status_color = self.GREEN if self.controls_enabled else self.RED
        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (20, 20))
        
        # Draw movement status
        y_pos = 80
        movement_text = f"Movement: {'Active' if any(self.movement.values()) else 'Stopped'}"
        movement = self.font.render(movement_text, True, self.WHITE)
        self.screen.blit(movement, (20, y_pos))
        
        # Draw active keys
        y_pos += 50
        key_text = "Active Keys: " + ", ".join(self.active_keys)
        keys = self.font.render(key_text, True, self.WHITE)
        self.screen.blit(keys, (20, y_pos))
        
        # Draw control instructions
        y_pos += 100
        instructions = [
            "Controls:",
            "W/S - Forward/Backward",
            "A/D - Turn Left/Right",
            "I/K - Shoulder Up/Down",
            "O/L - Elbow Up/Down",
            "P/; - Wrist Up/Down",
            "Space - Gripper",
            "X - Toggle Controls",
            "R - Reset",
            "Q - Quit"
        ]
        
        for instruction in instructions:
            text = self.font.render(instruction, True, self.GRAY)
            self.screen.blit(text, (20, y_pos))
            y_pos += 30
        
        # Draw movement indicator
        center_x = self.width - 150
        center_y = 150
        radius = 50
        pygame.draw.circle(self.screen, self.GRAY, (center_x, center_y), radius, 2)
        
        if any(self.movement.values()):
            indicator_x = center_x + self.movement['turn'] * radius
            indicator_y = center_y - self.movement['forward'] * radius
            pygame.draw.circle(self.screen, self.GREEN, (int(indicator_x), int(indicator_y)), 5)
        
        # Update display
        pygame.display.flip()
    
    def handle_input(self):
        """Handle keyboard input and update control states"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Key press events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    self.controls_enabled = not self.controls_enabled
                    print(f"Controls {'enabled' if self.controls_enabled else 'disabled'}")
                self.active_keys.add(pygame.key.name(event.key))
            
            # Key release events
            if event.type == pygame.KEYUP:
                self.active_keys.discard(pygame.key.name(event.key))
        
        if not self.controls_enabled:
            self.reset_states()
            return True
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Movement controls
        self.movement['forward'] = 0.0
        if keys[pygame.K_w]:
            self.movement['forward'] = self.movement_speed
        elif keys[pygame.K_s]:
            self.movement['forward'] = -self.movement_speed
            
        self.movement['turn'] = 0.0
        if keys[pygame.K_a]:
            self.movement['turn'] = -self.turn_speed
        elif keys[pygame.K_d]:
            self.movement['turn'] = self.turn_speed
        
        # Arm controls
        if keys[pygame.K_i]:
            self.arm['shoulder'] += self.arm_speed
        elif keys[pygame.K_k]:
            self.arm['shoulder'] -= self.arm_speed
            
        if keys[pygame.K_o]:
            self.arm['elbow'] += self.arm_speed
        elif keys[pygame.K_l]:
            self.arm['elbow'] -= self.arm_speed
            
        if keys[pygame.K_p]:
            self.arm['wrist'] += self.arm_speed
        elif keys[pygame.K_SEMICOLON]:
            self.arm['wrist'] -= self.arm_speed
        
        # Gripper controls
        self.gripper['position'] = 1.0 if keys[pygame.K_SPACE] else 0.0
        self.gripper['spin'] = 1.0 if keys[pygame.K_r] else 0.0
        
        # Send control state to simulation
        control_state = {
            'movement': self.movement.copy(),
            'arm': self.arm.copy(),
            'gripper': self.gripper.copy(),
            'commands': {
                'reset': keys[pygame.K_BACKSPACE],
                'quit': keys[pygame.K_ESCAPE]
            }
        }
        
        self.command_queue.put(control_state)
        return not keys[pygame.K_ESCAPE]
    
    def reset_states(self):
        """Reset all control states"""
        self.movement = {'forward': 0.0, 'turn': 0.0}
        self.arm = {'shoulder': 0.0, 'elbow': 0.0, 'wrist': 0.0}
        self.gripper = {'position': 0.0, 'spin': 0.0}
        self.command_queue.put({
            'movement': self.movement.copy(),
            'arm': self.arm.copy(),
            'gripper': self.gripper.copy(),
            'commands': {'reset': False, 'quit': False}
        })
    
    def run(self):
        """Main GUI loop"""
        running = True
        while running:
            running = self.handle_input()
            self.draw_control_status()
            time.sleep(1/60)  # 60 FPS
        
        pygame.quit()
    
    def start(self):
        """Start GUI in a separate thread"""
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
    
    def stop(self):
        """Stop GUI thread"""
        if hasattr(self, 'thread'):
            self.thread.join()