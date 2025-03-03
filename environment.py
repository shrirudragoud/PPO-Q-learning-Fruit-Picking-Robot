#!/usr/bin/env python3

import pybullet as p
import numpy as np
import os
import random
from PIL import Image

class FarmEnvironment:
    def __init__(self, client):
        self.client = client
        self.ground_size = 20
        self.tree_positions = []
        self.fruit_ids = []
        self.visual_items = []
        
        # Environment settings
        self.num_trees = 12
        self.num_fruits_per_tree = 8
        
        # Optimization: Cache for nearest fruit calculations
        self.nearest_fruit_cache = None
        self.last_query_pos = None
        self.cache_threshold = 0.1  # Distance threshold for cache invalidation
        
        # Initialize environment
        self.setup_lighting()
        self.create_ground()
        self.create_boundary()
        self.create_trees()
        self.add_decorative_elements()
        
        print("Farm environment initialized")

    def setup_lighting(self):
        """Setup enhanced lighting for better visuals"""
        # Main light
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=15,  # Further back for better overview
            cameraYaw=60,      # Better angle to see fruits
            cameraPitch=-35,   # Slightly higher angle
            cameraTargetPosition=[0, 0, 1]  # Look at height 1 instead of ground
        )

    def create_ground(self):
        """Create textured ground plane"""
        # Load ground mesh
        ground_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.ground_size/2, self.ground_size/2, 0.1]
        )
        
        # Create ground with grass color
        ground_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.ground_size/2, self.ground_size/2, 0.1],
            rgbaColor=[0.2, 0.6, 0.2, 1]
        )
        
        self.ground_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, -0.1]
        )
        
        # Add grid pattern for visual reference
        grid_spacing = 2  # Increased spacing for fewer lines
        for i in range(-self.ground_size//2, self.ground_size//2 + 1, grid_spacing):
            p.addUserDebugLine(
                [i, -self.ground_size/2, 0.01],
                [i, self.ground_size/2, 0.01],
                [0.4, 0.4, 0.4]
            )
            p.addUserDebugLine(
                [-self.ground_size/2, i, 0.01],
                [self.ground_size/2, i, 0.01],
                [0.4, 0.4, 0.4]
            )

    def create_boundary(self):
        """Create farm boundary with fences"""
        fence_height = 1.0
        fence_width = 0.1
        
        # Create fence posts at corners and intervals
        for x in [-self.ground_size/2, self.ground_size/2]:
            for y in [-self.ground_size/2, self.ground_size/2]:
                self.create_fence_post([x, y, fence_height/2])

        # Create connecting beams with increased spacing
        spacing = 4.0
        for x in np.arange(-self.ground_size/2, self.ground_size/2, spacing):
            self.create_fence_beam([x, -self.ground_size/2, fence_height*0.7])
            self.create_fence_beam([x, self.ground_size/2, fence_height*0.7])
            self.create_fence_beam([-self.ground_size/2, x, fence_height*0.7])
            self.create_fence_beam([self.ground_size/2, x, fence_height*0.7])

    def create_fence_post(self, position):
        """Create a fence post"""
        fence_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.5],
            rgbaColor=[0.4, 0.2, 0.0, 1]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision for optimization
            baseVisualShapeIndex=fence_visual,
            basePosition=position
        )

    def create_fence_beam(self, position):
        """Create a horizontal fence beam"""
        beam_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[2.0, 0.05, 0.05],
            rgbaColor=[0.4, 0.2, 0.0, 1]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision for optimization
            baseVisualShapeIndex=beam_visual,
            basePosition=position
        )

    def create_trees(self):
        """Create orange trees in a grid pattern"""
        spacing = 4.0
        offset = -self.ground_size/2 + spacing
        
        for row in range(3):
            for col in range(4):
                x = offset + col * spacing
                y = offset + row * spacing
                self.create_single_tree([x, y, 0])

    def create_single_tree(self, position):
        """Create a detailed orange tree using compound shapes"""
        trunk_height = 1.0  # Reduced from 2.0
        trunk_radius = 0.2
        
        # Create trunk collision that matches visual size
        tree_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=trunk_radius,
            height=trunk_height
        )
        
        # Create visual shapes separately for better aesthetics
        visual_shapes = []
        visual_poses = []

        # Trunk visual
        trunk_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=trunk_radius,
            length=trunk_height,
            rgbaColor=[0.4, 0.2, 0.0, 1]
        )
        visual_shapes.append(trunk_visual)
        visual_poses.append([0, 0, trunk_height/2, 0, 0, 0, 1])
        
        # Create denser, more natural-looking foliage
        foliage_positions = [
            # Center cluster
            [0, 0, 0.6],
            # Mid-level clusters
            [0.3, 0.3, 0.4],
            [-0.3, 0.3, 0.4],
            [0.3, -0.3, 0.4],
            [-0.3, -0.3, 0.4],
            # Top clusters
            [0, 0, 0.8],
            [0.2, 0.2, 0.7],
            [-0.2, 0.2, 0.7],
            [0.2, -0.2, 0.7],
            [-0.2, -0.2, 0.7]
        ]
        
        for pos in foliage_positions:
            radius = random.uniform(0.4, 0.6)  # Smaller, denser clusters
            foliage_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[0.1, 0.35, 0.1, 0.9]  # Slightly more opaque
            )
            visual_shapes.append(foliage_visual)
            visual_poses.append([
                pos[0], pos[1],
                trunk_height + radius + pos[2],
                0, 0, 0, 1
            ])
        
        # Create base tree with collision
        tree_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tree_collision,
            baseVisualShapeIndex=-1,
            basePosition=position
        )

        # Add visual elements as separate bodies
        for shape, pose in zip(visual_shapes, visual_poses):
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=shape,
                basePosition=[
                    position[0] + pose[0],
                    position[1] + pose[1],
                    position[2] + pose[2]
                ]
            )
        
        # Add fruits
        self.add_fruits_to_tree(position, trunk_height)
        self.tree_positions.append(position)

    def add_fruits_to_tree(self, tree_position, trunk_height):
        """Add oranges to a tree with stem joints"""
        tree_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, trunk_height/2]),
            basePosition=tree_position
        )
        
        for _ in range(self.num_fruits_per_tree):
            # Create fruit with stem
            fruit_radius = 0.08  # Slightly smaller oranges
            stem_length = 0.1
            
            # Random position within tree canopy
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0.3, 1.0)  # Closer to trunk
            height = random.uniform(1.0, 1.8)  # Lower heights due to shorter trunk
            
            # Fruit position
            x = tree_position[0] + r * np.cos(angle)
            y = tree_position[1] + r * np.sin(angle)
            z = trunk_height + height
            
            # Create fruit with stem as single body
            fruit_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=fruit_radius
            )
            
            fruit_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=fruit_radius,
                rgbaColor=[1.0, 0.5, 0.0, 1]
            )
            
            # Create fruit
            fruit_id = p.createMultiBody(
                baseMass=0.05,  # Lighter mass
                baseCollisionShapeIndex=fruit_shape,
                baseVisualShapeIndex=fruit_visual,
                basePosition=[x, y, z]
            )
            
            # Create joint between tree and fruit
            p.createConstraint(
                tree_id, -1, fruit_id, -1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[r * np.cos(angle), r * np.sin(angle), height],
                childFramePosition=[0, 0, 0],
                physicsClientId=self.client
            )
            
            # Set dynamics properties
            p.changeDynamics(
                fruit_id,
                -1,
                mass=0.05,
                lateralFriction=0.8,
                spinningFriction=0.1,
                rollingFriction=0.1,
                restitution=0.3,
                linearDamping=0.9,  # Add damping to reduce swinging
                angularDamping=0.9
            )
            
            self.fruit_ids.append(fruit_id)

    def add_decorative_elements(self):
        """Add decorative elements to the farm"""
        # Add fewer decorative elements with only visual shapes (no collision)
        for _ in range(5):  # Reduced from 10
            size = random.uniform(0.2, 0.5)
            x = random.uniform(-self.ground_size/2, self.ground_size/2)
            y = random.uniform(-self.ground_size/2, self.ground_size/2)
            
            rock_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size,
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,  # No collision shape
                baseVisualShapeIndex=rock_visual,
                basePosition=[x, y, size/2]
            )
        
        # Add fewer bushes with only visual shapes
        for _ in range(8):  # Reduced from 15
            size = random.uniform(0.3, 0.6)
            x = random.uniform(-self.ground_size/2, self.ground_size/2)
            y = random.uniform(-self.ground_size/2, self.ground_size/2)
            
            bush_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size,
                rgbaColor=[0.1, 0.3, 0.1, 1]
            )
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,  # No collision shape
                baseVisualShapeIndex=bush_visual,
                basePosition=[x, y, size/2]
            )

    def get_closest_fruit(self, position):
        """Find the closest fruit to a given position using caching"""
        if not self.fruit_ids:
            return None, float('inf'), None

        # Check cache validity
        if (self.nearest_fruit_cache is not None and
            self.last_query_pos is not None and
            np.linalg.norm(np.array(position) - np.array(self.last_query_pos)) < self.cache_threshold):
            # Use cached result if robot hasn't moved much
            fruit_id = self.nearest_fruit_cache
            if fruit_id in self.fruit_ids:  # Verify fruit still exists
                fruit_pos, _ = p.getBasePositionAndOrientation(fruit_id)
                return (fruit_id,
                        np.linalg.norm(np.array(position) - np.array(fruit_pos)),
                        fruit_pos)

        # Cache miss or invalid cache - find nearest fruit
        closest_fruit = None
        min_distance = float('inf')
        closest_pos = None
        fruit_positions = {fruit_id: p.getBasePositionAndOrientation(fruit_id)[0]
                         for fruit_id in self.fruit_ids}
        
        for fruit_id, fruit_pos in fruit_positions.items():
            distance = np.linalg.norm(np.array(position) - np.array(fruit_pos))
            if distance < min_distance:
                min_distance = distance
                closest_fruit = fruit_id
                closest_pos = fruit_pos

        # Update cache
        self.nearest_fruit_cache = closest_fruit
        self.last_query_pos = position
        
        return closest_fruit, min_distance, closest_pos

    def remove_fruit(self, fruit_id):
        """Remove a fruit from the simulation"""
        if fruit_id in self.fruit_ids:
            p.removeBody(fruit_id)
            self.fruit_ids.remove(fruit_id)
            if fruit_id == self.nearest_fruit_cache:
                self.nearest_fruit_cache = None  # Invalidate cache
            return True
        return False

    def reset(self):
        """Reset the environment"""
        # Remove existing fruits
        for fruit_id in self.fruit_ids:
            p.removeBody(fruit_id)
        self.fruit_ids = []
        
        # Reset cache
        self.nearest_fruit_cache = None
        self.last_query_pos = None
        
        # Add new fruits to trees
        for position in self.tree_positions:
            self.add_fruits_to_tree(position, 2.0)  # 2.0 is trunk height