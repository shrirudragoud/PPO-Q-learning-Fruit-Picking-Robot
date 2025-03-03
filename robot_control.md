# Fruit Harvesting Robot Control Guide

## 1. Mobile Platform Control

### Wheel Control Variables
- **front_left_wheel_joint**: Velocity control [-5.0, 5.0] rad/s
- **front_right_wheel_joint**: Velocity control [-5.0, 5.0] rad/s
- **rear_left_wheel_joint**: Velocity control [-5.0, 5.0] rad/s
- **rear_right_wheel_joint**: Velocity control [-5.0, 5.0] rad/s

### Platform Movement Functions
- Forward/Backward: Synchronize all wheels at same velocity
- Turn Left/Right: Differential velocity between left and right wheels
- Spot Turn: Opposite velocities for left and right sides
- Strafe: Not supported (non-mecanum wheels)

## 2. Robotic Arm Control

### Joint Variables
- **arm_to_platform**: Position control [-3.14159, 3.14159] rad
  - Base rotation for arm positioning
  - Effort limit: 100 Nm
  
- **shoulder_joint**: Position control [-1.57079, 1.57079] rad
  - Vertical lift control
  - Effort limit: 100 Nm

- **upper_arm_joint**: Position control [-1.57079, 1.57079] rad
  - Forward reach control
  - Effort limit: 100 Nm

- **elbow_joint**: Position control [-2.0, 2.0] rad
  - Fine positioning
  - Effort limit: 100 Nm

- **wrist_pitch_joint**: Position control [-1.57079, 1.57079] rad
  - End effector angle control
  - Effort limit: 50 Nm

- **wrist_roll_joint**: Position control (continuous)
  - End effector rotation
  - Effort limit: 50 Nm

### Arm Functions
1. Home Position: All joints to 0 position
2. Pre-grasp: Position arm for approaching fruit
3. Reach: Extend arm towards target
4. Retract: Return to safe position after picking

## 3. End Effector Control

### Gripper Variables
- **spin_joint**: Velocity control [-5.0, 5.0] rad/s
  - Continuous rotation for fruit detachment
  - Dynamic properties: damping=0.1, friction=0.1

- **left_finger_joint**: Position control [0, 0.03] m
  - Linear motion for grasping
  - Dynamic properties: damping=0.5, friction=0.5
  
- **right_finger_joint**: Position control [0, 0.03] m
  - Linear motion for grasping
  - Dynamic properties: damping=0.5, friction=0.5

### Gripper Functions
1. Open Gripper: Both fingers to max position
2. Close Gripper: Both fingers to target width
3. Twist Operation: Controlled rotation after grasp
4. Quick Release: Rapid opening for fruit deposit

## 4. Control Interfaces

### Hardware Interfaces
1. **VelocityJointInterface**:
   - All wheel joints
   - Spin joint for fruit detachment

2. **PositionJointInterface**:
   - All arm joints
   - Gripper finger joints

### Parameters for Fruit Handling
- **Soft Contact Properties**:
  - Friction coefficients (mu1, mu2): 1.2
  - Contact stiffness (kp): 1000000.0
  - Contact damping (kd): 1.0
  - Maximum velocity: 0.1 m/s
  - Minimum contact depth: 0.001 m

### Control Modes
1. **Manual Control**: Direct joint control
2. **Semi-Autonomous**: Preset movement patterns
3. **Fully Autonomous**: Vision-guided operation

## 5. Safety Features

### Joint Limits
- Software position limits on all joints
- Velocity limits for safe operation
- Effort limits to prevent damage

### Emergency Stops
1. Immediate wheel stop
2. Arm freeze in current position
3. Gripper release function

## 6. ROS Control Integration

### Namespace
- /fruit_harvesting_robot

### Control Plugins
- gazebo_ros_control
- DefaultRobotHWSim

### Available Topics
1. Joint States
2. Joint Commands
3. Controller States
4. Robot Status
