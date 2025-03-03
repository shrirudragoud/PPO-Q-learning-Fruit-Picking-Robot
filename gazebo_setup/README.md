# Fruit Harvesting Robot - Gazebo Simulation

A ROS/Gazebo simulation of a fruit harvesting robot with realistic physics and control.

## Prerequisites

1. ROS (tested on ROS Noetic)
2. Gazebo
3. Required ROS packages:
```bash
sudo apt-get install ros-noetic-ros-control ros-noetic-ros-controllers
sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
```

## Building

1. Create a catkin workspace (if you don't have one):
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
```

2. Clone this repository into the src directory:
```bash
cp -r /path/to/gazebo_setup ~/catkin_ws/src/fruit_harvesting_robot
```

3. Build the workspace:
```bash
cd ~/catkin_ws
catkin_make
```

4. Source the workspace:
```bash
source devel/setup.bash
```

## Running the Simulation

1. Launch the main simulation:
```bash
roslaunch fruit_harvesting_robot fruit_harvesting_robot.launch
```

2. Optional: Launch with RViz visualization:
```bash
roslaunch fruit_harvesting_robot fruit_harvesting_robot.launch rviz:=true
```

## Controls

### Robot Movement
- **W/S** - Forward/Backward
- **A/D** - Turn Left/Right

### Arm Control
- **I/K** - Shoulder Up/Down
- **O/L** - Elbow Up/Down
- **P/;** - Wrist Up/Down

### Gripper Control
- **Space** - Open/Close Gripper

### System Controls
- **R** - Reset Robot Position
- **Q** - Quit

## Project Structure

```
fruit_harvesting_robot/
├── config/
│   └── controllers.yaml     # Controller configurations
├── launch/
│   └── fruit_harvesting_robot.launch
├── src/
│   ├── robot_control_plugin.cpp
│   └── keyboard_control_node.cpp
├── worlds/
│   └── orange_farm.world
├── CMakeLists.txt
└── package.xml
```

## Features

1. **Physics Simulation**
   - Realistic robot dynamics
   - Collision detection
   - Terrain interaction
   - Fruit physics

2. **Control System**
   - ROS Control integration
   - Velocity-based movement
   - Position control for arm
   - Force-based gripper

3. **Environment**
   - Procedural orange trees
   - Harvestable fruits
   - Dynamic lighting
   - Ground physics

## Customization

### Modifying Controllers
Edit `config/controllers.yaml` to adjust:
- PID parameters
- Control types
- Joint properties

### World Configuration
Edit `worlds/orange_farm.world` to change:
- Tree placement
- Lighting conditions
- Physics properties
- Environment features

### Robot Properties
Modify the URDF file to adjust:
- Joint limits
- Physical properties
- Sensor configurations
- Visual appearance

## Troubleshooting

1. **Controller Issues**
   - Check controller configuration in yaml file
   - Verify joint names match URDF
   - Monitor controller states with:
     ```bash
     rostopic echo /fruit_harvesting_robot/controller_state
     ```

2. **Robot Movement**
   - Ensure keyboard control node is running
   - Check topic connections:
     ```bash
     rostopic echo /cmd_vel
     ```
   - Verify joint states:
     ```bash
     rostopic echo /joint_states
     ```

3. **Simulation Performance**
   - Reduce physics update rate in launch file
   - Lower visual quality in Gazebo
   - Check CPU/GPU usage

## Development

### Adding Features
1. Modify robot_control_plugin.cpp for new behaviors
2. Update keyboard_control_node.cpp for new controls
3. Adjust controllers.yaml for new joints
4. Update launch file for new components

### Testing
1. Use RViz for visualization
2. Monitor joint states and transforms
3. Check controller performance
4. Verify physics interactions