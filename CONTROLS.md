# Robot Control Guide

## Getting Started

1. Launch the simulation:
```bash
python main.py
```

2. Enable Controls:
   - Press 'X' key or use the "Toggle Controls" button in the UI
   - Watch the status indicator turn green when controls are enabled
   - Press 'X' again to disable controls (status turns red)

## Movement Controls

### Base Movement
- **W** - Move forward
- **S** - Move backward
- **A** - Turn left
- **D** - Turn right

### Arm Control
- **I/K** - Shoulder joint up/down
- **O/L** - Elbow joint up/down
- **P/;** - Wrist joint up/down

### Gripper Control
- **Space** - Close/Open gripper
- **R** - Spin gripper (for fruit detachment)

## System Controls
- **X** - Toggle control system
- **T** - Toggle debug visualization
- **R** - Reset simulation
- **Q** - Quit simulation

## Visual Feedback

1. Status Display:
   - Red: Controls disabled
   - Green: Controls enabled
   - Current action displayed
   - Joint positions shown

2. Motion Indicators:
   - Green arrow shows forward motion
   - Red arrow shows backward motion
   - Yellow lines show turning direction
   - Blue lines show distance to nearest fruit

3. Debug Information:
   - Joint angles
   - Robot position
   - Gripper state
   - Movement speed

## Gamepad Support (if available)

1. Movement:
   - Left stick: Robot movement
   - Right stick: Arm control
   - Triggers: Gripper control

2. Buttons:
   - A: Reset simulation
   - B: Quit
   - Bumpers: Camera control

## Safety Features

1. Control Lock:
   - Controls start disabled
   - Must explicitly enable controls
   - Can quickly disable with 'X'

2. Movement Limits:
   - Maximum speed capped
   - Joint limits enforced
   - Collision detection active

3. Emergency Controls:
   - 'X' - Immediate control disable
   - Space - Release gripper
   - 'R' - Full reset

## Common Issues

1. If controls don't respond:
   - Check if controls are enabled (status should be green)
   - Press 'X' to toggle controls
   - Ensure no UI elements are focused

2. If movement seems stuck:
   - Press 'R' to reset simulation
   - Toggle controls off/on with 'X'
   - Check for collisions

3. If arm behaves unexpectedly:
   - Reset to home position
   - Check joint limit indicators
   - Reduce movement speed

## Tips

1. Smooth Operation:
   - Enable controls before moving
   - Watch status display for feedback
   - Use debug visualization for precision

2. Efficient Control:
   - Position robot before using arm
   - Align properly with fruits
   - Use camera controls for better view

3. Practice Sequence:
   1. Enable controls (X)
   2. Position robot (WASD)
   3. Adjust arm (IOKL)
   4. Control gripper (Space)
   5. Monitor feedback
   6. Reset if needed (R)