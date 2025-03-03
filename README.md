# Orange Farm Robot Simulation

A PyBullet-based simulation of a fruit harvesting robot with integrated controls.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the simulation:

```bash
python main.py
```

policy and value functins

## Controls

### Getting Started

1. Launch simulation
2. Press 'X' to enable controls (status turns green)
3. Use controls listed below
4. Press 'X' again to disable controls

### Control Scheme

```
Movement:          Arm Control:         System:
W - Forward       I/K - Shoulder       X - Toggle Controls
S - Backward      O/L - Elbow         R - Reset
A - Turn Left     P/; - Wrist         Q - Quit
D - Turn Right    Space - Gripper
```

## Visual Feedback

### Status Display

- Status text shows current action
- Green: Controls enabled
- Red: Controls disabled
- Yellow: System messages

### Movement Indicators

- Forward/backward arrows
- Turn direction display
- Arm joint positions
- Gripper state

## Features

1. **Integrated Control System**

   - Direct keyboard input
   - Real-time status updates
   - Visual feedback
   - Safe control toggling

2. **Robot Control**

   - Smooth base movement
   - Precise arm control
   - Gripper manipulation
   - State monitoring

3. **Environment**
   - Physics simulation
   - Tree and fruit interactions
   - Collision detection
   - Dynamic responses

## Troubleshooting

### Controls Not Responding

1. Check status text color:
   - Red = Disabled
   - Green = Enabled
2. Press 'X' to toggle controls
3. Ensure simulation window is focused
4. Try resetting with 'R'

### Common Issues

1. Window Focus

   - Click on simulation window
   - Ensure window is active
   - Check key event messages

2. Movement Issues

   - Check control status
   - Reset simulation if stuck
   - Verify no collisions

3. Performance
   - Close other applications
   - Reduce graphics settings
   - Check system resources

## Development

### Project Structure

```
.
├── main.py              # Main simulation and controls
├── robot_control.py     # Robot movement functions
├── environment.py       # Farm environment
└── requirements.txt     # Dependencies
```

### Adding Features

1. Modify main.py for new controls
2. Update robot_control.py for movements
3. Extend environment.py for interactions
4. Update documentation

## Requirements

- Python 3.6+
- PyBullet
- NumPy

## Performance Notes

- Recommended: 60 FPS minimum
- GPU acceleration supported
- Multi-core CPU recommended

## Safety Features

1. Control Lock

   - Controls start disabled
   - Explicit enabling required
   - Quick disable available
   - Auto-stop on disable

2. Movement Limits

   - Speed caps
   - Turn rate limits
   - Joint constraints
   - Collision prevention

3. Emergency Controls
   - X - Disable all controls
   - R - Full reset
   - Q - Safe quit
