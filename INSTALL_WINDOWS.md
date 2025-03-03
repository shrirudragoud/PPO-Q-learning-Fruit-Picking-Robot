# Windows Installation Guide

## Using PyBullet Implementation

Since Gazebo and ROS are primarily designed for Linux, we'll use the PyBullet implementation for Windows users.

### Prerequisites

1. Install Python (3.8 or newer):
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. Install Visual Studio Build Tools:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

3. Install Git:
   - Download from: https://git-scm.com/download/win

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation

1. With PyBullet:
```bash
python main.py
```

## Controls

1. Enable controls:
   - Press 'X' to toggle controls
   - Watch for green "Controls Enabled" message

2. Movement:
   - W/S - Forward/Backward
   - A/D - Turn Left/Right

3. Arm Control:
   - I/K - Shoulder Up/Down
   - O/L - Elbow Up/Down
   - P/; - Wrist Up/Down

4. Gripper:
   - Space - Open/Close

5. System:
   - R - Reset
   - Q - Quit

## Troubleshooting

### Missing DLLs
If you see "Missing DLL" errors:
1. Reinstall Visual C++ Redistributable
2. Ensure Python and pip are up to date
3. Reinstall PyBullet: `pip install --upgrade pybullet`

### Import Errors
If you see import errors:
1. Verify virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python PATH in system environment variables

### Performance Issues
1. Update graphics drivers
2. Close unnecessary applications
3. Reduce simulation complexity in config.py

## Development Setup

### VS Code Configuration

1. Install Extensions:
   - Python
   - C/C++
   - CMake Tools
   - Python Test Explorer

2. Configure VS Code settings:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.analysis.extraPaths": ["${workspaceFolder}"],
    "editor.formatOnSave": true
}
```

### Debug Configuration

Add this to `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Main",
            "type": "python",
            "request": "launch",
            "program": "main.py",
            "console": "integratedTerminal"
        }
    ]
}
```

## Gazebo Note

If you need Gazebo functionality:
1. Consider using WSL2 (Windows Subsystem for Linux)
2. Install Ubuntu 20.04 in WSL2
3. Follow ROS/Gazebo installation for Ubuntu
4. Use the gazebo_setup folder in Linux environment

## Project Structure

```
project/
├── main.py              # Main simulation (PyBullet)
├── robot_control.py     # Robot control logic
├── environment.py       # Environment setup
├── config.py           # Configuration
└── requirements.txt    # Python dependencies
```

## Next Steps

1. Start with PyBullet implementation
2. Test robot control and visualization
3. Implement custom features
4. If needed, set up WSL2 for Gazebo development