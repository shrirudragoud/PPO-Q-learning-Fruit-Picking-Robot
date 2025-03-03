@echo off
echo Setting up Python environment for Robot Simulation...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8 or newer.
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install/upgrade pip
python -m pip install --upgrade pip

:: Install requirements
echo Installing required packages...
pip install -r requirements.txt

:: Check if installation was successful
if errorlevel 1 (
    echo Error installing packages! Please check error messages above.
    pause
    exit /b 1
)

echo.
echo Setup complete! Starting simulation...
echo.
echo Controls:
echo - Press X to enable controls
echo - WASD for movement
echo - IOKL for arm control
echo - Space for gripper
echo - R to reset, Q to quit
echo.
echo Press any key to start...
pause >nul

:: Run simulation
python main.py

:: Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Simulation ended with an error! Check messages above.
    pause
)

:: Deactivate virtual environment
call venv\Scripts\deactivate.bat