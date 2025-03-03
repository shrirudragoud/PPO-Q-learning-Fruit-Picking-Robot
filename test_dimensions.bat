@echo off
echo Testing Environment Dimensions
echo ===========================

:: Check Python installation
python --version
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.8 or later.
    pause
    exit /b 1
)

:: Install requirements
echo Installing requirements...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install requirements!
    pause
    exit /b 1
)

:: Run dimension analysis
echo.
echo Running dimension analysis...
python debug_dims.py

echo.
echo Testing complete! Check the output above for any dimension mismatches.
pause