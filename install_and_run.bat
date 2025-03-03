@echo off
echo Installing dependencies for Orange Harvesting Robot Training
echo ========================================================

:: Install Python packages
echo Installing Python packages...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install requirements!
    pause
    exit /b 1
)

:: Install project package
echo Installing project package...
python setup.py develop
if %ERRORLEVEL% neq 0 (
    echo Failed to install project package!
    pause
    exit /b 1
)

:: Create necessary directories
if not exist "models" mkdir models
if not exist "logs" mkdir logs

echo.
echo Setup complete! Starting training...
echo.

:: Run training
python train_harvester.py --phase 1

pause