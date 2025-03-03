@echo off
echo Orange Harvesting Robot Training Setup
echo ====================================

:: Create directories
if not exist "models" mkdir models
if not exist "logs" mkdir logs

:: Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.8 or later.
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install requirements!
    exit /b 1
)

python setup.py install
if %ERRORLEVEL% neq 0 (
    echo Failed to install project!
    exit /b 1
)

:menu
cls
echo.
echo Orange Harvesting Robot Training
echo ==============================
echo 1. Run Phase 1 Training (Basic Movement)
echo 2. Run Phase 2 Training (Precise Control)
echo 3. Run Phase 3 Training (Full Task)
echo 4. Evaluate Model
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Starting Phase 1 Training...
    python train_harvester.py --phase 1
    pause
    goto menu
)
if "%choice%"=="2" (
    echo Starting Phase 2 Training...
    python train_harvester.py --phase 2
    pause
    goto menu
)
if "%choice%"=="3" (
    echo Starting Phase 3 Training...
    python train_harvester.py --phase 3
    pause
    goto menu
)
if "%choice%"=="4" (
    echo.
    set /p model="Enter model path (e.g., models/phase_3_final.pt): "
    python evaluate_harvester.py %model% --episodes 10
    pause
    goto menu
)
if "%choice%"=="5" (
    echo Exiting...
    exit /b 0
)

echo Invalid choice!
pause
goto menu