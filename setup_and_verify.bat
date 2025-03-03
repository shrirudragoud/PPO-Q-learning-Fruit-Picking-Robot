@echo off
echo Setting up Orange Harvesting Robot Training Environment
echo ===================================================

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch first (with CUDA support)
echo.
echo Installing PyTorch...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

:: Install other requirements
echo.
echo Installing other requirements...
python -m pip install -r requirements.txt

:: Install project in development mode
echo.
echo Installing project in development mode...
python setup.py develop

:: Create necessary directories
if not exist "models" mkdir models
if not exist "logs" mkdir logs

:: Run verification steps
echo.
echo Running environment verification...

:: Check Python dependencies
python setup_env.py
if %ERRORLEVEL% neq 0 (
    echo Failed to verify environment!
    pause
    exit /b 1
)

:: Test environment dimensions
echo.
echo Testing environment dimensions...
python debug_dims.py
if %ERRORLEVEL% neq 0 (
    echo Failed to verify environment dimensions!
    pause
    exit /b 1
)

:: If everything succeeded
echo.
echo Environment setup and verification complete!
echo You can now run the training using:
echo   train.bat
echo.
echo Or test the environment using:
echo   test_environment.bat
echo.

pause