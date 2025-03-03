@echo off
echo Testing PPO Implementation
echo ========================

:: Check Python installation
python --version
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.8 or later.
    pause
    exit /b 1
)

:: Install requirements if needed
echo.
echo Checking dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install requirements!
    pause
    exit /b 1
)

:: Run dimension analysis first
echo.
echo Running dimension analysis...
python analyze_dimensions.py
if %ERRORLEVEL% neq 0 (
    echo Dimension analysis failed!
    pause
    exit /b 1
)

:: Run PPO tests
echo.
echo Running PPO implementation tests...
python test_ppo.py
if %ERRORLEVEL% neq 0 (
    echo PPO tests failed!
    pause
    exit /b 1
)

echo.
echo All tests completed successfully!
echo You can now proceed with training using:
echo   train.bat

pause