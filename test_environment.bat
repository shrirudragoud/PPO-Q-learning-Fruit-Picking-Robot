@echo off
echo Testing Orange Harvesting Robot Environment
echo =========================================

:: Install dependencies first if not already installed
echo Installing required packages...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install requirements!
    pause
    exit /b 1
)

echo.
echo Running environment test...
python test_env.py
if %ERRORLEVEL% neq 0 (
    echo Test failed! Check the error messages above.
) else (
    echo.
    echo Environment test completed successfully!
    echo You can now proceed with training using train.bat
)

pause