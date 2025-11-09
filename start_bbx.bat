@echo off
echo ========================================
echo Starting Beatbox Audio Processor
echo ========================================
echo.
echo Launching GUI...
echo.

REM Start the advanced GUI application
python advanced_gui.py

REM If the app exits with an error, pause to see the error message
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Application exited with an error
    echo ========================================
    pause
)
