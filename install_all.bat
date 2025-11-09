@echo off
echo ========================================
echo Beatbox Audio Processing - Installation
echo ========================================
echo.

REM Install Python requirements
echo [1/4] Installing Python requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo Requirements installed successfully!
echo.

REM Create necessary directories
echo [2/4] Creating directories...
if not exist "presets" mkdir presets
if not exist "recordings" mkdir recordings
if not exist "logs" mkdir logs
echo Directories created!
echo.

REM Enable diagnostics and loudness matching in config.py
echo [3/4] Configuring diagnostics and loudness matching...
powershell -Command "(gc config.py) -replace 'DIAGNOSTIC_MODE_ENABLED = False', 'DIAGNOSTIC_MODE_ENABLED = True' | Out-File -encoding ASCII config.py"
powershell -Command "(gc config.py) -replace 'LOUDNESS_MATCHING_ENABLED = False', 'LOUDNESS_MATCHING_ENABLED = True' | Out-File -encoding ASCII config.py"
echo Configuration updated!
echo.

REM Verify installation
echo [4/4] Verifying installation...
python -c "import librosa, numpy, scipy, pedalboard, sounddevice; print('All core packages imported successfully!')"
if %errorlevel% neq 0 (
    echo WARNING: Some packages may not have installed correctly
) else (
    echo Verification complete!
)
echo.

echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Quick Start:
echo 1. Run 'start_bbx.bat' to launch the application
echo 2. Analyze reference audio: python advanced_analyzer.py path/to/audio.wav
echo 3. Load presets and start processing!
echo.
echo For diagnostics, check the logs/ directory
echo.
pause
