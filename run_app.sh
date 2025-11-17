#!/bin/bash
# Beatbox Audio Style Transfer Application Launcher
# This script ensures the application runs with the correct Python version (3.12)

echo "==================================="
echo "Beatbox Audio Style Transfer"
echo "Professional Edition"
echo "==================================="
echo ""

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 is not installed."
    echo "Please install Python 3.12 first."
    exit 1
fi

# Navigate to the script directory
cd "$(dirname "$0")"

echo "Starting application with Python 3.12..."
echo ""

# Launch the advanced GUI
python3.12 advanced_gui.py

exit 0
