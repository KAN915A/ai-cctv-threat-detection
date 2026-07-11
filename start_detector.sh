#!/bin/bash

# Weapon Detection System - macOS/Linux Quick Start

echo ""
echo "==================================="
echo "  Weapon Detection System"
echo "==================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.8+ from https://www.python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Check if dependencies are installed
python3 -c "import cv2, ultralytics, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "Installing dependencies (this may take a few minutes)..."
    echo ""
    pip3 install -r requirements_weapon_detector.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

echo ""
echo "Options:"
echo "1. Webcam Detection"
echo "2. Image Detection"
echo "3. View Logs"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Starting webcam detection..."
        echo "Press 'q' to quit, 's' to save frame"
        echo ""
        python3 weapon_detector.py --mode webcam --model small
        ;;
    2)
        read -p "Enter image path: " image_path
        python3 weapon_detector.py --mode image --image "$image_path" --model small
        ;;
    3)
        python3 weapon_detector.py --mode logs
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
