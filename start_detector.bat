@echo off
REM Weapon Detection System - Windows Quick Start

echo.
echo ===================================
echo  Weapon Detection System
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import cv2, ultralytics, torch" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_weapon_detector.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Options:
echo 1. Webcam Detection (Press 1)
echo 2. Image Detection (Press 2)
echo 3. View Logs (Press 3)
echo 4. Exit (Press 4)
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting webcam detection...
    echo Press 'q' to quit, 's' to save frame
    echo.
    python weapon_detector.py --mode webcam --model small
) else if "%choice%"=="2" (
    set /p image="Enter image path: "
    python weapon_detector.py --mode image --image "%image%" --model small
) else if "%choice%"=="3" (
    python weapon_detector.py --mode logs
) else if "%choice%"=="4" (
    exit /b 0
) else (
    echo Invalid choice
)

pause
