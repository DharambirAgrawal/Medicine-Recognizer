@echo off
REM AAMA Application Launcher for Windows
REM This script activates the virtual environment and runs the GUI

echo ========================================
echo  AAMA - AI-Assisted Medical Assistant
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "env\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create a virtual environment first:
    echo   python -m venv env
    echo   .\env\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo Activating virtual environment...
call env\Scripts\activate.bat

echo.
echo Checking dependencies...
python -c "import cv2, numpy, mediapipe, PIL, tkinter" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Missing dependencies detected!
    echo Installing required packages...
    pip install opencv-contrib-python numpy mediapipe pillow
)

echo.
echo Starting AAMA Application...
echo.
python gui_app.py

if errorlevel 1 (
    echo.
    echo ERROR: Application crashed or exited with error!
    pause
)

deactivate
