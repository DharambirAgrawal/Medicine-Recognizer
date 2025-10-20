# AAMA Application Launcher for PowerShell
# This script activates the virtual environment and runs the GUI

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AAMA - AI-Assisted Medical Assistant" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "env\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create a virtual environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv env" -ForegroundColor Yellow
    Write-Host "  .\env\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\env\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Green
$deps = python -c "import cv2, numpy, mediapipe, PIL, tkinter" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Missing dependencies detected!" -ForegroundColor Yellow
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install opencv-contrib-python numpy mediapipe pillow
}

Write-Host ""
Write-Host "Starting AAMA Application..." -ForegroundColor Green
Write-Host ""
python gui_app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Application crashed or exited with error!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}

deactivate
