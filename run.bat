@echo off
title Oral Cancer Screening App
echo ===================================================
echo Starting Oral Cancer Screening Web Application...
echo ===================================================
cd /d "%~dp0"

:: Stop any old server listening on port 8000
echo Checking for existing server on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    echo Stopping old process on port 8000 - PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: Use the existing venv
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    echo WARNING: Virtual environment not found at venv\. Using system python.
    set "PYTHON_EXE=python"
)

:: Start uvicorn on port 8000 and open http://127.0.0.1:8000/screening via run.py
"%PYTHON_EXE%" run.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
)

echo.
pause
