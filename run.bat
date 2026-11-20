@echo off
:: ========================================================
:: One-Click Launch Script for Driver Drowsiness Detection (CMD)
:: ========================================================

cd /d "%~dp0"
echo ========================================================
echo  Starting Driver Drowsiness Detection System...
echo ========================================================

set "VENV_PYTHON=.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [Setup] .venv not found. Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [Error] Python not found in system PATH.
        pause
        exit /b 1
    )
    echo [Setup] Installing requirements...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    "%VENV_PYTHON%" -m pip install -r requirements.txt
)

echo [Launch] Launching main.py...
"%VENV_PYTHON%" main.py %*

if errorlevel 1 (
    echo.
    echo [Notice] Application exited with an error.
    pause
)
