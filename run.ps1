<#
.Synopsis
One-Click Launch Script for Driver Drowsiness Detection System (PowerShell)
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting Driver Drowsiness Detection System...        " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

# If .venv doesn't exist, create it automatically
if (-not (Test-Path $VenvPython)) {
    Write-Host "[Setup] .venv not found. Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Error] Python was not found on your system PATH." -ForegroundColor Red
        Read-Host "Press Enter to exit..."
        exit 1
    }
    Write-Host "[Setup] Installing requirements..." -ForegroundColor Yellow
    & "$VenvPython" -m pip install --upgrade pip
    & "$VenvPython" -m pip install -r requirements.txt
}

Write-Host "[Launch] Starting detection system..." -ForegroundColor Green
& "$VenvPython" main.py @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[Notice] Application exited with error code $LASTEXITCODE." -ForegroundColor Red
    Read-Host "Press Enter to close this window..."
}
