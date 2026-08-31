@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer first.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 exit /b 1

call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
call ".venv\Scripts\python.exe" -m pip install -r "backend\requirements.txt"
if errorlevel 1 exit /b 1

if not exist "backend\.env" copy "backend\.env.example" "backend\.env" >nul
echo AquaMind source environment is ready.
pause
