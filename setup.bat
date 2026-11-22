@echo off
REM Smart Farm Risk Management System - setup script (Windows cmd)
REM Creates .venv and installs packages.
REM   setup.bat        - base packages (requirements.txt)
REM   setup.bat gpu    - also LLM/GPU packages (requirements-gpu.txt, NVIDIA GPU required)
REM
REM NOTE: keep this file ASCII-only. cmd.exe parses .bat files with the console
REM code page (cp949 on Korean Windows); UTF-8 Korean text breaks line parsing
REM (lines get merged/split and commands such as "call activate" are skipped).

REM Run relative to the repository root no matter where it is called from
cd /d "%~dp0"

echo.
echo ========================================
echo Smart Farm Risk Management System setup
echo ========================================
echo.

python --version
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3.10+ and add it to PATH.
    goto :fail
)
echo.

echo [1/3] Creating virtual environment...
if exist .venv\Scripts\python.exe (
    echo .venv already exists. Skipping.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] failed to create .venv
        goto :fail
    )
    echo .venv created.
)
echo.

echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

echo Upgrading pip...
python -m pip install --upgrade pip setuptools
echo.

echo [3/3] Installing packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] installing requirements.txt failed
    goto :fail
)
if /i "%~1"=="gpu" (
    echo Installing LLM/GPU packages - torch etc., this takes a while...
    python -m pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [ERROR] installing requirements-gpu.txt failed
        goto :fail
    )
) else (
    echo For the LLM explanation feature run: setup.bat gpu   - NVIDIA GPU required
)
echo.

echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the virtual environment later:
echo   .venv\Scripts\activate.bat
echo.
echo Run the Streamlit app:
echo   streamlit run streamlit_app.py
echo.
pause
exit /b 0

:fail
echo.
echo Setup aborted.
pause
exit /b 1
