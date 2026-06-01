@echo off
REM ============================================================
REM  AI Clipper — Windows Launcher
REM  Starts the AI Clipper Flask server + opens browser
REM ============================================================

setlocal

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

title AI Clipper

echo.
echo  ============================================
echo    AI Clipper — Starting...
echo  ============================================
echo.

REM ── Activate venv ──
if exist "%PROJECT_DIR%\venv-win\Scripts\activate.bat" (
    call "%PROJECT_DIR%\venv-win\Scripts\activate.bat"
) else (
    echo [ERROR] Virtual environment not found.
    echo   Run setup.bat first, or create it manually:
    echo   python -m venv venv-win
    echo   venv-win\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM ── Check Ollama (optional) ──
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Ollama is running — AI captions enabled
) else (
    echo [WARN] Ollama not detected — captions will use transcript fallback
    echo        Install Ollama for AI captions: https://ollama.com/download
)
echo.

REM ── Start server ──
echo Starting Flask server...
echo.

cd /d "%PROJECT_DIR%"
python main.py

pause
