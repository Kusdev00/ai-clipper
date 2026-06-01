@echo off
REM ============================================================
REM  AI Clipper — Windows Setup Script
REM  Installs everything: Python deps, Ollama, model, FFmpeg
REM  Run this once, then run ai-clipper.bat to start
REM ============================================================

setlocal EnableDelayedExpansion

title AI Clipper Setup
color 0A

echo.
echo  ============================================
echo    AI Clipper — Windows Setup
echo  ============================================
echo.

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "MODEL=llama3.1:8b"
set "OLLAMA_URL=http://localhost:11434"

REM ── 1. Check Python ──
echo [1/5] Checking Python...

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    echo   OK — %PY_VER%
) else (
    echo   NOT FOUND
    echo.
    echo   Python 3.11+ is required. Download it:
    echo   https://www.python.org/downloads/
    echo.
    echo   IMPORTANT: Check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

REM ── 2. Create venv ──
echo.
echo [2/5] Setting up virtual environment...

if exist "%PROJECT_DIR%\venv-win" (
    echo   OK — venv-win already exists
) else (
    echo   Creating venv-win...
    python -m venv "%PROJECT_DIR%\venv-win"
    if %ERRORLEVEL% NEQ 0 (
        echo   FAILED — could not create venv
        pause
        exit /b 1
    )
    echo   OK — venv created
)

REM ── 3. Install Python dependencies ──
echo.
echo [3/5] Installing Python dependencies...

call "%PROJECT_DIR%\venv-win\Scripts\activate.bat"

python -m pip install --upgrade pip -q
pip install -r "%PROJECT_DIR%\requirements.txt" -q

if %ERRORLEVEL% NEQ 0 (
    echo   FAILED — pip install error
   echo   Try running manually:
   echo   venv-win\Scripts\activate
   echo   pip install -r requirements.txt
    pause
    exit /b 1
)
echo   OK — dependencies installed

REM ── 4. Install FFmpeg (via winget) ──
echo.
echo [4/5] Checking FFmpeg...

ffmpeg -version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   OK — FFmpeg already installed
) else (
    echo   FFmpeg not found. Installing via winget...
    
    winget --version >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo   WARNING: winget not available. Install FFmpeg manually:
        echo   https://www.gyan.dev/ffmpeg/builds/
        echo   Add ffmpeg.exe to your system PATH.
    ) else (
        winget install ffmpeg --accept-package-agreements --accept-source-agreements
        if %ERRORLEVEL% EQU 0 (
            echo   OK — FFmpeg installed
            echo   NOTE: You may need to restart your terminal for PATH to update.
        ) else (
            echo   WARNING: winget install failed. Install manually:
            echo   https://www.gyan.dev/ffmpeg/builds/
        )
    )
)

REM ── 5. Install Ollama ──
echo.
echo [5/5] Checking Ollama...

ollama --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%v in ('ollama --version 2^>^&1') do set "OLLAMA_VER=%%v"
    echo   OK — Ollama already installed (!OLLAMA_VER!)
) else (
    echo   Ollama not found. Downloading installer...
    
    set "OLLAMA_INSTALLER=%TEMP%\OllamaSetup.exe"
    
    curl -L -o "%OLLAMA_INSTALLER%" https://ollama.com/download/OllamaSetup.exe 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo   FAILED — could not download Ollama installer.
        echo   Download manually from: https://ollama.com/download
        pause
        exit /b 1
    )
    
    echo   Running Ollama installer...
    echo   (Follow the on-screen prompts)
    start /wait "" "%OLLAMA_INSTALLER%"
    del "%OLLAMA_INSTALLER%" 2>nul
    
    ollama --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo   OK — Ollama installed
    ) else (
        echo   Ollama installed but not in PATH yet.
        echo   Restart your terminal and run this script again.
        echo   Or add Ollama to PATH manually.
        pause
        exit /b 0
    )
)

REM ── 6. Start Ollama server ──
echo.
echo Starting Ollama server...

tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo   OK — Ollama server already running
) else (
    start "" ollama serve
    echo   Waiting for Ollama to start...
    
    set "RETRIES=0"
    :OLLAMA_WAIT
    timeout /t 3 /nobreak >nul
    curl -s "%OLLAMA_URL%/api/tags" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        set /a "RETRIES+=1"
        if !RETRIES! LSS 10 (
            echo   Waiting... (!RETRIES!x)
            goto OLLAMA_WAIT
        ) else (
            echo   WARNING: Ollama didn't respond within 30s.
            echo   You may need to start it manually: ollama serve
            goto PULL_MODEL
        )
    )
    echo   OK — Ollama server running
)

:PULL_MODEL
REM ── 7. Pull model ──
echo.
echo Checking model: %MODEL%

ollama list 2>nul | findstr /I "%MODEL%" >nul
if %ERRORLEVEL% EQU 0 (
    echo   OK — Model %MODEL% already pulled
) else (
    echo   Pulling model: %MODEL%...
    echo   (This may take several minutes depending on your internet)
    ollama pull %MODEL%
    if %ERRORLEVEL% NEQ 0 (
        echo   WARNING: Model pull failed. Pull manually:
        echo   ollama pull %MODEL%
    ) else (
        echo   OK — Model pulled
    )
)

REM ── 8. Test ──
echo.
echo Testing Ollama...

curl -s -X POST "%OLLAMA_URL%/api/generate" ^
    -H "Content-Type: application/json" ^
    -d "{\"model\":\"%MODEL%\",\"prompt\":\"Reply with OK.\",\"stream\":false,\"options\":{\"num_predict\":5}}" ^
    --max-time 30 >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo   OK — Ollama is responding
) else (
    echo   NOTE: Ollama responded but model may still be loading.
    echo   Try again in 30 seconds.
)

REM ── Done ──
echo.
echo  ============================================
echo    Setup Complete!
echo  ============================================
echo.
echo  To start AI Clipper:
echo    1. Run: ai-clipper.bat
echo    2. Open: http://127.0.0.1:7878
echo.
pause
