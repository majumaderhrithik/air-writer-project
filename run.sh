@echo off
title Air Writer - Virtual Writing System
echo.
echo  ✍️   Air Writer — Virtual Writing System
echo  ────────────────────────────────────────────
echo.

:: Check Python
python --version >nul 2>&1
if errorlive 1 (
    echo  ❌  Python not found. Install from https://python.org
    pause
    exit /b 1
)

:: Install/upgrade deps
echo  📦  Checking dependencies...
pip install opencv-python mediapipe numpy --quiet --upgrade

echo.
echo  🎥  Launching Air Writer...
echo  ────────────────────────────────────────────
echo   GESTURES
echo   ☝  Index finger up           ^→ DRAW
echo   ✌  Index + Middle close      ^→ LIFT pen
echo   🖐  Open palm (hold ~0.5s)    ^→ ERASE all
echo.
echo   KEYBOARD
echo   [Q] Quit   [C] Clear   [S] Save PNG
echo   [1-7] Colour   [+/-] Brush width
echo  ────────────────────────────────────────────
echo.

python air_writer.py
pause