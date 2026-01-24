@echo off
echo ============================================================
echo ML Learning App - Starting...
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Flask not found. Installing Flask...
    python -m pip install flask numpy --quiet
    echo.
)

echo Starting ML Learning App...
echo.
echo The app will open in your browser automatically.
echo If it doesn't, go to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server.
echo ============================================================
echo.

REM Run the app
python ml_learning_app.py

pause
