@echo off
echo ====================================
echo Running Medical Verifier
echo ====================================

call venv\Scripts\activate

echo.
echo Checking if Ollama is running...

curl http://localhost:11434 >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Ollama is not running!
    echo Run start_ollama.bat first.
    pause
    exit /b
)

echo.
echo Running main pipeline...
python main.py --visualize

echo.
echo Done!
pause