@echo off
echo ====================================
echo Setting up Medical Verifier Project
echo ====================================

echo.
echo [1] Creating virtual environment with Python 3.10...
py -3.10 -m venv venv

echo.
echo [2] Activating virtual environment...
call venv\Scripts\activate

echo.
echo [3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4] Installing dependencies...
pip install requests networkx matplotlib spacy scispacy numpy

echo.
echo [5] Installing SciSpacy model...
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

echo.
echo ====================================
echo IMPORTANT:
echo Install Ollama from: https://ollama.com
echo Then run: ollama pull phi4
echo ====================================

echo.
echo Setup complete!
pause