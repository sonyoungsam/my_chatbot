@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [My First Chat Bot] Virtual environment not found. Run this first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting My First Chat Bot...
streamlit run app.py

pause
