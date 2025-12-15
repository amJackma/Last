@echo off
echo Setting up and running Flight Delay Predictor...
cd /d %~dp0

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Please ensure Python is installed.
        pause
        exit /b 1
    )
    echo Installing requirements...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
) else (
    echo Activating existing virtual environment...
    call .venv\Scripts\activate.bat
)

echo Running Streamlit app...
streamlit run streamlit_app.py
pause