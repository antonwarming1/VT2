@echo off
echo ============================================
echo   Installing dependencies for VT2 project
echo            (Anaconda environment)
echo ============================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo Conda is not installed or not in the system's PATH.
    echo Make sure Anaconda/Miniconda is installed and run this
    echo from an Anaconda Prompt.
    pause
    exit /b 1
)

echo Found Conda:
call conda --version
echo.

echo Installing packages via conda...
call conda install -y pandas numpy matplotlib 


echo.
echo Installing remaining packages via pip...
pip install tensorflow
pip install scipy
pip install scikit-learn
pip install seaborn
pip install joblib
pip install fastapi
pip install uvicorn
pip install librosa
pip install requests
pip install pydantic
pip install tsfresh
pip install pyModbusTCP
pip install xmltodict
pip install noisereduce
pip install optuna
pip install soundfile
pip install python-snap7
pip install pyaudio

echo.
echo ============================================
echo   All dependencies installed successfully!
echo ============================================
pause
