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
conda --version
echo.

echo Installing packages via conda...
conda install -y pandas numpy matplotlib

echo.
echo Installing remaining packages via pip...
pip install pyModbusTCP
pip install python-snap7
pip install PyAudio
pip install xmltodict

echo.
echo ============================================
echo   All dependencies installed successfully!
echo ============================================
pause
