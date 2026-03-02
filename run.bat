@echo off

REM Get the current directory where the batch file is located


set "batch_dir=%~dp0"

REM Check if Python is installed
where python >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in the system's PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)
echo Running the Python script "ConverterKxmlToJson.py"...
start python ConverterKxmlToJson.py
echo Running the Python script "task_and_acustics_data_collection.py"
start python task_and_acustics_data_collection.py
pause