@echo off
REM Start the intranet segmentation web app on a Windows host.
REM
REM Double-click this file, or run it from a scheduled task set to "Run whether
REM user is logged on or not" so the service survives sign-out.
REM
REM Expects a virtual environment at .venv in the repository root. Edit
REM PYTHON_EXE below if your environment lives somewhere else.

setlocal

set "REPO_DIR=%~dp0.."
set "PYTHON_EXE=%REPO_DIR%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo.
    echo   Could not find the Python environment at:
    echo     %PYTHON_EXE%
    echo.
    echo   Create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements-web.txt
    echo     .venv\Scripts\pip install -e .
    echo.
    pause
    exit /b 1
)

cd /d "%REPO_DIR%"

REM Uncomment to override the packaged configuration.
REM set MICROSEG_WEB_PORT=8080
REM set MICROSEG_WEB_CONFIG=C:\microseg\web_server.yml

echo Starting the segmentation web app...
"%PYTHON_EXE%" scripts\run_web_server.py

REM Keep the window open if the server exits so the error stays readable.
if errorlevel 1 pause
endlocal
