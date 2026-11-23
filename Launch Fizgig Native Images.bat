@echo off
rem Experimental native Qt image shell. The normal Fizgig launcher is unchanged.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Fizgig's environment is missing. Run install_fizgig.bat first.
    pause
    exit /b 1
)
rem Do not install Qt for everyone: this optional experiment needs PySide6.
rem Start Fizgig's local web server first with "Launch Fizgig in browser.bat".
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
"venv\Scripts\python.exe" -m fizgig.web.qt_caption_trial %*
if errorlevel 1 pause
