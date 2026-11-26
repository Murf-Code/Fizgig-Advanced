@echo off
setlocal
cd /d "%~dp0"
set "PY=H:\Stable_Diffusion\Fizgig\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Fizgig Python not found at "%PY%".
  echo Run from PowerShell with your own Python path, or edit PY in this file.
  pause
  exit /b 1
)
powershell -NoProfile -Command "& '%PY%' -u '.\run_cpu.py' 2>&1 | Tee-Object -FilePath 'cpu-results-v2.log'; exit $LASTEXITCODE"
set "RC=%ERRORLEVEL%"
echo Exit code: %RC%
echo Report: cpu_results_v2.json
pause
exit /b %RC%
