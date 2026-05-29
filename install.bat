@echo off
rem One-command native install for the local voice assistant (Windows / cmd.exe).
rem
rem   install.bat               full install (venv, deps, models, doctor)
rem   install.bat --recreate    rebuild the venv from scratch (fixes conda/venv mixes)
rem   install.bat --dry-run     print the plan, change nothing
rem   install.bat --skip-models deps + venv only
rem
rem Double-clickable. PortAudio ships in the sounddevice wheel on Windows, so
rem there is no system step -- this finds Python and runs the cross-platform
rem installer (tools\install.py), the same code path Linux/macOS use.
setlocal
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py") || (
  where python >nul 2>nul && (set "PY=python") || (
    echo No Python found on PATH. Install Python 3.10+ from https://python.org ^(tick "Add to PATH"^).
    exit /b 1
  )
)

echo ==^> Cross-platform install ^(venv, deps, models, doctor^)
%PY% tools\install.py %*
exit /b %ERRORLEVEL%
