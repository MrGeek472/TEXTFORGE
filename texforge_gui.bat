@echo off
rem TEXFORGE : interface graphique, sans fenêtre de console
cd /d "%~dp0"
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw "%~dp0app_gui.py"
) else (
    python "%~dp0app_gui.py"
)
