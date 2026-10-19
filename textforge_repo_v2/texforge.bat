@echo off
rem TEXFORGE : lance l'outil depuis n'importe quel dossier
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
python -m texforge %*
