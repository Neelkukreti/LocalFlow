@echo off
REM Build LocalFlow.exe locally. Produces dist\LocalFlow.exe.
REM The build type (CPU vs GPU) matches whatever is installed in .venv:
REM   pip install -r requirements.txt      -> CPU exe
REM   pip install -r requirements-gpu.txt  -> GPU exe
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pip install --quiet pyinstaller==6.11.1
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean localflow.spec
echo.
echo Done -^> dist\LocalFlow.exe
pause
