@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Clean reinstall isolated external SAM2 runtime
echo ============================================================
echo This deletes runtime_sam2 and rebuilds it.
echo.
if exist "%~dp0runtime_sam2" (
  echo Removing old runtime_sam2...
  rmdir /s /q "%~dp0runtime_sam2"
)
set PYEXE=
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if "%PYEXE%"=="" set PYEXE=python
"%PYEXE%" "%~dp0scripts\install_external_sam2_runtime.py" --root "%~dp0runtime_sam2" --model base_plus --clean
echo.
echo Finished. Press any key.
pause >nul
