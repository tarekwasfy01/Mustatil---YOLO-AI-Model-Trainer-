@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0runtime\python\python.exe"
if not exist "%PY%" (
  echo Runtime missing. Run INSTALL_EXTERNAL_RUNTIME.bat first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag a GeoTIFF onto this BAT, or call it with input path as first argument.
  pause
  exit /b 1
)
set "IN=%~1"
set "OUT=%~dpn1_mustatil_detection.gpkg"
"%PY%" "%~dp0scripts\mustatil_external_analyzer.py" --input "%IN%" --output "%OUT%" --model "%~dp0models\bestf260.onnx"
pause