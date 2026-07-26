@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set PYTHONHASHSEED=42
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python could not be found. Install a compatible Python version or create .venv first.
        exit /b 1
    )
    set "PYTHON=py"
)

echo === PREPARING LEGACY ARCHIVE ===
"%PYTHON%" tools\prepare_rebuild.py
if errorlevel 1 exit /b 1

echo.
echo === RUNNING TESTS ===
"%PYTHON%" -m pytest tests -q
if errorlevel 1 exit /b 1

echo.
echo === RUNNING CORRECTED PIPELINE ===
"%PYTHON%" src\corrected_master_pipeline.py --config config\config.yaml --overwrite
if errorlevel 1 exit /b 1

echo.
echo === COMPLETE ===
echo Final outputs: results\final, figures\final, models\final
endlocal
