@echo off
echo === RUNNING TESTS ===
venv\Scripts\python.exe -m pytest tests\ -v
echo.
echo === TESTS COMPLETE ===
