@echo off
title SEVAMITR - Smart Income Allocation & Recovery Engine
echo =====================================================================
echo  SEVAMITR: AI-Powered Cashflow Protection & Gig Recovery Engine
echo =====================================================================
echo Starting FastAPI Web Server at http://127.0.0.1:8000 ...
echo Press Ctrl+C to stop the server.
echo.

set PYTHON_CMD="C:\Users\Manoj M U\AppData\Local\Python\bin\python.exe"

if exist %PYTHON_CMD% (
    %PYTHON_CMD% main.py
) else (
    python main.py
)

pause
