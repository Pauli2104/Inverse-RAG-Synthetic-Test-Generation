@echo off
REM Part B: run the exported suite natively under Inspect AI (and promptfoo if Node.js is installed)
REM for the reference agent and the 15 design faults, and compare every verdict with the harness.
setlocal
cd /d "%~dp0\.."
if not exist .venv\Scripts\python.exe ( py -3.12 -m venv .venv 2>nul || python -m venv .venv )
set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
%PY% -m pip install -q -r requirements-frameworks.txt
%PY% export_suite.py
set FW=inspect
where npx >nul 2>&1 && set FW=inspect promptfoo
%PY% frameworks\measure_agreement.py --frameworks %FW%
echo Agreement: frameworks\agreement\summary.json
pause
