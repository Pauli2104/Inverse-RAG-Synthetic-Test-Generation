@echo off
REM Part A: controlled study, statistics and figures (Python only, no internet). About 30 minutes.
REM Double-click, or run from cmd.exe. Creates .venv in the repository root on first use.
setlocal
cd /d "%~dp0\.."
if not exist .venv\Scripts\python.exe ( py -3.12 -m venv .venv 2>nul || python -m venv .venv )
set PY=.venv\Scripts\python.exe
%PY% -m pip install -q -r requirements.txt || goto fail
echo [A1] Main study: coverage, budget sweep, ablations, noise, Shapley, amplification
%PY% experiments.py || goto fail
echo [A2] Critic comparison
%PY% e4b.py || goto fail
echo [A3] Size-matched diversity
%PY% diversity.py || goto fail
echo [A4] Independent faults and contemporary-style generators (longest step)
%PY% revision_experiments.py 2 || goto fail
echo [A5] Critics on less structured documentation
%PY% messy_oracle.py || goto fail
echo [A6] Uniform versus stratified selection
%PY% selection_experiment.py || goto fail
echo [A7] Hypothesis tests, post hoc tests, sensitivity analyses
%PY% stats_analysis.py >nul || goto fail
%PY% make_stats_tables.py >nul || goto fail
echo [A8] Figures
%PY% figures.py || goto fail
%PY% figures_revision.py || goto fail
echo [A9] Framework export
%PY% export_suite.py || goto fail
echo Done: results\ and figures\ regenerated.
pause
exit /b 0
:fail
echo A step failed; see the message above.
pause
exit /b 1
