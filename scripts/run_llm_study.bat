@echo off
REM ======================================================================
REM  Part C: LLM-agent study with free local open models (Ollama, no API key).
REM  Install Ollama from https://ollama.com/download first. The four models need about 11 GB.
REM  On a laptop CPU the study takes 12-24 hours; keep the computer awake. If it stops, run this
REM  file again: finished model calls are cached in llmstudy\cache and are not repeated.
REM ======================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
if not exist .venv\Scripts\python.exe ( py -3.12 -m venv .venv 2>nul || python -m venv .venv )
set PY=.venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set "NO_PROXY=localhost,127.0.0.1"
set "no_proxy=localhost,127.0.0.1"
if not exist logs mkdir logs
%PY% -m pip install -q -r requirements-frameworks.txt > logs\00_pip.txt 2>&1

where ollama >nul 2>&1 || if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "PATH=%LOCALAPPDATA%\Programs\Ollama;!PATH!"
where ollama >nul 2>&1 || (
  echo ERROR: Ollama is not installed. Install it from https://ollama.com/download and run this file again.
  pause
  exit /b 1
)
ollama list >nul 2>&1 && goto ready
echo Starting the Ollama server...
start "Ollama" /min ollama serve
set /a TRIES=0
:wait
timeout /t 3 /nobreak >nul
ollama list >nul 2>&1 && goto ready
set /a TRIES+=1
if !TRIES! lss 20 goto wait
echo ERROR: the Ollama server did not answer. Open Ollama from the Start menu and run this file again.
pause
exit /b 1
:ready
set KEYFIXED=
:pull
set MISSING=
set KEYERR=
for %%M in (qwen2.5:7b qwen2.5:3b llama3.1:8b nomic-embed-text) do (
  ollama list | findstr /B /C:"%%M" >nul && (echo   %%M: available) || (
    echo   %%M: downloading ...
    ollama pull %%M > logs\04_ollama_pull_last.txt 2>&1
    ollama list | findstr /B /C:"%%M" >nul || (
      set "MISSING=!MISSING! %%M"
      findstr /C:"no key found" logs\04_ollama_pull_last.txt >nul && set KEYERR=1
    )
  )
)
if not defined MISSING goto models_ok
if defined KEYERR if not defined KEYFIXED (
  REM "ssh: no key found": Ollama's identity key is missing or empty. Set it aside and restart Ollama,
  REM which then creates a new key.
  echo Repairing Ollama's identity key and restarting Ollama...
  taskkill /F /IM "ollama app.exe" >nul 2>&1
  taskkill /F /IM ollama.exe >nul 2>&1
  timeout /t 3 /nobreak >nul
  if exist "%USERPROFILE%\.ollama\id_ed25519" move /Y "%USERPROFILE%\.ollama\id_ed25519" "%USERPROFILE%\.ollama\id_ed25519.bak" >nul
  if exist "%USERPROFILE%\.ollama\id_ed25519.pub" move /Y "%USERPROFILE%\.ollama\id_ed25519.pub" "%USERPROFILE%\.ollama\id_ed25519.pub.bak" >nul
  set KEYFIXED=1
  start "Ollama" /min ollama serve
  timeout /t 8 /nobreak >nul
  goto pull
)
echo ERROR: these models could not be downloaded:!MISSING!  See logs\04_ollama_pull_last.txt.
pause
exit /b 1
:models_ok

echo [1/7] Installation check with the mock provider (no model)
%PY% -m llmstudy.study --provider mock --n 5 --repeats 1 > logs\59_llm_mock_test.txt 2>&1 || (
  echo ERROR: see logs\59_llm_mock_test.txt
  pause
  exit /b 1
)
echo [2/7] Ollama connection
%PY% -m llmstudy.llm_client ollama qwen2.5:7b qwen2.5:3b llama3.1:8b nomic-embed-text
echo [3/7] Native garak + PyRIT suite
if exist llmstudy\native_suites\pyrit_garak.jsonl (echo   using the released suite) else (
  %PY% -m llmstudy.native_baselines --redteam > logs\30_redteam.txt 2>&1 )
echo [4/7] Native Ragas suite
if exist llmstudy\native_suites\ragas.jsonl (echo   using the released suite) else (
  %PY% -m llmstudy.native_baselines --ragas --llm ollama --model qwen2.5:7b --size 40 > logs\40_ragas.txt 2>&1 )
echo [5/7] Native suites on the deterministic agent
%PY% -m llmstudy.evaluate_extra_suites > logs\50_native_deterministic.txt 2>&1
echo [6/7] LLM study (long)
%PY% -m llmstudy.study --provider ollama --agent-model qwen2.5:7b --small-model qwen2.5:3b --teacher-model llama3.1:8b --judge-model llama3.1:8b --n 20 --repeats 2 --extra-suites llmstudy\native_suites\pyrit_garak.jsonl llmstudy\native_suites\ragas.jsonl > logs\60_llm_study.txt 2>&1
type logs\60_llm_study.txt | findstr /C:"done" /C:"Error" /C:"tables ->"
echo [7/7] Stale Ragas references and statistics
%PY% -m llmstudy.check_ragas_references
%PY% stats_analysis.py >nul
%PY% make_stats_tables.py >nul
echo Done: results_llm\ and results\stats.json updated; logs in logs\.
pause
endlocal
