#!/usr/bin/env bash
# Part C: LLM-agent study with local open models (Ollama). 12-24 hours on a laptop CPU.
# Interrupted runs resume from the response cache in llmstudy/cache/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 PYTHONIOENCODING=utf-8
mkdir -p logs
command -v ollama >/dev/null || { echo "Install Ollama from https://ollama.com/download first."; exit 1; }
ollama list >/dev/null 2>&1 || { (ollama serve > logs/ollama_serve.txt 2>&1 &); sleep 5; }
for m in qwen2.5:7b qwen2.5:3b llama3.1:8b nomic-embed-text; do
  ollama list | grep -q "^$m" || ollama pull "$m"
done
echo "== check: mock run without any model";  $PY -m llmstudy.study --provider mock --n 5 --repeats 1 > logs/59_llm_mock_test.txt
echo "== check: Ollama connection";            $PY -m llmstudy.llm_client ollama qwen2.5:7b qwen2.5:3b llama3.1:8b nomic-embed-text
if [ ! -f llmstudy/native_suites/pyrit_garak.jsonl ]; then
  echo "== native garak + PyRIT suite";       $PY -m llmstudy.native_baselines --redteam | tee logs/30_redteam.txt
fi
if [ ! -f llmstudy/native_suites/ragas.jsonl ]; then
  echo "== native Ragas suite";               $PY -m llmstudy.native_baselines --ragas --llm ollama --model qwen2.5:7b --size 40 | tee logs/40_ragas.txt
fi
echo "== native suites on the deterministic agent"; $PY -m llmstudy.evaluate_extra_suites | tee logs/50_native_deterministic.txt
echo "== LLM study (long)"
$PY -m llmstudy.study --provider ollama --agent-model qwen2.5:7b --small-model qwen2.5:3b \
    --teacher-model llama3.1:8b --judge-model llama3.1:8b --n 20 --repeats 2 \
    --extra-suites llmstudy/native_suites/pyrit_garak.jsonl llmstudy/native_suites/ragas.jsonl | tee logs/60_llm_study.txt
echo "== stale references in the native Ragas suite"; $PY -m llmstudy.check_ragas_references
echo "== statistics";                         $PY stats_analysis.py > /dev/null; $PY make_stats_tables.py > /dev/null
echo "Done: results_llm/ and results/stats.json updated."
