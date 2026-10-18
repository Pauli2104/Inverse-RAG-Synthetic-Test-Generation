# Reproduction guide

All commands run from the repository root. Section, table and figure numbers refer to the article; numbers with
the prefix S refer to its Online Resource 1.

## Part A: controlled study (Python only)

```bash
pip install -r requirements.txt
scripts/run_controlled.sh          # Windows: scripts\run_controlled.bat
```

| Step | Command | Writes | Article |
|---|---|---|---|
| A1 | `python experiments.py` | `results/results.json`, `results/datasets/*.jsonl` | Tables 5-7; Sect. 5.5; Figs 2-4; Fig. S1; Tables S2, S6 |
| A2 | `python e4b.py` | `results/e4b_critic_modes.json` | Fig. 4; Table S2 |
| A3 | `python diversity.py` | `results/diversity.json` | Table 5 (diversity) |
| A4 | `python revision_experiments.py 2` | `results/revision.json`, `results/revision_permutant.json` | Tables 8-9; Fig. 5 |
| A5 | `python messy_oracle.py` | `results/messy_oracle.json` | Table 10 |
| A6 | `python selection_experiment.py` | `results/selection.json` | Table S1 |
| A7 | `python stats_analysis.py` | `results/stats.json` | Sect. 3.9, 5.10; Tables 12-13; Tables S4-S5 |
| A8 | `python make_stats_tables.py` | `results/stats_tables.tex` | LaTeX of Tables 12-13, S4-S5 |
| A9 | `python figures.py`, `python figures_revision.py` | `figures/Fig2-Fig5`, `figures/FigS1` | Figs 2-5, Fig. S1 |
| A10 | `pdflatex figures/Fig1_source.tex` | `Fig1_source.pdf` | Fig. 1 |
| A11 | `python export_suite.py` | `export/` | Sect. 3.2 (framework export) |

`experiments.py --quick` runs a reduced version (5 seeds) in a few minutes; it overwrites `results/results.json`.
`revision_experiments.py` takes the number of worker processes as its argument; with 2 workers it runs in about
16 minutes. Steps A7-A9 need only the JSON files, which are included, so statistics and figures can be
regenerated without re-running the experiments.

Table 1 (frameworks), Table 2 (generation families), Table 3 (design faults) and Table 4 (held-out faults)
describe the design; their content is defined in `sdgtest/generator.py` and `sdgtest/sut.py` (`MUTANTS`,
`HOLDOUT`). The generator was frozen before the held-out faults were written; its SHA-256 hash is in
`results/frozen_generator.sha256` and can be checked with `sha256sum -c results/frozen_generator.sha256`.

## Part B: native framework execution (Sect. 4)

```bash
pip install -r requirements-frameworks.txt
scripts/run_frameworks.sh          # Windows: scripts\run_frameworks.bat
```

`frameworks/measure_agreement.py` runs the exported suite (541 tests) under Inspect AI for the reference agent
and all 15 design faults and compares every (test, agent) verdict with the harness. The released run
(`frameworks/agreement/summary.json`, logs in `inspect_logs.zip`) agreed on all 8,656 verdicts. promptfoo needs
Node.js (`npx promptfoo@latest eval -c frameworks/promptfooconfig.yaml`); in the article its configuration was
validated syntactically only. The DeepEval and Ragas adapters (`frameworks/deepeval_run.py`,
`frameworks/ragas_run.py`) are provided for completeness and were not part of the reported agreement.

## Part C: LLM-agent study (Sect. 5.9)

Install [Ollama](https://ollama.com/download), then

```bash
pip install -r requirements-frameworks.txt
scripts/run_llm_study.sh           # Windows: scripts\run_llm_study.bat
```

| Step | Command | Writes | Article |
|---|---|---|---|
| C1 | `ollama pull qwen2.5:7b qwen2.5:3b llama3.1:8b nomic-embed-text` | local models (about 11 GB) | |
| C2 | `python -m llmstudy.native_baselines --redteam` | `llmstudy/native_suites/pyrit_garak.jsonl` | native garak + PyRIT suite |
| C3 | `python -m llmstudy.native_baselines --ragas --llm ollama --model qwen2.5:7b --size 40` | `llmstudy/native_suites/ragas.jsonl` | native Ragas suite |
| C4 | `python -m llmstudy.evaluate_extra_suites` | `results_llm/native_on_deterministic.json` | native suites on the deterministic agent |
| C5 | `python -m llmstudy.study --provider ollama --agent-model qwen2.5:7b --small-model qwen2.5:3b --teacher-model llama3.1:8b --judge-model llama3.1:8b --n 20 --repeats 2 --extra-suites llmstudy/native_suites/pyrit_garak.jsonl llmstudy/native_suites/ragas.jsonl` | `results_llm/llm_study_ollama_<time>.json`, `summary.json`, `llm_tables.tex` | Table 11, Table S3 |
| C6 | `python -m llmstudy.check_ragas_references` | console | stale Ragas references (Sect. 5.9) |
| C7 | `python stats_analysis.py` | `results/stats.json` (LLM part) | Table 13 |

The scripts in `scripts/` skip C2 and C3 when the native suites already exist, so by default the study is run
on exactly the suites used in the article; delete them to regenerate. Every model call is cached in
`llmstudy/cache/`, so an interrupted run continues where it stopped. On a laptop CPU, C5 takes 12-24 hours.
`python -m llmstudy.study --provider mock --n 5 --repeats 1` checks the installation in a minute without any
model; mock runs write to `results_llm/mock/` and never touch the released results. `python -m llmstudy.diagnose` prints step-by-step traces of the agent for a few questions.

The released run is `results_llm/llm_study_ollama_20261018-122418.json`; its console output is in
`results_llm/run_logs/60_llm_study.txt`. LLM outputs can differ across hardware, operating systems and Ollama
versions even at temperature 0, so a re-run may differ in individual verdicts.

Other providers are supported (`--provider openai | anthropic | deepseek` with the corresponding API key in an
environment variable; `SDG_MAX_USD` caps the spend). Keys are read from the environment only and are never
written to files or logs.
