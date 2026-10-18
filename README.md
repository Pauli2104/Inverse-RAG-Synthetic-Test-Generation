# Inverse-RAG Synthetic Test Generation for Agentic AI Assistants

Code, data and results for the article

> **Inverse-RAG Synthetic Test Generation for Agentic AI Assistants in Intelligent Environments:
> A Mutation-Based Evaluation of Fault-Detection Effectiveness**
> Paula Otaño and Cristian Rodriguez Rivero.
> *Journal of Reliable Intelligent Environments* (Springer), special issue "Engineering Reliable Intelligent
> Environments: Quality, Technical Debt and Trustworthy AI". Under review.

The repository contains everything needed to regenerate every table and figure of the article and of its
Online Resource 1: the generation pipeline, the agents under test and their fault sets, the experiment
scripts, the generated test suites, the raw results and the scripts that turn them into tables and figures.

## What the study does

An *inverse-RAG* pipeline turns knowledge-base context into persona-conditioned questions with verifiable
answers, expands them along multi-hop, linguistic, edge-case, adversarial and privacy dimensions, filters them
with a claim-level critic and exports them to agent-testing frameworks. Mutation analysis measures how many
seeded faults a suite detects:

| Study | Agent under test | Faults | Where |
|---|---|---|---|
| Controlled study | deterministic tool-using campus assistant (`sdgtest/sut.py`) | 15 design faults M01-M15, 12 held-out taxonomy faults H01-H12, 572 automatic code mutants | `sdgtest/`, root scripts, `results/` |
| LLM-agent study | retrieve-then-act LLM agent over realistic facility documentation (`llmstudy/agent.py`) | 12 configuration faults L01-L12 | `llmstudy/`, `results_llm/` |

Suites compared at equal budget: S0 happy path, S1 rule-based, S2 synthetic pipeline, S2-LLM (LLM teacher and
critic), BR/BP/BRP (re-implemented Ragas-style and garak/PyRIT-style generators) and natively generated Ragas,
garak and PyRIT suites. See [docs/PIPELINE.md](docs/PIPELINE.md) and [docs/FAULTS.md](docs/FAULTS.md).

## Quick start (5 minutes, no internet, no API key)

```bash
git clone https://github.com/Pauli2104/Inverse-RAG-Synthetic-Test-Generation.git
cd Inverse-RAG-Synthetic-Test-Generation
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                             # smoke tests (about 1 minute)
python stats_analysis.py && python make_stats_tables.py   # hypothesis tests from the released results
python figures.py && python figures_revision.py           # all figures from the released results
```

`pytest` regenerates the released synthetic suite bit for bit from its seed, checks that it detects all 15
design faults without false alarms, and checks that the statistics in `results/stats.json` are reproduced.

## Reproducing the article

| Part | What | Needs | Time (laptop CPU) | Command |
|---|---|---|---|---|
| A | Controlled study, statistics, figures (Sect. 5.1-5.8, 5.10) | Python only | about 30 min | `scripts/run_controlled.sh` (Windows: `scripts\run_controlled.bat`) |
| B | Native execution of the exported suite under Inspect AI (Sect. 4) | Python, internet once | about 30 min | `scripts/run_frameworks.sh` |
| C | LLM-agent study with local open models (Sect. 5.9) | [Ollama](https://ollama.com), about 11 GB of models | 12-24 h on CPU | `scripts/run_llm_study.sh` |

Every number in the article comes from a file in this repository; [docs/REPRODUCE.md](docs/REPRODUCE.md) maps
each table and figure to the script that produces it and the file it reads. All randomness is seeded: Part A
reproduces the released results exactly. Part C uses LLMs whose outputs can vary across hardware and Ollama
versions; the released run is in `results_llm/`.

## Repository layout

```
sdgtest/                 generation pipeline and controlled-study agent
  kb.py                  synthetic campus knowledge base rendered from a hidden fact table
  generator.py           inverse-RAG pipeline: sampler, grammar teacher, personas, 5 families, critic, S0/S1
  baselines_ext.py       BR (Ragas-style), BP (garak/PyRIT-style) and BRP generators
  sut.py                 deterministic agent, design faults M01-M15, held-out faults H01-H12
  astmut.py              automatic AST mutation of the agent (572 viable mutants)
  harness.py             assertion executor (promptfoo vocabulary) and mutation analysis
  exporters.py           export to promptfoo, Inspect AI, DeepEval, Ragas and PyRIT formats
  metrics.py, textutil.py
llmstudy/                LLM-agent study
  agent.py               LLM agent, tools and faults L01-L12
  docs.py, facility_docs/, ground_truth.json   realistic documentation and its hidden fact table
  study.py               suites S0-S2-LLM, BRP and native suites; oracle, LLM judge, two runs per test
  native_baselines.py    native garak probes + PyRIT converters, native Ragas TestsetGenerator
  native_suites/         the native suites used in the article
  llm_client.py          Ollama / OpenAI / Anthropic / DeepSeek / mock clients with a response cache
  evaluate_extra_suites.py, check_ragas_references.py, diagnose.py
frameworks/              Inspect AI, promptfoo, DeepEval and Ragas adapters; agreement measurement
  agreement/             native Inspect AI verdicts (summary.json, inspect_logs.zip)
export/                  the synthetic suite (seed 0) exported to every framework format
*.py (root)              experiment, statistics and figure scripts (see docs/REPRODUCE.md)
results/                 controlled-study results, statistics and the released test suites (datasets/)
results_llm/             LLM-study results, LaTeX tables and run logs
figures/                 Fig1-Fig5 and FigS1 (PDF, EPS, PNG); Fig1_source.tex is the TikZ diagram
scripts/                 one-command runners for Linux/macOS (.sh) and Windows (.bat)
tests/                   smoke and reproducibility tests (run in CI)
docs/                    pipeline, fault catalogue, data description, reproduction guide
```

## Software and hardware used

The controlled study was run with Python 3.12 (Windows) and 3.13 (Linux) on 2 CPU cores without a GPU. The LLM study
and the native framework runs were run on a Windows laptop without a GPU (Python 3.12.10, Ollama with
`qwen2.5:7b` as agent, `qwen2.5:3b` as the downgraded model, `llama3.1:8b` as teacher, critic and judge and
`nomic-embed-text` for Ragas embeddings; Ragas 0.4.3, garak 0.9.0.4, PyRIT 1.1.0, Inspect AI 0.3.276). The
exact package versions are in `results_llm/run_logs/01_pip_freeze.txt` and `requirements-frameworks.txt`.

## Data

All data are synthetic. Buildings, rooms, people, e-mail addresses and phone numbers are fictitious and were
generated by the code; no personal data and no human participants are involved. See [docs/DATA.md](docs/DATA.md).

## Citation

If you use this code or data, please cite the article (see [CITATION.cff](CITATION.cff); GitHub's
"Cite this repository" button uses it). The citation will be updated with the DOI once the article is published.

## Licence

Code: [MIT](LICENSE). Data, test suites, results and figures: [CC BY 4.0](LICENSE-DATA).

## Contact

Paula Otaño (corresponding author), Systems Engineering Department, National Technological University,
Faculty Córdoba (UTN-FRC), Argentina. Questions and problems: please open an issue.
