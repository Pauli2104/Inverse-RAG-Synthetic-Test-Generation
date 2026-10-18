# Framework adapters

`python export_suite.py` writes the synthetic suite (seed 0, 541 tests) to `export/` in the formats of five
agent-testing frameworks. The files here run that export natively against the deterministic agent
(`sdgtest/sut.py`), the reference version or one of its design faults.

| File | Framework | Status in the article |
|---|---|---|
| `inspect_task.py` | Inspect AI task: the agent is the solver, the test assertions the scorer | executed natively; all 8,656 (test, agent) verdicts agreed with the harness |
| `promptfooconfig.yaml`, `promptfoo_provider.py` | promptfoo with a Python provider | validated syntactically (needs Node.js) |
| `deepeval_run.py` | DeepEval custom metric | provided, not part of the reported agreement |
| `ragas_run.py` | Ragas string metrics for answer-type tests | provided, not part of the reported agreement |
| `measure_agreement.py` | runs Inspect AI (and promptfoo) for the reference agent and every design fault and compares each verdict with the harness | writes `agreement/summary.json` |

```bash
pip install -r requirements-frameworks.txt
python export_suite.py
python frameworks/measure_agreement.py --frameworks inspect          # add promptfoo if Node.js is installed
inspect eval frameworks/inspect_task.py --model mockllm/model -T mutant=M05_guard_off
SDG_MUTANT=M06_guard_no_decode npx promptfoo@latest eval -c frameworks/promptfooconfig.yaml
python frameworks/deepeval_run.py --mutant M09_no_abstain
python frameworks/ragas_run.py --mutant M11_truncate_context
```

`agreement/summary.json` and `agreement/inspect_logs.zip` hold the released native run (Inspect AI 0.3.276,
Windows, Python 3.12). The PyRIT export (`export/pyrit/seed_prompts.yaml`) is a seed-prompt dataset for PyRIT
orchestrators; its YAML schema has changed across PyRIT releases.
