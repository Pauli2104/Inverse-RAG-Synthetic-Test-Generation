"""Ragas run with non-LLM metrics over answer-type test cases.

  python frameworks/ragas_run.py [--mutant M11_truncate_context]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.metrics import NonLLMStringSimilarity, StringPresence  # noqa: E402

from sdgtest.kb import build_kb  # noqa: E402
from sdgtest.sut import CampusAgent, safe_call  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--mutant"); args = ap.parse_args()
agent = CampusAgent(build_kb(seed=7), args.mutant)
rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "..", "export", "ragas", "testset.jsonl"), encoding="utf-8")]
data = [{"user_input": r["user_input"], "response": safe_call(agent, r["user_input"]), "reference": r["reference"],
         "retrieved_contexts": r["reference_contexts"]} for r in rows]
print(evaluate(EvaluationDataset.from_list(data), metrics=[StringPresence(), NonLLMStringSimilarity()]))
