"""Evaluate native generator suites (native_suites/*.jsonl) on the deterministic agent and its
design (M) and held-out (H) mutants, at the same budgets as the main study.

  python -m llmstudy.evaluate_extra_suites            -> results_llm/native_on_deterministic.json
Free-text Ragas references are checked with a numeric/key-phrase match (same rule as the LLM study critic).
"""
import glob
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sdgtest.harness import check  # noqa: E402
from sdgtest.kb import build_kb  # noqa: E402
from sdgtest.sut import HOLDOUT, MUTANTS, CampusAgent, safe_call_any  # noqa: E402

from .study import same_answer  # noqa: E402


def passes(t, out):
    for a in t["assertions"]:
        if a["type"] == "judge":
            if not same_answer(a["value"], out):
                return False
        elif not check(a, out):
            return False
    return True


def main():
    kb = build_kb(seed=7)
    agents = {"ref": CampusAgent(kb), **{m: CampusAgent(kb, m) for m in list(MUTANTS) + list(HOLDOUT)}}
    res = {}
    for path in glob.glob(os.path.join(os.path.dirname(__file__), "native_suites", "*.jsonl")):
        name = os.path.basename(path)[:-6]
        suite = [json.loads(l) for l in open(path, encoding="utf-8")]
        V = {a: np.array([passes(t, safe_call_any(ag, t["query"], t.get("attachments") or [])) for t in suite]) for a, ag in agents.items()}
        K = {m: V["ref"] & ~V[m] for m in agents if m != "ref"}
        out = {"n": len(suite), "ref_pass": float(V["ref"].mean())}
        for N in (25, 50, 100):
            for fam, ms in (("M", list(MUTANTS)), ("H", list(HOLDOUT))):
                vals = []
                for r in range(30):
                    idx = random.Random(7000 + r).sample(range(len(suite)), min(N, len(suite)))
                    vals.append(np.mean([K[m][idx].any() for m in ms]))
                out[f"MS_{fam}_N{N}"] = [float(np.mean(vals)), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
        out["killed_full"] = [m for m in K if K[m].any()]
        res[name] = out
        print(name, json.dumps(out))
    os.makedirs(os.path.join(ROOT, "results_llm"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results_llm", "native_on_deterministic.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
