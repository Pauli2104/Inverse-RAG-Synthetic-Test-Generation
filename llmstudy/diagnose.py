"""Short diagnostic run of the LLM agent (reference configuration) with full step-by-step traces.

  python -m llmstudy.diagnose --provider ollama --agent-model qwen2.5:7b --judge-model llama3.1:8b --k 8

Uses the same test samples as llmstudy.study (same seed and order) and the same cache, and prints for each test
the question, expected answer, every model reply, the tools called, the exit reason and the oracle verdict.
"""
import argparse
import json
import random

from sdgtest.generator import SDGPipeline, happy_path_suite, rule_based_suite
from sdgtest.kb import build_kb

from . import docs as docs_mod
from .agent import LLMCampusAgent
from .llm_client import LLM
from .study import Oracle, make_mock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--agent-model", default="qwen2.5:7b")
    ap.add_argument("--judge-model", default="llama3.1:8b")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=8, help="tests per suite to trace")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    docs_mod.render()
    kb = build_kb(seed=7)
    rng = random.Random(a.seed)
    s0 = rng.sample(happy_path_suite(kb), a.n)
    rng.sample(rule_based_suite(kb), a.n)  # keep the same random sequence as llmstudy.study
    s2 = rng.sample(SDGPipeline(kb, seed=a.seed).run(600), a.n)
    mock = make_mock() if a.provider == "mock" else None
    llm = LLM(a.provider, a.agent_model, mock_fn=mock)
    replies = []
    orig = llm.chat

    def logged(system, messages, sample=0, json_mode=False):
        r = orig(system, messages, sample=sample, json_mode=json_mode)
        replies.append(r)
        return r
    llm.chat = logged
    oracle = Oracle(LLM(a.provider, a.judge_model, mock_fn=mock))
    agent = LLMCampusAgent(llm)
    tally = {}
    for name, cases in (("S0", s0), ("S2", s2)):
        for tc in cases[:a.k]:
            replies.clear()
            r = agent.answer(tc.query, tc.attachments, sample=0)
            ok = oracle.passes(tc, r["output"])
            tally[(name, r.get("exit"), ok)] = tally.get((name, r.get("exit"), ok), 0) + 1
            print("=" * 100)
            print(f"{name} {tc.id} [{tc.dimension}/{tc.subtype}]  expected={tc.expected_behavior}: {tc.ground_truth}")
            print("Q:", tc.query[-400:].replace("\n", " "))
            first = r["trace"][0]["result"].get("results", []) if r["trace"] else []
            print("retrieved:", [h["id"] for h in first])
            for i, rep in enumerate(replies):
                print(f"  step {i + 1} model reply: {rep[:400]!r}")
            print("  tools:", [t["tool"] for t in r["trace"]], " exit:", r.get("exit"))
            print("  FINAL:", r["output"][:300].replace("\n", " "))
            print("  PASS" if ok else "  FAIL")
    print("=" * 100)
    print("summary (suite, exit, pass): count")
    for k, v in sorted(tally.items(), key=str):
        print("  ", k, v)


if __name__ == "__main__":
    main()
