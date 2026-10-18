"""DeepEval run: deterministic assertion metric (always) + LLM-judged metrics (optional, needs a judge model key).

  python frameworks/deepeval_run.py [--mutant M09_no_abstain] [--llm-judge]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deepeval import evaluate  # noqa: E402
from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402

from sdgtest.harness import check  # noqa: E402
from sdgtest.kb import build_kb  # noqa: E402
from sdgtest.sut import CampusAgent, safe_call  # noqa: E402


class SuiteAssertionMetric(BaseMetric):
    """Applies the promptfoo-vocabulary assertions stored with each golden."""
    threshold = 1.0

    def __init__(self):
        self.score, self.success, self.reason = 0.0, False, ""

    def measure(self, tc: LLMTestCase, *a, **k):
        ok = all(check(x, tc.actual_output) for x in tc.additional_metadata["assertions"])
        self.score, self.success = float(ok), ok
        self.reason = "all assertions passed" if ok else "assertion failure"
        return self.score

    async def a_measure(self, tc, *a, **k):
        return self.measure(tc)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "SuiteAssertions"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mutant"); ap.add_argument("--llm-judge", action="store_true")
    args = ap.parse_args()
    goldens = json.load(open(os.path.join(os.path.dirname(__file__), "..", "export", "deepeval", "goldens.json"), encoding="utf-8"))
    agent = CampusAgent(build_kb(seed=7), args.mutant)
    cases = [LLMTestCase(input=g["input"] or " ", actual_output=safe_call(agent, g["input"], g["additional_metadata"]["attachments"]),
                         expected_output=g["expected_output"], context=g["context"] or None,
                         retrieval_context=g["context"] or None, additional_metadata=g["additional_metadata"]) for g in goldens]
    metrics = [SuiteAssertionMetric()]
    if args.llm_judge:  # requires OPENAI_API_KEY (or a configured local judge)
        from deepeval.metrics import GEval
        metrics.append(GEval(name="Correctness", criteria="Is the actual output consistent with the expected output and context?",
                             evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]))
    evaluate(test_cases=cases, metrics=metrics)


if __name__ == "__main__":
    main()
