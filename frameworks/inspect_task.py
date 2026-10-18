"""Inspect AI task: the SUT is plugged in as a solver; the scorer applies the suite assertions.

  inspect eval frameworks/inspect_task.py --model mockllm/model
  inspect eval frameworks/inspect_task.py --model mockllm/model -T mutant=M05_guard_off
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inspect_ai import Task, task  # noqa: E402
import json  # noqa: E402

from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import ModelOutput  # noqa: E402
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr  # noqa: E402
from inspect_ai.solver import Generate, TaskState, solver  # noqa: E402

from sdgtest.harness import check  # noqa: E402
from sdgtest.kb import build_kb  # noqa: E402
from sdgtest.sut import CampusAgent, safe_call  # noqa: E402

DATA = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export", "inspect", "dataset.jsonl"))


def load_dataset():
    """Build the dataset in memory (no path patterns: avoids Inspect/pathlib glob errors on Windows paths)."""
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"{DATA} not found - run 'python export_suite.py' first")
    samples = []
    with open(DATA, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                samples.append(Sample(id=r["id"], input=r["input"] or " ", target=str(r.get("target") or ""),
                                      metadata=r.get("metadata") or {}))
    return MemoryDataset(samples, name="sdg_suite")


@solver
def campus_agent(mutant=None):
    agent = CampusAgent(build_kb(seed=7), mutant)

    async def solve(state: TaskState, generate: Generate):
        q = state.input_text if state.input_text.strip() else ""
        out = safe_call(agent, q, state.metadata.get("attachments") or [])
        state.output = ModelOutput.from_content(model="campus-agent", content=out)
        return state
    return solve


@scorer(metrics=[accuracy(), stderr()])
def suite_assertions():
    async def score(state: TaskState, target: Target):
        out = state.output.completion
        ok = all(check(a, out) for a in state.metadata["assertions"])
        return Score(value=CORRECT if ok else INCORRECT, answer=out)
    return score


@task
def sdg_suite(mutant: str = ""):
    return Task(dataset=load_dataset(), solver=campus_agent(mutant or None), scorer=suite_assertions())
