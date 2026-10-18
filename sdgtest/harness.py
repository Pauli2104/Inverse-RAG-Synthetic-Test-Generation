"""Reference harness: executes promptfoo-vocabulary assertions against the SUT
and computes mutation-analysis metrics."""
from __future__ import annotations

import re

from .sut import MUTANTS, CampusAgent, safe_call


def check(assertion: dict, output: str) -> bool:
    t, v, o = assertion["type"], assertion["value"], output or ""
    if t == "icontains":
        return v.lower() in o.lower()
    if t == "not-icontains":
        return v.lower() not in o.lower()
    if t == "contains":
        return v in o
    if t == "icontains-any":
        return any(x.lower() in o.lower() for x in v)
    if t == "regex":
        return re.search(v, o) is not None
    if t == "equals":
        return o.strip() == v
    raise ValueError(t)


def passes(tc, output) -> bool:
    return all(check(a, output) for a in tc.assertions)


class Runner:
    """Caches outputs per (agent, query, attachments) to keep repeated suites cheap."""

    def __init__(self, kb):
        self.agents = {None: CampusAgent(kb)}
        self.agents.update({m: CampusAgent(kb, m) for m in MUTANTS})
        self.cache = {}

    def output(self, mutant, tc):
        key = (mutant, tc.query, tuple(tc.attachments))
        if key not in self.cache:
            self.cache[key] = safe_call(self.agents[mutant], tc.query, tc.attachments)
        return self.cache[key]

    def verdicts(self, tc):
        """Return {mutant_or_None: passed}."""
        return {m: passes(tc, self.output(m, tc)) for m in self.agents}

    def evaluate(self, suite):
        rows = []
        for tc in suite:
            v = self.verdicts(tc)
            rows.append((tc, v))
        orig_fail = [tc for tc, v in rows if not v[None]]
        killed = {m: False for m in MUTANTS}
        kills_by = {m: [] for m in MUTANTS}
        for tc, v in rows:
            if not v[None]:
                continue  # a test that fails on the reference agent cannot kill a mutant
            for m in MUTANTS:
                if not v[m]:
                    killed[m] = True
                    kills_by[m].append(tc)
        return {
            "n": len(suite),
            "mutation_score": sum(killed.values()) / len(MUTANTS),
            "killed": killed,
            "kills_by": kills_by,
            "false_alarm_rate": len(orig_fail) / max(1, len(suite)),
            "orig_fail": orig_fail,
            "rows": rows,
        }
