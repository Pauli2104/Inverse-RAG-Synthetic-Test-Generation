"""Smoke and reproducibility tests for the generation pipeline and the controlled-study agent."""
import hashlib
import json
import os
import tempfile

from sdgtest.generator import SDGPipeline, happy_path_suite, save_jsonl
from sdgtest.harness import Runner
from sdgtest.kb import build_kb
from sdgtest.sut import HOLDOUT, MUTANTS

KB = build_kb(seed=7)


def test_knowledge_base_size():
    assert len(KB.facts) == 132 and len(KB.chunks) == 54 and len(KB.people) == 24


def test_generator_reproduces_released_suite():
    """The released S2 suite (seed 0) is regenerated bit for bit."""
    pool = SDGPipeline(KB, seed=0).run(600)
    path = os.path.join(tempfile.mkdtemp(), "s2.jsonl")
    save_jsonl(pool, path)
    assert open(path, encoding="utf-8").read() == open("results/datasets/sdg_seed0.jsonl", encoding="utf-8").read()


def test_released_suite_detects_all_design_faults_without_false_alarms():
    pool = SDGPipeline(KB, seed=0).run(600)
    res = Runner(KB).evaluate(pool)
    assert res["false_alarm_rate"] == 0.0
    assert res["mutation_score"] == 1.0


def test_happy_path_suite_is_weak():
    res = Runner(KB).evaluate(happy_path_suite(KB))
    assert res["mutation_score"] < 0.2


def test_fault_catalogues():
    assert len(MUTANTS) == 15 and len(HOLDOUT) == 12


def test_generator_matches_frozen_hash():
    """The generator has not changed since the held-out faults were written."""
    for line in open("results/frozen_generator.sha256", encoding="utf-8"):
        digest, path = line.split()
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest, path
