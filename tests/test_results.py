"""The released statistics and the LLM-study files are consistent with the article."""
import json

import stats_analysis


def test_statistics_are_reproduced_from_released_results():
    fresh = json.loads(json.dumps({"controlled": stats_analysis.controlled(), "llm": stats_analysis.llm()}))
    assert fresh == json.load(open("results/stats.json", encoding="utf-8"))


def test_headline_numbers():
    rev = json.load(open("results/revision.json", encoding="utf-8"))
    ms = rev["ms"]
    assert round(ms["design (M)"]["S2"]["100"]["mean"], 2) == 0.93
    assert round(ms["held-out taxonomy (H)"]["S2"]["100"]["mean"], 2) == 0.70
    assert round(ms["automatic AST, killable non-trivial (A*)"]["S2"]["100"]["mean"], 2) == 0.92


def test_llm_study_detection_counts():
    summ = json.load(open("results_llm/summary.json", encoding="utf-8"))["summary"]
    detected = {s: sum(v["killed"].values()) for s, v in summ.items()}
    assert detected["S2_LLM"] == 8 and detected["S2"] == 5 and detected["pyrit_garak"] == 0


def test_native_agreement():
    agr = json.load(open("frameworks/agreement/summary.json", encoding="utf-8"))["frameworks"]["inspect"]
    assert agr["pairs_expected"] == agr["agreeing"] == 8656
