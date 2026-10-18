"""Export a synthetic suite into the native formats of agent-testing frameworks.

promptfoo  -> frameworks/promptfoo/tests.yaml   (vars + assert, same assertion vocabulary)
Inspect AI -> frameworks/inspect/dataset.jsonl  (input / target / metadata)
DeepEval   -> frameworks/deepeval/goldens.json  (input / expected_output / context / additional_metadata)
Ragas      -> frameworks/ragas/testset.jsonl    (user_input / reference / reference_contexts)
PyRIT      -> frameworks/pyrit/seed_prompts.yaml (adversarial seeds with harm categories)
Also writes the suite as JSONL (and Parquet when pyarrow is available).
"""
from __future__ import annotations

import json
import os


def _yaml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # JSON strings are valid YAML scalars


def export_all(suite, kb, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = [t.to_json() for t in suite]
    with open(f"{out_dir}/suite.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        import pandas as pd
        pd.DataFrame(rows).astype({"assertions": str, "contexts": str, "attachments": str, "meta": str}).to_parquet(f"{out_dir}/suite.parquet")
    except Exception:
        pass  # pyarrow/fastparquet not installed

    ctx = lambda t: [kb.chunks[c].text for c in t.contexts]

    # promptfoo -------------------------------------------------------------
    d = f"{out_dir}/promptfoo"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/tests.yaml", "w", encoding="utf-8") as fh:
        for t in suite:
            fh.write(f"- description: {_yaml_str(t.id + ' ' + t.dimension + '/' + t.subtype)}\n")
            fh.write(f"  vars:\n    query: {_yaml_str(t.query)}\n    attachments: {json.dumps(t.attachments, ensure_ascii=False)}\n")
            fh.write("  assert:\n")
            for a in t.assertions:
                fh.write(f"    - type: {a['type']}\n      value: {json.dumps(a['value'], ensure_ascii=False)}\n")
            fh.write(f"  metadata: {json.dumps({'dimension': t.dimension, 'subtype': t.subtype, 'persona': t.persona})}\n")

    # Inspect AI ------------------------------------------------------------
    d = f"{out_dir}/inspect"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/dataset.jsonl", "w", encoding="utf-8") as fh:
        for t in suite:
            fh.write(json.dumps({"id": t.id, "input": t.query or " ", "target": t.ground_truth or t.expected_behavior,
                                 "metadata": {"assertions": t.assertions, "attachments": t.attachments,
                                              "dimension": t.dimension, "subtype": t.subtype}}, ensure_ascii=False) + "\n")

    # DeepEval ----------------------------------------------------------------
    d = f"{out_dir}/deepeval"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/goldens.json", "w", encoding="utf-8") as fh:
        json.dump([{"input": t.query, "expected_output": t.ground_truth or t.expected_behavior, "context": ctx(t),
                    "additional_metadata": {"id": t.id, "assertions": t.assertions, "attachments": t.attachments,
                                            "dimension": t.dimension, "subtype": t.subtype}} for t in suite],
                  fh, ensure_ascii=False, indent=1)

    # Ragas -------------------------------------------------------------------
    d = f"{out_dir}/ragas"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/testset.jsonl", "w", encoding="utf-8") as fh:
        for t in suite:
            if t.expected_behavior == "answer":
                fh.write(json.dumps({"user_input": t.query, "reference": t.ground_truth, "reference_contexts": ctx(t),
                                     "synthesizer_name": f"sdgtest:{t.dimension}/{t.subtype}"}, ensure_ascii=False) + "\n")

    # PyRIT -------------------------------------------------------------------
    d = f"{out_dir}/pyrit"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/seed_prompts.yaml", "w", encoding="utf-8") as fh:
        fh.write("dataset_name: sdgtest_campus_adversarial\nprompts:\n")
        for t in suite:
            if t.dimension in ("adversarial", "privacy"):
                harm = "prompt_injection" if t.dimension == "adversarial" else "privacy"
                fh.write(f"  - value: {_yaml_str(t.query)}\n    data_type: text\n    harm_categories: [{harm}]\n"
                         f"    metadata: {{id: {t.id}, subtype: {t.subtype}}}\n")
    return out_dir
