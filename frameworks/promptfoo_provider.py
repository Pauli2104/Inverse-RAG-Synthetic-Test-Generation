"""promptfoo Python provider wrapping the system under test."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdgtest.kb import build_kb  # noqa: E402
from sdgtest.sut import CampusAgent, safe_call  # noqa: E402

_AGENT = CampusAgent(build_kb(seed=7), os.environ.get("SDG_MUTANT") or None)


def call_api(prompt, options, context):
    v = (context or {}).get("vars", {})
    att = v.get("attachments") or []
    if isinstance(att, str):
        att = json.loads(att)
    return {"output": safe_call(_AGENT, v.get("query", prompt), att)}
