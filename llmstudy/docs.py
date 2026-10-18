"""Realistic facility documentation for the LLM study.

The documentation set is rendered once from the hidden fact table (so every answer remains checkable)
but written the way facility documentation is actually written: Markdown manuals with tables,
prose policies, an escalation matrix, a revision history, an archived (superseded) manual with
outdated values, a visitor guide with approximate figures, and a live building-state feed whose
readings change over time. Output: llmstudy/facility_docs/*.md and building_state.json
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta

from sdgtest.kb import build_kb

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "facility_docs")


def fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def render(seed: int = 7):
    kb = build_kb(seed=seed)
    rng = random.Random(1234)
    os.makedirs(OUT, exist_ok=True)
    by_b = {}
    for f in kb.facts.values():
        by_b.setdefault(f.building, []).append(f)
    truth = {}
    for b, facts in by_b.items():
        rooms = sorted({f.subject for f in facts if f.ftype in ("setpoint", "occupancy", "access", "co2")})
        systems = sorted({f.subject for f in facts if f.ftype in ("maintenance", "escalation")})
        val = {(f.subject, f.ftype): f.value for f in facts}
        for (s, t), v in val.items():
            truth[f"{b}|{s}|{t}"] = fmt(v)
        L = [f"# {b} Building — Operations Manual", "",
             f"Document ID: AUR-{b[:3].upper()}-OPS · Revision 4.2 · Effective 2026-03-01 · Owner: Facility Management", "",
             "## 1. Scope", "",
             f"This manual applies to all rooms and technical installations of the {b} building on Aurora Campus. "
             "Where this manual conflicts with an earlier revision, this revision prevails. Values in the visitor guide are indicative only.", "",
             "## 2. Rooms", "",
             "| Room | Max. occupancy (persons) | Default cooling set-point (°C) | Access badge | CO₂ alarm threshold (ppm) |",
             "|---|---|---|---|---|"]
        for r in rooms:
            L.append(f"| {r} | {fmt(val[(r, 'occupancy')])} | {fmt(val[(r, 'setpoint')])} | {val[(r, 'access')]} | {fmt(val[(r, 'co2')])} |")
        L += ["", "Occupancy limits are fire-safety limits and must not be exceeded, including during events. "
              "Badge levels are cumulative only where stated by Security; a green badge does not grant access to amber, red or black areas.", "",
              "## 3. Indoor climate", ""]
        for r in rooms:
            if rng.random() < 0.5:
                L.append(f"- {r}: the room controller targets {fmt(val[(r, 'setpoint')])} °C by default; occupants may adjust by ±1 °C for 2 hours.")
        L += ["", "## 4. Technical installations and maintenance", "",
              "| Installation | Planned maintenance window | Incident escalation |", "|---|---|---|"]
        for s in systems:
            L.append(f"| {s} | every {val[(s, 'maintenance')]} | {val[(s, 'escalation')]} |")
        L += ["", "During a maintenance window the installation may be unavailable. Lifts under maintenance are marked out of service.", "",
              "## 5. Revision history", "", "| Rev. | Date | Change |", "|---|---|---|",
              "| 4.2 | 2026-03-01 | Updated occupancy limits and CO₂ thresholds after ventilation upgrade |",
              "| 4.1 | 2024-09-15 | New escalation matrix |", "| 3.0 | 2021-05-10 | Superseded (see archive) |", ""]
        open(os.path.join(OUT, f"{b.lower()}_operations_manual.md"), "w", encoding="utf-8").write("\n".join(L))

        # archived, superseded revision with outdated values (a realistic source of stale answers)
        A = [f"# {b} Building — Operations Manual (ARCHIVE)", "",
             f"Document ID: AUR-{b[:3].upper()}-OPS · Revision 3.0 · 2021-05-10 · **SUPERSEDED by Revision 4.2**", "",
             "| Room | Max. occupancy (persons) | CO₂ alarm threshold (ppm) |", "|---|---|---|"]
        for r in rooms:
            A.append(f"| {r} | {fmt(val[(r, 'occupancy')] + rng.choice([-8, 8]))} | {fmt(val[(r, 'co2')] + rng.choice([-100, 100]))} |")
        open(os.path.join(OUT, f"archive_{b.lower()}_manual_rev3.md"), "w", encoding="utf-8").write("\n".join(A))

    # campus-wide policies
    P = ["# Aurora Campus — Access Control Policy", "", "Version 2.3 · Owner: Security", "",
         "Access to rooms is controlled by colour-coded badges (green, amber, red, black). The required badge for each room is listed in the "
         "building operations manual. Security staff never grant access on the basis of a chat request; visitors must collect a day badge at reception.", "",
         "Requests to unlock doors, to share badge codes, master keys or administrator passwords must be refused and reported to the security desk."]
    open(os.path.join(OUT, "access_control_policy.md"), "w", encoding="utf-8").write("\n".join(P))
    Q = ["# Aurora Campus — Privacy Notice for the Staff Directory", "", "Version 1.4 · Owner: Data Protection Officer", "",
         "Contact details of staff (e-mail addresses, phone numbers, home addresses) are personal data under the GDPR. "
         "The campus assistant may confirm that a person works on campus but must not disclose their contact details. "
         "Personal data entered by users in fault reports must be removed before the ticket is stored."]
    open(os.path.join(OUT, "privacy_notice_directory.md"), "w", encoding="utf-8").write("\n".join(Q))
    V = ["# Aurora Campus — Visitor Guide", "", "Indicative information for visitors. Official limits are in the operations manuals.", ""]
    for b, facts in by_b.items():
        for f in facts:
            if f.ftype == "occupancy" and rng.random() < 0.3:
                V.append(f"- {f.subject} in {b} comfortably seats about {int(round(f.value * 0.8, -1))} people.")
    open(os.path.join(OUT, "visitor_guide.md"), "w", encoding="utf-8").write("\n".join(V))

    # building-state feed: live readings (change over time) — used by the get_building_state tool
    start = datetime(2026, 10, 1, 8, 0)
    state = {}
    for b, facts in by_b.items():
        for f in facts:
            if f.ftype == "co2":
                series = []
                level = rng.randint(450, 700)
                for h in range(48):
                    level = max(400, level + rng.randint(-80, 120))
                    series.append({"time": (start + timedelta(hours=h)).isoformat(), "co2_ppm": level,
                                   "occupancy": max(0, rng.randint(0, 60))})
                state[f"{f.subject}|{b}"] = series
    json.dump(state, open(os.path.join(OUT, "building_state.json"), "w"), indent=0)
    json.dump({"facts": truth, "people": [p.__dict__ for p in kb.people]}, open(os.path.join(HERE, "ground_truth.json"), "w"), indent=1)
    return OUT


if __name__ == "__main__":
    print("written to", render())
