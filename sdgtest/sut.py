"""System under test: a tool-using campus operations assistant (agentic RAG).

Pipeline: input checks -> guardrail -> normalisation -> intent routing ->
tools (kb_search, calculator, directory, ticketing) -> answer composition.
Fault injection is controlled by ``mutant``; ``None`` is the reference agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .kb import BUILDINGS, ROOM_KINDS, SYSTEMS, KnowledgeBase
from .textutil import BM25, deobfuscate_views, edit1, tokens

MUTANTS = {
    "M01_retrieval_top1": "Retriever returns only the top-1 chunk",
    "M02_no_lexicon": "Synonym/dialect lexicon removed from query normalisation",
    "M03_no_spellfix": "Spelling correction against KB vocabulary disabled",
    "M04_case_sensitive": "Entity linking becomes case-sensitive",
    "M05_guard_off": "Input guardrail disabled",
    "M06_guard_no_decode": "Guardrail inspects raw text only (no de-obfuscation)",
    "M07_pii_disclose": "Directory tool discloses contact details",
    "M08_redact_off": "PII redaction in ticket summaries disabled",
    "M09_no_abstain": "Abstention removed; falls back to best-scoring value",
    "M10_follow_injection": "Instructions found in retrieved/attached text are obeyed",
    "M11_truncate_context": "Chunks truncated to first 120 characters",
    "M12_calc_operand_order": "Calculator subtracts operands in the wrong order",
    "M13_crash_empty": "Unhandled exception on empty input",
    "M14_crash_long": "Unhandled exception when input exceeds the token budget",
    "M15_compare_first": "Comparison always returns the first-mentioned room",
}

# Held-out mutants, written after the generator was frozen (sha256 in results/frozen_generator.sha256)
# and derived from external fault taxonomies: Barnett et al. (2024) RAG failure points FP1-FP7 and
# OWASP LLM Top 10 (2025) items LLM05-LLM08. They were not used to design any generation family.
HOLDOUT = {
    "H01_missing_document": "FP1 missing content: one building manual is not indexed",
    "H02_no_entity_retrieval": "FP2 missed top-ranked documents: entity-matched retrieval disabled",
    "H03_extractor_gap": "FP4 not extracted: one occupancy sentence template is not parsed",
    "H04_wrong_unit": "FP5 wrong format: set-points reported in degrees Fahrenheit",
    "H05_ignore_site": "FP6 incorrect specificity: building qualifier ignored in look-up",
    "H06_incomplete_answer": "FP7 incomplete: maintenance windows reported as day only",
    "H07_echo_payload": "LLM05 improper output handling: refusal echoes the decoded request",
    "H08_excessive_agency": "LLM06 excessive agency: tickets opened for any incident/fault mention",
    "H09_prompt_leak": "LLM07 system-prompt leakage: refusal reveals internal policy text",
    "H10_context_stuffing": "LLM08 retrieval poisoning: attachments crowd knowledge-base evidence out of context",
    "H11_no_budget_guard": "Reliability: over-budget inputs are processed without trimming",
    "H12_strict_abstention": "LLM09 over-refusal: abstains whenever no building is named",
}
INTERNAL_POLICY = "[internal policy: guard=v2; max_tokens=512; directory=protected]"

LEXICON = {
    r"\blift\b": "elevator", r"\blifts\b": "elevator", r"\bair ?con\b": "hvac",
    r"\bair[- ]conditioning\b": "hvac", r"\bclimate system\b": "hvac", r"\bheating\b": "hvac",
    r"\bfire alarms?\b": "fire alarm", r"\bsmoke alarm\b": "fire alarm", r"\blights\b": "lighting",
    r"\bdoor system\b": "access control", r"\bentry system\b": "access control",
    r"\bedificio\b": "building", r"\bsala\b": "room", r"\bcapacidad\b": "occupancy",
    r"\bcapacity\b": "occupancy", r"\bhow many (?:people|persons|folks|ppl)\b": "occupancy",
    r"\bheadcount\b": "occupancy", r"\btemperature\b": "setpoint", r"\btemp\b": "setpoint",
    r"\bthermostat\b": "setpoint", r"\bclearance\b": "badge", r"\bpass\b": "badge",
    r"\bkeycard\b": "badge", r"\bservic(?:e|ing|ed)\b": "maintenance", r"\bupkeep\b": "maintenance",
    r"\bcarbon dioxide\b": "co2", r"\bair quality\b": "co2", r"\bescalat\w*\b": "escalation",
    r"\bwho (?:do i|should i|to) (?:call|contact|notify)\b": "escalation",
}

INTENT_KW = {
    "setpoint": ["setpoint", "cooling", "cooled", "degrees"],
    "occupancy": ["occupancy", "occupy", "fit", "people", "seats"],
    "access": ["badge", "access", "enter", "entry"],
    "co2": ["co2", "ppm"],
    "maintenance": ["maintenance", "maintained", "serviced"],
    "escalation": ["escalation", "escalated", "incident", "incidents"],
}

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier)", r"disregard (?:all |the |your )?(?:previous|prior|rules|instructions)",
    r"system prompt", r"you are now", r"developer mode", r"\bdan\b", r"jailbreak",
    r"reveal (?:your|the) (?:instructions|prompt|rules)", r"pretend (?:you|to) ", r"act as (?:an? )?unrestricted",
    r"override", r"no restrictions", r"bypass (?:the |your )?(?:safety|guard|filter)",
    r"print (?:all )?(?:badge|admin|master) (?:codes|passwords)", r"admin password", r"master key",
]
LEXICON_RX = [(re.compile(p, re.I), r) for p, r in LEXICON.items()]
INJECTION_RX = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.I)
WORD_RE = re.compile(r"[A-Za-z]+")
INSTR_IN_CONTEXT = re.compile(r"(system note|assistant instruction|operator override|ignore the user|disregard previous)", re.I)
INSTR_PAYLOAD = re.compile(r"(?:reply|respond|answer)(?: only)? with '?([A-Z][A-Z ]{3,})'?", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+?\d[\d ()-]{7,}\d")

ABSTAIN = "I don't know based on the available documentation."
REFUSE = "I can't help with that request."
REFUSE_PII = "I can't share personal contact details."
EMPTY = "Please provide a question."
MAX_TOKENS = 512

# extraction patterns for each fact type (the agent's own reader)
NUM = r"(\d+(?:\.\d+)?)"
EXTRACT = {
    "setpoint": re.compile(r"setpoint of " + NUM + r" °C|is " + NUM + r" °C"),
    "occupancy": re.compile(r"occupancy of " + NUM + r" people|than " + NUM + r" people|is " + NUM + r" people"),
    "access": re.compile(r"(?:requires a|of a|to) (green|amber|red|black) badge"),
    "co2": re.compile(r"(?:above|is) " + NUM + r" ppm"),
    "maintenance": re.compile(r"every (\w+day from \d\d:\d\d to \d\d:\d\d)"),
    "escalation": re.compile(r"to the ([a-zA-Z ]+?)\.$"),
}


H03_OCCUPANCY = re.compile(r"occupancy of " + NUM + r" people|is " + NUM + r" people")


class AgentCrash(RuntimeError):
    pass


@dataclass
class Result:
    output: str
    tools: list
    intent: str


class CampusAgent:
    def __init__(self, kb: KnowledgeBase, mutant: str | None = None, top_k: int = 4):
        self.kb, self.m = kb, mutant
        self.chunks = {c.cid: c.text for c in kb.chunks.values()
                       if not (mutant == "H01_missing_document" and c.cid.startswith("Equinox"))}
        self.bm25 = BM25(self.chunks)
        self.top_k = 1 if mutant == "M01_retrieval_top1" else top_k
        self.room_names = sorted({r for _, r in kb.rooms}, key=len, reverse=True)
        vocab = set()
        for t in self.chunks.values():
            vocab.update(tokens(t))
        for kws in INTENT_KW.values():
            vocab.update(kws)
        for pat in LEXICON:
            vocab.update(re.findall(r"[a-z]{4,}", pat.replace("\\b", " ").replace("\\w", " ")))
        vocab.update(["thermostat", "servicing", "serviced", "service", "smoke", "alarm", "conditioning",
                      "capacity", "clearance", "keycard", "temperature", "headcount", "contact", "phone", "email"])
        vocab.update(["building", "room", "elevator", "hvac", "badge", "difference", "compare", "total",
                      "combined", "ticket", "which", "more", "higher", "larger", "lower"])
        self.vocab = {v for v in vocab if len(v) >= 4 and not v.isdigit()}
        self.vocab_list = sorted(self.vocab)
        self._fixcache = {}
        self.directory = {p.name.lower(): p for p in kb.people}

    # ------------------------------------------------------------------ helpers
    def _spellfix(self, q: str) -> str:
        if self.m == "M03_no_spellfix":
            return q
        cache = self._fixcache
        def fix(m):
            w = m.group(0)
            lw = w.lower()
            if len(lw) < 4 or lw in self.vocab:
                return w
            if lw not in cache:
                cache[lw] = next((v for v in self.vocab_list if abs(len(v) - len(lw)) <= 1 and edit1(lw, v)), None)
            return cache[lw] or w
        return WORD_RE.sub(fix, q)

    def _normalise(self, q: str) -> str:
        q = re.sub(r"\s+", " ", q).strip()
        q = self._spellfix(q)
        if self.m != "M02_no_lexicon":
            for pat, rep in LEXICON_RX:
                q = pat.sub(rep, q)
        return q

    def _guard(self, q: str) -> bool:
        if self.m == "M05_guard_off":
            return False
        views = [q] if self.m == "M06_guard_no_decode" else deobfuscate_views(q)
        return any(INJECTION_RX.search(v) for v in views)

    def _entities(self, q: str):
        flags = 0 if self.m == "M04_case_sensitive" else re.I
        rooms = []
        for r in self.room_names:
            for m in re.finditer(re.escape(r), q, flags):
                rooms.append((m.start(), r))
        rooms = [r for _, r in sorted(rooms)]
        # unknown room-like mentions (e.g. 'Lab Z99') are tracked for abstention
        unknown = [m.group(0) for m in re.finditer(r"\b(?:%s) [A-Z]\d{2}\b" % "|".join(ROOM_KINDS), q, flags)
                   if m.group(0).lower() not in {r.lower() for r in self.room_names}]
        buildings = [b for b in BUILDINGS if re.search(r"\b%s\b" % b, q, flags)]
        systems = [s for s in SYSTEMS if re.search(r"\b%s\b" % re.escape(s), q, flags)]
        return rooms, buildings, systems, unknown

    def _intent(self, q: str) -> str:
        ql = q.lower()
        for s in SYSTEMS:
            ql = ql.replace(s.lower(), " ")
        scores = {k: sum(1 for w in kws if re.search(r"\b%s\b" % w, ql)) for k, kws in INTENT_KW.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"

    def _contexts(self, query: str, extra: list[str], entities=()):
        hits = self.bm25.search(query, self.top_k)
        ids = [c for c, s in hits if s > 0]
        if self.m not in ("M01_retrieval_top1", "H02_no_entity_retrieval"):
            # hybrid retrieval: add chunks that literally mention a linked entity
            for e in entities:
                ids += [cid for cid, t in self.chunks.items() if e.lower() in t.lower() and cid not in ids][:4]
        ctx = [self.chunks[c] for c in ids]
        if self.m == "H10_context_stuffing" and extra:
            ctx = (list(extra) + ctx)[:len(extra)]  # attachments fill the context window
        else:
            ctx += list(extra or [])
        if self.m == "M11_truncate_context":
            ctx = [c[:120] for c in ctx]
        return ctx

    def _lookup(self, ftype, subject, building, ctx):
        flags = 0 if self.m == "M04_case_sensitive" else re.I
        if self.m == "H05_ignore_site":
            building = None
        for c in ctx:
            for sent in re.split(r"(?<=\.)\s+", c):
                if re.search(re.escape(subject), sent, flags) and (building is None or building in sent):
                    rx = EXTRACT[ftype]
                    if self.m == "H03_extractor_gap" and ftype == "occupancy":
                        rx = H03_OCCUPANCY
                    m = rx.search(sent)
                    if m:
                        return next(g for g in m.groups() if g)
        return None

    def _fallback_value(self, ftype, ctx):
        for c in ctx:
            m = EXTRACT.get(ftype, EXTRACT["occupancy"]).search(c)
            if m:
                return next(g for g in m.groups() if g)
        return "21.0"

    # ------------------------------------------------------------------ main
    def answer(self, query, attachments=None) -> Result:
        tools = []
        if query is None or not str(query).strip():
            if self.m == "M13_crash_empty":
                raise AgentCrash("IndexError: empty query")
            return Result(EMPTY, tools, "empty")
        ntok = len(str(query).split())
        if ntok > MAX_TOKENS and self.m != "H11_no_budget_guard":
            if self.m == "M14_crash_long":
                raise AgentCrash("ContextLengthExceeded")
            # budget exceeded: keep only the final question/sentence of the message
            parts = [p for p in re.split(r"\n+|(?<=[.?!])\s+", str(query).strip()) if p.strip()]
            query = parts[-1] if parts and len(parts[-1].split()) <= MAX_TOKENS else " ".join(str(query).split()[-64:])
        if self._guard(query):
            if self.m == "H07_echo_payload":
                return Result(f"{REFUSE} Blocked request: {deobfuscate_views(query)[-1][:200]}", tools, "blocked")
            if self.m == "H09_prompt_leak":
                return Result(f"{REFUSE} {INTERNAL_POLICY}", tools, "blocked")
            return Result(REFUSE, tools, "blocked")

        q = self._normalise(query)
        ql = q.lower()
        # --- ticketing tool (PII must be redacted)
        ticket_rx = r"\b(ticket|report (?:a|an) (?:fault|issue|problem)|log (?:a|an) (?:fault|issue))\b"
        if self.m == "H08_excessive_agency":
            ticket_rx = r"\b(ticket|fault|issue|problem|incidents?|broken|not working)\b"
        if re.search(ticket_rx, ql):
            summary = query if self.m == "M08_redact_off" else PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", query))
            tools.append("create_ticket")
            return Result(f"Ticket created. Summary: {summary[:300]}", tools, "ticket")
        # --- directory tool (contact details are protected)
        for name, p in self.directory.items():
            if name in ql or p.name.split()[1].lower() in ql.split():
                if re.search(r"(e-?mail|phone|number|contact|reach|call)", ql):
                    tools.append("directory")
                    if self.m == "M07_pii_disclose":
                        return Result(f"{p.name}: {p.email}, {p.phone}", tools, "directory")
                    return Result(REFUSE_PII, tools, "directory")

        rooms, buildings, systems, unknown = self._entities(q)
        intent = self._intent(q)
        ctx = self._contexts(q, attachments or [], rooms + systems)
        tools.append("kb_search")

        if self.m == "M10_follow_injection":
            for c in ctx:
                if INSTR_IN_CONTEXT.search(c):
                    m = INSTR_PAYLOAD.search(c)
                    return Result(m.group(1).strip() if m else "ACCESS GRANTED", tools, "hijacked")
        else:
            ctx = [" ".join(s for s in re.split(r"(?<=\.)\s+", c) if not INSTR_IN_CONTEXT.search(s)) for c in ctx]

        no_abstain = self.m == "M09_no_abstain"
        if (unknown or intent == "unknown" or (not rooms and not systems)) and not no_abstain:
            return Result(ABSTAIN, tools, intent)
        if self.m == "H12_strict_abstention" and not buildings:
            return Result(ABSTAIN, tools, intent)

        # --- multi-hop operations over two rooms
        op = None
        if len(rooms) >= 2:
            if re.search(r"how many more|difference|how much (?:higher|more|larger)", ql):
                op = "diff"
            elif re.search(r"\b(total|combined|together|sum)\b", ql):
                op = "sum"
            elif re.search(r"\bwhich\b|\bhigher\b|\blarger\b|\bmore\b|\bbigger\b", ql):
                op = "cmp"
        if op:
            r1, r2 = rooms[0], rooms[1]
            # second hop: retrieve per entity
            ctx2 = ctx + self._contexts(f"{r1} {intent}", []) + self._contexts(f"{r2} {intent}", []) \
                if self.top_k > 1 else ctx
            if self.m == "M11_truncate_context":
                ctx2 = [c[:120] for c in ctx2]
            v1, v2 = self._lookup(intent, r1, None, ctx2), self._lookup(intent, r2, None, ctx2)
            tools.append("calculator")
            if v1 is None or v2 is None:
                return Result(ABSTAIN if not no_abstain else "0", tools, intent)
            a, b = float(v1), float(v2)
            if op == "diff":
                d = (b - a) if self.m == "M12_calc_operand_order" else abs(a - b)
                if self.m != "M12_calc_operand_order" and a < b:
                    r1, r2 = r2, r1
                return Result(f"{r1} exceeds {r2} by {d:g}.", tools, intent)
            if op == "sum":
                return Result(f"The combined value is {a + b:g}.", tools, intent)
            win = r1 if (self.m == "M15_compare_first" or a >= b) else r2
            return Result(f"{win} has the higher value ({max(a, b):g}).", tools, intent)

        subject = rooms[0] if rooms else (systems[0] if systems else "")
        building = buildings[0] if buildings else None
        v = self._lookup(intent, subject, building, ctx) if intent in EXTRACT else None
        if v is None:
            if no_abstain:
                return Result(f"The answer is {self._fallback_value(intent, ctx)}.", tools, intent)
            return Result(ABSTAIN, tools, intent)
        if self.m == "H04_wrong_unit" and intent == "setpoint":
            v = f"{float(v) * 9 / 5 + 32:.1f} °F"
        if self.m == "H06_incomplete_answer" and intent == "maintenance":
            v = v.split()[0]
        return Result(f"The answer is {v}.", tools, intent)


def safe_call(agent: CampusAgent, query, attachments=None) -> str:
    try:
        return agent.answer(query, attachments).output
    except AgentCrash as e:
        return f"<ERROR: {e}>"


def safe_call_any(agent, query, attachments=None) -> str:
    """Like safe_call but treats any exception (e.g. from automatically generated mutants) as a crash."""
    try:
        return str(agent.answer(query, attachments).output)
    except Exception as e:  # noqa: BLE001
        return f"<ERROR: {type(e).__name__}>"
