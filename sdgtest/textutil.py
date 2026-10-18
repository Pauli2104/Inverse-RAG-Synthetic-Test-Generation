"""Small, dependency-free text utilities shared by the SUT and the harness."""
from __future__ import annotations

import base64
import binascii
import codecs
import math
import re
import unicodedata
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = set("""a an the of in on at to for is are was were be been what which who whom whose how
when where do does did can could would should may might will shall i you me my we our it its
this that these those please tell give show with by from and or as about there their them
""".split())

HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "ɡ": "g", "ո": "n", "ʀ": "r", "ⅼ": "l", "Ι": "I", "Α": "A", "Ε": "E",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Ѕ": "S", "Ν": "N", "Μ": "M", "Κ": "K", "Η": "H",
})
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOP]


def deobfuscate_views(text: str) -> list[str]:
    """Return alternative readings of a string used by the input guardrail."""
    views = [text]
    t = unicodedata.normalize("NFKC", text).translate(HOMOGLYPHS)
    views.append(t)
    views.append(t.translate(LEET))
    # collapse s p a c e d   l e t t e r s
    views.append(re.sub(r"(?<=\b\w) (?=\w\b)", "", t))
    views.append(codecs.decode(t, "rot13"))
    for m in re.findall(r"[A-Za-z0-9+/=]{16,}", text):
        try:
            dec = base64.b64decode(m + "=" * (-len(m) % 4), validate=False).decode("utf-8", "ignore")
            if dec and sum(ch.isprintable() for ch in dec) / len(dec) > 0.9:
                views.append(dec)
        except (binascii.Error, ValueError):
            pass
    return views


def edit1(a: str, b: str) -> bool:
    """True if a and b are within Damerau edit distance 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return True
        return False
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]:
            return True
    return False


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.4, b: float = 0.75):
        self.ids = list(docs)
        self.toks = {i: content_tokens(docs[i]) for i in self.ids}
        self.tf = {i: Counter(t) for i, t in self.toks.items()}
        self.avgdl = sum(len(t) for t in self.toks.values()) / max(1, len(self.ids))
        df = Counter()
        for t in self.toks.values():
            df.update(set(t))
        n = len(self.ids)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        q = content_tokens(query)
        scores = []
        for i in self.ids:
            tf, dl, s = self.tf[i], len(self.toks[i]), 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def char_ngrams(s: str, n: int = 3) -> set:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0
