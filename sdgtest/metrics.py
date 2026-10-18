"""Coverage, diversity and statistics helpers."""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .textutil import tokens


def distinct_n(queries, n):
    grams, total = set(), 0
    for q in queries:
        t = tokens(q)
        g = list(zip(*[t[i:] for i in range(n)]))
        grams.update(g); total += len(g)
    return len(grams) / total if total else 0.0


def semantic_spread(queries, max_n=600, seed=0):
    qs = [q for q in queries if q.strip()]
    if len(qs) > max_n:
        qs = random.Random(seed).sample(qs, max_n)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(qs)
    S = cosine_similarity(X)
    iu = np.triu_indices(len(qs), 1)
    return float(1 - S[iu].mean())


def cell_coverage(suite, cells):
    seen = {(tc.dimension, tc.ftype or tc.subtype) for tc in suite}
    return len(seen & cells) / len(cells)


def fact_coverage(suite, kb):
    touched = set()
    for tc in suite:
        subj = (tc.meta or {}).get("subject")
        subs = (tc.meta or {}).get("subjects", [])
        for f in kb.facts.values():
            if f.ftype == tc.ftype and (f.subject == subj or f.subject in subs) and f.chunk_id in tc.contexts:
                touched.add(f.fid)
    return len(touched) / len(kb.facts)


def bootstrap_ci(x, b=5000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (b, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def a12(x, y):
    """Vargha-Delaney A12: probability that a value from x exceeds one from y."""
    gt = sum(1 for a in x for b in y if a > b)
    eq = sum(1 for a in x for b in y if a == b)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def wilcoxon_p(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def shapley(players, value):
    """Exact Shapley values for a small cooperative game value(frozenset)."""
    n = len(players)
    phi = {}
    for p in players:
        others = [q for q in players if q != p]
        s = 0.0
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                s += w * (value(frozenset(S) | {p}) - value(frozenset(S)))
        phi[p] = s
    return phi
