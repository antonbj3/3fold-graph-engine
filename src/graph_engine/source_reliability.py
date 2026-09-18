#!/usr/bin/env python3
"""
source_reliability.py — how often each independent ORIGIN is right, estimated from agreement alone.

claim_federation and regime_posterior use one fixed reliability r for every report (0.75 by default). Origins differ.
When several independent origins report on the same questions, their agreement identifies each origin's accuracy without
any labels (Dawid & Skene 1979; identifiable with ≥ 3 conditionally independent origins per question, which is what the
lineage tree already guarantees for distinct roots — copies of one root are collapsed to one vote first).

Model. Question j has a hidden true sign t_j ∈ {±1}; origin s answers a_sj = t_j with probability r_s, else −t_j.
EM: E-step P(t_j = +) ∝ π Π_s r_s^{[a_sj=+]} (1−r_s)^{[a_sj=−]}, M-step r_s = (Σ_j P(t_j = a_sj) + a − 1) / (n_s + a + b − 2)
with a Beta(a, b) prior on r_s (partial pooling toward the prior when an origin has answered few questions).
The result feeds Federation(reliability_by_root=...) so that a root's log-odds weight is log(r_s/(1−r_s)) instead of a common value.

Limits: origins must be independent given the truth (a shared misreading across origins is invisible, as everywhere else in
this repository); a question with one origin contributes nothing to r; symmetric errors (same rate on + and −) are assumed.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

__all__ = ["estimate", "root_votes"]


def root_votes(fed, at_point: dict | None = None) -> dict[tuple, dict[str, int]]:
    """{pair: {root: majority sign of that root's reports}} from a claim_federation.Federation (copies collapsed)."""
    votes: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for c in fed.claims.values():
        if at_point is not None and not all(lo <= at_point.get(v, lo) <= hi for v, (lo, hi) in c["box"].items()):
            continue
        for src in c["evidence"]:
            for r in fed.roots(src):
                votes[c["pair"]][r].append(c["sign"])
    return {p: {r: (1 if sum(v) > 0 else -1) for r, v in d.items() if sum(v) != 0} for p, d in votes.items()}


def estimate(votes: dict[tuple, dict[str, int]], prior=(3.0, 1.5), iters: int = 50, base: float = 0.5) -> dict[str, dict]:
    """votes: {question: {origin: ±1}} → {origin: {"r": posterior mean, "n": questions answered, "alpha", "beta"}}."""
    a, b = prior
    qs = [q for q, d in votes.items() if len(d) >= 2]; origins = sorted({s for q in qs for s in votes[q]})
    r = {s: a / (a + b) for s in origins}
    for _ in range(iters):
        post = {}
        for q in qs:
            lo = math.log(base / (1 - base))
            for s, v in votes[q].items():
                lo += v * math.log(r[s] / (1 - r[s]))
            post[q] = 1 / (1 + math.exp(-lo))
        cnt, hit = defaultdict(float), defaultdict(float)
        for q in qs:
            for s, v in votes[q].items():
                cnt[s] += 1; hit[s] += post[q] if v > 0 else 1 - post[q]
        r = {s: (hit[s] + a - 1) / (cnt[s] + a + b - 2) for s in origins}
        r = {s: min(max(x, 0.02), 0.98) for s, x in r.items()}
    return {s: {"r": r[s], "n": int(cnt.get(s, 0)), "alpha": hit.get(s, 0) + a, "beta": cnt.get(s, 0) - hit.get(s, 0) + b} for s in origins}
