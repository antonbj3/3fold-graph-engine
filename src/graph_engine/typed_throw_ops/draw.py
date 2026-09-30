"""draw.py — drawing triples and chains with KNOWN inclusion probabilities.

The three distances of throws.py are kept apart, now for the two ENDS of a triple (A, C) — the pair the middle part B
is supposed to join:
    graph     effective resistance between the reports of A and C in the corpus reference graph (report → report
              mentions of U###/A###/P1–P4/FB_* ids and engine module names). Mentions are structural metadata, not
              physics. Reports in different components get R = ∞ and are flagged; for scoring the log term is capped.
    subject   word distance of the two parts' prose (signature.word_distance) — a negative filter, as in throws.py
    mechanism the triple's own value (value.py): complementarity bits of {A,B,C} for the best target, and whether
              A and C compose directly (a direct link makes B redundant)
Score (declared, heuristic, only the SAMPLER's input — the probabilities are exact given the score):
    s = w_v · min(max_complementarity, cap) + a · log(min(R_AC, R_cap)/median R) + b · log(subject_AC + ε)
        − w_d · [A,C compose directly] − w_w · [any member reach 'words']
Draw: throws.draw_pairs over the enumerated universe (triple t ↦ pair (t, 0)): systematic sampling with exact
inclusion probabilities π_t, softmax temperature T and floor λ, so every triple in the declared universe has π > 0
and a 1/π-weighted fit of outcomes is valid (throws.py, e8b). The universe is declared per run and recorded.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .engine import draw_pairs, resistance_sketch

ID_RE = re.compile(r"\b(U\d{3}|A\d{3}|FB_[A-Z0-9_]+?_2026\d{4}|P[1-4]\b)")


class CorpusGraph:
    def __init__(self, mentions: dict[str, set[str]], k: int = 64, seed: int = 0):
        nodes = sorted(set(mentions) | {m for v in mentions.values() for m in v})
        self.idx = {n: i for i, n in enumerate(nodes)}
        edges = sorted({tuple(sorted((self.idx[a], self.idx[b]))) for a, v in mentions.items() for b in v if a != b})
        self.n, self.edges = len(nodes), np.array(edges, np.int64).reshape(-1, 2)
        A = sp.coo_matrix((np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])), shape=(self.n, self.n))
        ncomp, self.comp = connected_components(A + A.T, directed=False)
        self.sketch = {}
        for c in range(ncomp):
            members = np.flatnonzero(self.comp == c)
            if len(members) < 2:
                continue
            loc = {g: i for i, g in enumerate(members)}
            e = np.array([(loc[a], loc[b]) for a, b in self.edges if self.comp[a] == c], np.int64)
            sk = resistance_sketch.ResistanceSketch.build(len(members), e, k=k, seed=seed)
            self.sketch[c] = (loc, sk)
        self.ncomp = ncomp

    def resistance(self, r1: str, r2: str) -> float:
        if r1 == r2:
            return 0.0
        if r1 not in self.idx or r2 not in self.idx:
            return float("inf")
        i, j = self.idx[r1], self.idx[r2]
        if self.comp[i] != self.comp[j] or self.comp[i] not in self.sketch:
            return float("inf")
        loc, sk = self.sketch[self.comp[i]]
        return float(sk.resistance(np.array([loc[i]]), np.array([loc[j]]))[0])


def score_terms(R_ac: float, subj_ac: float, comp_bits: float, direct_ac: bool, any_words: bool, medR: float,
                w_v: float = 1.0, a: float = 1.0, b: float = 1.0, w_d: float = 2.0, w_w: float = 1.0,
                cap_bits: float = 6.0, R_cap_factor: float = 10.0, eps: float = 1e-3) -> dict:
    R = min(R_ac, R_cap_factor * medR) if np.isfinite(R_ac) else R_cap_factor * medR
    t = {"value": w_v * min(max(comp_bits, 0.0), cap_bits),
         "graph": a * math.log(max(R, 1e-9) / medR),
         "subject": b * math.log(subj_ac + eps),
         "direct_penalty": -w_d * float(direct_ac),
         "words_penalty": -w_w * float(any_words)}
    t["score"] = sum(t.values())
    t["R_ac"] = R_ac
    t["R_ac_infinite"] = not np.isfinite(R_ac)
    return t


def draw_from_universe(scores: np.ndarray, k: int, temperature: float = 1.0, floor: float = 0.1, seed: int = 0):
    """k members of the universe as (index, π) with exact inclusion probabilities (throws.draw_pairs)."""
    ii = np.arange(len(scores)); jj = np.zeros(len(scores), int)
    got = draw_pairs(ii, jj, np.asarray(scores, float), k, temperature=temperature, floor=floor, seed=seed)
    return [(int(t), float(pi)) for t, _, pi in got]


def all_inclusion_probabilities(scores: np.ndarray, k: int, temperature: float = 1.0, floor: float = 0.1) -> np.ndarray:
    """π for EVERY member of the universe (the same computation draw_pairs does internally)."""
    from .engine import inclusion_probabilities
    s = np.asarray(scores, float) / temperature
    p = np.exp(s - s.max()); p = (1 - floor) * p / p.sum() + floor / len(p)
    return inclusion_probabilities(p, min(k, len(p)))
