#!/usr/bin/env python3
"""
throws.py — propose NEW links: far apart in the graph, far apart in subject, close in mechanism; drawn with KNOWN probabilities.

Three distances, kept separate because they answer different questions:
    graph      resistance distance R_ij (resistance_sketch): how weakly the two nodes are already connected
    subject    1 − similarity of their descriptions (any text measure, optional): same field or not
    mechanism  mechanism_signature.distance, or any other computed signature distance: same way of behaving or not
A throw is worth making when the first two are LARGE and the third is SMALL:
        score_ij = −mechanism_ij / τ  +  a·log(R_ij / median R)  +  b·log(subject_ij + ε)

Choice rule. The K pairs are not the top K. They are drawn without replacement with probability
        π_ij = (1 − λ)·softmax(score / T)_ij + λ / N ,
and each drawn pair is returned with its inclusion probability. Reason (measured on two public citation graphs,
examples/engine_experiments/e8, e8b): when the outcome of a throw is known only for the throws that were made, top-K
selection finds the most links during the run but the model fitted on those outcomes ranks an unseen period at
AP 0.130, against 0.162 for uniformly random throws; softmax draws (T = 0.5) with a 1/π-weighted fit reach 0.163 while finding
4.7 times as many links as random (on the second graph: 2.9 times, 0.190 against 0.192). Top-K gives π ∈ {0, 1}: no weight can correct for pairs that could never be drawn.
"""
from __future__ import annotations

import numpy as np

__all__ = ["throw_scores", "draw", "draw_pairs", "inclusion_probabilities"]


def throw_scores(mechanism: np.ndarray, graph: np.ndarray, subject: np.ndarray | None = None,
                 tau: float = 0.25, a: float = 1.0, b: float = 1.0, eps: float = 1e-3) -> np.ndarray:
    """Scores for candidate pairs. Inputs are arrays of one shape: either n×n distance matrices (upper triangle used), or
    1-D arrays over a LIST of candidate pairs. Use the list form on large graphs — an n×n matrix at n = 27 400 is 6 GB —
    with distances taken per pair (resistance_sketch.resistance(i, j) needs no matrix)."""
    mechanism, graph = np.asarray(mechanism, float), np.asarray(graph, float)
    g = graph[np.triu_indices(len(graph), 1)] if graph.ndim == 2 else graph
    med = np.median(g[np.isfinite(g) & (g > 0)])
    s = -mechanism / tau + a * np.log(np.maximum(graph, 1e-12) / med)
    if subject is not None:
        s = s + b * np.log(subject + eps)
    return s


def inclusion_probabilities(p: np.ndarray, k: int) -> np.ndarray:
    """π_i = min(1, c·p_i) with c chosen so that Σπ = k (units that would exceed 1 are taken with certainty and the rest
    rescaled). These are the EXACT inclusion probabilities of `draw`."""
    p = np.asarray(p, float) / np.sum(p); pi = np.minimum(k * p, 1.0)
    for _ in range(len(p)):
        sure = pi >= 1.0; rest = k - sure.sum()
        if rest <= 0 or not (~sure).any():
            break
        new = np.where(sure, 1.0, np.minimum(rest * p / p[~sure].sum(), 1.0))
        if np.allclose(new, pi):
            break
        pi = new
    return pi


def draw_pairs(i: np.ndarray, j: np.ndarray, scores: np.ndarray, k: int, temperature: float = 1.0, floor: float = 0.05,
               seed: int = 0) -> list[tuple[int, int, float]]:
    """k of the listed candidate pairs, as (i, j, π). Systematic sampling on a random permutation: fixed size k, and
    P(pair drawn) = π exactly (Madow 1949). A first version drew sequentially without replacement and reported min(k·p, 1),
    which is not the inclusion probability of that scheme (review: Σπ = 56.9 for k = 60, up to 1.43× off per pair)."""
    i, j, s = np.asarray(i), np.asarray(j), np.asarray(scores, float) / temperature
    p = np.exp(s - s.max()); p = (1 - floor) * p / p.sum() + floor / len(p)
    k = min(k, len(p)); pi = inclusion_probabilities(p, k)
    rng = np.random.default_rng(seed); order = rng.permutation(len(p)); cum = np.cumsum(pi[order])
    pick = order[np.minimum(np.searchsorted(cum, rng.random() + np.arange(k), side="right"), len(p) - 1)]
    return [(int(i[t]), int(j[t]), float(pi[t])) for t in pick]


def draw(scores: np.ndarray, k: int, exclude: np.ndarray | None = None, temperature: float = 1.0, floor: float = 0.05,
         seed: int = 0) -> list[tuple[int, int, float]]:
    """`draw_pairs` over the upper triangle of an n×n score matrix; `exclude` masks pairs that are already linked."""
    iu = np.triu_indices(len(scores), 1); ok = np.ones(len(iu[0]), bool) if exclude is None else ~exclude[iu]
    return draw_pairs(iu[0][ok], iu[1][ok], scores[iu][ok], k, temperature, floor, seed)
