#!/usr/bin/env python3
"""
record_guarantee.py — a distribution-free admission rule for extracted records: "of the records admitted, at least 1 − α are
ENTIRELY right", with probability ≥ 1 − δ over the calibration draw.

typed_extraction gates on Π_f max(p_f, 1 − p_f) ≥ 0.6, which reads as P(all fields right) only if the fields are calibrated and
independent (e12: a misreading shared by the fields breaks that). Here the gate is set from data instead:

  score        s(record) = 1 − min_f max(p_f, 1 − p_f)      (the least certain field; any score works, this one is monotone)
  calibration  n labelled records with (s_k, ok_k), ok = every field right
  threshold    λ̂ = the largest of `n_grid` candidate values (quantiles of the calibration scores) whose Clopper–Pearson upper
               (1 − δ/n_grid) confidence bound on the error rate among calibration records with s ≤ λ is ≤ α. The Bonferroni
               split of δ over the candidates is what makes picking the largest passing one valid (Learn-then-Test,
               Angelopoulos, Bates, Candès, Jordan, Lei 2021/2022). Two earlier versions were wrong and were caught by the
               held-out test: scanning upward from the most certain record admitted nothing (one record's bound is 0.95), and
               scanning downward until a candidate passed broke the level in 8 of 20 splits (uncorrected multiple testing).
  guarantee    if future records are exchangeable with the calibration set, P(error rate among admitted ≤ α) ≥ 1 − δ.
               No independence between fields, no calibration of p, no model of the judge is assumed.

What it does not do: it says nothing about coverage (how many are admitted); a bad judge is handled by admitting little.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

__all__ = ["least_certain_field_score", "fit_threshold", "admit"]


def least_certain_field_score(P: np.ndarray) -> np.ndarray:
    """P: n×K field probabilities → s ∈ [0, 0.5], smaller = more certain record."""
    return 1.0 - np.max(np.stack([P, 1 - P]), 0).min(1)


def _ucb(errors: int, n: int, delta: float) -> float:
    return 1.0 if n == 0 else float(beta.ppf(1 - delta, errors + 1, n - errors)) if errors < n else 1.0


def fit_threshold(scores: np.ndarray, all_right: np.ndarray, alpha: float = 0.1, delta: float = 0.05, n_grid: int = 20) -> dict:
    """Largest candidate λ (quantiles of the calibration scores, `n_grid` of them) whose Clopper–Pearson upper (1 − δ/n_grid)
    bound on the admitted error rate is ≤ α. Bonferroni over the grid makes the selection valid (Learn-then-Test)."""
    s = np.asarray(scores, float); ok = np.asarray(all_right, bool)
    grid = np.unique(np.quantile(s, np.linspace(0, 1, n_grid + 1)[1:]))
    best = {"threshold": -1.0, "admitted_in_calibration": 0, "ucb_at_threshold": 1.0}
    for lam in grid:
        m = s <= lam; k = int(m.sum()); e = int((~ok[m]).sum()); u = _ucb(e, k, delta / len(grid))
        if u <= alpha and k > best["admitted_in_calibration"]:
            best = {"threshold": float(lam), "admitted_in_calibration": k, "ucb_at_threshold": u}
    return {**best, "n": int(len(s)), "alpha": alpha, "delta": delta, "n_grid": int(len(grid))}


def admit(scores: np.ndarray, fit: dict) -> np.ndarray:
    return np.asarray(scores) <= fit["threshold"]
