"""Stratified calibration and pooling for multi-lens yes/no questions.

Measured background (small language model, same yes/no question asked
through K "lenses" = different phrasings crossed with the two orders of
the answer options):

1. The same lens gives bit-identical answers on repeat. Repeating a lens
   adds no information.
2. Errors of two lenses that share the same option order correlate at
   +0.61. Errors of two lenses with swapped option order correlate at
   -0.08. Lens errors are not independent, and the sign of the
   correlation depends on a nuisance factor (option order), not on
   content.
3. On some sentence forms (a "stratum") the model is reliably WRONG:
   accuracy 0.21-0.35. Per-stratum recalibration turns a 0.54-accurate
   lens into 0.84, and turns a pool of 8 such lenses into 0.93 accuracy
   on held-out items.
4. Naive pooling (sum of per-lens log-odds) is over-confident when lens
   errors correlate, because it treats correlated evidence as if it were
   independent evidence.

This module provides: a clipped logit, per-stratum-per-lens Platt
calibration, an error-correlation matrix, two summaries of that matrix
(effective_lenses and participation_ratio, which disagree on purpose),
naive/tempered pooling, and a balanced-design helper for choosing lenses
so nuisance factors cancel by construction.

Relation to existing code. `effective_lenses` is the Kish form and `participation_ratio` the participation-ratio form
that neff_form.py already defines and contrasts for equicorrelation (N/(1+(N−1)ρ) against N/(1+(N−1)ρ²)); here they take
a full correlation matrix. That the ρ² form cannot see the sign of a correlation was measured earlier on real optical
data in examples/optical_constants/qc_neff_sign_blindness.py. llm_federation_analysis.py measures the error correlation
between model views as a diagnostic; `tempered_weight` is the correction that uses it.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p, eps: float = 1e-6):
    """Clipped logit: log(p / (1 - p)) with p clipped to [eps, 1 - eps]."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


class StratifiedPlatt:
    """Per-(stratum, lens) 1-D Platt calibration of logit probabilities.

    For every (stratum, lens) pair a 1-D logistic regression is fit on
    logit(P[:, lens]) restricted to that stratum's labelled rows. If a
    stratum has fewer than `min_items` labelled items for a lens, or the
    labels for that (stratum, lens) subset are a single class, the fit
    falls back to a global logistic regression over all strata for that
    lens.

    A NEGATIVE fitted slope is an expected and useful outcome: it means
    the lens is reliably wrong on that stratum, and flipping its sign
    turns a systematically-wrong signal into a systematically-right one.
    A lens that is reliably wrong is as informative as one that is
    reliably right; only a lens whose slope is near zero (no relation
    between its logit and the truth) is uninformative.
    """

    def __init__(self, min_items: int = 8):
        self.min_items = min_items
        self.models_: Dict[tuple, LogisticRegression] = {}
        self.global_models_: Dict[int, LogisticRegression] = {}

    @staticmethod
    def _fit_one(x, y) -> LogisticRegression:
        model = LogisticRegression(C=1.0)
        model.fit(x.reshape(-1, 1), y)
        return model

    def fit(self, P, y, strata) -> "StratifiedPlatt":
        P = np.asarray(P, dtype=float)
        y = np.asarray(y).astype(int)
        strata = np.asarray(strata)
        n, k = P.shape
        L = logit(P)

        self.models_ = {}
        self.global_models_ = {}

        for lens in range(k):
            x_all = L[:, lens]
            if len(np.unique(y)) >= 2:
                self.global_models_[lens] = self._fit_one(x_all, y)
            else:
                self.global_models_[lens] = None

        for lens in range(k):
            for s in np.unique(strata):
                mask = strata == s
                x_s = L[mask, lens]
                y_s = y[mask]
                if x_s.shape[0] >= self.min_items and len(np.unique(y_s)) >= 2:
                    self.models_[(s, lens)] = self._fit_one(x_s, y_s)
                else:
                    self.models_[(s, lens)] = None
        return self

    def transform(self, P, strata):
        P = np.asarray(P, dtype=float)
        strata = np.asarray(strata)
        n, k = P.shape
        L = logit(P)
        out = np.zeros((n, k), dtype=float)

        for lens in range(k):
            global_model = self.global_models_.get(lens)
            for i in range(n):
                s = strata[i]
                model = self.models_.get((s, lens))
                if model is None:
                    model = global_model
                x = L[i, lens]
                if model is None:
                    # No usable fit at all (e.g. single-class overall):
                    # pass the raw logit through unchanged.
                    out[i, lens] = x
                else:
                    out[i, lens] = model.decision_function(
                        np.array([[x]])
                    )[0]
        return out


def error_correlation(L, y):
    """(K, K) correlation matrix of the per-lens error indicators.

    L is (n_items, K) log-odds; y is (n_items,) bool ground truth. The
    error indicator for lens k on item i is 1 if sign(L[i, k]) (read as
    class 1 when > 0) disagrees with y[i], else 0. A lens with a constant
    error indicator (always right or always wrong on this sample) has
    zero variance; its correlation with every other lens is defined as 0
    rather than NaN.
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y).astype(int)
    pred = (L > 0).astype(int)
    err = (pred != y[:, None]).astype(float)

    k = err.shape[1]
    C = np.eye(k)
    stds = err.std(axis=0)
    for i in range(k):
        for j in range(i + 1, k):
            if stds[i] == 0.0 or stds[j] == 0.0:
                c = 0.0
            else:
                c = np.corrcoef(err[:, i], err[:, j])[0, 1]
                if np.isnan(c):
                    c = 0.0
            C[i, j] = c
            C[j, i] = c
    return C


def effective_lenses(C) -> float:
    """Kish equal-weight effective sample size of the K lenses.

    effective_lenses(C) = K**2 / sum(C). This form sees the SIGN of the
    pairwise correlations: for an equicorrelated matrix with common
    correlation r it reduces to K / (1 + (K-1) r), so anticorrelated
    lenses (r < 0) count as MORE than K independent lenses, because their
    errors cancel when pooled. Answers: "how many independent votes does
    this pool of lenses behave like, for the purpose of pooling them?"
    """
    C = np.asarray(C, dtype=float)
    k = C.shape[0]
    tot = C.sum()
    # Σ C is k²·Var(mean of the standardized errors); it reaches 0 when the errors cancel exactly (K = 2, r = −1).
    # A non-positive total means an infinite count in the Kish sense; return inf instead of a negative or exploding number.
    return float("inf") if tot <= 1e-9 * k else (k ** 2) / tot


def participation_ratio(C) -> float:
    """Participation ratio of the correlation matrix's eigenvalues.

    participation_ratio(C) = (sum(eigvals))**2 / sum(eigvals**2). For an
    equicorrelated matrix with common correlation r it reduces to
    K / (1 + (K-1) r**2): it depends on r only through r**2, so it is
    BLIND to the sign of the correlation (r and -r give the same value).
    Answers: "how many effective independent DIMENSIONS does the error
    covariance span?" -- a dimensionality question, not a pooling-benefit
    question. Neither summary is recommended over the other; they answer
    different questions and are provided side by side.
    """
    C = np.asarray(C, dtype=float)
    eigvals = np.linalg.eigvalsh(C)
    return float((eigvals.sum() ** 2) / np.sum(eigvals ** 2))


def pool(L, weight: float = 1.0):
    """Pooled probability = sigmoid(weight * sum_k L[:, k])."""
    L = np.asarray(L, dtype=float)
    return sigmoid(weight * L.sum(axis=1))


def tempered_weight(C) -> float:
    """effective_lenses(C) / K, clipped to (0, 1].

    Used as the `weight` in `pool` to down-weight the sum of log-odds
    when lens errors correlate positively (effective_lenses < K), which
    corrects the naive pool's over-confidence. Clipped above at 1 because
    a pool of independent or anticorrelated lenses should not be
    up-weighted beyond the naive sum; clipped below (away from 0 and from
    negative values, which can occur algebraically for extreme
    anticorrelation) so the weight stays a legitimate scaling factor.
    """
    C = np.asarray(C, dtype=float)
    k = C.shape[0]
    w = effective_lenses(C) / k
    return float(np.clip(w, 1e-6, 1.0))


def _sylvester_hadamard(n: int) -> np.ndarray:
    """Sylvester construction of a Hadamard matrix of order n (n a power of 2)."""
    H = np.array([[1]], dtype=int)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def balanced_lenses(factors: Dict[str, int]) -> List[Dict[str, int]]:
    """Rows of a two-level orthogonal array balanced across `factors`.

    `factors` maps a nuisance-factor name to its number of levels, which
    must be 2 for every factor (e.g. {"option_order": 2, "polarity": 2}).
    Returns a list of dicts factor -> level (0 or 1), one per row, such
    that every factor is balanced (equal count of each level) and every
    pair of factors is orthogonal (all four level combinations appear
    equally often).

    Construction: take a Sylvester Hadamard matrix of the smallest order
    n (a power of 2) with n >= len(factors) + 1, drop its all-ones
    column, and assign each remaining column (there are n - 1 >=
    len(factors) of them; only the first len(factors) are used) to one
    factor. Any two distinct columns of a Hadamard matrix are orthogonal
    to each other and to the all-ones vector, which gives balance and
    pairwise orthogonality directly.

    Proof that additive nuisance biases cancel: suppose each factor f
    shifts the measured log-odds by an additive bias b_f * s_f(row),
    where s_f(row) in {-1, +1} is the level of factor f on that row (the
    Hadamard column). Summing over the n rows of a balanced set gives
    sum_row b_f * s_f(row) = b_f * sum_row s_f(row) = b_f * 0 = 0, because
    s_f is a non-trivial Hadamard column and therefore orthogonal to the
    all-ones column, i.e. sums to zero. This holds independently for
    every factor f, so the total additive bias sum_f b_f * s_f(row),
    summed over rows, is also exactly 0.
    """
    names = list(factors.keys())
    for name, levels in factors.items():
        if levels != 2:
            raise ValueError(f"factor {name!r} has {levels} levels; only 2-level factors are supported")

    m = len(names)
    n = 1
    while n < m + 1:
        n *= 2

    H = _sylvester_hadamard(n)
    cols = H[:, 1:1 + m]  # drop the all-ones column, take m orthogonal columns

    rows = []
    for r in range(n):
        row = {}
        for j, name in enumerate(names):
            row[name] = 0 if cols[r, j] == 1 else 1
        rows.append(row)
    return rows
