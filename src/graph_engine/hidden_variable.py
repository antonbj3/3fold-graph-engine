#!/usr/bin/env python3
"""
hidden_variable.py — a disagreement between reports whose declared validity boxes overlap is read as a candidate for an
UNDECLARED variable, not as a contradiction to settle.

margin_net and claim_federation classify a disagreeing edge as REGIME-BOUNDARY when the discordant reports' boxes are
disjoint and CONTRADICTION when they overlap. In fields where sources describe their setting loosely (clinical: age, dose,
sex, comorbidity, follow-up time; manufacturing: batch, supplier, ambient humidity) the second case is usually the first
case on an axis nobody declared. The move that keeps the node open is to find that axis, not to choose a side.

Given the reports of one edge, each with a `margin` (or a `sign`) and a dict of `attributes` (whatever is known about
the source's setting, numeric or categorical, missing allowed), rank the candidate splits:

    for each attribute and each threshold (numeric) or level (categorical):
        the reports are divided into two groups; the fit of "one number per group" is compared with "one number for all"
        score = Q_all − (Q_left + Q_right)   the drop of the weighted residual sum of squares (χ² statistic with σ)
        for signs: the drop of the impurity (Gini) of the sign vote

The best split is not a finding; it is the throw: the value of DECLARING that variable is the χ² drop, and the report says
what the boxes would become. Two guards: (1) an attribute that is missing on most reports cannot score (it would split
"known" from "unknown", which is the reporting habit, not the mechanism); (2) with n reports there are n − 1 numeric
thresholds per attribute, so the score is compared with the best split of the same data under random relabelling
(permutation p-value) — a split that beats chance by nothing is listed with p ≈ 1, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["HiddenVariableCandidate", "candidates", "from_mechanism"]


@dataclass
class HiddenVariableCandidate:
    attribute: str
    split: Any                     # threshold (numeric: left = value ≤ threshold) or level (categorical: left = value == level)
    kind: str                      # "numeric" | "categorical"
    score: float                   # χ² drop (margins) or impurity drop (signs)
    p_value: float                 # permutation p-value of that score
    n_left: int
    n_right: int
    left: dict = field(default_factory=dict)    # {"mean": m̂, "s": s} or {"sign_share": …}
    right: dict = field(default_factory=dict)
    boxes: dict = field(default_factory=dict)   # the validity axis the declaration would add: {attribute: [lo, hi]} per side


def _gls(m, s):
    w = 1.0 / s ** 2
    mh = float((w * m).sum() / w.sum())
    return mh, float(np.sqrt(1.0 / w.sum())), float((w * (m - mh) ** 2).sum())


def _score_split(mask, m, s, signs):
    if signs is None:
        _, _, q_all = _gls(m, s)
        _, _, ql = _gls(m[mask], s[mask]); _, _, qr = _gls(m[~mask], s[~mask])
        return q_all - ql - qr
    g = lambda v: 1.0 - ((v > 0).mean() ** 2 + (v < 0).mean() ** 2)
    n = len(signs)
    return g(signs) - (mask.sum() / n) * g(signs[mask]) - ((~mask).sum() / n) * g(signs[~mask])


def candidates(reports: list[dict], min_side: int = 2, min_known: float = 0.7, n_perm: int = 200, seed: int = 0,
               default_sigma: float = 0.1) -> list[HiddenVariableCandidate]:
    """reports: [{"margin": float, "sigma": float, "attributes": {name: value}}] or with "sign": ±1 instead of margin.
    Returns candidates sorted by score, each with a permutation p-value; empty when no attribute is known on enough reports."""
    n = len(reports)
    if n < 2 * min_side:
        return []
    use_signs = all("margin" not in r or r.get("margin") is None for r in reports)
    signs = np.array([r["sign"] for r in reports], float) if use_signs else None
    m = None if use_signs else np.array([r["margin"] for r in reports], float)
    s = None if use_signs else np.array([default_sigma if r.get("sigma") is None else r["sigma"] for r in reports], float)
    if s is not None and (s <= 0).any():
        raise ValueError("sigma must be > 0")
    names = sorted({k for r in reports for k in (r.get("attributes") or {})})
    rng = np.random.default_rng(seed)
    out = []
    for name in names:
        vals = [(r.get("attributes") or {}).get(name) for r in reports]
        known = np.array([v is not None for v in vals])
        if known.mean() < min_known or known.sum() < 2 * min_side:
            continue
        idx = np.flatnonzero(known); v = [vals[i] for i in idx]
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
        mk = (m[idx] if m is not None else None, s[idx] if s is not None else None, signs[idx] if signs is not None else None)
        splits = []
        if numeric:
            u = np.unique(np.array(v, float))
            for a, b in zip(u[:-1], u[1:]):
                splits.append(("numeric", float((a + b) / 2), np.array(v, float) <= (a + b) / 2))
        else:
            for lvl in sorted(set(map(str, v))):
                splits.append(("categorical", lvl, np.array([str(x) == lvl for x in v])))
        splits = [(k, t, mask) for k, t, mask in splits if mask.sum() >= min_side and (~mask).sum() >= min_side]
        if not splits:
            continue
        scored = [(_score_split(mask, *mk), k, t, mask) for k, t, mask in splits]
        best = max(scored, key=lambda x: x[0]); score, kind, thr, mask = best
        # permutation: best split score of the same attribute after shuffling the outcomes
        null = []
        for _ in range(n_perm):
            perm = rng.permutation(len(idx))
            mkp = tuple(None if x is None else x[perm] for x in mk)
            null.append(max(_score_split(mask2, *mkp) for _, _, mask2 in splits))
        p = float((np.array(null) >= score - 1e-12).mean())
        side = lambda msk: ({"sign_share": float((mk[2][msk] > 0).mean())} if use_signs else dict(zip(("mean", "s"), _gls(mk[0][msk], mk[1][msk])[:2])))
        if numeric:
            arr = np.array(v, float)
            boxes = {name: {"left": [float(arr[mask].min()), float(arr[mask].max())], "right": [float(arr[~mask].min()), float(arr[~mask].max())]}}
        else:
            boxes = {name: {"left": [thr], "right": sorted({str(x) for x in np.array(v, object)[~mask]})}}
        out.append(HiddenVariableCandidate(name, thr, kind, float(score), p, int(mask.sum()), int((~mask).sum()), side(mask), side(~mask), boxes))
    return sorted(out, key=lambda c: -c.score)


def from_mechanism(model, x_range: tuple[float, float], params: dict[str, tuple[float, float]], n_x: int = 41, n_p: int = 21,
                   fixed: dict | None = None) -> list[dict]:
    """Candidates from FIRST PRINCIPLES, before any report: `model(x, **params) -> y`. For each parameter θ, scan its range with
    the other parameters at their midpoints (or `fixed`) and record over what share of (x, θ) the sign of dy/dx differs from
    the sign at the parameter's midpoint. A parameter that can flip the relation inside the declared ranges is a hidden
    axis the boxes must carry; the score is that share (0 = the parameter never changes the sign, so leaving it out of the
    box costs nothing for the sign). Also returned: the threshold in θ where the flip first appears, if any.
    This is the mechanism-side counterpart of `candidates` (data side); a variable that both nominate is the throw."""
    fixed = fixed or {}
    xs = np.linspace(*x_range, n_x); h = (x_range[1] - x_range[0]) / (n_x * 10)
    mid = {k: fixed.get(k, (lo + hi) / 2) for k, (lo, hi) in params.items()}
    sign_at = lambda p: np.sign(np.array([model(x + h, **p) - model(x - h, **p) for x in xs]))
    base = sign_at(mid)
    out = []
    for k, (lo, hi) in params.items():
        grid = np.linspace(lo, hi, n_p); flips = np.zeros((n_p, n_x), bool); first = None
        for i, t in enumerate(grid):
            p = dict(mid); p[k] = float(t)
            s = sign_at(p); flips[i] = (s != base) & (s != 0)
            if first is None and flips[i].any():
                first = float(t)
        out.append({"attribute": k, "score": float(flips.mean()), "flip_threshold": first,
                    "x_share_flipped_at_extremes": [float(flips[0].mean()), float(flips[-1].mean())]})
    return sorted(out, key=lambda d: -d["score"])
