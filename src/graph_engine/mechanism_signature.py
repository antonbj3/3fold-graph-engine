#!/usr/bin/env python3
"""
mechanism_signature.py — "mechanically close" as a COMPUTED property of a model, not a judgement about its description.

A one-state equilibrium model F(x, μ) = 0 (x = response, μ = load / control) is traced along its solution curve and
reduced to a short signature of how the solution behaves where it stops being unique:

    n_limit        number of limit points (dμ/dx = 0 with a sign change): places where the solution turns back
    order          order of tangency at the first limit point: μ_c − μ ∝ |x − x_c|^order   (2 = generic fold)
    beta           response exponent  |x − x_c| ∝ |μ_c − μ|^beta  = 1/order             (fold: 1/2)
    gamma          stiffness exponent |∂F/∂x| ∝ |μ_c − μ|^gamma                           (fold: 1/2)
    odd            F(−x, μ) = −F(x, μ) on a grid of (x, μ), not only on the branch: a symmetric (pitchfork-type) model, which also has order 2 but keeps the
                   x = 0 branch for every μ
    x_c, mu_c      where it happens (NOT part of the distance: location is field-specific, the mechanism is not)

Two models with the same (n_limit > 0, order, gamma, odd) lose uniqueness the same way, whatever their subject.
`distance` compares only those entries. The three textbook cases named in this repository's README — thermal runaway,
snap-through buckling and electrostatic pull-in — come out identical (one fold, order 2, β = γ = 1/2), and differ from a
symmetric buckling column (odd, γ = 1), and from a cusp, a linear and a saturating model, which have no limit point (tests).

How it is computed. The curve is parametrized by x: for each x the load μ(x) solving F(x, μ) = 0 is found by
bracketing; limit points are sign changes of dμ/dx; exponents are least-squares slopes in log–log coordinates on a
window next to the limit point. Scope: one state variable, F monotone in μ on the bracket. For a vector state the
same quantities exist (limit point = zero of the smallest singular value of ∂F/∂x along an arclength continuation);
that is not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
from scipy.optimize import brentq

__all__ = ["Signature", "signature", "signature_vector", "distance"]


@dataclass
class Signature:
    n_limit: int
    order: float
    beta: float
    gamma: float
    odd: bool
    x_c: float
    mu_c: float


def signature(F: Callable[[float, float], float], x_range: tuple[float, float], mu_bracket: tuple[float, float],
              n: int = 4001, window: tuple[float, float] = (1e-3, 3e-2)) -> Signature:
    xs = np.linspace(*x_range, n); mu = np.full(n, np.nan)
    for k, x in enumerate(xs):
        a, b = F(x, mu_bracket[0]), F(x, mu_bracket[1])
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            mu[k] = brentq(lambda m: F(x, m), *mu_bracket, xtol=1e-14, rtol=1e-13)
    ok = np.isfinite(mu); xs, mu = xs[ok], mu[ok]
    if len(xs) < 20:
        raise ValueError("no equilibrium found: F(x, mu) = 0 has no sign change in mu_bracket for (almost) any x in x_range")
    # oddness is a property of F, so it is tested OFF the solution branch too (on the branch F = 0 = −F trivially for any
    # F whose branch is symmetric; review gave F = (μ−x²)x + (μ−x²)² as a non-odd F that passed the on-branch test)
    gx = np.linspace(max(abs(x_range[0]), abs(x_range[1])) * 0.05, max(abs(x_range[0]), abs(x_range[1])), 9)
    gm = np.linspace(mu_bracket[0], mu_bracket[1], 7)
    vals = np.array([[F(x, m) for m in gm] for x in gx]); mirror = np.array([[F(-x, m) for m in gm] for x in gx])
    fin = np.isfinite(vals) & np.isfinite(mirror)
    odd = bool(fin.any()) and bool(np.all(np.abs(vals + mirror)[fin] <= 1e-9 * (1 + np.abs(vals[fin]))))
    d = np.gradient(mu, xs); idx = [k for k in range(2, len(d) - 2) if d[k - 1] * d[k + 1] < 0 and abs(d[k]) <= min(abs(d[k - 1]), abs(d[k + 1]))]
    idx = [k for j, k in enumerate(idx) if j == 0 or k - idx[j - 1] > 3]
    if not idx:
        return Signature(0, float("nan"), float("nan"), float("nan"), odd, float("nan"), float("nan"))
    k0 = idx[0]; sl = slice(max(k0 - 3, 0), k0 + 4); c = np.polyfit(xs[sl], mu[sl], 2); xc = -c[1] / (2 * c[0]); muc = float(np.polyval(c, xc))
    span = xs[-1] - xs[0]; dx = np.abs(xs - xc); dm = np.abs(muc - mu); m = (dx > window[0] * span) & (dx < window[1] * span) & (dm > 0)
    order = float(np.polyfit(np.log(dx[m]), np.log(dm[m]), 1)[0])
    h = 1e-6 * span; stiff = np.array([abs(F(x + h, mm) - F(x - h, mm)) / (2 * h) for x, mm in zip(xs[m], mu[m])]); g = stiff > 0
    gamma = float(np.polyfit(np.log(dm[m][g]), np.log(stiff[g]), 1)[0])
    return Signature(len(idx), order, 1.0 / order, gamma, odd, float(xc), muc)


def distance(a: Signature, b: Signature) -> float:
    """0 = the same way of losing uniqueness. Location (x_c, mu_c) is ignored on purpose."""
    if (a.n_limit == 0) != (b.n_limit == 0):
        return 4.0
    if a.n_limit == 0:
        return 0.0 if a.odd == b.odd else 1.0
    return abs(a.order - b.order) + 2 * abs(a.gamma - b.gamma) + (0.0 if a.odd == b.odd else 1.0)


def signature_vector(F: Callable[[np.ndarray, float], np.ndarray], x0: np.ndarray, mu0: float, ds: float = 0.01, n_steps: int = 3000,
                     mu_stop: tuple[float, float] = (-np.inf, np.inf), window: tuple[float, float] = (1e-3, 3e-2)) -> Signature:
    """The same signature for a VECTOR state x ∈ ℝⁿ, F(x, μ) = 0 traced by pseudo-arclength continuation (Keller 1977).
    A limit point is where the tangent's μ-component changes sign; there σ_min(∂F/∂x) → 0 and the curve turns back.
    order:  μ_c − μ ∝ |x − x_c|^order along the curve (x measured by arclength from the fold)
    gamma:  σ_min(∂F/∂x) ∝ |μ_c − μ|^gamma
    odd:    F(−x, μ) = −F(x, μ) on a grid.
    Scalar models give the same numbers as `signature` (test); the fold of a coupled two-state model reads order 2, γ = 1/2."""
    from scipy.optimize import fsolve
    x = np.asarray(x0, float); n = len(x); mu = float(mu0)
    x = fsolve(lambda v: F(v, mu), x, xtol=1e-12)
    def J(x, mu, h=1e-6):
        Jx = np.zeros((n, n)); Jm = np.zeros(n); f0 = F(x, mu)
        for k in range(n):
            e = np.zeros(n); e[k] = h; Jx[:, k] = (F(x + e, mu) - F(x - e, mu)) / (2 * h)
        Jm[:] = (F(x, mu + h) - F(x, mu - h)) / (2 * h); return Jx, Jm
    Jx, Jm = J(x, mu); t = np.linalg.lstsq(np.c_[Jx, Jm], -np.zeros(n), rcond=None)[0]
    null = np.linalg.svd(np.c_[Jx, Jm])[2][-1]; t = null / np.linalg.norm(null); t = t if t[-1] > 0 else -t
    xs, mus, sig = [x.copy()], [mu], [np.linalg.svd(Jx, compute_uv=False)[-1]]
    for _ in range(n_steps):
        guess = np.r_[x, mu] + ds * t
        def G(v):
            return np.r_[F(v[:n], v[n]), t @ (v - np.r_[x, mu]) - ds]
        v = fsolve(G, guess, xtol=1e-12); x, mu = v[:n], float(v[n])
        if not (mu_stop[0] <= mu <= mu_stop[1]) or not np.all(np.isfinite(v)):
            break
        Jx, Jm = J(x, mu); null = np.linalg.svd(np.c_[Jx, Jm])[2][-1]; tn = null / np.linalg.norm(null)
        t = tn if tn @ t > 0 else -tn
        xs.append(x.copy()); mus.append(mu); sig.append(np.linalg.svd(Jx, compute_uv=False)[-1])
    xs, mus, sig = np.array(xs), np.array(mus), np.array(sig)
    s_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xs, axis=0), axis=1))]      # arclength of the STATE: near a fold dμ ≈ 0
    d = np.gradient(mus, s_arc); idx = [k for k in range(2, len(d) - 2) if d[k - 1] * d[k + 1] < 0 and abs(d[k]) <= min(abs(d[k - 1]), abs(d[k + 1]))]
    idx = [k for j, k in enumerate(idx) if j == 0 or k - idx[j - 1] > 3]
    gx = np.linspace(0.05, 1.0, 7)[:, None] * (np.abs(xs).max(0) + 1e-9)[None, :]; gm = np.linspace(mus.min(), mus.max(), 5)
    odd = all(np.allclose(F(-g, m), -np.asarray(F(g, m)), atol=1e-9 * (1 + np.abs(F(g, m)).max())) for g in gx for m in gm)
    if not idx:
        return Signature(0, float("nan"), float("nan"), float("nan"), odd, float("nan"), float("nan"))
    k0 = idx[0]; sl = slice(max(k0 - 3, 0), k0 + 4); c = np.polyfit(s_arc[sl], mus[sl], 2); sc = -c[1] / (2 * c[0]); muc = float(np.polyval(c, sc))
    span = s_arc[-1] - s_arc[0]; dxs = np.abs(s_arc - sc); dm = np.abs(muc - mus); m = (dxs > window[0] * span) & (dxs < window[1] * span) & (dm > 0) & (sig > 0)
    order = float(np.polyfit(np.log(dxs[m]), np.log(dm[m]), 1)[0]); gamma = float(np.polyfit(np.log(dm[m]), np.log(sig[m]), 1)[0])
    xc = xs[np.argmin(np.abs(s_arc - sc))]
    return Signature(len(idx), order, 1.0 / order, gamma, bool(odd), float(np.linalg.norm(xc)), muc)
