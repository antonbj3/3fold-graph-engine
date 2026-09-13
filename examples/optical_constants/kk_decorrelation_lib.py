 #!/usr/bin/env python3
"""kk_decorrelation_lib — cert B: a finite-parameter causal-oscillator regression that predicts
kappa FROM n (opposite direction + opposite numerical machinery vs kk_cert_lib's kk_re_from_im,
which predicts n FROM kappa via an exact PV singular-integral quadrature). See
kk_decorrelation_PREREG.md Sec.1 for the full derivation and justification.

Reuses (imports, does not reimplement) kk_cert_lib's lorentz_eps, interior_mask, relative_residual,
load_real_instance, inject_bump, make_log_grid, OSC_PARAMS_DEFAULT, metal_nk, DRUDE_AU, INTERBAND_AU.
"""
import numpy as np
from scipy.optimize import least_squares

from kk_cert_lib import lorentz_eps # noqa: F401  (reused, not reimplemented)


# =====================================================================================
# Causal 3-oscillator regression fit: params (E_j, gamma_j, f_j) x M=3, gamma_j>0 ENFORCED
# (the fit is literally incapable of producing an acausal curve -- the causality constraint
# is structural, not a post-hoc check).
# =====================================================================================
M_OSC = 3


def _params_to_vec(params):
    return np.array([v for triple in params for v in triple], dtype=float)


def _vec_to_params(x):
    return [(x[3 * j], x[3 * j + 1], x[3 * j + 2]) for j in range(M_OSC)]


def _bounds(E0, E1):
    """
    Frozen bound formula (Sec.1): E_j allowed somewhat outside the measured band (poles just
    below/above are physically common -- IR phonon / deep-UV interband), gamma_j and f_j generic
    wide ranges. NOT tuned per-instance after seeing any residual -- a fixed formula of (E0,E1) only.
    """
    lb = np.array([0.05 * E0, 1e-4, 1e-6] * M_OSC)
    ub = np.array([20.0 * E1, 50.0, 200.0] * M_OSC)
    return lb, ub


def _init_guess(E0, E1, rng):
    """
    One random restart's initial guess: E_j at log-spaced fractions {0.15,0.5,0.85} of the
    band, jittered log-normal (sigma=0.3); gamma_j, f_j drawn from generic wide ranges.
    """
    logE0, logE1 = np.log(E0), np.log(E1)
    fracs = [0.15, 0.5, 0.85]
    x0 = []
    for frac in fracs:
        Ej = float(np.exp(logE0 + frac * (logE1 - logE0) + rng.normal(0, 0.3)))
        Ej = max(Ej, 1e-6)
        gj = float(rng.uniform(0.02, 2.0) * (E1 - E0 + 1e-9))
        gj = np.clip(gj, 1e-3, 40.0)
        fj = float(rng.uniform(0.1, 5.0))
        x0 += [Ej, gj, fj]
    return np.array(x0)


def fit_causal_oscillators_to_n(E, n_target, n_restarts=10, seed=0):
    """
    Nonlinear-least-squares fit of M_OSC causal Lorentz oscillators to n_target(E) ONLY
    (kappa never enters). Multi-restart (fixed seed => deterministic), keeps best-loss result.
    Returns (params_best, n_model, k_model, loss_best).
    """
    E0, E1 = float(E.min()), float(E.max())
    lb, ub = _bounds(E0, E1)
    rng = np.random.default_rng(seed)

    def resid(x):
        params = _vec_to_params(x)
        eps = lorentz_eps(E, params, acausal=False)
        nk = np.sqrt(eps)
        return nk.real - n_target

    best = None
    for r in range(n_restarts):
        x0 = np.clip(_init_guess(E0, E1, rng), lb, ub)
        try:
            sol = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=4000)
        except Exception:
            continue
        loss = float(np.sum(sol.fun ** 2))
        if best is None or loss < best[1]:
            best = (sol.x, loss)
    x_best, loss_best = best
    params_best = _vec_to_params(x_best)
    eps_best = lorentz_eps(E, params_best, acausal=False)
    nk_best = np.sqrt(eps_best)
    return params_best, nk_best.real, nk_best.imag, loss_best


# --- metal-appropriate variant (Drude + interband), diagnostic only (mirrors kk_cert's own
# Part C metal diagnostic; NOT used in the headline decorrelation test -- Sec.3/PREREG). ---
def _metal_bounds(E0, E1):
    # x = [Ep, gamma_D, E_ib, gamma_ib, f_ib]
    lb = np.array([1.0, 1e-4, 0.05 * E0, 1e-4, 1e-6])
    ub = np.array([30.0, 10.0, 20.0 * E1, 20.0, 200.0])
    return lb, ub


def fit_causal_drude_interband_to_n(E, n_target, n_restarts=10, seed=0):
    E0, E1 = float(E.min()), float(E.max())
    lb, ub = _metal_bounds(E0, E1)
    rng = np.random.default_rng(seed)

    def eps_of(x):
        Ep, gD, Eib, gib, fib = x
        eps = np.ones_like(E, dtype=complex) - Ep ** 2 / (E ** 2 + 1j * gD * E)
        eps = eps + fib * Eib ** 2 / (Eib ** 2 - E ** 2 - 1j * gib * E)
        return eps

    def resid(x):
        eps = eps_of(x)
        nk = np.sqrt(eps)
        return nk.real - n_target

    best = None
    for r in range(n_restarts):
        x0 = np.array([
            rng.uniform(3.0, 15.0), rng.uniform(0.02, 1.0),
            float(np.exp(np.log(E0) + rng.uniform(0.2, 0.8) * (np.log(E1) - np.log(E0)))),
            rng.uniform(0.1, 3.0), rng.uniform(0.1, 5.0),
        ])
        x0 = np.clip(x0, lb, ub)
        try:
            sol = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=4000)
        except Exception:
            continue
        loss = float(np.sum(sol.fun ** 2))
        if best is None or loss < best[1]:
            best = (sol.x, loss)
    x_best, loss_best = best
    eps_best = eps_of(x_best)
    nk_best = np.sqrt(eps_best)
    return x_best, nk_best.real, nk_best.imag, loss_best


# =====================================================================================
# Metric (frozen, Sec.1): log-residual for real (large-dynamic-range) kappa data.
# =====================================================================================
def log_relative_residual(k_pred, k_target, k_norm_ref, mask):
    """
    median|log10(k_pred)-log10(k_target)| over mask, normalized by log10-range of k_norm_ref
    (ALWAYS the original/unperturbed kappa array -- fixed throughout an amplitude sweep, mirrors
    kk_cert_lib.relative_residual's own convention of normalizing against the never-perturbed n).
    """
    m = mask & np.isfinite(k_pred) & np.isfinite(k_target) & (k_pred > 0) & (k_target > 0)
    finite_ref = k_norm_ref[np.isfinite(k_norm_ref) & (k_norm_ref > 0)]
    if len(finite_ref) == 0 or not np.any(m):
        return float("nan")
    rng = np.log10(np.max(finite_ref)) - np.log10(np.min(finite_ref))
    if rng <= 0:
        return float("nan")
    return float(np.median(np.abs(np.log10(k_pred[m]) - np.log10(k_target[m])))) / rng
