"""leg_admission — the LEG-ADMISSION GATE for the ensemble scene-eyes trust-map (an agent worktree contribution).

WHY (validated on real eth3d held-out, commit f43c976a1): a candidate cert-leg being PAIRWISE-DECORRELATED from the
existing legs is NECESSARY BUT NOT SUFFICIENT to admit it. A leg can be pairwise-decorrelated AND individually predictive
yet add ~zero UNIQUE COVERAGE beyond the current ENSEMBLE (its signal is already inside the ensemble). Admitting such a leg
does NOT lift out-of-sample prediction and, under MAX/OR composition, HURTS it (dilution). So gate on PARTIAL held-out
COVERAGE (unique predictive variance beyond the ensemble), not on pairwise rho. This composes with (does not duplicate) a
fusion module like trustmap_compose: admission decides WHICH legs enter; fusion decides HOW to combine them.

Deployment: pass each candidate leg's per-region softness scores, the current ensemble's scores, and a held-out reliability
TRUTH (e.g. reproj error in held-out views). admits returns the decision + diagnostics. strength_weighted_compose
is the fusion a weak-but-covering leg needs (equal-weight/OR dilutes it).

Numpy-only, additive, 0-collision. Selftest: `python -m graph_engine.leg_admission`.
Ties  (decorr-signal != decorr-coverage),, D's ensemble scene-eyes directive.
"""
from __future__ import annotations
import numpy as np


def _rank(x):
    """Rank-transform x, NaN-safe. FAIL-OPEN BUG FIXED: plain
    np.argsort(np.argsort(x)) does NOT propagate NaN -- it fabricates a full, plausible-looking rank spread
    from all-NaN or zero-variance input (e.g. all-NaN -> [0,1,2,...,n-1]; one NaN mixed in gets silently
    assigned the MAXIMUM rank, as if it were the single largest value). This cascaded into strength_weighted_
    compose/admits/safe_fuse/regime_adaptive_fence, all of which import this function, producing confident-
    looking fused scores/admission decisions from garbage/degenerate input. Fixed: non-finite entries stay
    NaN; an all-NaN or zero-variance array returns all-NaN (no fabricated signal)."""
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if not np.any(m) or np.ptp(x[m]) < 1e-12:
        return np.full(len(x), np.nan)
    r = np.full(len(x), np.nan)
    r[m] = np.argsort(np.argsort(x[m])).astype(float)
    return r


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(_rank(a[m]), _rank(b[m]))[0, 1])


def _partial_spearman(a, b, ctrl):
    """Partial Spearman(a, b | ctrl): rank-residualize a and b on ctrl, then correlate. ctrl may be a 2D array (n, k)
    of several controls (the ensemble's constituent legs) -> residualize on all of them."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    C = np.asarray(ctrl, float)
    if C.ndim == 1:
        C = C[:, None]
    m = np.isfinite(a) & np.isfinite(b) & np.all(np.isfinite(C), axis=1)
    if m.sum() < 5:
        return float("nan")
    ra, rb = _rank(a[m]), _rank(b[m])
    Rc = np.column_stack([_rank(C[m, j]) for j in range(C.shape[1])])
    Rc = np.column_stack([np.ones(len(Rc)), Rc])                       # design matrix with intercept
    # least-squares residuals of ra, rb on the control ranks
    beta_a, *_ = np.linalg.lstsq(Rc, ra, rcond=None)
    beta_b, *_ = np.linalg.lstsq(Rc, rb, rcond=None)
    resa, resb = ra - Rc @ beta_a, rb - Rc @ beta_b
    if np.std(resa) < 1e-12 or np.std(resb) < 1e-12:
        return 0.0
    return float(np.corrcoef(resa, resb)[0, 1])


def partial_coverage(candidate_scores: np.ndarray, ensemble_legs: np.ndarray, truth: np.ndarray) -> float:
    """The candidate leg's UNIQUE predictive coverage of `truth` beyond the current ensemble legs.
    ensemble_legs: 1D (single ensemble score) or 2D (n, k) of the ensemble's constituent leg scores (residualize on all).
    Returns partial Spearman(candidate, truth | ensemble_legs) in [-1, 1]. This is the admission signal."""
    return _partial_spearman(candidate_scores, truth, ensemble_legs)


def pairwise_min_decorrelation(candidate_scores: np.ndarray, existing_legs: np.ndarray) -> float:
    """max_i |Spearman(candidate, leg_i)| over existing legs -> the pairwise-decorrelation check (NECESSARY-not-sufficient).
    existing_legs: 2D (n, k). Small value = pairwise-decorrelated. Reported to show pairwise-alone can mislead."""
    L = np.asarray(existing_legs, float)
    if L.ndim == 1:
        L = L[:, None]
    # P5b guard (repository-wide): bare max(generator) silently skips a non-leading NaN leg (_spearman
    # returns nan on <3 finite pairs) -- np.max surfaces it instead. Non-gating (admits's decision reads only
    # `gain`, not this diagnostic), hardened anyway so the reported/printed pairwise_min_decorr is never a
    # silently-incomplete max over a leg subset.
    return float(np.max([abs(_spearman(candidate_scores, L[:, j])) for j in range(L.shape[1])]))


def admits(candidate_scores: np.ndarray, existing_legs: np.ndarray, truth: np.ndarray, min_gain: float = 0.01) -> tuple:
    """Admission decision for a candidate cert-leg -- the DIRECT test: does adding it to the strength-weighted fused ensemble
    IMPROVE held-out prediction of `truth`? (This is what actually matters; partial coverage & pairwise rho are diagnostics.)
    candidate_scores: (n) per-region softness of the candidate leg.
    existing_legs: (n, k) per-region softness of the k legs already in the ensemble.
    truth: (n) held-out reliability truth (higher = less reliable), e.g. held-out reproj error. For an honest
                       decision use a CALIBRATION split for the weights and a disjoint TEST split for the gain (cross-validate);
                       passing one array measures in-sample gain (optimistic) -- fine as a screen, not a final claim.
    Admits iff the fused-ensemble held-out prediction improves by > min_gain when the candidate is added.
    Returns (admit: bool, info: dict)."""
    L = np.asarray(existing_legs, float)
    if L.ndim == 1:
        L = L[:, None]
    fused_before = strength_weighted_compose(L, truth=truth)
    fused_after = strength_weighted_compose(np.column_stack([L, candidate_scores]), truth=truth)
    gain = _spearman(fused_after, truth) - _spearman(fused_before, truth)
    cov = partial_coverage(candidate_scores, L, truth)
    solo = _spearman(candidate_scores, truth)
    pair = pairwise_min_decorrelation(candidate_scores, L)
    admit = bool(np.isfinite(gain) and gain > min_gain)
    reason = ("admit: adding it lifts the fused held-out prediction by %+.3f (unique coverage %.3f)" % (gain, cov)
              if admit else
              "reject: fused-prediction gain %+.3f <= %.2f (unique coverage %.3f) -- pairwise min|rho|=%.2f, solo=%.3f: "
              "pairwise-decorr and/or solo-predictive is NOT enough (decorr-signal != decorr-coverage); it adds no lift"
              % (gain, min_gain, cov, pair, solo))
    return admit, {"fused_gain": gain, "partial_coverage": cov, "solo_predictiveness": solo,
                   "pairwise_min_decorr": pair, "min_gain": min_gain, "reason": reason}


def strength_weighted_compose(legs: np.ndarray, truth: np.ndarray | None = None, weights: np.ndarray | None = None) -> np.ndarray:
    """Fuse leg softness scores weighted by predictive strength (the fusion a weak-but-covering leg needs; equal-weight/OR
    dilutes it). legs: (n, k). If weights is None and truth given, weight_i = max(0, Spearman(leg_i, truth)) (a held-out
    calibration split should provide `truth`, not the test set). Returns a fused per-region softness score (n)."""
    L = np.asarray(legs, float)
    if L.ndim == 1:
        L = L[:, None]
    R = np.column_stack([_rank(L[:, j]) / max(1, len(L) - 1) for j in range(L.shape[1])])   # rank-normalized [0,1]
    if weights is None:
        if truth is None:
            weights = np.ones(L.shape[1])
        else:
            weights = np.array([max(0.0, _spearman(L[:, j], truth)) for j in range(L.shape[1])])
    weights = np.asarray(weights, float)
    if weights.sum() <= 0:
        weights = np.ones(L.shape[1])
    return R @ (weights / weights.sum())


def _load_monotone_agnostic():
    """Load `monotone_agnostic_admits` (partial distance correlation + label-permutation significance
    gate) from the sibling module and verify the resolution before use.

    Returns (fn_or_None, method_str_or_None, error_str_or_None). If the module or either symbol is
    missing, returns None so `admits_v2` degrades to the Spearman-only gate instead of crashing.
    """
    try:
        from . import monotone_agnostic_admission as mod
    except ImportError:
        try:
            import monotone_agnostic_admission as mod
        except Exception as e:
            return None, None, "monotone_agnostic_admission unavailable: %r" % (e,)
    except Exception as e:
        return None, None, "monotone_agnostic_admission unavailable: %r" % (e,)
    fn = getattr(mod, "monotone_agnostic_admits", None)
    fingerprint = getattr(mod, "partial_distance_correlation", None)
    if callable(fn) and callable(fingerprint):
        return fn, "monotone_agnostic_admission.monotone_agnostic_admits", None
    return None, None, ("resolved a module but monotone_agnostic_admits / partial_distance_correlation "
                        "are not both callable")


def admits_v2(candidate_scores: np.ndarray, existing_legs: np.ndarray, truth: np.ndarray,
              controls: np.ndarray | None = None, alpha: float = 0.05, n_perm: int = 500,
              min_gain: float = 0.01) -> tuple:
    """UPGRADE path over admits -- ADDITIVE, admits itself is left completely UNCHANGED (backward-compat).

    Fixes the cross-verified gap in admits: gating via partial SPEARMAN ONLY is (a) MONOTONE-BLIND
    (misses a strongly-predictive but non-monotone leg, e.g. folded/|distance|/extreme-value signals) and
    (b) has NO SIGNIFICANCE FLOOR (measured: a pure-noise/null-control leg admits via admits's min_gain-
    only screen far above the nominal false-positive rate a real gate should have -- cross-pool finding:
    ~13/20 by chance; see scripts/physics_exp/admits_v2_monotone_agnostic_validation.py for A's own
    reproduction). The original gap was cross-verified independently before this fix landed.

    Pipeline:
      (1) Load monotone_agnostic_admits (Szekely-Rizzo partial distance-correlation + label-permutation
          significance gate -- catches non-monotone dependence too, a strict superset of Spearman) via the
          sibling-module loader (_load_monotone_agnostic).
      (2) PRE-FILTER on SIGNIFICANCE: does the candidate carry ANY real, possibly-non-monotone, dependence
          on `truth` beyond the existing/control legs, beyond CHANCE (permutation-calibrated, not just a
          fixed min_gain screen)? This is the part that fixes the false-admit-on-noise bug -- the old gate
          had no significance floor at all.
      (3) The ORIGINAL admits (Spearman fused-gain) is ALSO run and reported as a side-by-side legacy
          diagnostic (unchanged, not gating admits_v2's decision) -- this is what surfaces the monotone-
          agnostic benefit concretely: a real non-monotone leg that the OLD gate rejects (~0 rank-gain,
          Spearman is blind to it) but that the NEW significance-floored gate correctly admits. Note
          `info["monotone_benefit_demo"]` flags exactly this case.
      (4) GRACEFUL DEGRADATION: if the module is unavailable at runtime (import fails in _load_monotone_agnostic) -- fall back to the ORIGINAL admits alone, with a clear
          warning in info["fallback"] (not a hard crash on a missing cross-worktree dependency).

    controls: partial out against these (n) or (n,k) controls for the dcor test; defaults to `existing_legs`
    (the same role existing_legs plays in the original admits/partial_coverage -- "unique beyond the
    ensemble"). alpha/n_perm: passed straight to the permutation significance test. min_gain: applied to
    the pdcor (same semantic role as the original admits's min_gain -- screens out significant-but-tiny
    effects that a huge n could otherwise rubber-stamp).

    Returns (admit: bool, info: dict)."""
    L = np.asarray(existing_legs, float)
    if L.ndim == 1:
        L = L[:, None]
    ctrl = L if controls is None else np.asarray(controls, float)

    fn, method, err = _load_monotone_agnostic()

    # legacy diagnostic: ALWAYS computed for side-by-side comparison; does not gate admits_v2's decision
    # when the module is available (it IS the decision when the module is unavailable -- see below).
    legacy_admit, legacy_info = admits(candidate_scores, L, truth, min_gain=min_gain)

    if fn is None:
        return legacy_admit, {
            "mode": "fallback_spearman_only",
            "b_module_available": False,
            "warning": ("monotone_agnostic_admission unavailable (%s) -- graceful degradation to the "
                        "ORIGINAL Spearman-only admits() gate. NOTE: this fallback has NEITHER the "
                        "significance floor NOR the monotone-agnostic benefit this function exists to add."
                        % err),
            "legacy_admits_v1": {"admit": legacy_admit, **legacy_info},
        }

    sig = fn(np.asarray(candidate_scores, float), np.asarray(truth, float), controls=ctrl,
              n_perm=n_perm, alpha=alpha)
    pdcor, p = sig.get("pdcor"), sig.get("p")
    sig_admits = bool(sig.get("admits", False))
    passes_min_gain = (pdcor is not None) and (pdcor > min_gain)
    admit = bool(sig_admits and passes_min_gain)

    if admit:
        reason = ("admit: monotone-agnostic significance gate clears -- pdcor=%.3f > min_gain=%.2f, "
                   "perm-p=%.3f < alpha=%.2f (possibly non-monotone dependence beyond controls, confirmed "
                   "above chance by label-permutation)" % (pdcor, min_gain, p, alpha))
    elif not sig_admits:
        reason = ("reject: no significant beyond-chance dependence (pdcor=%r, perm-p=%r vs alpha=%.2f) -- "
                   "this is the fix for the false-admit-on-noise bug" % (pdcor, p, alpha))
    else:
        reason = "reject: significant but below min_gain (pdcor=%.3f <= %.2f)" % (pdcor, min_gain)

    return admit, {
        "mode": "monotone_agnostic_significance_gated",
        "b_module_available": True,
        "b_import_method": method,
        "pdcor": pdcor, "perm_p": p, "alpha": alpha, "n_perm": n_perm, "min_gain": min_gain,
        "significance_gate_admits": sig_admits,
        "reason": reason,
        "legacy_admits_v1": {"admit": legacy_admit, **legacy_info},
        # True exactly when the new gate admits something the OLD Spearman-only gate rejected --
        # the concrete monotone-agnostic-benefit signal (not just the significance-floor benefit).
        "monotone_benefit_demo": bool((not legacy_admit) and admit),
    }


def _selftest():
    rng = np.random.default_rng(0)
    n = 3000
    # three independent latent failure sources; overall unreliability = worst-of-three (union of blind spots)
    a, b, c = rng.random(n), rng.random(n), rng.random(n)
    truth = np.maximum.reduce([a, b, c]) + rng.normal(0, 0.05, n)     # held-out reliability truth
    leg1 = a + rng.normal(0, 0.08, n)                                 # the ensemble captures failure sources a and b ...
    leg2 = b + rng.normal(0, 0.08, n)                                 # ... but is BLIND to source c
    ensemble = np.column_stack([leg1, leg2])

    # (1) COVERING candidate = source c (the axis the ensemble misses) -> ADMIT (real unique coverage -> fused lift)
    cand_cover = c + rng.normal(0, 0.08, n)
    # (2) NOISE candidate = pure independent noise: PAIRWISE-DECORRELATED from everything (would pass a naive rho gate!)
    # but predicts nothing -> REJECT (no coverage, no gain). Shows pairwise-decorr alone is NOT admissible.
    cand_noise = rng.random(n)
    # (3) REDUNDANT candidate = a linear combo of what the ensemble already has -> solo-predictive of truth (would pass a
    # solo gate!) but adds NO unique coverage -> REJECT. Shows solo-predictiveness alone is NOT admissible.
    cand_redun = 0.6 * leg1 + 0.4 * leg2 + rng.normal(0, 0.05, n)

    res = {}
    for name, cand in (("covering", cand_cover), ("noise", cand_noise), ("redundant", cand_redun)):
        ok, info = admits(cand, ensemble, truth)
        res[name] = (ok, info)
        print("%-10s: %s" % (name, info["reason"]))
        print("            gain=%+.3f coverage=%.3f solo=%.3f pairwise_min_decorr=%.3f"
              % (info["fused_gain"], info["partial_coverage"], info["solo_predictiveness"], info["pairwise_min_decorr"]))

    # the NOISE candidate is pairwise-decorrelated (fools a rho gate) yet must be REJECTED
    assert res["noise"][1]["pairwise_min_decorr"] < 0.15, "noise candidate is pairwise-decorrelated"
    assert not res["noise"][0], "pairwise-decorrelated pure noise must be REJECTED (no coverage)"
    # the REDUNDANT candidate is solo-predictive (fools a solo gate) yet must be REJECTED
    assert res["redundant"][1]["solo_predictiveness"] > 0.4, "redundant candidate is solo-predictive"
    assert not res["redundant"][0], "solo-predictive but redundant leg must be REJECTED (no unique coverage)"
    # the COVERING candidate must be ADMITTED
    assert res["covering"][0], "covering leg (unique axis) must be ADMITTED"

    # strength-weighted fusion must be >= equal-weight when legs differ in strength (a weak covering leg gets small weight)
    weak_cover = c + rng.normal(0, 0.5, n)                            # weak but genuinely covering
    legs_all = np.column_stack([leg1, leg2, weak_cover])
    sw = _spearman(strength_weighted_compose(legs_all, truth=truth), truth)
    se = _spearman(strength_weighted_compose(legs_all), truth)
    print("strength-weighted fusion vs truth=%.3f ; equal-weight=%.3f (weighted >= equal when strengths differ)" % (sw, se))
    assert sw >= se - 0.02, "strength-weighted should be >= equal-weight"

    print("leg_admission selftest: PASS -- the direct fused-gain gate ADMITS the unique-coverage leg and REJECTS both the "
          "pairwise-decorrelated pure-noise leg AND the solo-predictive-but-redundant leg (neither pairwise rho nor solo "
          "predictiveness is a sufficient admission criterion).")
    return True


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="leg_admission selftest runner")
    _ap.add_argument("--selftest", action="store_true", help="run the selftest (also the default action)")
    _ap.parse_args()
    _selftest()
