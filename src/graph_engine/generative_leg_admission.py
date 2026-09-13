"""generative_leg_admission -- the admission GATE for GENERATIVE leg-discovery (coordinator mandate 'generativ ben-upptaeckt').
Generation (distillation, augmentation, re-parameterization) is a COMMON-MODE SOURCE that the weight-lineage gate misses.

★THE GENERATIVE-SPECIFIC FAIL-OPEN: a leg DISTILLED/generated from a pool member (a student trained to mimic a teacher's
outputs) reproduces the member's ERROR STRUCTURE -- its errors correlate ~0.9+ with the teacher -- while sharing ZERO
weights. D's layer-1 weight-lineage gate (shared backbone = one leg) PASSES it (different weights), but it is JOINTLY BLIND:
booking it as a new decorrelated leg is a fail-open. Generated legs MUST be admitted on ERROR-CORRELATION with the pool
(layer-2, D's error-corr concept), not weight-lineage alone. Weight-lineage catches shared WEIGHTS; error-corr catches shared
ERRORS -- and generation shares errors without weights.

★Small-n discipline (CI-bound routing): LOW error-corr is the OPTIMISTIC (admittable) direction, so gate the
bootstrap CI-UPPER of the max error-corr with the pool -- admit only if even the upper bound is below tol. numpy only, no GPU.
selftest at bottom."""
from __future__ import annotations
import numpy as np


def _corr(a, b):
    if a.size < 3:
        return np.nan
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def bonferroni_ci_pctl(family_alpha, n_candidates):
    """Multiplicity-corrected CI percentile for a family of n_candidates generated legs: a GENERATIVE discovery run produces
    MANY candidates and admits those passing the error-corr gate, so the family-wise false-admit grows with n_candidates.
    Tighten the per-candidate CI-upper to the Bonferroni level so the EXPECTED number of false-admits stays < family_alpha
    regardless of how many candidates were generated. (@ -- winner's-curse on the admission gate itself.)"""
    k = max(int(n_candidates), 1)
    return 100.0 * (1.0 - family_alpha / k)


def _max_abs_corr(cand, pool):
    """MAX |error-corr| between cand and any pool member. Bare max/NaN interaction (the -155 class, python
    builtin min/max variant): NaN only 'wins' if it is the FIRST item evaluated; otherwise it is SILENTLY SKIPPED
    (nan > x is always False) -- so a NaN correlation from ONE corrupted pool member can be invisibly dropped from
    consideration, understating the true max whenever the corrupted member would have been the actual worst case.
    Guard explicitly: any non-finite pairwise corr makes the pool-max UNMEASURABLE -> return nan deterministically,
    not order-dependent."""
    vals = [abs(_corr(cand, p)) for p in pool]
    if any(not np.isfinite(v) for v in vals):
        return float("nan")
    return max(vals)


def _max_abs_corr_ci_upper(cand, pool, n_boot, seed, pctl=97.5):
    """bootstrap CI-upper of the MAX |error-corr| between the candidate and any pool member (held-out errors)."""
    m = cand.size
    rng = np.random.default_rng(seed)
    point = _max_abs_corr(cand, pool)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        c = cand[idx]
        v = _max_abs_corr(c, [p[idx] for p in pool])
        if np.isfinite(v):
            boots.append(v)
        # non-finite draws are SKIPPED (not silently coerced to a low value) -- if too many are skipped, `boots`
        # stays small/empty and the caller's isfinite-on-hi guard below forces ABSTAIN instead of a false CI-upper.
    boots = np.array(boots)
    hi = float(np.percentile(boots, pctl)) if boots.size >= max(10, int(0.1 * n_boot)) else float("nan")
    return point, hi


def generative_leg_admission(candidate_errvec, pool_errvecs, weight_lineage_shared=False, supervision_lineage_shared=False,
                             error_vs_decorrelated_source=True, corr_tol=0.30, min_n=30, n_boot=200, seed=0, ci_pctl=97.5):
    """candidate_errvec: the candidate leg's error vector on held-out data (estimate - ground_truth). pool_errvecs: list of
    the existing pool legs' held-out error vectors. weight_lineage_shared: True if D's layer-1 finds a shared backbone.
    supervision_lineage_shared: True if the candidate was trained on the SAME label source as a pool member (layer-1b --
    shared supervision = shared label bias = common-mode even with different weights AND different training samples, @).
    error_vs_decorrelated_source: the error vectors MUST be computed vs a DECORRELATED supervision source (true gt or an
    independent label set); if they are vs the SHARED (possibly-biased) labels, the systematic bias CANCELS and layer-2 is
    BLIND to shared-supervision common-mode -- set this False to force ABSTAIN (the layer-2 precondition is not met).
    Returns dict(verdict ADMIT-DECORRELATED / REJECT-LAYER1-WEIGHT-LINEAGE / REJECT-LAYER1B-SUPERVISION-LINEAGE /
    REJECT-LAYER2-ERROR-CORR / ABSTAIN-LOW-N / ABSTAIN-LAYER2-PRECONDITION, caught_layer, max_error_corr,...)."""
    cand = np.asarray(candidate_errvec, float)
    pool = [np.asarray(p, float) for p in pool_errvecs]
    cand = cand[np.isfinite(cand)]
    if not pool or cand.size < 3:
        return dict(verdict="ABSTAIN-LOW-N", caught_layer=None, reason="empty pool or too few samples")
    # LAYER-1: weight lineage (shared backbone = one leg) -- runs FIRST (standing rule)
    if weight_lineage_shared:
        return dict(verdict="REJECT-LAYER1-WEIGHT-LINEAGE", caught_layer=1,
                    reason="candidate shares a weight backbone with a pool member -> one leg regardless of declaration")
    # LAYER-1b: supervision lineage (shared label source = shared bias) -- weight-decorrelated but supervision-common-mode
    if supervision_lineage_shared:
        return dict(verdict="REJECT-LAYER1B-SUPERVISION-LINEAGE", caught_layer=1.5,
                    reason="candidate was trained on the SAME label source as a pool member -> shares the label bias "
                           "(common-mode) even with different weights and different training samples")
    # LAYER-2 PRECONDITION: the error must be vs a DECORRELATED supervision source, else the shared bias cancels (blind)
    if not error_vs_decorrelated_source:
        return dict(verdict="ABSTAIN-LAYER2-PRECONDITION", caught_layer=None,
                    reason="error vectors are vs the SHARED label source -> a shared supervision bias would CANCEL and "
                           "layer-2 error-corr is blind; recompute error vs a decorrelated source (true gt / independent labels)")
    if cand.size < min_n:
        return dict(verdict="ABSTAIN-LOW-N", caught_layer=None, max_error_corr=None,
                    reason=f"{cand.size} held-out samples < min_n {min_n}: cannot certify error-decorrelation")
    # LAYER-2: error correlation (catches distilled/generated common-mode that shares errors, not weights)
    n = min(cand.size, min(p.size for p in pool))
    cand2 = cand[:n]
    pool2 = [p[:n] for p in pool]
    point, hi = _max_abs_corr_ci_upper(cand2, pool2, n_boot, seed, pctl=ci_pctl)
    if not np.isfinite(hi):
        # `nan > corr_tol` is silently False in Python -- would otherwise fall through to ADMIT. A non-finite bound
        # means a pool member's held-out error vector is too NaN-corrupted to certify the max-corr bound at all.
        return dict(verdict="ABSTAIN-NONFINITE-CORR", caught_layer=None,
                    max_error_corr=(round(point, 4) if np.isfinite(point) else None), max_error_corr_ci_upper=None,
                    reason="max error-corr with the pool is UNMEASURABLE (NaN in a pool member's held-out errors "
                           "corrupts the bound) -- cannot certify decorrelation; ABSTAIN rather than silently ADMIT")
    if hi > corr_tol:
        return dict(verdict="REJECT-LAYER2-ERROR-CORR", caught_layer=2, max_error_corr=round(point, 4),
                    max_error_corr_ci_upper=round(hi, 4),
                    reason=(f"candidate error correlates with the pool (max |corr| point {point:.3f}, CI-upper {hi:.3f} > tol "
                            f"{corr_tol}) -> shares the pool's error structure (a generated/distilled common-mode), jointly "
                            f"blind despite no shared weights"))
    return dict(verdict="ADMIT-DECORRELATED", caught_layer=None, max_error_corr=round(point, 4),
                max_error_corr_ci_upper=round(hi, 4),
                reason=(f"candidate passes layer-1 (no shared weights) AND layer-2 (max error-corr CI-upper {hi:.3f} <= tol "
                        f"{corr_tol}) -> a genuinely decorrelated new leg"))


def _legs(n, seed):
    rng = np.random.default_rng(seed)
    x, x2, y = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    eT = 0.7 * x + 0.3 * rng.standard_normal(n)              # teacher: structured error on axis x
    eS = eT + 0.15 * rng.standard_normal(n)                  # distilled student: mimics teacher's error, 0 shared weights
    eI = 0.7 * x2 + 0.3 * rng.standard_normal(n)             # independent: structured error on a DIFFERENT axis
    return eT, eS, eI


def selftest():
    eT, eS, eI = _legs(500, 0)
    checks = []
    r_s = generative_leg_admission(eS, [eT], weight_lineage_shared=False)
    checks.append(("★distilled student passes layer-1 (0 shared weights) but REJECTED by layer-2 error-corr (common-mode)", r_s["verdict"] == "REJECT-LAYER2-ERROR-CORR" and r_s["caught_layer"] == 2))
    r_i = generative_leg_admission(eI, [eT], weight_lineage_shared=False)
    checks.append(("both-ways: independent leg ADMITTED (passes both layers; not over-rejected)", r_i["verdict"] == "ADMIT-DECORRELATED"))
    r_w = generative_leg_admission(eI, [eT], weight_lineage_shared=True)
    checks.append(("control: a shared-weight candidate rejected at LAYER-1 (before layer-2)", r_w["verdict"] == "REJECT-LAYER1-WEIGHT-LINEAGE" and r_w["caught_layer"] == 1))
    print("generative_leg_admission selftest:")
    for nm, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", nm))
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
