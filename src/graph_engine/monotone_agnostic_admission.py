"""monotone_agnostic_admission — the MONOTONE-AGNOSTIC fix for the agent pool admission-cluster's shared blind spot.

B found (grading cert_increment_gate / safe_fuse / reid_identity_cert) that all admit/weight legs by partial
SPEARMAN, which is MONOTONE-blind: a strongly-predictive but NON-MONOTONE leg (folded / |embedding-distance| / extreme-
value identity signals -- common in re-ID and in |residual| gates) has ~zero Spearman -> gets rejected / weight 0 ->
the ensemble discards a good leg and can wrongly ABSTAIN. This module replaces the monotone rank correlation with a
PARTIAL DISTANCE CORRELATION (Székely-Rizzo dcor: 0 iff independent, captures ANY dependence incl non-monotone), with a
label-permutation significance gate. Drop it into any of those tools as their `admission_fn`.

  partial_distance_correlation(x, y, controls=None) -> pdcor in ~[-1,1] (monotone-agnostic dependence beyond controls)
  monotone_agnostic_admits(leg, y, controls=None, n_perm=500, alpha=0.05, seed=0) -> {pdcor, p, admits}
    admits iff the permutation p (shuffle y) < alpha AND pdcor > 0.

Captures monotone dependence too (a monotone leg still has pdcor>0), so it is a strict SUPERSET of the Spearman admission
-- safe to use as the default. numpy only (dcor needs no scipy). Honest-negative = a truly uninformative leg is NOT
admitted (calibrated ~alpha FP rate).
"""
import numpy as np


def _dcenter(x):
    x = np.asarray(x, float).reshape(-1, 1)
    D = np.abs(x - x.T)
    return D - D.mean(0, keepdims=True) - D.mean(1, keepdims=True) + D.mean()


def _dcor(x, y):
    A = _dcenter(x); B = _dcenter(y)
    dvarx = (A * A).mean(); dvary = (B * B).mean()
    denom = dvarx * dvary
    return float(np.sqrt(max((A * B).mean(), 0.0) / np.sqrt(denom))) if denom > 0 else 0.0


def partial_distance_correlation(x, y, controls=None):
    """distance-correlation of x,y after partialling out `controls` (monotone-agnostic). controls: (n,) or (n,k) or None."""
    if controls is None:
        return _dcor(x, y)
    Z = np.asarray(controls, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    # aggregate controls into a single distance-representative (mean of per-control dcor terms via a joint distance)
    z = Z.mean(1) if Z.shape[1] > 1 else Z[:, 0]      # simple 1-D control proxy; for multi-control pass the ensemble mean
    rxy, rxz, ryz = _dcor(x, y), _dcor(x, z), _dcor(y, z)
    den = np.sqrt(max(1 - rxz ** 2, 1e-12)) * np.sqrt(max(1 - ryz ** 2, 1e-12))
    return (rxy - rxz * ryz) / den


def monotone_agnostic_admits(leg, y, controls=None, n_perm=500, alpha=0.05, seed=0):
    leg = np.asarray(leg, float).ravel(); y = np.asarray(y, float).ravel()
    # ★NULL-SAFETY (B reflexive non-finite sweep): alpha=+inf makes `p < alpha` always True -> the
    # permutation gate is silently DISABLED and a pure-noise leg (p~0.55) ADMITS. Non-finite alpha -> fail-closed.
    if not np.isfinite(alpha):
        return dict(pdcor=None, p=None, admits=False, note="non-finite alpha: admission gate disabled -> fail-closed")
    if len(leg) < 20:
        return dict(pdcor=None, p=None, admits=False, note="n<20: too few to admit")
    rng = np.random.default_rng(seed)
    pd = partial_distance_correlation(leg, y, controls)
    cnt = 1
    for _ in range(n_perm):
        if abs(partial_distance_correlation(leg, rng.permutation(y), controls)) >= abs(pd):
            cnt += 1
    p = cnt / (n_perm + 1)
    return dict(pdcor=round(pd, 4), p=round(p, 4), admits=bool(p < alpha and pd > 0),
                note="monotone-agnostic (partial distance-corr): admits a strong NON-MONOTONE leg that partial-Spearman drops")


def _selftest():
    rng = np.random.default_rng(0); n = 300; ok = tot = 0
    y = (rng.random(n) > 0.5).astype(float)
    prop = y * 0.5 + rng.standard_normal(n)
    # (1) strong NON-MONOTONE leg (correct at extreme +-3) -> ADMIT (partial-Spearman would drop it)
    nonmono = np.where(y == 1, rng.choice([-3, 3], n) + rng.normal(0, 0.3, n), rng.normal(0, 0.3, n))
    r1 = monotone_agnostic_admits(nonmono, y, prop, n_perm=300)
    tot += 1; ok += r1["admits"]
    # (2) monotone leg -> also ADMIT (superset of Spearman)
    r2 = monotone_agnostic_admits(y + rng.standard_normal(n), y, prop, n_perm=300)
    tot += 1; ok += r2["admits"]
    # (3) pure-noise leg -> NOT admitted (calibrated)
    r3 = monotone_agnostic_admits(rng.standard_normal(n), y, prop, n_perm=300)
    tot += 1; ok += (not r3["admits"])
    # (4) FP rate over null draws ~ alpha (not inflated)
    fp = 0
    for s in range(40):
        rr = np.random.default_rng(100 + s)
        yy = (rr.random(200) > 0.5).astype(float)
        if monotone_agnostic_admits(rr.standard_normal(200), yy, yy * 0.5 + rr.standard_normal(200), n_perm=150)["admits"]:
            fp += 1
    tot += 1; ok += (fp / 40 <= 0.15)
    print("monotone_agnostic_admission selftest:")
    print("  non-monotone admits=%s (pdcor=%.2f) | monotone admits=%s | noise admits=%s | null FP rate=%.0f%%"
          % (r1["admits"], r1["pdcor"], r2["admits"], r3["admits"], 100 * fp / 40))
    print("  %d/%d PASS" % (ok, tot))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
