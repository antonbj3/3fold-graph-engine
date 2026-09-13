"""DECORRELATED ABSTAIN-FIELD CERT — ship L's fluctuation-gauge discipline as a callable primitive.

The cert: a per-pixel/per-item ABSTAIN field = rank-ENSEMBLE of DECORRELATED uncertainty gauges (input-perturbation
fluctuation PERP cross-member disagreement), VALID only if it clears a MANDATORY FLOOR-CHECK — it must beat AND add
BEYOND a naive difficulty baseline (edge/gradient content). A gauge that only matches the baseline is NOT a certificate.
This is L's  cross-channel finding (real depth + SR + EuRoC video); the floor-check is first-class because its
baseline strength is SCENE/CHANNEL-dependent (near-zero on textured SR, moderate on motion-blur video / depth-gradient),
so it is a genuine discriminator, never skip it.

  python -u -m graph_engine.decorrelated_abstain_field     # known-answer selftest (self-contained, no external data)

Empirical grounding: SR ens 0.818 vs edge-floor 0.071; depth ens 0.306 vs
depth-gradient 0.236; EuRoC-video ens 0.652 vs edge-floor 0.365 (all certify; all beat + add beyond their baseline)."""
import numpy as np

try:
    from scipy import stats
    from scipy.stats import rankdata as _rankdata            # ★L116 argsort-tie-bug fix: rankdata AVERAGES ties (argsort(argsort) does NOT)
    def _spearman(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
        return float(stats.spearmanr(a[m], b[m]).correlation)
except Exception:
    def _rankdata(a):                                        # numpy fallback: average-rank for ties (matches scipy 'average')
        a = np.asarray(a, float)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt); starts = csum - cnt
        return ((starts + csum + 1) / 2.0)[inv]
    def _spearman(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
        ra = _rankdata(a[m]); rb = _rankdata(b[m])           # ★tie-safe (was argsort(argsort), mis-ranks tied/discrete scores)
        return float(np.corrcoef(ra, rb)[0, 1])

def _rank_std(v):
    # fix (NaN -> identity-rank class): _rankdata on a raw gauge with NON-FINITE values ranks NaN as
    # an ordinary orderable value (np.unique sorts it) -> FABRICATES a position-keyed 0->1 ramp from ZERO information (an
    # all-NaN gauge became [0,.167,..,.833], and mixing it CORRUPTED the ensemble). Unlike _spearman (which isfinite-masks),
    # _rank_std did not. Fix: rank only the FINITE values; map non-finite -> 1.0 (MAX abstain -- an unmeasurable uncertainty
    # is maximally untrustworthy, fail-CLOSED, not a fabricated mid-range signal).
    v = np.asarray(v, float); n = v.size
    out = np.ones(n)                                          # non-finite -> 1.0 = max abstain (conservative)
    fin = np.isfinite(v)
    if fin.any():
        out[fin] = (_rankdata(v[fin]) - 1.0) / max(int(fin.sum()), 1)
    return out

def abstain_field(gauges):
    """gauges: list of ≥1 per-element uncertainty arrays. Returns the equal-weight rank-mean ensemble in [0,1] (high=abstain)."""
    # assert-under--O class (D/H/J agent pool disclosure, this tick): plain assert is stripped entirely by python -O.
    if len(gauges) < 1:
        raise ValueError("abstain_field requires at least 1 gauge")
    # ★: drop ENTIRELY non-finite gauges (they carry no information; keeping one would pull the whole field to abstain).
    fields = [_rank_std(np.asarray(g, float).ravel()) for g in gauges if np.isfinite(np.asarray(g, float)).any()]
    if not fields:
        return np.ones(np.asarray(gauges[0], float).ravel().shape)   # all gauges non-finite -> abstain everywhere
    return np.mean(fields, axis=0)

def decorrelation_matrix(gauges):
    """pairwise Spearman between gauges (near-1 => redundant/common-mode = not a real fusion)."""
    n = len(gauges); M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = _spearman(np.ravel(gauges[i]), np.ravel(gauges[j]))
    return M

def floor_check(field, error, baseline):
    """MANDATORY. Does `field` beat AND add BEYOND a naive difficulty `baseline` at predicting `error`?
    Returns {sp_field, sp_baseline, beats_baseline, partial_beyond_baseline, survives}."""
    error = np.asarray(error, float).ravel(); baseline = np.asarray(baseline, float).ravel(); field = np.asarray(field, float).ravel()
    sp_field = _spearman(field, error); sp_base = _spearman(baseline, error)
    A = np.c_[_rank_std(baseline), np.ones(baseline.size)]
    # C class (this tick): np.linalg.lstsq(NaN) can HANG (LAPACK DLASCL) rather than raise or return -- a DoS risk
    # worse than inv/solve's silent-NaN (those at least complete). Guard isfinite BEFORE the call, never rely on it.
    _rhs = _rank_std(error)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(_rhs))):
        raise ValueError("A/error contains non-finite values -- lstsq could hang")
    sol, *_ = np.linalg.lstsq(A, _rhs, rcond=None)
    partial = _spearman(field, _rank_std(error) - A @ sol)
    return {"sp_field": sp_field, "sp_baseline": sp_base, "beats_baseline": bool(sp_field > sp_base),
            "partial_beyond_baseline": partial, "survives": bool(sp_field > sp_base and abs(partial) > 0.10)}

def tail_heaviness(x):
    """Cheap tail-index proxy: 99pct(|x-median|) / MAD. Gaussian ~3.4; heavy-tailed (Student-t ν<2, α-stable) >> that."""
    d = np.abs(np.asarray(x, float).ravel() - np.median(x))
    return float(np.percentile(d, 99) / (np.median(d) + 1e-12))

def _trimmed_mean(M, tau):
    if tau <= 0: return M.mean(axis=0)
    Ms = np.sort(M, axis=0); n = M.shape[0]; k = int(round(n * tau))
    return Ms[k:n - k].mean(axis=0) if n - 2 * k >= 1 else np.median(M, axis=0)

def robust_ensemble_aggregate(members, heavy_thresh=6.0, force=None, light_trim=0.1):
    """thread-17/19 LOAD-BEARING sharpening + L's thread-20 breakdown-point refinement, deployed. Aggregating N DECORRELATED
    members buys √N precision under MEAN only in LIGHT tails; in HEAVY tails (ν<~1) the mean REVERSES (single big-jump dominates,
    diversity HURTS). ★L thread-20 finding: it is NEAR-BINARY in the breakdown point — the MEAN (bp=0) is the UNIQUE failure; ANY
    positive breakdown point (even a 5-10% trim) recovers it, and a light trim BEATS the mean even in light tails at ~zero cost.
    So this NEVER uses the pure mean: MEDIAN for estimated-heavy tails, a LIGHT TRIMMED-MEAN otherwise (robust even if the
    tail-heaviness estimate MISFIRES). members = list of N per-item arrays (or N×M array). Returns (aggregate, info)."""
    M = np.asarray(members, float)
    th = tail_heaviness(M)
    heavy = (th > heavy_thresh) if force is None else (force == "median")
    agg = np.median(M, axis=0) if heavy else _trimmed_mean(M, light_trim)
    return agg, {"tail_heaviness": round(th, 2), "aggregator": ("median" if heavy else "trim%d" % int(light_trim * 100)),
                 "n_members": M.shape[0], "breakdown_point": (0.5 if heavy else light_trim),
                 "reason": "heavy→median (bp0.5)" if heavy else "light→light-trim (bp>0, never pure mean — safe under estimator misfire)"}

def tail_coupling_check(gauges, error, tail_q=0.90):
    """thread-15/16 TAIL-COUPLING guard: two gauges can be 2nd-moment-DECORRELATED (low ρ) yet co-FAIL on the same rare
    high-error tail (copula tail-dependence = a common-mode INVISIBLE to ρ). Needs a labelled `error` (validation-time).
    tail_excess = P(all gauges below-median | high-error tail) / P(all below-median overall); >1 => tail-coupled hole.
    A cert is validated tail-SAFE only if tail_excess is not >1 (the gauges FIRE on hard cases, not co-certify them)."""
    error = np.asarray(error, float).ravel()
    lows = [np.asarray(g, float).ravel() < np.median(np.asarray(g, float).ravel()) for g in gauges]
    all_low = np.logical_and.reduce(lows)
    # fix (reject-polarity percentile-NaN class): np.quantile(error, q)
    # propagates NaN if error has ANY non-finite value -> `error >= NaN` is False for EVERY item -> tail empty ->
    # tail_cm=0 -> excess=0 -> tail_coupled=False -> ONE NaN error value HID a genuinely tail-coupled hole
    # (measured True->False). Fix: nanquantile threshold (the FINITE tail is still detected); and if too few finite error
    # values remain to assess the tail, FAIL-CLOSED (tail_coupled=True, cannot validate tail-safety) rather than report safe.
    finite_err = np.isfinite(error)
    if finite_err.sum() < max(3, int(0.05 * error.size)):
        return {"tail_excess": float("inf"), "bulk_co_certify": float(all_low.mean()), "tail_co_certify": float("nan"),
                "tail_coupled": True, "reason": "insufficient finite error values -- cannot validate tail-safety (fail-closed)"}
    bulk = float(all_low.mean()); tail = error >= np.nanquantile(error, tail_q)
    tail_cm = float(all_low[tail].mean()) if tail.any() else 0.0
    excess = tail_cm / (bulk + 1e-9)
    return {"tail_excess": excess, "bulk_co_certify": bulk, "tail_co_certify": tail_cm,
            "tail_coupled": bool(excess > 1.3)}   # >1.3 = co-certify the hard tail MORE than the bulk = a tail hole

def certify_abstain_field(gauges, error=None, baseline=None, decorr_max=0.85):
    """Full cert. gauges -> ensemble abstain field + (if error given) validation. `certified` iff the ensemble beats
    every single gauge AND (if a baseline is given) clears the mandatory floor-check."""
    field = abstain_field(gauges)
    D = decorrelation_matrix(gauges); off = D[np.triu_indices(len(gauges), 1)]
    out = {"n_gauges": len(gauges), "decorrelation_matrix": D.tolist(),
           "max_pairwise_corr": float(np.max(np.abs(off))) if len(off) else 0.0,
           "genuinely_decorrelated": bool(len(off) == 0 or np.max(np.abs(off)) < decorr_max)}
    if error is not None:
        error = np.asarray(error, float).ravel()
        sp_singles = [_spearman(np.ravel(g), error) for g in gauges]
        out["sp_singles"] = sp_singles; out["sp_ensemble"] = _spearman(field, error)
        # NaN comparison hazard: Python's builtin max over sp_singles
        # is order-dependent under NaN -- if a corrupted/degenerate gauge's NaN score is not the FIRST list element,
        # max silently drops it and returns the largest REAL score, fabricating ensemble_beats_singles=True even
        # though the ensemble's true standing against the corrupted gauge's (unknown) score can't be verified.
        # Fail-closed: any non-finite single score means "beats singles" cannot be certified.
        out["ensemble_beats_singles"] = bool(np.isfinite(out["sp_ensemble"])
                                             and all(np.isfinite(s) for s in sp_singles)
                                             and out["sp_ensemble"] > max(sp_singles))
        if baseline is not None:
            out["floor_check"] = floor_check(field, error, baseline)
        out["tail_coupling"] = tail_coupling_check(gauges, error) if len(gauges) >= 2 else {"tail_coupled": False}
        out["certified"] = bool(out["ensemble_beats_singles"]
                                and (baseline is None or out["floor_check"]["survives"])
                                and not out["tail_coupling"]["tail_coupled"])   # tail-coupled => a tail hole => NOT certified
    return field, out

def _selftest():
    """KNOWN-ANSWER (self-contained, no external data): (POS) two decorrelated gauges that both track a known error, with
    a moderate baseline, CERTIFY. (NEG-1) two pure-noise gauges do NOT certify. (NEG-2) a gauge that IS the baseline FAILS
    the floor-check. Proves the cert accepts real decorrelated gauges and rejects both noise and an edge-detector-in-disguise."""
    rng = np.random.RandomState(0); n = 3000
    err = rng.rand(n)
    g1 = err + 0.8 * rng.randn(n)                 # decorrelated gauge 1 (tracks err, indep noise)
    g2 = err + 0.8 * rng.randn(n)                 # decorrelated gauge 2 (tracks err, indep noise)
    base = 0.4 * err + 1.2 * rng.randn(n)         # a MODERATE naive baseline (partly predictive, like depth-gradient)
    noise1, noise2 = rng.randn(n), rng.randn(n)   # pure-noise gauges
    field, cert = certify_abstain_field([g1, g2], error=err, baseline=base)
    _, cert_noise = certify_abstain_field([noise1, noise2], error=err, baseline=base)
    edge_only = floor_check(_rank_std(base), err, base)
    # NEG-3: a TAIL-COUPLED camouflage pair (thread-15) -- ρ~0 but both gauges suppressed low on rare high-error cases.
    camo = rng.rand(n) < 0.04
    tg1 = np.where(camo, rng.rand(n) * 0.12, rng.rand(n)); tg2 = np.where(camo, rng.rand(n) * 0.12, rng.rand(n))
    terr = np.where(camo, 0.75 + 0.25 * rng.rand(n), 0.30 * rng.rand(n))
    _, cert_tail = certify_abstain_field([tg1, tg2], error=terr, baseline=0.4 * terr + rng.randn(n))
    # NEG-4 / thread-17: robust aggregation of N decorrelated members. HEAVY (Cauchy) -> median chosen + far closer to truth than mean.
    truth = 5.0; K = 16
    heavy_members = [truth + rng.standard_t(1, 400) for _ in range(K)]     # Cauchy-ish noise around truth
    light_members = [truth + rng.randn(400) for _ in range(K)]
    agg_h, info_h = robust_ensemble_aggregate(heavy_members)
    agg_l, info_l = robust_ensemble_aggregate(light_members)
    mean_h = np.mean(heavy_members, axis=0)
    err_median_h = float(np.median(np.abs(agg_h - truth))); err_mean_h = float(np.median(np.abs(mean_h - truth)))
    checks = {
        "POS: real decorrelated gauges certify": cert["certified"],
        "POS: ensemble beats both singles": cert["ensemble_beats_singles"],
        "POS: floor-check survives moderate baseline": cert["floor_check"]["survives"],
        "POS: real gauges NOT tail-coupled": (not cert["tail_coupling"]["tail_coupled"]),
        "NEG-1: pure-noise gauges do NOT certify": (not cert_noise["certified"]),
        "NEG-2: baseline-only gauge FAILS floor-check": (not edge_only["survives"]) and abs(edge_only["partial_beyond_baseline"]) < 0.05,
        "NEG-3: TAIL-COUPLED camouflage is FLAGGED + NOT certified (hidden hole)": cert_tail["tail_coupling"]["tail_coupled"] and (not cert_tail["certified"]),
        "THREAD-17: heavy-tail members -> robust_aggregate picks MEDIAN": info_h["aggregator"] == "median",
        "THREAD-17: median beats mean on heavy-tail members (closer to truth)": err_median_h < err_mean_h,
        "THREAD-17/20: light-tail members -> light-trim (bp>0, NEVER pure mean, safe under estimator misfire)": info_l["breakdown_point"] > 0 and info_l["aggregator"] != "mean",
        "field in [0,1]": bool(field.min() >= 0 and field.max() <= 1),
    }
    ok = all(checks.values())
    print("decorrelated_abstain_field selftest: %s" % ("PASS %d/%d" % (sum(checks.values()), len(checks)) if ok else "FAIL %d/%d" % (sum(checks.values()), len(checks))))
    for k, v in checks.items(): print("  [%s] %s" % ("x" if v else " ", k))
    print("  pos: sp_ensemble=%.3f singles=%s floor(sp_base=%.3f partial=%.3f) | neg-noise certified=%s" % (
        cert["sp_ensemble"], [round(s, 3) for s in cert["sp_singles"]], cert["floor_check"]["sp_baseline"],
        cert["floor_check"]["partial_beyond_baseline"], cert_noise["certified"]))
    return ok

if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
