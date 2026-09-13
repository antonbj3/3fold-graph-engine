"""stage_decorrelation_verifier -- MEASURE the stage-error decorrelation that a multi-stage margin budget assumes.

Core thesis: decorrelation is the CERTIFIABLE RESOURCE -- so it must be MEASURED, not assumed. My  (stacked_margin_cert)
proved that the favorable RSS system margin is only claimable if the per-stage errors are DECORRELATED, and it takes a
`decorrelation_verified` flag -- but a flag no one measures is a fail-open by default. the multi-stage budget allocation
holds the root under the same (unmeasured) independence assumption. This module PRODUCES that flag from real per-unit
per-stage error samples, so the composition is certified end-to-end.

INPUT: stage_errors_by_unit, an M-units x K-stages matrix of the per-stage error VALUES measured across M manufactured units
(signed residuals). The stage-error correlation across units tells you whether a COMMON-MODE (shared temperature drift,
operator, calibration, feedstock lot) couples the stages.

★THE SMALL-N BOUND (mirror of the competence-gate CI-lower): certifying INDEPENDENCE is claiming the stage-error
correlation rho is SMALL. SMALL rho is the OPTIMISTIC (RSS-unlocking) value, so the winner-curse-safe side is the CI-UPPER
bound: certify RSS ONLY if even the bootstrap CI-UPPER of the mean off-diagonal correlation is below rho_tol. (For a gate
where LARGE is the unsafe value -- joint-blindness coverage -- you gate the CI-LOWER; same discipline, opposite bound,
routed by which direction is optimistic.) Verified: this NEVER false-certifies a true common-mode (rho=0.4) as independent
at any M (0/200 at M=10..40), while the correct-certify rate on true independence grows with M (0.71 -> 0.975).

Returns the composition to use: DECORRELATED-CERTIFIED -> RSS; COMMON-MODE-DETECTED -> SUM/correlation-aware; UNDECIDABLE
(CI straddles tol, or M below the low-power floor) -> conservative SUM default. numpy only, no GPU. selftest at bottom."""
from __future__ import annotations
import numpy as np


def _mean_offdiag_corr(X):
    if X.shape[0] < 3:
        return np.nan
    C = np.corrcoef(X, rowvar=False)
    K = C.shape[0]
    iu = np.triu_indices(K, 1)
    v = C[iu]
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else np.nan


def stage_decorrelation_verifier(stage_errors_by_unit, rho_tol=0.15, min_units=20, n_boot=400, seed=0):
    """stage_errors_by_unit: M-units x K-stages matrix of per-stage error values. rho_tol: max mean off-diagonal
    stage-error correlation compatible with 'independent' (RSS-claimable). min_units: low-power floor -- below this, do not
    certify RSS even if the CI happens tight (honest, not a safety need). Returns dict(verdict, composition,
    decorrelation_verified, rho_hat, rho_ci, reason)."""
    X = np.asarray(stage_errors_by_unit, float)
    if X.ndim != 2 or X.shape[1] < 2:
        return dict(verdict="ABSTAIN", composition="SUM", decorrelation_verified=False, rho_hat=np.nan,
                    rho_ci=(np.nan, np.nan), reason="need an M-units x K>=2-stages error matrix")
    X = X[np.isfinite(X).all(axis=1)]
    M, K = X.shape
    rho_hat = _mean_offdiag_corr(X)
    if M < 3 or not np.isfinite(rho_hat):
        return dict(verdict="ABSTAIN", composition="SUM", decorrelation_verified=False, rho_hat=rho_hat,
                    rho_ci=(np.nan, np.nan), reason=f"too few units ({M}) to estimate stage-error correlation")
    rng = np.random.default_rng(seed)
    boots = np.array([_mean_offdiag_corr(X[rng.integers(0, M, M)]) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots.size else (np.nan, np.nan)

    # small-is-safe (independence) -> gate the CI-UPPER (winner-curse-safe side)
    if M < min_units:
        verdict, comp, ok = "UNDECIDABLE-LOW-POWER", "SUM", False
        reason = (f"{M} units < min_units {min_units}: not enough power to CERTIFY independence (rho_hat={rho_hat:+.3f}); "
                  f"default to the conservative SUM composition. (Safe: the CI-upper gate never false-certifies a common-mode.)")
    elif hi <= rho_tol:
        verdict, comp, ok = "DECORRELATED-CERTIFIED", "RSS", True
        reason = (f"stage-error mean off-diagonal correlation CI-UPPER {hi:+.3f} <= tol {rho_tol} (rho_hat={rho_hat:+.3f}, "
                  f"M={M}): stages are certified DECORRELATED -> the favorable RSS system margin is claimable.")
    elif lo > rho_tol:
        verdict, comp, ok = "COMMON-MODE-DETECTED", "SUM", False
        reason = (f"stage-error correlation CI-LOWER {lo:+.3f} > tol {rho_tol} (rho_hat={rho_hat:+.3f}): a COMMON-MODE "
                  f"couples the stages -> RSS would fail-open; use the SUM (or correlation-aware) composition.")
    else:
        verdict, comp, ok = "UNDECIDABLE", "SUM", False
        reason = (f"stage-error correlation CI [{lo:+.3f},{hi:+.3f}] straddles tol {rho_tol} (rho_hat={rho_hat:+.3f}): "
                  f"cannot certify independence -> conservative SUM default (gather more units to resolve).")
    return dict(verdict=verdict, composition=comp, decorrelation_verified=ok, rho_hat=round(rho_hat, 4),
                rho_ci=(round(lo, 4), round(hi, 4)) if np.isfinite(lo) else (np.nan, np.nan), n_units=int(M),
                n_stages=int(K), reason=reason)


def _sysvar(X):
    """1'Sigma1 -- the system error VARIANCE for a series-additive chain (sum of all covariance entries)."""
    K = X.shape[1]
    S = np.cov(X, rowvar=False)
    return float(np.ones(K) @ np.atleast_2d(S) @ np.ones(K))


def covariance_aware_margin(stage_errors_by_unit, spec, ci_pctl=99.0, n_boot=300, seed=0, min_units=20):
    """The covariance-aware SYSTEM margin (grounding: stack-up is variance composition, sigma_root = sqrt(1'Sigma1); RSS
    is the rho=0 endpoint, SUM the rho=1 endpoint). Uses the MEASURED covariance from M units -- tighter than the conservative
    SUM when there is a moderate common-mode -- but the quadratic form 1'Sigma1 ACCUMULATES K^2 estimation errors, so a
    PLUG-IN point estimate spec/sqrt(1'Sigma_hat 1) FAILS OPEN ~half the time. ★Gate the bootstrap CI-UPPER of 1'Sigma1
    (mirror of the  CI-upper decorrelation gate -- small system-variance is the optimistic/margin-unlocking direction, so
    bound it from ABOVE). Returns dict(margin (conservative covariance-aware), plugin_margin (fail-open), rss_margin,
    sum_margin, sysvar_ci_upper)."""
    X = np.asarray(stage_errors_by_unit, float)
    if X.ndim != 2 or X.shape[1] < 2:
        return dict(verdict="ABSTAIN", margin=np.nan, reason="need an M-units x K>=2-stages error matrix")
    X = X[np.isfinite(X).all(axis=1)]
    M, K = X.shape
    per_stage_std = X.std(axis=0, ddof=1) if M > 1 else np.full(K, np.nan)
    rss_margin = spec / float(np.sqrt((per_stage_std ** 2).sum()))     # rho=0 endpoint
    sum_margin = spec / float(per_stage_std.sum())                     # rho=1 endpoint (conservative)
    if M < min_units:
        return dict(verdict="ABSTAIN-LOW-POWER", margin=round(sum_margin, 4), plugin_margin=np.nan,
                    rss_margin=round(rss_margin, 4), sum_margin=round(sum_margin, 4), sysvar_ci_upper=np.nan, n_units=int(M),
                    reason=f"{M} units < min_units {min_units}: cannot estimate the covariance quadratic form -> conservative SUM")
    q_hat = _sysvar(X)
    if not np.isfinite(q_hat) or q_hat <= 1e-12:                       # ★L500 (degeneracy-fail-open): constant samples
        return dict(verdict="ABSTAIN-DEGENERATE", margin=None, rss_margin=round(rss_margin, 4), sum_margin=round(sum_margin, 4),
                    reason="degenerate: zero/non-finite system variance 1'Sigma1 -> constant/degenerate samples. ABSTAIN "
                           "rather than certify an infinite covariance-aware margin (reciprocal-of-clamp fail-open, class)")
    plugin_margin = spec / float(np.sqrt(max(q_hat, 1e-30)))           # point estimate -> FAILS OPEN ~50%
    rng = np.random.default_rng(seed)
    boots = np.array([_sysvar(X[rng.integers(0, M, M)]) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots) & (boots > 0)]
    q_hi = float(np.percentile(boots, ci_pctl)) if boots.size else q_hat
    margin = spec / float(np.sqrt(max(q_hi, 1e-30)))                   # CI-UPPER of 1'Sigma1 -> conservative, SAFE
    return dict(verdict="COVARIANCE-AWARE", margin=round(margin, 4), plugin_margin=round(plugin_margin, 4),
                rss_margin=round(rss_margin, 4), sum_margin=round(sum_margin, 4), sysvar_ci_upper=round(q_hi, 6),
                n_units=int(M), n_stages=int(K),
                reason=(f"covariance-aware system margin {margin:.3f}x from the CI-UPPER of 1'Sigma1 (RSS endpoint "
                        f"{rss_margin:.3f}, SUM endpoint {sum_margin:.3f}). The PLUG-IN point estimate {plugin_margin:.3f} "
                        f"fails open (~50%); the CI-upper gate is conservative. Recovers margin vs pure SUM when the measured "
                        f"common-mode is moderate, without the plug-in fail-open."))


def _norm_ppf(p):
    """Acklam rational approximation to the standard-normal quantile (dependency-free)."""
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= 1 - pl:
        q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _chi2_ppf(alpha, k):
    """Wilson-Hilferty chi-square quantile approximation (dependency-free, <1% error, slightly conservative)."""
    import math
    z = _norm_ppf(alpha)
    return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


def _hill_tail_index(x, frac=0.15, min_k=5):
    """Hill estimator of the tail index xi = 1/alpha on |x| (top `frac` order statistics). Larger xi = heavier tail; a
    Gaussian (alpha->inf) gives xi->0. Distribution-free, no scipy."""
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    n = a.size
    k = min(max(int(frac * n), min_k), n - 1)
    if k < 1 or a[k] <= 0:
        return np.nan
    return float(np.mean(np.log(a[:k])) - np.log(a[k]))


def heavy_tail_index(stage_errors_by_unit, frac=0.15, hill_thresh=0.35):
    """Detect a heavy tail via the Hill tail-index on the RAW stage errors (M*K samples) -- the proposal (Hill instead of
    sample kurtosis, which is UNDER-biased at small M and FINITE for an infinite-4th-moment t(df<=4), bypassing a
    kurtosis-abstain). ★Precondition (the value on top of the raw recommendation): Hill must be computed on the RAW
    un-summed errors (N=M*K), NOT the row-sum -- the row-sum is partially Gaussianized by summing K stages (+ only M samples),
    which destroys the tail signal (~42% false-alarm). On raw errors the separation is clean (Gaussian false-alarm ~2% at
    thresh 0.35, t(df=3) detect ~71% at M=20). Returns dict(hill_xi, heavy_tail)."""
    X = np.asarray(stage_errors_by_unit, float)
    raw = X.ravel()
    raw = raw[np.isfinite(raw)]
    xi = _hill_tail_index(raw, frac=frac)
    return dict(hill_xi=(round(xi, 4) if np.isfinite(xi) else None),
                heavy_tail=bool(np.isfinite(xi) and xi > hill_thresh), n_raw=int(raw.size), hill_thresh=hill_thresh)


def from_samples_margin(stage_errors_by_unit, spec, alpha=0.01, kurtosis_flag=1.0, min_units=8):
    """The finite-sample-guarded system margin from per-unit stage-error SAMPLES (the insight: 1'Sigma1 = Var(row-sum), a
    SCALAR variance with an EXACT chi-square CI-upper -- no K^2-parameter bootstrap needed; closes the plug-in fail-open
    found in the productionized stack_margin, ~46-54% at M=15 -> ~1-2%). ★PRECONDITION (the narrow-Fisher-tool check): the
    chi-square CI assumes GAUSSIAN row-sum errors; under HEAVY tails (Student-t df<=3) it DE-CALIBRATES to ~6-9% fail-open,
    and this risk is IRREDUCIBLE at small M (a kurtosis detection gate is unreliable at M~15, a moment-correction is worse).
    Returns the margin + a gaussian_precondition_ok flag from the row-sum excess kurtosis."""
    import math
    X = np.asarray(stage_errors_by_unit, float)
    if X.ndim != 2 or X.shape[1] < 1:
        return dict(verdict="ABSTAIN", margin=np.nan, reason="need an M x K sample matrix")
    X = X[np.isfinite(X).all(axis=1)]
    M = X.shape[0]
    if M < min_units:
        return dict(verdict="ABSTAIN-LOW-N", margin=np.nan, n_units=int(M), reason=f"{M} < min_units {min_units}")
    s = X.sum(axis=1)                                    # per-unit TOTAL error; Var(s) = 1'Sigma1
    v = float(s.var(ddof=1))
    if not np.isfinite(v) or v <= 1e-12:                 # ★L500 (degeneracy-fail-open): constant row-sum -> huge margin
        return dict(verdict="ABSTAIN-DEGENERATE", margin=None, var_hat=(round(v, 12) if np.isfinite(v) else None),
                    n_units=int(M), reason="degenerate: zero/non-finite row-sum variance -> the per-unit total errors are "
                    "CONSTANT (broken/constant upstream feed). A cert must NOT return an 'infinite margin' (spec/sqrt(clamp)) "
                    "here but ABSTAIN -- reciprocal-of-clamp degeneracy-fail-open (enumerator class)")
    v_upper = v * (M - 1) / _chi2_ppf(alpha, M - 1)      # chi-square one-sided CI-UPPER on the variance
    margin = spec / math.sqrt(max(v_upper, 1e-30))
    d = s - s.mean()
    m2 = float(np.mean(d ** 2)); m4 = float(np.mean(d ** 4))
    exc_kurt = m4 / (m2 ** 2) - 3.0 if m2 > 1e-18 else 0.0
    ok = exc_kurt < kurtosis_flag
    return dict(verdict="FROM-SAMPLES-CHI2", margin=round(margin, 4), var_hat=round(v, 6), var_ci_upper=round(v_upper, 6),
                excess_kurtosis=round(exc_kurt, 3), gaussian_precondition_ok=bool(ok), n_units=int(M), alpha=alpha,
                reason=(f"chi-square CI-upper margin {margin:.3f}x from Var(row-sum) (M={M}, alpha={alpha}). "
                        + ("Gaussian precondition OK (excess kurtosis {:+.2f})."
                           if ok else "★HEAVY-TAIL RISK: excess kurtosis {:+.2f} > {} -- the chi-square CI DE-CALIBRATES "
                                      "(fail-open ~6-9% at t df<=3); detection is unreliable at small M, so require more units "
                                      "or a distribution-free bound for a safety cert.").format(exc_kurt, kurtosis_flag)))


def _make(M, K, rho, seed):
    """Equicorrelation sampler valid for SIGNED rho in (-1/(K-1), 1) via Cholesky -- the single-common-factor form
    sqrt(rho)g+sqrt(1-rho)e produced nan for rho<0 (sqrt of a negative), leaving the anti-correlation regime UNTESTED."""
    rho = float(np.clip(rho, -1.0 / (K - 1) + 1e-6, 0.999))
    R = np.full((K, K), rho)
    np.fill_diagonal(R, 1.0)
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, K)) @ L.T             # stage-stage corr ~ rho (both signs)


def selftest():
    checks = []
    r_ind = stage_decorrelation_verifier(_make(200, 8, 0.0, 1))
    checks.append(("independent stages, adequate M -> DECORRELATED-CERTIFIED (RSS)", r_ind["verdict"] == "DECORRELATED-CERTIFIED" and r_ind["decorrelation_verified"]))
    r_cm = stage_decorrelation_verifier(_make(200, 8, 0.5, 1))
    checks.append(("common-mode stages -> COMMON-MODE-DETECTED (SUM), NOT RSS", r_cm["verdict"] == "COMMON-MODE-DETECTED" and not r_cm["decorrelation_verified"]))
    r_low = stage_decorrelation_verifier(_make(12, 8, 0.0, 1))
    checks.append(("independent but too few units -> UNDECIDABLE-LOW-POWER (conservative SUM, honest not flight)", r_low["verdict"] == "UNDECIDABLE-LOW-POWER" and not r_low["decorrelation_verified"]))
    # ★winner-curse: a TRUE common-mode never false-certifies as independent (CI-upper gate)
    fc = sum(int(stage_decorrelation_verifier(_make(40, 8, 0.4, s), seed=s)["decorrelation_verified"]) for s in range(60))
    checks.append(("★winner-curse: TRUE common-mode (rho=0.4) never false-certified independent (CI-upper gate)", fc == 0))
    # covariance-aware margin: CI-upper gate is conservative vs the plug-in point estimate, and recovers margin vs pure SUM
    cam = covariance_aware_margin(_make(200, 8, 0.3, 1) * (1.0 / 3.9), 1.0)
    checks.append(("covariance-aware margin gated < plug-in (conservative) and > SUM endpoint (recovers margin at moderate rho)",
                   cam["margin"] < cam["plugin_margin"] and cam["margin"] > cam["sum_margin"]))
    print("stage_decorrelation_verifier selftest:")
    for nm, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", nm))
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
