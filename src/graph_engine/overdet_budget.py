"""overdet_budget — reusable OVER-DETERMINATION BUDGET + free-param-nullspace checker for any recon/render cert (an agent worktree).

The agent pool converged  on a general law: a FREE-FIT forward-model/nuisance parameter is a
NULL-SPACE that DROPS the over-determination ratio r=M/N -- it absorbs the very error the cert should catch (free-fit PSF hides a
wrong-HF scene; free-fit exposure/scale/gain hides a material error), re-opening a fail-open. The fix: EXTERNALLY ANCHOR the
param (measure it from an independent calibration, bounded), NEVER free-fit to the residual the cert gates on. Each lane has been
applying this ad-hoc; this module makes it a reusable check, the productionization of the frame (like eps_min_cert did for resolution).

TWO checks:
  overdet_ratio(M, free_params, anchored_params) -- the BUDGET: r = M / N_free. r>1 => FALSIFIABLE; r<=1 => UNDER-DETERMINED
    (the free params can fit any residual -> unfalsifiable). Anchored params do NOT count against N (fixed by external measurement).
  falsification_power(observable_of, wrong_input, fit_free, anchored_value,...) -- the MECHANISM: measures the residual on a
    KNOWN-WRONG input under a FREE-FIT param vs the ANCHORED param. If free-fit drives the wrong-input residual below the fence
    (hides the error) while anchored leaves it above, the free param RE-OPENS a fail-open -> returns power_loss ratio + fail_open flag.

Ties [[a-single-free-parameter-over-determination-claim-must-be-red-teamed, ...]] (fixed-param absorption),
[[over-determination-ratio-quantifies-observability-vs-corroboration, ...]], [[a-learned-prior-is-not-a-measurement, ...]].
Composes eps_min_cert (the anchored param's required calibration accuracy = its eps_min). repro of self-test: deterministic."""
import math
import numpy as np

FALSIFIABLE = "FALSIFIABLE"
UNDERDETERMINED = "UNDERDETERMINED"


def overdet_ratio(n_observables, n_free_params, n_anchored_params=0):
    """r = M / N_free. Anchored params are fixed by external calibration -> excluded from N. r>1 => falsifiable."""
    Nf = max(int(n_free_params), 0)
    r = float(n_observables) / Nf if Nf > 0 else float("inf")
    return dict(r=r, M=int(n_observables), N_free=Nf, N_anchored=int(n_anchored_params),
                status=FALSIFIABLE if r > 1.0 else UNDERDETERMINED)


def falsification_power(observable_of, wrong_input, param_grid, anchored_value, fence,
                        correct_input=None, anchor_tol=0.0):
    """Does a FREE-FIT param hide a KNOWN-WRONG input (fail-open), vs an ANCHORED param catching it?
    observable_of(inp, param) -> a residual-like scalar (>=0; 0 = perfect match to the reference the cert gates on).
    param_grid: candidate param values the free-fit would search. anchored_value: the externally-measured param.
    fence: the residual threshold below which the cert CERTIFIES (accepts). correct_input: optional, to check no-false-alarm.
    anchor_tol: relative bound the anchored param is allowed to roam (0 = pinned).
    Returns dict(freefit_residual, anchored_residual, freefit_false_certifies, anchored_catches, power_loss, fail_open,...)."""
    pg = np.asarray(param_grid, float)
    # FREE-FIT: pick the param minimizing the residual on the WRONG input (claimant-tunable).
    # ★NaN HYGIENE (fix, D-OVERDET-BUDGET-...-NAN-IN-ARGMIN): a NaN residual (model failure at that param) WINS np.argmin
    # -> would set freefit_res=NaN -> NaN<fence is False -> the fail-open detector ITSELF fails open. Mask non-finite to +inf
    # so failed-model params never win the free-fit, and surface n_nonfinite so an all-failed sweep isn't a silent clean.
    resids = np.array([observable_of(wrong_input, p) for p in pg], dtype=float)
    n_nonfinite = int(np.sum(~np.isfinite(resids)))
    resids_masked = np.where(np.isfinite(resids), resids, np.inf)
    if not np.isfinite(resids_masked).any():
        return dict(freefit_residual=float("inf"), freefit_param=float("nan"), anchored_param=anchored_value,
                    anchored_residual=float("inf"), freefit_false_certifies=False, anchored_catches=False,
                    power_loss=1.0, fail_open=False, n_nonfinite=n_nonfinite, status="UNRESOLVED_NAN")
    i_fit = int(np.argmin(resids_masked))
    freefit_res = float(resids_masked[i_fit]); p_fit = float(pg[i_fit])
    # ANCHORED: param bounded to anchored_value*(1 +/- anchor_tol); best the wrong input can do within that bound (finite only)
    lo, hi = anchored_value * (1 - anchor_tol), anchored_value * (1 + anchor_tol)
    in_bound = pg[(pg >= min(lo, hi)) & (pg <= max(lo, hi))]
    if in_bound.size == 0:
        in_bound = np.array([anchored_value])
    a_res_all = [observable_of(wrong_input, p) for p in in_bound]
    a_res = [v for v in a_res_all if np.isfinite(v)]
    # ★ANCHORED-PATH NaN HYGIENE (fix degeneracy-enumerator: asymmetric with the free-fit guard above). If the anchored
    # forward model is NON-FINITE across the ENTIRE anchored bound, a_res is empty -> the old code set anchored_res=inf ->
    # anchored_catches=True: it FALSELY credits the safety leg with catching the wrong input on ZERO valid evaluations (and
    # status stayed "OK", masking it). We have NO basis to claim the anchored leg catches -> ABSTAIN, symmetric with the
    # free-fit UNRESOLVED_NAN early return, do not credit.
    if not a_res:
        return dict(freefit_residual=freefit_res, freefit_param=p_fit, anchored_param=anchored_value,
                    anchored_residual=float("nan"), freefit_false_certifies=bool(freefit_res < fence),
                    anchored_catches=False, power_loss=float("nan"), fail_open=False,
                    n_nonfinite=n_nonfinite, n_anchored_nonfinite=int(len(in_bound)),
                    status="ANCHORED_UNRESOLVED_NAN")
    anchored_res = float(min(a_res))
    freefit_false = freefit_res < fence                      # free-fit accepts the WRONG input -> fail-open
    anchored_catch = anchored_res >= fence                   # anchored rejects the WRONG input -> caught
    out = dict(freefit_residual=freefit_res, freefit_param=p_fit, anchored_param=anchored_value,
               anchored_residual=anchored_res, freefit_false_certifies=bool(freefit_false),
               anchored_catches=bool(anchored_catch),
               power_loss=(anchored_res / (freefit_res + 1e-12)),
               fail_open=bool(freefit_false and anchored_catch), n_nonfinite=n_nonfinite, status="OK")
    if correct_input is not None:
        cc = [observable_of(correct_input, p) for p in in_bound]; cc = [v for v in cc if np.isfinite(v)]
        out["anchored_no_false_alarm"] = bool(cc and min(cc) < fence)
    return out


def _chi2_lower_quantile(alpha, dof):
    """Lower alpha-quantile of chi-square(dof) via Wilson-Hilferty (no scipy; accurate for dof>=2). For the CI-UPPER of an
    estimated variance we divide sigma2_hat*dof by this LOWER chi2 quantile (small quantile -> large upper bound)."""
    # inverse-normal (Acklam-ish rational approx) for the standard normal quantile z_alpha
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if alpha < pl:
        q = math.sqrt(-2 * math.log(alpha)); z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif alpha <= 1 - pl:
        q = alpha - 0.5; r = q*q; z = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - alpha)); z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    t = 1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))
    return dof * (t ** 3) if t > 0 else 1e-9


def hill_tail_index(x, k_frac=0.15):
    """Hill estimator of the TAIL INDEX alpha from the top order statistics of |x| (red-team: sample KURTOSIS is
    unreliable -- under-biased at small n, and FINITE for infinite-4th-moment t3 -> bypasses the heavy-tail abstain. The tail
    index does NOT bypass: alpha<=4 <=> the 4th moment (kurtosis) is INFINITE, so the chi2/kurtosis finite-sample CI is INVALID).
    ★: run this on the RAW per-stage errors, NEVER the row-sum (the row-sum is CLT-Gaussianized -> tail hidden -> 50% FA).
    Returns (alpha, k_used); alpha=inf for a light/bounded tail, nan if too few samples."""
    a = np.sort(np.abs(np.asarray(x, float).ravel()))
    a = a[np.isfinite(a) & (a > 0)]
    n = len(a)
    if n < 20:
        return (float("nan"), 0)
    k = max(5, int(k_frac * n))
    k = min(k, n - 1)
    top = a[-(k + 1):]                                                # the k+1 largest |x|
    xk = top[0]                                                       # the (k+1)-th largest = threshold
    hill = float(np.mean(np.log(top[1:]) - np.log(xk)))              # mean log-excess over threshold
    # hill<=0 means a DEGENERATE tail (all top order-stats equal -> zero log-excess): the tail index is UNDEFINED, not "light".
    # Return nan (not inf) so a consumer's finite-check ABSTAINs; the stack_margin Hill guard already fail-closes on non-finite.
    return (1.0 / hill if hill > 0 else float("nan"), k)


def stack_margin(per_stage_margins, rho=1.0, corr=None, n_samples=None, ci=0.95, excess_kurtosis=0.0, stage_error_samples=None):
    """Compose PER-STAGE margins into a ROOT/SYSTEM margin (multi-stage stack-up).
    Each stage i contributes a fractional error e_i = 1/m_i. The root error is a VARIANCE COMPOSITION: sigma_root =
    sqrt(e' R e) where R is the stage-error CORRELATION matrix (SIGNED, -- errors can add OR cancel; n_eff/rho^2 is
    sign-blind and WRONG here). root_margin = 1/sigma_root.
      rho=1 (default, conservative) -> R=all-ones -> sigma_root = sum(e_i) = SUM/common-mode (the shared-factor case).
      rho=0  -> R=I -> RSS (fully DECORRELATED -- only valid if stage independence is VERIFIED).
      corr=<matrix> -> exact signed covariance. ★DEFAULT to SUM unless decorrelation is verified (RSS over-claims:
      quantified RSS fails open 1%->29% at rho=0.7).
    ★HARDENED (red-team): DEGENERATE inputs (non-finite margins, non-finite/non-symmetric/NON-PSD corr, negative variance)
      -> status ABSTAIN (NOT a numeric SAFE margin) so a degenerate covariance can NEVER fail open to SAFE.
    ★FINITE-SAMPLE (red-team): if n_samples is given, the per-stage errors are PLUG-IN estimates -> inflate sigma_root to
      its chi-square CI-UPPER (gate on the CI-upper, not the point estimate) so a small-sample SAFE does not fail open.
    Returns dict(root_margin, sigma_root, rho, status SAFE/FAIL_OPEN/ABSTAIN, dominant_stage, reason/note)."""
    def _abstain(reason):
        return dict(root_margin=float("nan"), sigma_root=float("nan"), rho=(None if corr is not None else float(rho)),
                    status="ABSTAIN", dominant_stage=-1, reason=reason,
                    note="DEGENERATE input -> ABSTAIN (fail-CLOSED): a non-finite/non-PSD covariance must never yield a numeric SAFE margin.")
    m = np.asarray(per_stage_margins, float)
    if m.size == 0 or not np.all(np.isfinite(m)) or np.any(m <= 0):    #: non-finite/non-positive per-stage margin
        return _abstain("non-finite or non-positive per-stage margin(s)")
    e = 1.0 / m
    k = len(e)
    if corr is not None:
        R = np.asarray(corr, float)
        if R.shape != (k, k) or not np.all(np.isfinite(R)):           #: non-finite / wrong-shape corr
            return _abstain("correlation matrix non-finite or shape != (%d,%d)" % (k, k))
        if not np.allclose(R, R.T, atol=1e-9):                        #: non-symmetric corr
            return _abstain("correlation matrix not symmetric")
        eigmin = float(np.linalg.eigvalsh(0.5 * (R + R.T)).min())
        if eigmin < -1e-9:                                            #: NON-PSD corr (e'Re could go negative -> fake-safe)
            return _abstain("correlation matrix not PSD (min eigenvalue %.2e < 0)" % eigmin)
    else:
        if not np.isfinite(rho):
            return _abstain("non-finite rho")
        R = np.full((k, k), float(rho)); np.fill_diagonal(R, 1.0)     # uniform correlation rho, unit diagonal
        if k > 1 and float(rho) < -1.0 / (k - 1) - 1e-9:              # the all-ones-off-diagonal matrix is PSD only for rho>=-1/(k-1)
            return _abstain("uniform rho=%.3f below PSD floor -1/(k-1)=%.3f for k=%d stages" % (rho, -1.0 / (k - 1), k))
    var = float(e @ R @ e)
    if not np.isfinite(var) or var < 0:                              #: degenerate variance -> ABSTAIN, never fake-safe
        return _abstain("degenerate root variance (%.3e)" % var)
    sigma_root = float(np.sqrt(var))
    fs_note = ""
    if n_samples is not None:                                        #: finite-sample CI-UPPER on the variance
        if not (isinstance(n_samples, (int, float)) and n_samples > 1):
            return _abstain("n_samples must be > 1 for a finite-sample CI")
        if not (np.isfinite(excess_kurtosis) and excess_kurtosis >= 0.0):
            return _abstain("excess_kurtosis must be finite and >= 0")
        # ★: if RAW per-stage error samples are supplied, GUARD the CI's finite-4th-moment precondition with the Hill
        # tail-index on the RAW pooled (per-stage-standardized) errors -- NOT sample kurtosis (bypasses on t3) and NOT the
        # row-sum (CLT-Gaussianized). alpha<=4 => infinite kurtosis => the chi2/kurtosis CI is INVALID => ABSTAIN.
        if stage_error_samples is not None:
            pooled = []
            for s in stage_error_samples:
                s = np.asarray(s, float); s = s[np.isfinite(s)]
                if s.size >= 8 and s.std() > 0:
                    pooled.append((s - s.mean()) / s.std())          # standardize per stage, then pool (shape-only, scale-free)
            if not pooled:
                return _abstain("stage_error_samples: no usable per-stage samples for a tail-index guard")
            alpha, k_used = hill_tail_index(np.concatenate(pooled))
            # ★HONEST CAVEAT: the Hill estimate is BIASED LOW at small n (n=80: Gaussian~4.3, t6~2.6 though true alpha=inf/6),
            # so this OPT-IN guard is CONSERVATIVE -- it fail-CLOSES (ABSTAIN) on borderline/finite-4th-moment tails, not just
            # genuinely-infinite ones. That trades the t3 fail-OPEN (22%) for some false-ABSTAIN (safe direction for a gate).
            # The PRECISE threshold/k calibration (bias-correct the Hill, reduce false-abstain) is the active tail-index work.
            if not np.isfinite(alpha) or alpha <= 4.0:               # infinite/near-infinite 4th moment -> CI precondition unmet
                return _abstain("heavy-tailed stage errors: Hill tail-index alpha=%.2f <= 4 (4th moment infinite/unresolved at this n, finite-sample CI precondition unmet -- CONSERVATIVE opt-in guard; precise calibration = Hill work)" % alpha)
        n = float(n_samples)
        # red-team: the chi-square CI-UPPER ASSUMES GAUSSIAN stage errors; under heavy tails it UNDER-covers (t(df3): 6-9%
        # fail-open). Var(s^2) = sigma^4 * (2/(n-1) + kappa/n) (kappa = EXCESS kurtosis; 0 for Gaussian). Match a chi-square of
        # EFFECTIVE dof: 2/dof_eff = 2/(n-1) + kappa/n -> dof_eff = 2 / (2/(n-1) + kappa/n). kappa=0 -> dof_eff=n-1 (EXACT chi2,
        # continuous); kappa>0 -> smaller dof_eff -> WIDER CI-UPPER -> SMALLER (safer) margin. Closes the heavy-tail gap.
        dof_eff = 2.0 / (2.0 / (n - 1.0) + excess_kurtosis / n)
        chi2_lo = _chi2_lower_quantile(1.0 - ci, dof_eff)           # LOWER chi2 quantile -> UPPER bound on variance
        sigma_root = sigma_root * math.sqrt(dof_eff / chi2_lo)      # inflate to the CI-UPPER of sigma_root (conservative)
        prec = "GAUSSIAN stage errors" if excess_kurtosis == 0.0 else "excess_kurtosis=%.1f (heavy-tail corrected)" % excess_kurtosis
        fs_note = "; finite-sample: sigma at %.0f%% CI-UPPER, dof_eff=%.1f (n=%d); ★PRECONDITION: %s -- for heavier tails pass excess_kurtosis " % (100 * ci, dof_eff, int(n_samples), prec)
    root_margin = (1.0 / sigma_root) if sigma_root > 0 else float("inf")
    dom = int(np.argmax(e))                                          # the stage with the largest fractional error
    return dict(root_margin=root_margin, sigma_root=sigma_root, rho=(None if corr is not None else float(rho)),
                status=("SAFE" if root_margin >= 1.0 else "FAIL_OPEN"), dominant_stage=dom,
                note="DEFAULT rho=1 (SUM/common-mode) unless stage-error DECORRELATION is verified; RSS(rho=0) over-claims (1%->29% fail-open @rho0.7)" + fs_note)


def certify(ratio_result, power_result):
    """COMBINE the two checks per the red-team: overdet_ratio (the COUNT) is NECESSARY-NOT-SUFFICIENT -- r>1 can still fail
    open on a DEGENERATE free param. A cert is CERTIFIABLE iff r>1 (falsifiable budget) AND falsification_power shows the free
    param does NOT hide a wrong input (not fail_open) AND no all-NaN. Returns dict(certifiable, reason)."""
    r_ok = ratio_result.get("status") == FALSIFIABLE
    p = power_result
    pstatus = p.get("status", "")
    if not r_ok:
        return dict(certifiable=False, reason="UNDERDETERMINED (r=%.2f<=1: too many free params, unfalsifiable)" % ratio_result.get("r", 0))
    # ★any NON-OK power status = we could not cleanly assess -> NOT certifiable (fail-closed). Covers UNRESOLVED_NAN (free-fit
    # grid all-NaN) AND ANCHORED_UNRESOLVED_NAN (anchored bound all-NaN, fix degeneracy-enumerator) -- never certify on
    # a degenerate/unassessable power result.
    if pstatus != "OK":
        return dict(certifiable=False, reason="power status=%s (forward model degenerate/unassessable -- cannot certify)" % (pstatus or "MISSING"))
    if p.get("fail_open", True):
        return dict(certifiable=False, reason="FAIL-OPEN: a free param hides a wrong input despite r>1 (DEGENERATE param -- ratio blind). ANCHOR it.")
    return dict(certifiable=True, reason="r>1 AND the (anchored) forward-model param does not hide a wrong input")


# ------------------------------------- self-test (python overdet_budget.py) -------------------------------------
def _selftest():
    checks = []
    # (A) budget: M=3 observables, 1 free param -> r=3 falsifiable; 4 free -> r=0.75 underdetermined; anchoring one -> r=1.5
    a1 = overdet_ratio(3, 1); a2 = overdet_ratio(3, 4); a3 = overdet_ratio(3, 2, n_anchored_params=2)
    checks.append(("budget: 1 free->FALSIFIABLE, 4 free->UNDERDETERMINED",
                   a1["status"] == FALSIFIABLE and a2["status"] == UNDERDETERMINED))
    # (B) free-fit-hides vs anchored-catches, on a clean LINEAR toy (a free additive 'exposure/offset' nuisance param that
    # absorbs a wrong-input error). observable = |value(input) - real - param|. The true offset is 0 (anchored). A WRONG
    # input differs from real by DELTA; a FREE-FIT param = DELTA drives the residual to 0 (hides it), while the anchored
    # param (bounded near 0) leaves the residual ~DELTA (caught). Robust -- no PSF-cutoff/frequency finickiness.
    I_TRUE, P_TRUE = 1.0, 0.3; R = I_TRUE + P_TRUE          # real reference; forward model: predicted = input + param(exposure)
    I_WRONG = 1.5                                            # wrong input (error 0.5 vs I_TRUE)
    obs = lambda inp, param: abs(inp + param - R)           # residual the cert gates on
    fence = 0.5 * abs(I_WRONG - I_TRUE)                      # accept if residual < fence (=0.25)
    fp = falsification_power(obs, I_WRONG, np.linspace(-1.0, 2.0, 301), P_TRUE, fence,
                             correct_input=I_TRUE, anchor_tol=0.2)    # anchored = true exposure P_TRUE, bounded +/-20%
    checks.append(("free-fit hides wrong input, anchored catches it",
                   fp["fail_open"] and fp["anchored_catches"] and fp.get("anchored_no_false_alarm", False)))
    checks.append(("power_loss > 1 (free param destroys falsification power)", fp["power_loss"] > 1.5))
    # (C2) degeneracy-enumerator: anchored forward model NON-FINITE across the whole anchored bound must NOT falsely
    # credit anchored_catches=True on inf -- it must ABSTAIN (symmetric with the free-fit UNRESOLVED_NAN guard).
    obs_anchor_nan = lambda inp, p: (float("nan") if 0.8 <= p <= 1.2 else abs(inp + p - R))   # NaN only inside anchored bound
    fp_anan = falsification_power(obs_anchor_nan, I_WRONG, np.linspace(-2.0, 4.0, 601), anchored_value=1.0, fence=0.25, correct_input=1.0, anchor_tol=0.15)
    cert_anan = certify(overdet_ratio(3, 1), fp_anan)
    checks.append(("anchored-bound all-NaN -> ANCHORED_UNRESOLVED_NAN + not-certifiable, no false anchored_catches (fix L63)",
                   fp_anan["status"] == "ANCHORED_UNRESOLVED_NAN" and fp_anan["anchored_catches"] is False and not cert_anan["certifiable"]))
    # (C) NaN-in-argmin fix: an observable that returns NaN for some params must NOT mask a real fail-open
    obs_nan = lambda inp, p: (float("nan") if 0.4 < p < 0.6 else abs(inp + p - R))
    fp_nan = falsification_power(obs_nan, I_WRONG, np.linspace(-1.0, 2.0, 301), P_TRUE, fence, correct_input=I_TRUE, anchor_tol=0.2)
    checks.append(("NaN residual does NOT mask the fail-open (fix)", np.isfinite(fp_nan["freefit_residual"]) and fp_nan["fail_open"] and fp_nan.get("n_nonfinite", 0) > 0))
    # (D) degeneracy: certify gates on BOTH ratio AND falsification_power (r>1 but fail_open -> NOT certifiable)
    cert = certify(overdet_ratio(40, 1), fp)                 # r=40 FALSIFIABLE by count, but fp.fail_open=True (degenerate)
    checks.append(("certifycatches degenerate r>1 fail-open (ratio necessary-not-sufficient)", not cert["certifiable"]))
    # (E) stack_margin (multi-stage stack-up). Properties that MUST hold:
    m = [3.9, 6.8, 6.6, 10.8, 10.7, 18.4, 5.5]               # batch's 7 glass-chain stage margins
    s_cm = stack_margin(m, rho=1.0); s_rss = stack_margin(m, rho=0.0); s_mid = stack_margin(m, rho=0.7)
    # E1 common-mode(SUM) <= rho0.7 <= RSS: more correlation -> WORSE (smaller) margin (RSS is the rho=0 best case)
    checks.append(("stack: common-mode margin <= rho0.7 <= RSS (RSS over-claims)",
                   s_cm["root_margin"] <= s_mid["root_margin"] + 1e-9 <= s_rss["root_margin"] + 1e-9))
    # E2 common-mode margin == 1/sum(1/m) exactly (the shared-factor SUM), and default rho is 1 (conservative)
    inv_sum = sum(1.0 / x for x in m)
    checks.append(("stack: common-mode == 1/sum(1/m) (SUM), status SAFE (>1x)",
                   abs(s_cm["root_margin"] - 1.0 / inv_sum) < 1e-9 and s_cm["status"] == "SAFE"))
    # E3 SIGNED covariance: an ANTI-correlated pair CANCELS -> LARGER margin than RSS (n_eff/rho^2 is sign-blind -> would miss this)
    Ranti = np.array([[1.0, -0.9], [-0.9, 1.0]])
    s_anti = stack_margin([2.0, 2.0], corr=Ranti); s_indep = stack_margin([2.0, 2.0], rho=0.0)
    checks.append(("stack: signed anti-correlation improves margin beyond RSS (sign matters, n_eff blind)",
                   s_anti["root_margin"] > s_indep["root_margin"]))
    # E4 fail-open detection: stages that individually pass (>1x) but STACK below 1x under common-mode -> FAIL_OPEN flagged
    s_fo = stack_margin([1.5, 1.5, 1.5], rho=1.0)            # 1/(3*(1/1.5)) = 0.5x < 1 under common-mode
    checks.append(("stack: per-stage-safe but common-mode-stacked fail-open is FLAGGED ",
                   s_fo["status"] == "FAIL_OPEN" and s_fo["root_margin"] < 1.0))
    # (E5) degenerate-covariance fail-open FIX: non-PSD corr / NaN corr / NaN margin must ABSTAIN, NEVER numeric-SAFE
    Rnpsd = np.full((3, 3), -0.9); np.fill_diagonal(Rnpsd, 1.0)      # min-eig<0 -> non-PSD -> e'Re can go negative -> fake-safe
    s_npsd = stack_margin([2, 2, 2], corr=Rnpsd)
    Rnan = np.array([[1.0, float("nan")], [float("nan"), 1.0]]); s_cnan = stack_margin([2, 2], corr=Rnan)
    s_mnan = stack_margin([2, float("nan"), 2], rho=0.5)
    s_rholo = stack_margin([2, 2, 2], rho=-0.9)                       # uniform rho below -1/(k-1)=-0.5 PSD floor
    checks.append(("stack: degenerate cov (non-PSD/NaN-corr/NaN-margin/sub-floor-rho) -> ABSTAIN not SAFE (fix)",
                   all(s["status"] == "ABSTAIN" for s in (s_npsd, s_cnan, s_mnan, s_rholo))))
    # (E6) valid PSD corr still works (guard doesn't over-reject): rho=0.3, k=3 is PSD -> a numeric SAFE/FAIL_OPEN verdict
    s_ok = stack_margin([3, 3, 3], rho=0.3)
    checks.append(("stack: valid PSD corr still returns a numeric verdict (guard not over-strict)", s_ok["status"] in ("SAFE", "FAIL_OPEN") and np.isfinite(s_ok["root_margin"])))
    # (E7) finite-sample CI: a point-estimate SAFE margin should TIGHTEN (or flip toward FAIL_OPEN) under a small sample
    s_plug = stack_margin([1.3, 1.3], rho=0.0)                       # plug-in ~ 0.92x... actually RSS of 1.3,1.3 = 0.919 -> use safer
    s_plug2 = stack_margin([2.0, 2.0], rho=0.0)                      # plug-in RSS margin = 1.414x SAFE
    s_fs = stack_margin([2.0, 2.0], rho=0.0, n_samples=15)           # finite-sample CI-UPPER -> smaller margin than plug-in
    checks.append(("stack: finite-sample CI-UPPER margin < plug-in margin (fix: gate on CI not point-est)",
                   s_fs["status"] in ("SAFE", "FAIL_OPEN") and s_fs["root_margin"] < s_plug2["root_margin"]))
    # (E8) heavy-tail precondition: excess_kurtosis>0 must WIDEN the CI -> SMALLER (safer) margin than the Gaussian (kappa=0)
    # case, and kappa=0 must EXACTLY recover the plain chi-square (dof_eff=n-1, continuous).
    s_gauss = stack_margin([2.0, 2.0], rho=0.0, n_samples=15, excess_kurtosis=0.0)
    s_heavy = stack_margin([2.0, 2.0], rho=0.0, n_samples=15, excess_kurtosis=6.0)   # t(df3)-like heavy tails
    checks.append(("stack: heavy-tail excess_kurtosis widens CI -> smaller margin, kappa=0 recovers chi2 (fix)",
                   s_heavy["root_margin"] < s_gauss["root_margin"] and abs(s_gauss["root_margin"] - s_fs["root_margin"]) < 1e-9))
    # (E9) Hill tail-index guard: RAW t3-like samples (infinite 4th moment, sample-kurtosis would BYPASS) must ABSTAIN;
    # Gaussian raw samples must still return a numeric verdict. Deterministic via seeded rng.
    rng = np.random.default_rng(12345)
    gauss_samps = [rng.standard_normal(80), rng.standard_normal(80)]                 # light tails -> alpha large -> numeric
    t3_samps = [rng.standard_t(3, 80), rng.standard_t(3, 80)]                        # infinite 4th moment -> alpha~3 -> ABSTAIN
    s_hill_g = stack_margin([2.0, 2.0], rho=0.0, n_samples=80, stage_error_samples=gauss_samps)
    s_hill_t = stack_margin([2.0, 2.0], rho=0.0, n_samples=80, stage_error_samples=t3_samps)
    alpha_g = hill_tail_index(np.concatenate([(g - g.mean())/g.std() for g in gauss_samps]))[0]
    alpha_t = hill_tail_index(np.concatenate([(t - t.mean())/t.std() for t in t3_samps]))[0]
    checks.append(("stack: Hill guard ABSTAINs on t3 (alpha=%.1f<=4) numeric on Gaussian (alpha=%.1f) (fix bypass)" % (alpha_t, alpha_g),
                   s_hill_t["status"] == "ABSTAIN" and s_hill_g["status"] in ("SAFE", "FAIL_OPEN")))
    allok = all(ok for _, ok in checks)
    print("overdet_budget selftest: %s" % ("PASS" if allok else "FAIL"))
    for name, ok in checks:
        print("  [%s] %s" % ("ok" if ok else "XX", name))
    print("  budget r(1free)=%.1f/%s r(4free)=%.2f/%s ; freefit_res=%.4f (sig=%.2f) anchored_res=%.4f power_loss=%.1fx"
          % (a1["r"], a1["status"], a2["r"], a2["status"], fp["freefit_residual"], fp["freefit_param"],
             fp["anchored_residual"], fp["power_loss"]))
    print("DECISIVE overdet_budget_selftest_pass=%d n_checks=%d" % (1 if allok else 0, len(checks)))
    return 0 if allok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
