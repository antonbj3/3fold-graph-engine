"""overdet_precondition_gate — the runnable RIGOR GATE for D's cross-substrate over-determination program (the [D→@X STRESS]
"you and @Y hit the SAME object on decorrelated substrates — is it certified INDEPENDENTLY?" directives). Productized from G.

THE PROBLEM: two lanes AGREEING on a σ_min / cert object can be a SHARED-TOOL or shared-object common-mode
(FRAMING-ECHO, n_eff≈1), NOT independent confirmation. Raw-score agreement is INFLATED by both certs tracking the same object — a
celebrated cross-worktree agreement can be declared a breakthrough while adding zero decorrelated coverage.

THE GATE (decorrelation-that-counts): two certs over-DETERMINE an object iff
  (1) BOTH discriminate it (agreement on the object), AND
  (2) their ERRORS are decorrelated — the RESIDUAL correlation AFTER removing the shared object signal (partial correlation | GT,
      residualized at the object's TRUE granularity: a binary label OR a continuous estimate) is ≈0.
The lift lives in the decorrelated residuals; n_eff = 2/(1+|ρ_resid|). If BOTH discriminate but ρ_resid is HIGH → FRAMING-ECHO
(agree AND fail together), no lift. Use BEFORE declaring any [D→@X STRESS] cross-worktree agreement a breakthrough.

Pure numpy + scipy.stats (Mann-Whitney AUC). Grader; does not itself certify anything — it gates whether two certs' agreement lifts.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

__all__ = ["overdet_gate", "moment_exists", "partial_corr_given", "residualize", "selftest"]


def residualize(v, z):
    """Remove the shared-object signal z from v by LINEAR regression. z may be a BINARY indicator OR a CONTINUOUS object estimate;
    binary z reduces to class-mean removal, continuous z removes the shared latent so only the independent ERROR remains.
    ★: residualizing on a binary label when the object is CONTINUOUS leaves the within-class latent shared → false-echo;
    always residualize at the object's true granularity."""
    v = np.asarray(v, float); z = np.asarray(z, float)
    Z = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ beta


def partial_corr_given(x, y, z):
    """Correlation of the ERRORS of x and y after removing the shared object signal z (partial correlation | z) = the
    decorrelation-that-counts. Raw corr(x,y) is inflated by both tracking z; this isolates the residual (error) dependence."""
    rx = residualize(x, z); ry = residualize(y, z)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(score, truth):
    truth = np.asarray(truth).astype(bool)
    if truth.all() or (~truth).all():
        return 0.5
    u = stats.mannwhitneyu(np.asarray(score)[truth], np.asarray(score)[~truth], alternative="two-sided").statistic
    return float(u / (truth.sum() * (~truth).sum()))


def _cohens_d(score, truth):
    """Effect size (|Δmean| / pooled-std) — the ABSOLUTE separation AUC is blind to. ★B FLAG-1 ( grade): AUC RANKS, it does
    not MEASURE magnitude; a physically-negligible separation (spectral leakage 2e-5) ranks AUC 1.00 = winner's-curse. Gate
    discrimination on effect size, not raw AUC."""
    score = np.asarray(score, float); truth = np.asarray(truth).astype(bool)
    if truth.all() or (~truth).all():
        return 0.0
    a, b = score[truth], score[~truth]
    sp = np.sqrt((a.var() + b.var()) / 2.0) + 1e-12
    return float(abs(a.mean() - b.mean()) / sp)


def overdet_gate(s1, s2, truth, z_resid=None, tau=0.3, discrim_margin=0.25, min_effect=0.2):
    """Decide GENUINE cross-substrate over-det vs FRAMING-ECHO for two certs' per-item scores s1, s2 on a shared object.
    truth = shared-object GT (binary, for discrimination). z_resid = the object estimate to residualize on (defaults to truth;
    pass a CONTINUOUS object estimate when the shared object is continuous). Returns:
      genuine_lift: both discriminate AND residual(error) correlation < tau  → a real joint cert (n_eff→2)
      framing_echo: both discriminate but residuals correlated (≥tau)         → agreement is a shared-object common-mode, no lift
      rho_resid: the residual (error) correlation — the decorrelation-that-counts
      n_eff: 2/(1+|rho_resid|) — coverage the pair actually adds
    """
    truth = np.asarray(truth).astype(bool)
    z = truth.astype(float) if z_resid is None else np.asarray(z_resid, float)
    auc1, auc2 = _auc(s1, truth), _auc(s2, truth)
    d1, d2 = _cohens_d(s1, truth), _cohens_d(s2, truth)
    # ★B FLAG-1: discrimination needs BOTH an AUC margin AND a real effect size — a negligible-magnitude separation (winner's-curse)
    # ranks AUC~1 but carries no physical signal → NOT a discriminator, so its residual analysis is meaningless.
    both_discriminate = bool(abs(auc1 - 0.5) > discrim_margin and abs(auc2 - 0.5) > discrim_margin
                             and d1 > min_effect and d2 > min_effect)
    rho_resid = partial_corr_given(s1, s2, z)
    n_eff = 2.0 / (1.0 + abs(rho_resid))
    genuine = bool(both_discriminate and abs(rho_resid) < tau)
    return dict(genuine_lift=genuine, framing_echo=bool(both_discriminate and not genuine),
                rho_resid=rho_resid, n_eff=n_eff, auc1=auc1, auc2=auc2, cohens_d1=d1, cohens_d2=d2,
                both_discriminate=both_discriminate)


def moment_exists(x, m=2, n_shuffle=20, plateau_thresh=0.05):
    """Ferguson-Klass MAX-TO-SUM ratio test for whether the m-th moment E[|X|^m] is FINITE. Parameter-free (no fitted tail
    index): R_m(n) = max_i|X_i|^m / sum_i|X_i|^m -> 0 a.s. IFF E[|X|^m] < inf. Watch R_m as n grows over the sample; a
    plateau bounded away from 0 => INFINITE moment (the max keeps dominating the sum), decay toward 0 => FINITE.
    Returns {finite: True/False/None, plateau, curve, n_grid}; use m=1 for the mean, m=2 for the variance.
    ★BEATS fitted-Hill and running-variance for moment-existence on real heavy
    tails (193x real-vs-finite-control separation on avalanche energies); prefer it near the alpha=1 / alpha=2 boundary
    where a fitted tail index (Hill) is unreliable and a running variance is ambiguous. ★HONEST SCOPE: certifies moment
    EXISTENCE / heaviness, NOT the exact tail index; and it is an ASYMPTOTIC-IID statement — on mixtures/structured data it
    certifies the clean regime, it is not a repair of a fragile tail-index band (C's boundary). Report each moment separately
    (variance-nonexistence can be clear while mean-existence is borderline)."""
    x = np.abs(np.asarray(x, float).ravel()); x = x[x > 0] ** m
    n = len(x)
    if n < 100:
        return dict(finite=None, plateau=float("nan"), reason="too few samples (need >=100)")
    n_grid = sorted(set(int(g) for g in np.geomspace(100, n, 6)))
    rng = np.random.default_rng(0)
    curve = []
    for k in n_grid:
        if k >= n:
            curve.append(float(x.max() / x.sum())); continue
        vals = []
        for _ in range(n_shuffle):
            s = rng.choice(x, k, replace=False); vals.append(s.max() / s.sum())
        curve.append(float(np.mean(vals)))
    plateau = float(curve[-1])
    if plateau > plateau_thresh:
        finite = False                                                      # R_m bounded away from 0 => infinite moment
    elif plateau < 0.4 * curve[0]:
        finite = True                                                       # R_m decayed toward 0 => finite moment
    else:
        finite = None                                                       # inconclusive (borderline / underdetermined)
    return dict(finite=finite, plateau=plateau, curve=curve, n_grid=n_grid)


def tail_index_hill(x, k_frac=0.05, robust=True, gap_thresh=3.0):
    """Hill estimator of the tail index alpha (P(|X|>t) ~ t^-alpha) from the upper |x| order statistics. alpha<1 =
    infinite MEAN, alpha<2 = infinite VARIANCE (so Pearson rho / n_eff=2/(1+rho) are inapplicable). Returns alpha_hat
    (np.inf for a light/finite tail with no power law, e.g. Gaussian -> large alpha). ★Honest: Hill is biased/noisy for
    light tails (there is no true power-law tail) — treat alpha_hat>=2 as 'finite-variance-safe' and do NOT over-read
    its exact value; the DECISION (alpha<1 vs 1<alpha<2 vs >=2) is what is robust, not the point estimate.
    ★robust=True (fixes B's P1 gate-4): naive Hill is a MEAN of log-ratios => breakdown 0, so a few GROSS spurious
    spikes (e.g. 10x-max electronic mis-detections) inflate alpha_hat unboundedly (median shift ~54-72% at 3% contamination).
    GROSS spikes sit FAR above the genuine tail => a large RATIO JUMP at the spike/genuine boundary; detect the first
    ratio-jump > gap_thresh in the top 5% and TRIM above it before Hill. On clean data there is no gross jump => reduces
    EXACTLY to naive Hill (0 bias). This only defends against WELL-SEPARATED gross contamination (B's case); spikes blended
    into the genuine tail are statistically indistinguishable and no estimator can remove them — that is the honest floor.
    ★ real-data validation (11 micropillar AE avalanche pillars, Zenodo 5897653): the trim is EMPIRICALLY SAFE on
    well-sampled real power-law tails (deviates <0.6% from naive) even when it fires on a GENUINE dominant avalanche
    (dia32_p2 natural 8.65x top gap), because a deep Hill (k=hundreds) dilutes any single event. The gross-spike-vs-real-
    extreme confound is SAMPLE-SIZE-GATED: negligible when the tail is well-populated, decisive only for SPARSE tails
    (k of order tens), where one event materially moves alpha — there prefer a leave-one-out band + ABSTAIN over silent
    trimming. So robust=True is safe as a default; the caller should abstain on tail-underdetermined (small-k) samples."""
    a = np.sort(np.abs(np.asarray(x, float)).ravel())[::-1]
    a = a[a > 0]
    n = len(a)
    if n < 20:
        return np.inf
    if robust and n >= 40:                                                   # trim above a gross spike/genuine ratio-jump
        top = a[:max(6, int(0.05 * n))]
        ratios = top[:-1] / (top[1:] + 1e-300)
        j = int(np.argmax(ratios))
        if ratios[j] > gap_thresh:
            a = a[j + 1:]; n = len(a)
    k = min(max(10, int(k_frac * n)), n - 1)
    xi = float(np.mean(np.log(a[:k]) - np.log(a[k])))                        # Hill estimate of 1/alpha
    return (1.0 / xi) if xi > 1e-9 else np.inf


def tail_aware_overdet_guard(s1, s2, truth, z_resid=None, alpha_var=2.0, alpha_mean=1.15):
    """★thread-17 GUARD: overdet_gate's rho_resid and n_eff=2/(1+rho) are SECOND-MOMENT constructs — undefined for
    infinite-variance (tail index alpha<2) residual errors, and with a MEAN combine the decorrelation LEVERAGE INVERTS
    below alpha=1 (decorrelation HURTS: avg of n iid alpha-stable scales as n^(1/alpha-1)). This guard estimates the
    residual-error tail index and returns tail-aware advice so a heavy-tailed over-det is not mis-certified. Returns
    {tail_index_alpha, rho_valid (alpha>=alpha_var), decorrelation_helps (alpha>1), recommended_combine, warning}.
    ★Constructive (per the thread-17 resolution): a ROBUST/median combine over >=3 legs restores decorrelation's benefit;
    the linear average is what the heavy tail breaks."""
    truth_b = np.asarray(truth).astype(bool)
    z = truth_b.astype(float) if z_resid is None else np.asarray(z_resid, float)
    # ★robust location removal for the TAIL estimate: the tail index is location-invariant, and OLS residualize is
    # NON-robust — a single heavy-tailed outlier corrupts the regression and distorts the very tail we measure. Subtract
    # the class-conditional MEDIAN (binned for a continuous z) so the heavy tail is preserved.
    def _robust_resid(v):
        v = np.asarray(v, float); out = v.copy()
        uz = np.unique(z)
        levels = uz if len(uz) <= 2 else np.quantile(z, np.linspace(0, 1, 6))   # bin a continuous object estimate
        if len(uz) <= 2:
            for u in uz:
                m = z == u; out[m] = v[m] - np.median(v[m])
        else:
            idx = np.clip(np.digitize(z, levels[1:-1]), 0, len(levels) - 1)
            for b in np.unique(idx):
                m = idx == b; out[m] = v[m] - np.median(v[m])
        return out
    alpha = tail_index_hill(np.concatenate([_robust_resid(s1), _robust_resid(s2)]))
    rho_valid = bool(alpha >= alpha_var)
    decorr_helps = bool(alpha > 1.0)
    combine = "mean" if alpha >= alpha_var else "median(>=3 legs)"
    if alpha >= alpha_var:
        warn = "finite-variance (alpha~{:.1f}>={:.0f}): Pearson rho_resid / n_eff VALID, MEAN combine OK".format(min(alpha, 99), alpha_var)
    elif alpha > alpha_mean:
        warn = ("infinite-variance heavy tail (alpha~{:.2f}<2): Pearson rho_resid UNRELIABLE + n_eff=2/(1+rho) OVERSTATED "
                "-> use a ROBUST/median combine (>=3 legs); decorrelation still helps but weaker".format(alpha))
    else:
        warn = ("VERY heavy tail (alpha~{:.2f}<=1, infinite mean): with a MEAN combine decorrelation INVERTS and HURTS "
                "(thread-17/g988) -> MUST use a median combine over >=3 legs; the naive n_eff is meaningless".format(alpha))
    return dict(tail_index_alpha=float(alpha), rho_valid=rho_valid, decorrelation_helps=decorr_helps,
                recommended_combine=combine, warning=warn)


def hard_subset_decorrelation(s1, s2, truth, z_resid=None, hard_frac=0.25):
    """★HARD-SUBSET decorrelation check (heeds the coherent-adversarial / decorrelation-lives-on-the-hard-subset
    discipline): overdet_gate's rho_resid is BULK, so two certs can be bulk-decorrelated (rho_resid~0 -> gate says
    genuine_lift, n_eff~2) yet CO-FAIL on the HARD subset (a shared common-mode on the ambiguous/low-margin cases) where
    the lift actually matters -> the real n_eff there is ~1. This measures the residual correlation on the AMBIGUOUS-OBJECT
    subset = smallest |z - median(z)| (using the object estimate z, NOT the error-corrupted score — a shared error inflates
    the score margin and would hide itself). ★Needs a CONTINUOUS z_resid for a meaningful hard subset (binary z degenerates
    -> returns tail_coupled=False, use the score/other margin then). Returns {rho_bulk, rho_hard, n_eff_bulk, n_eff_hard,
    tail_coupled}; tail_coupled=True => bulk-decorrelated but tail-coupled -> DOWNGRADE n_eff to n_eff_hard. Complements
    tail_aware_overdet_guard (that = tail HEAVINESS / aggregation regime; this = tail COUPLING / shared blind spot)."""
    s1 = np.asarray(s1, float); s2 = np.asarray(s2, float); tb = np.asarray(truth).astype(bool)
    z = tb.astype(float) if z_resid is None else np.asarray(z_resid, float)
    r1, r2 = residualize(s1, z), residualize(s2, z)
    rho_bulk = partial_corr_given(s1, s2, z)
    zc = np.abs(z - np.median(z))                                      # object ambiguity: small = near the class boundary
    if float(np.ptp(zc)) < 1e-9:                                       # binary/degenerate z -> no meaningful hard subset
        return dict(rho_bulk=float(rho_bulk), rho_hard=float(rho_bulk), n_eff_bulk=2.0 / (1.0 + abs(rho_bulk)),
                    n_eff_hard=2.0 / (1.0 + abs(rho_bulk)), tail_coupled=False)
    k = max(10, int(hard_frac * len(s1)))
    hard = np.argsort(zc)[:k]
    a, b = r1[hard] - r1[hard].mean(), r2[hard] - r2[hard].mean()
    rho_hard = float(np.sum(a * b) / (np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-12))
    tail_coupled = bool(abs(rho_hard) > abs(rho_bulk) + 0.25 and abs(rho_hard) > 0.4)
    return dict(rho_bulk=float(rho_bulk), rho_hard=rho_hard, n_eff_bulk=2.0 / (1.0 + abs(rho_bulk)),
                n_eff_hard=2.0 / (1.0 + abs(rho_hard)), tail_coupled=tail_coupled)


def selftest() -> int:
    ok = True
    rng = np.random.default_rng(941)
    print("overdet_precondition_gate selftest (residual decorrelation, NOT raw agreement):")

    # (1) FRAMING-ECHO: two certs that BOTH discriminate the object but fail on the SAME items (correlated shared error) → no lift
    truth = rng.random(200) < 0.5
    shared_err = rng.normal(size=200)                                        # a common error both certs inherit
    e1 = truth.astype(float) * 2 + 0.8 * shared_err + 0.2 * rng.normal(size=200)
    e2 = truth.astype(float) * 2 + 0.8 * shared_err + 0.2 * rng.normal(size=200)
    echo = overdet_gate(e1, e2, truth)
    c1 = bool(echo["framing_echo"] and not echo["genuine_lift"] and abs(echo["rho_resid"]) > 0.3)
    ok &= c1; print(f"  [{'✓' if c1 else '✗'}] framing-echo (shared error): both discriminate {echo['both_discriminate']}, "
                    f"ρ_resid={echo['rho_resid']:+.2f} n_eff={echo['n_eff']:.2f} → framing_echo={echo['framing_echo']}")

    # (2) GENUINE LIFT: two independent-noise measurements of a CONTINUOUS latent → residualize on the latent → decorrelated errors
    latent = rng.normal(size=200); tc = latent > 0
    g1 = latent + 0.5 * rng.normal(size=200); g2 = latent + 0.5 * rng.normal(size=200)
    lift = overdet_gate(g1, g2, tc, z_resid=latent)
    c2 = bool(lift["genuine_lift"] and abs(lift["rho_resid"]) < 0.3 and lift["n_eff"] > 1.5)
    ok &= c2; print(f"  [{'✓' if c2 else '✗'}] genuine lift (independent errors, continuous residualize): "
                    f"ρ_resid={lift['rho_resid']:+.2f} n_eff={lift['n_eff']:.2f} → genuine_lift={lift['genuine_lift']}")

    # (3) the binary-vs-continuous residualization trap: residualizing the SAME genuine pair on the BINARY label leaves the
    # within-class continuous latent shared → false FRAMING-ECHO. Confirms the gate must residualize at the object's granularity.
    false_echo = overdet_gate(g1, g2, tc)                                    # z_resid defaults to the BINARY truth → wrong granularity
    c3 = bool(abs(false_echo["rho_resid"]) > abs(lift["rho_resid"]) + 0.2)
    ok &= c3; print(f"  [{'✓' if c3 else '✗'}] granularity trap: binary-residualized ρ_resid={false_echo['rho_resid']:+.2f} > "
                    f"continuous {lift['rho_resid']:+.2f} → residualize at the object's TRUE granularity (g941)")

    # (4) one-sided discrimination guard: if only ONE cert discriminates, it is NOT an over-det (no agreement to lift)
    noise = rng.normal(size=200)
    one = overdet_gate(g1, noise, tc, z_resid=latent)
    c4 = bool(not one["genuine_lift"] and not one["framing_echo"])
    ok &= c4; print(f"  [{'✓' if c4 else '✗'}] one-sided (2nd cert blind): both_discriminate={one['both_discriminate']} → "
                    f"neither lift nor echo (no agreement to assess)")

    # (5) WINNER'S-CURSE (B FLAG-1): two certs with HIGH AUC but NEGLIGIBLE absolute separation (magnitude ~machine-eps) must
    # NOT count as discriminators — AUC ranks floating-point noise. Effect-size gate rejects it.
    tw = rng.random(200) < 0.5
    tiny = 1e-15 * tw + rng.normal(scale=1e-16, size=200)                    # AUC≈1 but Cohen's d ~ 0 (physically negligible)
    wc = overdet_gate(tiny, tiny + rng.normal(scale=1e-16, size=200), tw)
    c5 = bool(not wc["both_discriminate"] and not wc["genuine_lift"] and not wc["framing_echo"])
    ok &= c5; print(f"  [{'✓' if c5 else '✗'}] winner's-curse (B FLAG-1): auc≈{wc['auc1']:.2f} but Cohen's d={wc['cohens_d1']:.1e} → "
                    f"both_discriminate={wc['both_discriminate']} (effect-size gate rejects negligible-magnitude AUC)")

    # (6) ★thread-17 TAIL-AWARE guard: the rho/n_eff machinery is 2nd-moment; it inverts in heavy tails. The guard
    # must (a) recover the tail-index ORDERING, (b) flag alpha<2 as rho-invalid + recommend a robust combine, (c) flag
    # alpha<=1 as 'decorrelation HURTS with mean', and (d) NOT over-warn on genuinely light (Gaussian) tails.
    def _sym_stable(alpha, n):                                               # CMS symmetric alpha-stable, scale 1
        U = rng.uniform(-np.pi / 2, np.pi / 2, n); W = rng.exponential(1.0, n)
        if abs(alpha - 1.0) < 1e-9:
            return np.tan(U)
        return (np.sin(alpha * U) / np.cos(U) ** (1 / alpha)) * (np.cos(U - alpha * U) / W) ** ((1 - alpha) / alpha)
    tw = rng.random(600) < 0.5
    g_heavy = tail_aware_overdet_guard(tw * 3 + _sym_stable(0.7, 600), tw * 3 + _sym_stable(0.7, 600), tw)   # alpha=0.7 infinite mean
    g_mid = tail_aware_overdet_guard(tw * 3 + _sym_stable(1.5, 600), tw * 3 + _sym_stable(1.5, 600), tw)     # alpha=1.5 infinite var
    g_light = tail_aware_overdet_guard(tw * 3 + rng.normal(size=600), tw * 3 + rng.normal(size=600), tw)     # Gaussian light tail
    c6 = bool((not g_heavy["rho_valid"]) and (not g_heavy["decorrelation_helps"]) and "median" in g_heavy["recommended_combine"]
              and (not g_mid["rho_valid"]) and g_mid["decorrelation_helps"]
              and g_light["rho_valid"] and g_light["recommended_combine"] == "mean" and g_light["decorrelation_helps"]
              and g_heavy["tail_index_alpha"] < g_mid["tail_index_alpha"] < g_light["tail_index_alpha"])
    ok &= c6; print(f"  [{'✓' if c6 else '✗'}] thread-17 tail-aware guard: alpha_hat heavy={g_heavy['tail_index_alpha']:.2f}<"
                    f"mid={g_mid['tail_index_alpha']:.2f}<light={g_light['tail_index_alpha']:.2f}; heavy→{g_heavy['recommended_combine']} "
                    f"(decorr_helps={g_heavy['decorrelation_helps']}), light→{g_light['recommended_combine']} (rho_valid={g_light['rho_valid']})")

    # (7) ★HARD-SUBSET coupling: a pair that is BULK-decorrelated but co-fails on the ambiguous-object subset must be
    # flagged tail_coupled; a genuinely decorrelated pair must NOT be.
    zc7 = rng.uniform(-3.0, 3.0, 800); t7 = zc7 > 0
    hardm = np.abs(zc7 - np.median(zc7)) < np.quantile(np.abs(zc7 - np.median(zc7)), 0.25)
    sh = rng.normal(size=800)
    c1_ = zc7 + np.where(hardm, 0.3 * rng.normal(size=800) + 0.85 * sh, 0.9 * rng.normal(size=800))
    c2_ = zc7 + np.where(hardm, 0.3 * rng.normal(size=800) + 0.85 * sh, 0.9 * rng.normal(size=800))
    hc = hard_subset_decorrelation(c1_, c2_, t7, z_resid=zc7)
    d1_ = zc7 + 0.9 * rng.normal(size=800); d2_ = zc7 + 0.9 * rng.normal(size=800)
    hd = hard_subset_decorrelation(d1_, d2_, t7, z_resid=zc7)
    c7 = bool(hc["tail_coupled"] and abs(hc["rho_bulk"]) < 0.3 and abs(hc["rho_hard"]) > 0.6 and not hd["tail_coupled"])
    ok &= c7; print(f"  [{'✓' if c7 else '✗'}] hard-subset coupling (g998): tail-coupled pair rho_bulk={hc['rho_bulk']:+.2f} "
                    f"(gate would say lift) → rho_HARD={hc['rho_hard']:+.2f} tail_coupled={hc['tail_coupled']}; decorrelated "
                    f"control rho_hard={hd['rho_hard']:+.2f} (not flagged)")

    # (8) ★ GAP-robust tail index (fixes B's P1 gate-4): naive Hill breaks under 3% gross 10x-max spurious spikes;
    # the robust=True gap-trim stays stable AND reduces to naive on clean data.
    xt = (1.0 - rng.uniform(size=3000)) ** (-1.0 / 1.9)                     # Pareto alpha=1.9
    xc = xt.copy(); sp = rng.choice(3000, 90, replace=False); xc[sp] = 10.0 * xt.max()   # 3% at 10x-max
    a_naive_clean = tail_index_hill(xt, robust=False); a_robust_clean = tail_index_hill(xt, robust=True)
    a_naive_contam = tail_index_hill(xc, robust=False); a_robust_contam = tail_index_hill(xc, robust=True)
    naive_shift = abs(a_naive_contam - a_naive_clean) / a_naive_clean
    robust_shift = abs(a_robust_contam - a_robust_clean) / a_robust_clean
    clean_agree = abs(a_naive_clean - a_robust_clean) / a_naive_clean       # robust==naive on clean (no gross jump)
    c8 = bool(naive_shift > 0.25 and robust_shift < 0.12 and clean_agree < 0.02 and abs(a_robust_contam - 1.9) < 0.4)
    ok &= c8; print(f"  [{'✓' if c8 else '✗'}] g1008 gap-robust tail index: naive Hill shift {naive_shift:.0%} (BREAKS under "
                    f"3% 10x spikes) vs robust {robust_shift:.0%} (STABLE, alpha_hat={a_robust_contam:.2f}); clean naive≡robust "
                    f"(Δ{clean_agree:.1%})")

    # (9) ★moment_exists (agent pool-consumed): parameter-free max-to-sum certifies moment (non)existence — Pareto(alpha=1.5)
    # has INFINITE variance (E[X^2]=inf), exponential has all moments FINITE.
    pareto = (1.0 - rng.uniform(size=8000)) ** (-1.0 / 1.5)                 # alpha=1.5 => E[X^2] infinite
    expo = rng.exponential(1.0, size=8000)                                  # all moments finite
    me_pareto = moment_exists(pareto, m=2); me_expo = moment_exists(expo, m=2)
    c9 = bool(me_pareto["finite"] is False and me_expo["finite"] is True)
    ok &= c9; print(f"  [{'✓' if c9 else '✗'}] g1012 moment_exists (max-to-sum): Pareto(a=1.5) E[X^2] finite={me_pareto['finite']} "
                    f"(plateau {me_pareto['plateau']:.3f}, INFINITE) vs exponential finite={me_expo['finite']} "
                    f"(plateau {me_expo['plateau']:.4f}, FINITE) — parameter-free, no fitted tail index")

    print(f"\n  SELFTEST overdet_precondition_gate: {'✓ ALL PASS — gates D over-det program: residual decorrelation + effect-size + thread-17 tail-awareness + hard-subset coupling + g1008 gap-robust tail index + g1012 moment-existence, not raw agreement' if ok else '✗ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
