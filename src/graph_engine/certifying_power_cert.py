#!/usr/bin/env python3
"""certifying_power_cert — a cert that gates on DIRECTLY-MEASURED certifying power
P(both correct | members agree) + reports the (rho, q) decomposition, instead of gating on correctness-correlation rho
alone (which  showed is INSUFFICIENT: at matched rho the shared-confusion rate q shifts certifying power by ~0.19).

Input contract (richer than decorrelation_validity_cert, which takes only correctness): member ANSWERS + true labels on
a calibration set. From these:
  - correctness_i = (answer_i == label)
  - rho  = worst-pair correctness-correlation (what decorrelation_validity_cert measures; do errors CO-OCCUR)
  - q    = P(agree | both wrong) = agreement-among-wrong (do co-occurring errors AGREE; the / deep-corner axis)
  - certifying_power = P(all correct | all members give the same answer)  <- the quantity the cert should actually gate on
validated iff certifying_power >= min_power AND enough agree-instances to estimate it (else ABSTAIN, not a silent pass).

Why this beats rho-only: two member pairs with IDENTICAL rho can have very different certifying power. rho-only
gives them the same verdict; this cert distinguishes them by the measured power. numpy only, provenance-clean.
Selftest: `python -m graph_engine.certifying_power_cert`."""
from __future__ import annotations
import numpy as np

MIN_POWER = 0.90      # default certifying-power bar to validate
MIN_AGREE = 30        # minimum agree-instances to estimate certifying power (else ABSTAIN -- thin-data guard)
MIN_BOTHWRONG = 10    # minimum both-wrong instances to estimate q (else q is reported as None)


def _worst_correctness_corr(correct):
    """Max pairwise phi-correlation of per-instance correctness across members (constant member -> 1.0, pessimistic)."""
    K = len(correct); worst = -np.inf; pair = None
    for i in range(K):
        for j in range(i + 1, K):
            ci, cj = correct[i].astype(float), correct[j].astype(float)
            if ci.std() < 1e-9 or cj.std() < 1e-9:
                r = 1.0
            else:
                r = float(np.corrcoef(ci, cj)[0, 1]); r = r if np.isfinite(r) else 1.0
            if r > worst:
                worst, pair = r, (i, j)
    return worst, pair


def _wilson_lower(k, n, z=1.96):
    """Wilson score LOWER bound on a binomial proportion k/n. At small n this is well below k/n, so gating on it makes
    validated=True mean 'certifying power is CONFIDENTLY >= bar', not 'the noisy point estimate cleared it' (audit: a
    point-estimate gate over-validated a true-0.80 pair 5.5% of the time at n_agree~35; the n=22 over-claim class)."""
    if n <= 0:
        return 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


STRATA_PRECONDITION = ("certifying power computed on a MIXED calibration set is an AGGREGATE: a 10-20%% subpopulation "
                       "with shared-confusion (q~1) can sit at true power ~0.4 beneath an aggregate power_lo>=0.90 "
                       "(L259 aggregation mirage). Pass strata=<subpopulation labels> to gate on the WORST stratum. "
                       "★L264 WITNESSABILITY BOUNDARY: label QUALITY is only partially self-checkable -- mis-labeled "
                       "strata hiding a SHARED-CONFUSION mode are caught by the label-independent q witness (L263), but "
                       "a hidden DISAGREEING-blind sub-mode (member blind there, errors disagree) is STRUCTURALLY "
                       "unwitnessable from (answers, labels): the iid mixture's contingency table is identical to a "
                       "homogeneous pair's (measured: hidden stratum true power 0.85 under aggregate 0.98, corr 0.009). "
                       "The 'labels must span the BLINDNESS regimes' precondition is therefore DECLARED, not gated.")


def certifying_power_cert(member_answers, true_labels, min_power=MIN_POWER, strata=None):
    """member_answers: list of K arrays (each len N) of predicted class labels; true_labels: array len N.
    Returns dict(validated, certifying_power, rho, q, n_agree, n_both_wrong, n_members, reason).
    validated=True iff certifying_power >= min_power on enough agree-instances; ABSTAIN (validated=False, reason names it)
    when too few agree-instances to estimate. rho/q are diagnostics (decomposition).
    ★: strata (optional, len N): subpopulation labels. When passed, validated additionally requires EVERY stratum's
    own Wilson-lower power >= min_power on that stratum's agree-set (a stratum too thin to estimate -> ABSTAIN, honest).
    When omitted, the AGGREGATION-MIRAGE precondition is surfaced (strata_precondition field): a mixed calibration can
    hide a q~1 subgroup at true power ~0.4 beneath an aggregate power_lo>=0.90 (measured; the aggregate is NOT wrong,
    it is just not a per-subpopulation guarantee)."""
    A = [np.asarray(a) for a in member_answers]
    y = np.asarray(true_labels)
    K = len(A)
    if K < 2:
        return dict(validated=False, certifying_power=None, rho=None, q=None, n_agree=0, n_both_wrong=0,
                    n_members=K, reason="need >=2 members")
 # ★ (inf-sibling) + (red-team HOLE2/4): min_power must be a FINITE SCALAR. NaN was safe (power_lo>=NaN
 # False) but -inf made `power_lo >= -inf` -> True -> FAIL-OPEN; None crashed np.isfinite(None); an ARRAY broke scalar
 # isfinite. np.isfinite (NOT isnan) closes {nan,+inf,-inf}.
    _mp_bad = min_power is None
    if not _mp_bad:
        try:
            _mp = np.asarray(min_power, float); _mp_bad = _mp.ndim != 0 or not bool(np.isfinite(_mp))
        except (ValueError, TypeError):
            _mp_bad = True
    if _mp_bad:
        return dict(validated=False, certifying_power=None, rho=None, q=None, n_agree=0, n_both_wrong=0,
                    n_members=K, reason="bad min_power=%r -- need finite scalar; ABSTAIN (fail-closed)" % (min_power,))
    correct = [a == y for a in A]
 # all members give the SAME answer:
    stacked = np.stack(A, 0)
    agree = np.all(stacked == stacked[0], axis=0)
    all_correct = np.all(np.stack(correct, 0), axis=0)
    n_agree = int(agree.sum())
    rho, _ = _worst_correctness_corr(correct)
 # q = P(agree | all wrong) on the WORST pair's both-wrong set (proxy for shared confusion). Use all-members-wrong.
    all_wrong = np.all(~np.stack(correct, 0), axis=0)
    n_bw = int(all_wrong.sum())
    q = float((agree & all_wrong).sum() / n_bw) if n_bw >= MIN_BOTHWRONG else None
 # ★ (subagent TEST4 dead-zone): MIN_AGREE=30 was INTERNALLY INCONSISTENT with min_power=0.90 -- the smallest n
 # at which even a PERFECT record (k=n) clears Wilson-lower>=0.90 is 35, so n_agree in [30,34] could NEVER validate
 # regardless of true power, and the reason string falsely implied "genuinely low power" at provably power=1.0.
 # Derive the structural minimum from min_power/z instead of the fixed constant, and name the state honestly.
    z2 = 1.96 * 1.96
    n_min_structural = int(np.ceil(z2 * min_power / max(1.0 - min_power, 1e-9)))   # k=n Wilson-lower >= min_power
    n_floor = max(MIN_AGREE, n_min_structural)
    if n_agree < n_floor:
        why = ("structurally under-powered: even a PERFECT record cannot clear Wilson-lower>=%.2f below n=%d"
               % (min_power, n_min_structural)) if n_agree >= MIN_AGREE else \
              ("too thin to estimate certifying power (<%d)" % MIN_AGREE)
        return dict(validated=False, certifying_power=None, rho=round(float(rho), 3), q=q, n_agree=n_agree,
                    n_both_wrong=n_bw, n_members=K,
                    reason="ABSTAIN: %d agree-instances -- %s; need n_agree>=%d" % (n_agree, why, n_floor))
    k = int((all_correct & agree).sum())
    power = float(k / n_agree)
    power_lo = _wilson_lower(k, n_agree)          # gate on the LOWER confidence bound, not the noisy point estimate
    validated = bool(power_lo >= min_power)
    reason = ("certifying power %.3f (Wilson lower %.3f) >= %.2f on %d agree-instances -> validated (rho %.2f, q %s)"
              % (power, power_lo, min_power, n_agree, rho, ("%.2f" % q) if q is not None else "n/a") if validated else
              "certifying power %.3f (Wilson lower %.3f) < %.2f -> NOT validated: not CONFIDENTLY above the bar "
              "(too few agree-instances or genuinely low power; rho %.2f, q %s -- L212: high q = correlated errors "
              "that also AGREE)"
              % (power, power_lo, min_power, rho, ("%.2f" % q) if q is not None else "n/a"))
 # ★ (subagent TEST1 aggregation mirage): gate on the WORST STRATUM when strata is supplied. The aggregate
 # power_lo is NOT a per-subpopulation guarantee: a 10-20% q~1 subgroup at true power ~0.4 hid beneath aggregate
 # power_lo=0.95 (measured on the committed reproducer). Per-stratum: each stratum's own agree-set must clear the
 # bar on ITS Wilson lower bound; a stratum too thin to estimate -> ABSTAIN (undecidable, NOT a pass).
 # ★ (residual 1): the q witness is LABEL-INDEPENDENT, so it must run on the strata-OMITTED path too
 # (the old code chained it inside `if strata is not None`, skipping it on exactly the path that has no labels --
 # q=1.0 sat in the result while validated=True). Compute the witness ONCE here; both paths consult it.
    c_obs = len(np.unique(np.concatenate([stacked.ravel(), y])))
    chance_q = 1.0 / max(c_obs - 1, 1)
    q_witness_fires = (q is not None and q > 0.5 and q > 3.0 * chance_q)
 # ★ (residual 2): for BINARY answers (c_obs==2) the witness is STRUCTURALLY uninformative (q~1 by chance --
 # agreeing-wrong always lands on the one other class), so the subpopulation-transfer claim is UNVERIFIABLE without
 # trusted strata. Consistent with the boundary this is DECLARED machine-readably (a hard ABSTAIN here would,
 # by the same disagreeing-mode argument, have to abstain EVERY no-strata call -- that is consumer policy).
 # ★ (the last open matrix cell): the flag keyed on `strata is None` was NOMINAL -- a COARSE one-label strata
 # washed it (reads 'stratified' while effectively unstratified, per-stratum gate identical to aggregate). Key the
 # flag on EFFECTIVE stratification (>=2 distinct labels) instead. (guard-flag-count-is-nominal, in my own cert.)
    _n_eff_strata = 0
    if strata is not None:
        try:
            _n_eff_strata = len(np.unique(np.asarray(strata)))
        except (ValueError, TypeError):
            _n_eff_strata = 0
    effectively_stratified = _n_eff_strata >= 2
    subpop_unverifiable = bool((not effectively_stratified) and (c_obs <= 2 or q is None))
    strata_worst = None
    if strata is None and q_witness_fires:
        validated = False
        reason = ("ABSTAIN (label-provenance witness, L263/L265): shared-confusion q=%.2f (chance ~%.2f) proves a "
                  "confusion attractor exists and NO strata were supplied to locate it -- the aggregate power %.3f is "
                  "an aggregation mirage candidate; supply strata spanning the confusion mode." % (q, chance_q, power_lo))
    if strata is not None:
        s = np.asarray(strata)
        if s.ndim != 1 or s.shape[0] != y.shape[0] or (s.dtype.kind == "f" and not bool(np.all(np.isfinite(s)))):
            return dict(validated=False, certifying_power=round(power, 3), certifying_power_lower_ci=round(power_lo, 3),
                        rho=round(float(rho), 3), q=q, n_agree=n_agree, n_both_wrong=n_bw, n_members=K,
                        reason="bad strata (need finite 1-D len-N labels); ABSTAIN (fail-closed)")
        per = []
        for lab in np.unique(s):
            m = (s == lab)
            na = int((agree & m).sum())
            if na < n_floor:
                return dict(validated=False, certifying_power=round(power, 3), certifying_power_lower_ci=round(power_lo, 3),
                            rho=round(float(rho), 3), q=q, n_agree=n_agree, n_both_wrong=n_bw, n_members=K,
                            strata_power_lo=None,
                            reason="ABSTAIN: stratum %r has only %d agree-instances (<%d) -- per-stratum power "
                                   "undecidable; gather data on that stratum" % (lab, na, n_floor))
            ka = int((all_correct & agree & m).sum())
            per.append((lab, _wilson_lower(ka, na), na))
        strata_worst = min(per, key=lambda t: t[1])
        if strata_worst[1] < min_power:
            validated = False
            reason = ("NOT validated: worst stratum %r certifying-power Wilson-lower %.3f < %.2f (n_agree=%d) while "
                      "the AGGREGATE was %.3f -- the aggregation mirage this strata gate exists to catch (L259)"
                      % (strata_worst[0], strata_worst[1], min_power, strata_worst[2], power_lo))
        elif q_witness_fires:
 # ★ (labeling-provenance rider): the strata gate TRUSTS the labels -- PERMUTED labels made
 # the mirage mixture validate again (worst stratum 0.93 while the true hard subgroup sits at 0.41). The
 # label-INDEPENDENT witness is q itself (survived the permutation, 0.987); chance-aware (computed
 # once above, shared with the strata-omitted path per residual 1).
            validated = False
            reason = ("ABSTAIN (label-provenance, L263): shared-confusion q=%.2f (chance ~%.2f) proves a confusion "
                      "attractor exists, yet EVERY supplied stratum clears the bar (worst %.3f) -- the strata labels "
                      "likely do not SPAN the confusion regime (mis-labeled/permuted labels reproduce this exactly); "
                      "cannot certify subpopulation transfer. Re-stratify by the actual confusion mode."
                      % (q, chance_q, strata_worst[1]))
    return dict(validated=validated, certifying_power=round(power, 3), certifying_power_lower_ci=round(power_lo, 3),
                rho=round(float(rho), 3), q=q, n_agree=n_agree, n_both_wrong=n_bw, n_members=K,
                strata_power_lo=(round(strata_worst[1], 3) if strata_worst else None),
 # ★: a COARSE one-label strata is EFFECTIVELY unstratified -- surface the precondition for it too.
                strata_precondition=(None if effectively_stratified else STRATA_PRECONDITION),
                subpopulation_unverifiable=subpop_unverifiable,   # ★L265/L267: True = NOT effectively stratified
 # (no strata OR <2 distinct labels) AND the q witness is structurally uninformative (binary / q
 # inestimable) -- consumers requiring a subpopulation-transfer guarantee should treat validated=True
 # as AGGREGATE-ONLY and demand a real (>=2-label) stratification.
                reason=reason)


def _make_pair(rho_target, q_target, p=0.6, C=5, n=4000, seed=0):
    """Generate a 2-member (answers, labels) sample with ~target correctness-correlation and shared-confusion q."""
    rng = np.random.default_rng(seed)
    r = rho_target  # shared-fraction ~ rho for this competence range (empirical, close enough for the demo)
    thr = 0.0 if abs(p - 0.5) < 1e-9 else -np.sqrt(2) * _erfinv(2 * p - 1)
    g = rng.normal(size=n)
    y = np.zeros(n, int)   # true label 0
    ans = []
    for m in range(2):
        s = np.sqrt(max(r, 0)) * g + np.sqrt(max(1 - r, 0)) * rng.normal(size=n)
        correct = s > thr
        a = np.zeros(n, int)
        wrong = ~correct; nw = int(wrong.sum())
        att = rng.random(nw) < q_target            # shared attractor wrong class 1 w.p. q
        a[wrong] = np.where(att, 1, rng.integers(1, C, nw))
        ans.append(a)
    return ans, y


def _erfinv(z):
    a = 0.147; ln = np.log(1 - z * z); t = 2 / (np.pi * a) + ln / 2
    return np.sign(z) * np.sqrt(np.sqrt(t * t - ln / a) - t)


def _selftest():
    ok = tot = 0

 # ★DECISIVE test: TWO pairs with MATCHED rho but different q. rho-only would give them the SAME verdict; this
 # cert must SEPARATE them by measured certifying power (low q -> high power -> validated; high q -> low -> not).
    ans_lowq, y = _make_pair(0.6, q_target=0.0, p=0.72, C=8, n=6000, seed=1)   # correlated correctness, wrong answers DISAGREE
    ans_hiq, y2 = _make_pair(0.6, q_target=1.0, p=0.72, C=8, n=6000, seed=1)   # SAME rho + competence, wrong answers AGREE
    r_low = certifying_power_cert(ans_lowq, y, min_power=0.9)
    r_hi = certifying_power_cert(ans_hiq, y2, min_power=0.9)
 # (1) rho is ~matched across the two pairs (the fair-comparison precondition)
    tot += 1; ok += abs(r_low["rho"] - r_hi["rho"]) < 0.08
 # (2) the low-q pair certifies (agreement implies correct)...
    tot += 1; ok += (r_low["certifying_power"] > r_hi["certifying_power"] + 0.05)
 # (3)... and the high-q pair has strictly LOWER power -> the cert SEPARATES matched-rho pairs rho-only cannot
    tot += 1; ok += (r_hi["q"] is not None and r_hi["q"] > r_low["q"])
 # (4) ABSTAIN on thin data (few agree-instances) rather than silently pass
    thin_ans = [np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 4])]
    r_thin = certifying_power_cert(thin_ans, np.array([0, 0, 0, 0, 0]), min_power=0.9)
    tot += 1; ok += (not r_thin["validated"] and "ABSTAIN" in r_thin["reason"])
 # (5) a genuinely decorrelated, low-q, competent pair validates (large n -> Wilson lower ~ point estimate)
    tot += 1; ok += (r_low["validated"] is True)
 # (6) ★n-CI guard: a SMALL-n pair whose POINT power clears the bar but whose Wilson LOWER bound does NOT must NOT
 # validate (audit: point-estimate gate over-validated a true-0.80 pair 5.5% at n_agree~35; the n=22 class).
 # 35 agree-instances, 33/35 both-correct-when-agree -> point 0.943 but Wilson lower ~0.82 < 0.90.
    sa = np.r_[np.zeros(33, int), np.full(2, 1), np.arange(2, 7)]      # 33 both-correct, 2 both-wrong-agree, 5 disagree
    sb = np.r_[np.zeros(33, int), np.full(2, 1), np.arange(7, 12)]
    ys = np.zeros(len(sa), int)
    r_smalln = certifying_power_cert([sa, sb], ys, min_power=0.90)
    tot += 1; ok += (r_smalln["certifying_power"] >= 0.90 and r_smalln["certifying_power_lower_ci"] < 0.90
                     and r_smalln["validated"] is False)
 # ★ inf-sibling sentinel (completeness-audit): a NON-FINITE min_power must fail CLOSED. NaN was already safe
 # but min_power=-inf made `power_lo >= -inf` -> True -> FAIL-OPEN. np.isfinite closes all of {nan,+inf,-inf}.
    ac2, y2 = _make_pair(0.6, q_target=0.0, p=0.72, C=8, n=6000, seed=1)
    tot += 1; ok += all(certifying_power_cert(ac2, y2, min_power=v)["validated"] is False
                        for v in (float("nan"), float("inf"), float("-inf")))
 # (8) ★ aggregation-mirage sentinel (subagent TEST1): a 90/10 easy/hard mixture where the hard subgroup has
 # shared confusion (q~1, true power ~0.4) must (a) VALIDATE on the aggregate WITHOUT strata (the mirage,
 # precondition surfaced) and (b) be CAUGHT (not validated) WITH strata passed -- the worst-stratum gate.
    rngm = np.random.default_rng(7); nm = 30000; Cm = 8
    ym = np.zeros(nm, int)
    hardm = rngm.random(nm) < 0.10
    a1, a2 = ym.copy(), ym.copy()
    ez = ~hardm & (rngm.random(nm) < 0.03)                  # rare decorrelated easy errors
    a1[ez] = rngm.integers(1, Cm, int(ez.sum())); ez2 = ~hardm & (rngm.random(nm) < 0.03)
    a2[ez2] = rngm.integers(1, Cm, int(ez2.sum()))
    att = (rngm.random(nm) < 0.6) & hardm                   # hard: shared-confusion attractor (both wrong, same answer)
    a1[att] = 1; a2[att] = 1
    r_mix = certifying_power_cert([a1, a2], ym, min_power=0.90)
    r_str = certifying_power_cert([a1, a2], ym, min_power=0.90, strata=hardm.astype(int))
 # ★ (residual 1): the label-independent witness now runs on the OMITTED path too -- the q~1 mirage mixture
 # WITHOUT strata must ABSTAIN (previously validated=True with q=0.99 sitting in the result).
    tot += 1; ok += (r_mix["validated"] is False and "label-provenance witness" in r_mix["reason"]
                     and r_str["validated"] is False and "stratum" in r_str["reason"])
 # (9) ★ label-provenance sentinel (rider): PERMUTED strata labels (uncorrelated with the true
 # confusion regime) previously re-validated the same mirage mixture (worst labeled stratum ~aggregate). The
 # label-independent q witness must catch it (q survives permutation) -> ABSTAIN; and a clean low-q ensemble
 # with arbitrary labels must NOT over-abstain.
    r_perm = certifying_power_cert([a1, a2], ym, min_power=0.90, strata=rngm.permutation(hardm.astype(int)))
    g1m, g2m = ym.copy(), ym.copy()
    wg1 = rngm.random(nm) < 0.05; g1m[wg1] = rngm.integers(1, Cm, int(wg1.sum()))
    wg2 = rngm.random(nm) < 0.05; g2m[wg2] = rngm.integers(1, Cm, int(wg2.sum()))
    r_ok = certifying_power_cert([g1m, g2m], ym, min_power=0.90, strata=rngm.integers(0, 2, nm))
    tot += 1; ok += (r_perm["validated"] is False and "label-provenance" in r_perm["reason"]
                     and r_ok["validated"] is True)
 # (10) ★ (residual 2): BINARY answers -> the witness is structurally uninformative (q~1 by chance) -> a
 # no-strata binary cert must surface subpopulation_unverifiable=True (machine-readable DECLARED boundary,
 # consistent with: declared not fake-gated); the multi-class low-q clean case must have it False.
    yb = rngm.integers(0, 2, nm); b1 = yb.copy(); b2 = yb.copy()
    wb1 = rngm.random(nm) < 0.05; b1[wb1] = 1 - yb[wb1]
    wb2 = rngm.random(nm) < 0.05; b2[wb2] = 1 - yb[wb2]
    r_bin = certifying_power_cert([b1, b2], yb, min_power=0.90)
    r_mc = certifying_power_cert([g1m, g2m], ym, min_power=0.90)
    tot += 1; ok += (r_bin["validated"] is True and r_bin["subpopulation_unverifiable"] is True
                     and r_mc["subpopulation_unverifiable"] is False)
 # (11) ★ (the last open matrix cell): binary + COARSE one-label strata must NOT wash the flag -- it is
 # EFFECTIVELY unstratified (flag keyed on >=2 distinct labels, not `strata is None`); precondition surfaced.
 # Both-ways: binary + a REAL 2-label stratification clears the flag (per-stratum gate genuinely ran).
    r_coarse = certifying_power_cert([b1, b2], yb, min_power=0.90, strata=np.zeros(nm, int))
    r_real2 = certifying_power_cert([b1, b2], yb, min_power=0.90, strata=rngm.integers(0, 2, nm))
    tot += 1; ok += (r_coarse["subpopulation_unverifiable"] is True and r_coarse["strata_precondition"] is not None
                     and r_real2["subpopulation_unverifiable"] is False)

    print("certifying_power_cert selftest: %d/%d (matched rho %.2f/%.2f | low-q power=%.3f lo=%.3f val=%s | "
          "high-q power=%.3f val=%s | thin->ABSTAIN | small-n point=%.3f lo=%.3f val=%s)"
          % (ok, tot, r_low["rho"], r_hi["rho"], r_low["certifying_power"], r_low["certifying_power_lower_ci"],
             r_low["validated"], r_hi["certifying_power"], r_hi["validated"],
             r_smalln["certifying_power"], r_smalln["certifying_power_lower_ci"], r_smalln["validated"]))
    return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
