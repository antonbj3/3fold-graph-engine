#!/usr/bin/env python3
"""member_decorrelation_cert — agent-worktree DEPLOYABLE distilling the n_eff analysis into ONE actionable pre-check: given candidate decorrelated
MEMBERS' per-item scores and a ground-truth correctness/error signal, does their AGREEMENT actually CERTIFY (predict correctness) -- or is
it a SHARED-CONFUSION illusion? Complements overdet_neff (effective leg COUNT) and my decorrelated_probe_value (operator-overlap of a
NEW observation); this is the EMPIRICAL cert-validity gate on real per-item data.

THE MAP (my session, decorrelation-certifies IFF members are genuinely independent ON THE CONFUSION MODE):
  CERTIFIES  — cross-MODEL agreement (AUC 0.91), complementary OBSERVATIONS, decorrelated recovery channels
  FAILS      — LAYERS of one model (AUC 0.38, share the commitment), same-OBSERVATION channels (info-floor), same-modality SENSORS
               (agreement=prominence), scalar-decorrelation (wrong operator)
Three gates a member-agreement cert must pass, all checkable from per-item data:
  (1) BEYOND-SINGLE-SIGNAL: agreement must predict correctness ABOVE the best single member's own confidence (else it is just
      that self-signal; layers FAIL this, being anti-correlated after partialling confidence).
  (2) HARD-SUBSET INDEPENDENCE: the members must be decorrelated (n_eff>1.5) ON THE HARD/decisive items (the wrong/high-error ones),
      not just the easy bulk (legs collinear on the hard cases over-state the over-determination).
  (3) COVERAGE / COMPETENCE: where members agree, coverage must be >0 (incompetent members never converge -> gate safely abstains).

  member_decorrelation_cert(member_scores, correct, single_confidence=None, hard_frac=0.3) -> verdict dict
numpy only (scipy optional for a cleaner AUC)."""
import numpy as np
try:
    from scipy.stats import rankdata as _rankdata            # ★fleet argsort-tie-bug fix: rankdata AVERAGES ties (argsort(argsort) does NOT)
except Exception:
    def _rankdata(a, axis=None):                             # numpy fallback: average-rank for ties
        a = np.asarray(a, float)
        if axis is None:
            order = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
            # average ties
            _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
            csum = np.cumsum(cnt); starts = csum - cnt
            avg = (starts + csum + 1) / 2.0
            return avg[inv]
        return np.apply_along_axis(lambda v: _rankdata(v), axis, a)


def _auc(score, label):
    label = np.asarray(label).astype(bool)
    if label.all() or not label.any():
        return float("nan")
    score = np.asarray(score, float)
    # class (this tick): scipy.stats.rankdata does NOT just poison the NaN position -- a SINGLE NaN entry makes
    # the ENTIRE ranked array NaN (verified: rankdata([1,2,nan,4])=[nan,nan,nan,nan]). Guard before ranking.
    if not np.all(np.isfinite(score)):
        raise ValueError("score contains non-finite values -- rankdata would poison the whole vector")
    r = _rankdata(score)                  # ★tie-safe (was argsort(argsort), mis-ranks tied/discrete scores)
    return float((r[label].mean() - (label.sum() + 1) / 2.0) / (~label).sum())


def _neff(scores):
    """participation ratio of the member score-correlation eigen-spectrum (= the joint n_eff, NOT pairwise-mean)."""
    scores = np.asarray(scores, float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values -- rankdata would poison the whole column")
    X = np.apply_along_axis(_rankdata, 0, scores)     # ★tie-safe spearman rank (was argsort(argsort))
    X = X - X.mean(0); sd = X.std(0)
    keep = sd > 1e-12
    if keep.sum() < 2:
        return float(keep.sum())
    Xk = X[:, keep] / sd[keep]
    C = Xk.T @ Xk / len(Xk); lam = np.linalg.eigvalsh(np.clip(C, -1, 1)); lam = lam[lam > 1e-12]
    return float((lam.sum() ** 2) / np.sum(lam ** 2)) if lam.size else float(keep.sum())


def member_decorrelation_cert(member_scores, correct, single_confidence=None, hard_frac=0.3, agree_tol=None, ground_truth=None):
    """member_scores: (n_items, n_members) each member's per-item answer/score. correct: (n_items,) bool GT-correctness.
    single_confidence: (n_items) optional best single member's own confidence (the baseline agreement must beat). hard_frac: fraction of
    items treated as the HARD/decisive subset (lowest agreement = the contested ones). ground_truth: (n_items) optional -- ★ audit fix:
    if the member SCORES scale with item difficulty/magnitude (arithmetic answers, regression outputs, confidence-on-a-scale), pass the
    ground-truth value so the tool operates on GT-CENTERED RESIDUALS (error-direction agreement), NOT raw scores. Otherwise members look
    'agreeing' merely because both track the item magnitude (raw rank-corr 0.88) not because they share an error mechanism (residual
    rank-corr 0.35) -> the tool false-labels genuinely-decorrelated members SHARED_CONFUSION. Same lesson as: gate on the outcome-
    relevant coordinate (error), not a proxy (raw scale). Returns whether member-AGREEMENT certifies."""
    S = np.asarray(member_scores, float); correct = np.asarray(correct).astype(bool)
    if ground_truth is not None:
        S = S - np.asarray(ground_truth, float)[:, None]     # ★residualize: remove the shared item-magnitude, keep the error direction
    # ★ EXPLICIT finite guard (argreduce/rank-on-NaN degeneracy class): a partially-failed member (NaN scores) would,
    # in an argsort/rank correlation, sort NaN to a fixed position -> BIAS the correlation DOWNWARD -> a truly-correlated
    # (shared-confusion) pair reads as more decorrelated = fail-open. Previously this cert was only ACCIDENTALLY fail-closed
    # (NaN -> neff NaN -> `neff>1.5` False). Guard it EXPLICITLY: drop non-finite rows; ABSTAIN if too few remain.
    if S.size:
        fin = np.isfinite(S).all(axis=1)
        if not fin.all():
            S = S[fin]; correct = correct[fin]
    if S.shape[0] < 10:
        return dict(verdict="ABSTAIN-NONFINITE", n_finite=int(S.shape[0]), n_members=int(S.shape[1] if S.ndim == 2 else 0),
                    reason="too few finite items after dropping non-finite member scores (a partially-failed leg) -> cannot "
                           "certify decorrelation; rank/argreduce on NaN silently biases toward spurious decorrelation (class)")
    n, k = S.shape
    if k < 2:
        return dict(verdict="INSUFFICIENT_MEMBERS", n_members=k)
    if n < 1:
        # ★ self-census: the member guard (k<2) did NOT cover the ITEM dimension, so an empty item-set (n=0)
        # crashed uncontrolled at np.percentile(disp,40) on empty disp -- a "false-admit-safe yet crashes uncontrolled"
        # gap (guarded on one dimension, unguarded on the other). Abstain cleanly instead of crashing.
        return dict(verdict="INSUFFICIENT_ITEMS", n_items=n, n_members=k)
    # AGREEMENT signal = negative mean pairwise dispersion (numeric) per item (high = members agree)
    disp = np.array([np.mean([abs(S[i, a] - S[i, b]) for a in range(k) for b in range(a + 1, k)]) for i in range(n)])
    agreement = -disp
    auc_agree = _auc(agreement, correct)
    # ★ (self-quantile class, resweep): the OLD default `disp <= percentile(disp,40)` is a SELF-QUANTILE ->
    # coverage == 0.40 for ANY data (a tautology: 40% of items are below their own 40th percentile), so the
    # INSUFFICIENT_COVERAGE guard was VACUOUS in the default path (only worked with an explicit agree_tol). Replace with a
    # SHUFFLED-NULL-relative coverage (scale-free AND meaningful): an item genuinely agrees if its dispersion is below the
    # dispersion expected under column-permuted (chance) members. coverage = fraction below the null's 5th percentile ->
    # varies with REAL agreement (genuine-agree ~0.24, random ~0.06) instead of always 0.40.
    if agree_tol is None:
        rng = np.random.default_rng(0)
        null_disp = []
        for _ in range(20):
            Sp = np.column_stack([rng.permutation(S[:, c]) for c in range(k)])
            null_disp.extend([np.mean([abs(Sp[i, a] - Sp[i, b]) for a in range(k) for b in range(a + 1, k)]) for i in range(n)])
        tol = float(np.percentile(null_disp, 5))
    else:
        tol = agree_tol
    coverage = float((disp <= tol).mean())
    # (1) beyond single-signal
    auc_conf = _auc(single_confidence, correct) if single_confidence is not None else float("nan")
    beyond = (auc_agree - auc_conf) if not np.isnan(auc_conf) else float("nan")
    beyond_ok = bool(np.isnan(beyond) or beyond > 0.03)
    # (2) hard-subset independence: n_eff of members on the HARD (lowest-agreement) items
    hard_idx = np.argsort(-disp)[:max(3, int(hard_frac * n))]
    neff_hard = _neff(S[hard_idx]); neff_full = _neff(S)
    hard_ok = bool(neff_hard > 1.5)
    # (3) coverage
    cov_ok = bool(coverage > 0.05)
    if not cov_ok:
        verdict = "INSUFFICIENT_COVERAGE"
    elif auc_agree > 0.6 and beyond_ok and hard_ok:
        verdict = "CERTIFIES"
    elif (not hard_ok) or (not beyond_ok) or auc_agree < 0.55:
        verdict = "SHARED_CONFUSION"
    else:
        verdict = "WEAK"
    return dict(verdict=verdict, auc_agreement=round(auc_agree, 3), auc_single_confidence=round(auc_conf, 3) if not np.isnan(auc_conf) else None,
                agreement_beyond_confidence=round(beyond, 3) if not np.isnan(beyond) else None,
                neff_hard_subset=round(neff_hard, 2), neff_full=round(neff_full, 2), coverage=round(coverage, 3),
                n_members=k, guidance={
                    "CERTIFIES": "members are independent on the hard subset AND agreement beats the single-member signal -> agreement certifies (abstain on disagreement)",
                    "SHARED_CONFUSION": "members collapse to one effective leg on the hard cases OR agreement adds nothing beyond single-member confidence -> agreement does NOT certify (like cross-LAYER L89 / same-observation L84); use genuinely independent members (cross-model, decorrelated observations)",
                    "INSUFFICIENT_COVERAGE": "members never agree (incompetent for the task) -> gate abstains everywhere (L86); need more competent members",
                    "WEAK": "borderline; more data or better members"}[verdict])


def _selftest():
    rng = np.random.default_rng(0); n = 400; ok = tot = 0
    truth = rng.random(n) < 0.6                                           # correct/wrong label
    base = rng.normal(size=n)                                             # the shared correct answer signal
    # (A) CERTIFIES (cross-model -like): when CORRECT all members ~ base (agree); when WRONG each member = base + big
    # INDEPENDENT noise (disagree, and independent on the wrong items). Agreement <-> correct; n_eff high on hard.
    MA = np.stack([np.where(truth, base, base + 2.5 * rng.normal(size=n)) for _ in range(3)], 1)
    conf = truth.astype(float) + 0.7 * rng.normal(size=n)                 # a WEAK single-member confidence (noisier than agreement)
    rA = member_decorrelation_cert(MA, truth, conf)
    tot += 1; ok += (rA["verdict"] == "CERTIFIES")
    # (B) SHARED_CONFUSION (cross-LAYER / same-observation): members ~always AGREE whether right or wrong
    # (they share the answer/commitment) -> agreement does NOT separate correct from wrong; collinear on the hard subset.
    MB = np.stack([base + 0.1 * rng.normal(size=n) for _ in range(3)], 1)
    rB = member_decorrelation_cert(MB, truth, conf)
    tot += 1; ok += (rB["verdict"] in ("SHARED_CONFUSION", "WEAK"))
    # (C) INSUFFICIENT_COVERAGE: members always disagree (never converge)
    MC = rng.normal(size=(n, 3)) * 5
    rC = member_decorrelation_cert(MC, truth, agree_tol=1e-6)
    tot += 1; ok += (rC["verdict"] == "INSUFFICIENT_COVERAGE")
    print("member_decorrelation_cert selftest: %d/%d  (A=%s[auc=%.2f neff_hard=%.2f] B=%s[auc=%.2f neff_hard=%.2f] C=%s)" % (
        ok, tot, rA["verdict"], rA["auc_agreement"], rA["neff_hard_subset"],
        rB["verdict"], rB["auc_agreement"], rB["neff_hard_subset"], rC["verdict"]))
    return ok == tot


if __name__ == "__main__":
    _selftest()
