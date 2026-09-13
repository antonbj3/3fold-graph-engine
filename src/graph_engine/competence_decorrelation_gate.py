"""competence_decorrelation_gate -- the DECORRELATION-STACK layer the uniform double-null exposed (CLAIM:D-UNIFORM-OFFVIEW-
DEPTH-DOUBLENULL-...-METHOD-DECORRELATED-LEGS-SHARE-FEATURE-DEPENDENCE-JOINTLY-BLIND): two legs can be METHOD-decorrelated
(distinct lineage, D layer-1) AND error-uncorrelated on their competent region (layer-2) yet be JOINTLY BLIND on an
input regime where they SHARE a competence-determining dependence. Classical SfM (needs texture features) and neural depth
(DINOv2, degrades on uniform) are method-decorrelated but BOTH lose competence on low-texture input -> jointly blind on
uniform. Neither the lineage gate nor the error-corr gate catches this: error-corr is measured on the COMPETENT region;
on the INCOMPETENT region both abstain/fail so their joint blindness is INVISIBLE to layer-2.

THE MISSING LAYER (this module): measure each leg's COMPETENCE as a function of an input CONDITION axis (e.g. texture,
lighting, motion) and require the legs' competence-dependences to be DECORRELATED -- i.e. one leg competent where the other
fails. If two legs' competence RISES/FALLS together along the same input axis, they share a failure mode and are jointly
blind where that axis is adverse, however decorrelated their methods. This is the FAILURE-MODE-DECORRELATION precondition
for generative leg-discovery: a covering decorrelated leg-SET must span the failure-mode axes, not just the method/lineage
axes. Serves the agent pool-spine (decorrelated channels need a COMPETENCE field, not only a lineage field).

numpy only, no GPU. selftest at bottom."""
from __future__ import annotations
import numpy as np


def competence_by_condition(competence, condition, bins=8):
    """Bin a per-sample competence (0/1 or continuous in [0,1]) over the CONDITION axis -> mean competence per bin.
    Returns (bin_centers, comp_per_bin) with NaN for empty bins (an honest gap, not a fabricated 0)."""
    competence = np.asarray(competence, float); condition = np.asarray(condition, float)
    m = np.isfinite(competence) & np.isfinite(condition)
    competence, condition = competence[m], condition[m]
    if competence.size < bins:
        return np.array([]), np.array([])
    edges = np.quantile(condition, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    centers, comp = [], []
    for i in range(bins):
        sel = (condition >= edges[i]) & (condition < edges[i + 1])
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        comp.append(float(competence[sel].mean()) if sel.any() else np.nan)
    return np.array(centers), np.array(comp)


def competence_correlation(comp_a_by_bin, comp_b_by_bin):
    """Correlation of two legs' competence across the condition axis (on the bins where BOTH are defined).
    HIGH (+) => shared failure axis (competence rises/falls together -> jointly blind where the axis is adverse).
    ~0 / negative => DECORRELATED competence (one covers where the other fails). NaN if <3 shared bins."""
    a, b = np.asarray(comp_a_by_bin, float), np.asarray(comp_b_by_bin, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _wilson_lower(k, n, z=1.96):
    """Wilson score interval LOWER bound for a binomial proportion k/n (default 95%). 0 if n==0."""
    if n <= 0:
        return 0.0
    p = k / n; z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def _worst_condition_single(comp_a, comp_b, condition, bins, min_bin_n, ci_lower):
    """★ distribution-INVARIANT joint-blindness: stratify by the CONDITION axis into FIXED-RANGE bins (robust to a
    zero-inflated axis), and return the WORST bin's joint-blind. ★ (adopting the red-team): gate on the WILSON
    CI-LOWER of each bin's joint-blind proportion, NOT the point estimate -- the worst-bin POINT-MAX false-flags a
    genuinely-decorrelated pair at small n (a lucky bin spikes above threshold from sampling noise; ~4-10% false-flag at
    n=80, 0% at n=800). The CI-lower requires EVIDENCE of joint-blindness (my gate-on-CI-lower-not-point-estimate rule),
    cutting the small-n false-flag to ~0 while keeping true positives (a genuine high-rate bin's CI-lower stays high).
    Returns (worst_joint_blind, adverse_regime_sampled). worst is the max over bins of the CI-lower (or point if ci_lower
    False); NaN if no bin adequately sampled; adverse_regime_sampled False if the most-adverse bin is under-sampled."""
    a = np.asarray(comp_a, float); b = np.asarray(comp_b, float); c = np.asarray(condition, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[m], b[m], c[m]
    if a.size < min_bin_n:
        return np.nan, False
    lo, hi = float(c.min()), float(c.max())
    if hi - lo < 1e-12:
        both = (a < 0.5) & (b < 0.5)
        return (_wilson_lower(int(both.sum()), both.size) if ci_lower else float(both.mean())), True
    edges = np.linspace(lo, hi, bins + 1); edges[-1] += 1e-9
    worst = np.nan; adverse_sampled = False
    for i in range(bins):
        sel = (c >= edges[i]) & (c < edges[i + 1])
        nb = int(sel.sum())
        if nb < min_bin_n:
            continue
        k = int(((a[sel] < 0.5) & (b[sel] < 0.5)).sum())
        jb = _wilson_lower(k, nb) if ci_lower else (k / nb)     # ★CI-lower = EVIDENCE of joint-blindness, small-n-safe
        worst = jb if not np.isfinite(worst) else max(worst, jb)
        if i == 0:                                    # the most-adverse (lowest-condition) bin is adequately sampled
            adverse_sampled = True
    return worst, adverse_sampled


def worst_condition_joint_blind(comp_a, comp_b, condition, bins=8, min_bin_n=10, ci_lower=True):
    """★ (adopting the red-team): MULTI-RESOLUTION worst-condition joint-blind. The single-resolution / gate at
    a COARSE bins misses a LOCALIZED-SEVERE joint-blindness NARROWER than the bin width -- the narrow severe band dilutes
    across the bin and its CI-lower falls below threshold (the '2 winner-curse fixes = different blind spots' insight: a
    coarse stratification/percentile is blind to a localized-severe fault). FIX: sweep FINER resolutions too and take the
    WORST CI-lower -- a localized-severe band CONCENTRATES at a finer resolution, and the WILSON CI-LOWER keeps the false-
    flag at ~0% at EVERY resolution (so finer bins do NOT reintroduce the winner-curse). Gets BOTH: catches localized-severe
    (severe-TP) AND does not false-flag a decorrelated pair (clean-FP). Resolutions swept: {bins, 2*bins, 4*bins} capped so
    each bin keeps >= min_bin_n samples. Returns (worst_joint_blind, adverse_regime_sampled) = the max over resolutions."""
    a = np.asarray(comp_a, float); b = np.asarray(comp_b, float); c = np.asarray(condition, float)
    n_fin = int((np.isfinite(a) & np.isfinite(b) & np.isfinite(c)).sum())
    max_bins = max(bins, n_fin // max(min_bin_n, 1))                # finest resolution keeping >= min_bin_n per bin
    resolutions = sorted({min(bins * f, max_bins) for f in (1, 2, 4)})
    worst = np.nan; adverse = False
    for nb in resolutions:
        w, adv = _worst_condition_single(comp_a, comp_b, condition, int(nb), min_bin_n, ci_lower)
        if np.isfinite(w):
            worst = w if not np.isfinite(worst) else max(worst, w)
        adverse = adverse or adv
    return worst, adverse


def competence_decorrelation_gate(comp_a, comp_b, condition, bins=8, joint_blind_thresh=0.15, stratify=True, min_bin_n=10):
    """Gate two legs for FAILURE-MODE (competence) decorrelation. comp_a/comp_b: per-sample competence in [0,1];
    condition: per-sample input condition (e.g. texture) -- reported as a diagnostic, the DECISION is condition-free and
    robust to a zero-inflated axis. PRIMARY signal = joint_blind_fraction: the fraction of samples where NEITHER leg is
    competent (a<0.5 & b<0.5) -- the coverage GAP the double-null is about. Returns dict(verdict, joint_blind_fraction,
    competence_corr (diagnostic), reason). FLAGGED-JOINTLY-BLIND if a substantial region has no competent leg (the pair
    does NOT span the input space, however decorrelated their methods); DECORRELATED-COMPETENCE if one leg covers where the
    other fails (small joint-blind region)."""
    a = np.asarray(comp_a, float); b = np.asarray(comp_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return dict(verdict="UNDECIDABLE", joint_blind_fraction=np.nan, competence_corr=np.nan,
                    reason="too few finite samples -> cannot assess failure-mode decorrelation")
    a, b = a[m], b[m]
    marginal_joint_blind = float(((a < 0.5) & (b < 0.5)).mean())
    _, ca = competence_by_condition(comp_a, condition, bins)
    _, cb = competence_by_condition(comp_b, condition, bins)
    rc = competence_correlation(ca, cb)

    if stratify:
        # ★ distribution-INVARIANT decision: use the WORST condition-bin joint-blind (detects a rare adverse regime the
        # marginal fraction under-counts on a biased sample). UNDECIDABLE if the adverse regime is under-sampled.
        worst_jb, adverse_sampled = worst_condition_joint_blind(comp_a, comp_b, condition, bins, min_bin_n)
        if not np.isfinite(worst_jb):
            return dict(verdict="UNDECIDABLE", joint_blind_fraction=round(marginal_joint_blind, 4), worst_condition_joint_blind=None,
                        competence_corr=(round(rc, 4) if np.isfinite(rc) else None),
                        reason="no condition bin adequately sampled -> cannot assess failure-mode decorrelation")
        if worst_jb > joint_blind_thresh:
            return dict(verdict="FLAGGED-JOINTLY-BLIND", joint_blind_fraction=round(marginal_joint_blind, 4),
                        worst_condition_joint_blind=round(worst_jb, 4), competence_corr=(round(rc, 4) if np.isfinite(rc) else None),
                        reason=f"worst-condition joint-blind {worst_jb:.0%} > {joint_blind_thresh:.0%}: there is an input "
                               f"regime where NEITHER leg is competent (jointly blind), even if rare in this sample "
                               f"(marginal {marginal_joint_blind:.0%}) -> method-decorrelation NOT sufficient.")
        if not adverse_sampled:
            return dict(verdict="UNDECIDABLE", joint_blind_fraction=round(marginal_joint_blind, 4),
                        worst_condition_joint_blind=round(worst_jb, 4), competence_corr=(round(rc, 4) if np.isfinite(rc) else None),
                        reason="the most-adverse condition regime is UNDER-SAMPLED -> cannot rule out joint blindness there "
                               "(distribution-invariant honest abstain, NOT a false-admit).")
        # ★ (declare the REGIME, winner-curse-fixes-are-regime-specific): this is a FRACTION-STRATA (COVERAGE)
        # verdict -- it certifies no jointly-blind REGION of >= min_bin_n samples, NOT the absence of individual jointly-
        # blind POINTS. The CI-lower/worst-condition statistic is regime-A (fraction-strata); it DEGENERATES on single
        # points. If jointly-blind samples EXIST below the coverage threshold, a SAFETY consumer (single-value tail)
        # must use a value-based (calib-null max) statistic -- not this coverage verdict.
        n_jb = int(((a < 0.5) & (b < 0.5)).sum())
        safety_note = (None if n_jb == 0 else
                       f"COVERAGE ok but {n_jb} individual jointly-blind sample(s) exist below the coverage-region "
                       f"threshold -- this is a FRACTION-STRATA verdict, NOT a single-value safety cert; for a safety tail "
                       f"use a value-based (calib-null max) statistic (regime-B).")
        return dict(verdict="DECORRELATED-COMPETENCE", joint_blind_fraction=round(marginal_joint_blind, 4),
                    worst_condition_joint_blind=round(worst_jb, 4), competence_corr=(round(rc, 4) if np.isfinite(rc) else None),
                    regime="fraction-strata-coverage", safety_note=safety_note,
                    reason=f"worst-condition joint-blind {worst_jb:.0%} <= {joint_blind_thresh:.0%} across all sampled "
                           f"regimes -> one leg covers where the other fails (failure-mode decorrelated).")

    # legacy MARGINAL mode (distribution-dependent; kept for a representative pre-registered sample only)
    if marginal_joint_blind > joint_blind_thresh:
        return dict(verdict="FLAGGED-JOINTLY-BLIND", joint_blind_fraction=round(marginal_joint_blind, 4),
                    competence_corr=(round(rc, 4) if np.isfinite(rc) else None),
                    reason=f"marginal joint-blind {marginal_joint_blind:.0%} > {joint_blind_thresh:.0%} (distribution-dependent).")
    return dict(verdict="DECORRELATED-COMPETENCE", joint_blind_fraction=round(marginal_joint_blind, 4),
                competence_corr=(round(rc, 4) if np.isfinite(rc) else None),
                reason=f"marginal joint-blind {marginal_joint_blind:.0%} <= {joint_blind_thresh:.0%} (distribution-dependent).")


def multi_axis_competence_gate(comp_a, comp_b, condition_axes, bins=8, joint_blind_thresh=0.15, min_bin_n=10):
    """★ -- the competence gate's verdict is AXIS-SCOPED: a DECORRELATED-COMPETENCE pass on ONE condition axis does NOT
    rule out joint blindness on an UNTESTED axis. Run the (worst-condition) gate on EACH supplied axis and compose
    weakest-link: FLAGGED-JOINTLY-BLIND if the pair is jointly blind on ANY axis; DECORRELATED-COMPETENCE only if
    decorrelated on ALL tested axes; UNDECIDABLE if any axis is undecidable and none flags. condition_axes: dict
    {axis_name: per-sample condition}. Returns dict(verdict, tested_axes, per_axis, flagged_axes) -- the verdict is SCOPED
    to tested_axes; an axis NOT in condition_axes is NOT certified (declare it)."""
    per_axis = {}
    flagged, undecidable = [], []
    for name, cond in condition_axes.items():
        g = competence_decorrelation_gate(comp_a, comp_b, cond, bins, joint_blind_thresh, stratify=True, min_bin_n=min_bin_n)
        per_axis[name] = g["verdict"]
        if g["verdict"] == "FLAGGED-JOINTLY-BLIND":
            flagged.append(name)
        elif g["verdict"] == "UNDECIDABLE":
            undecidable.append(name)
    tested = list(condition_axes.keys())
    if flagged:
        verdict = "FLAGGED-JOINTLY-BLIND"
        reason = f"jointly blind on axis/axes {flagged} (decorrelated on {[a for a in tested if a not in flagged]}); a pass on one axis does NOT cover another"
    elif undecidable:
        verdict = "UNDECIDABLE"; reason = f"undecidable on {undecidable} (adverse regime under-sampled)"
    else:
        verdict = "DECORRELATED-COMPETENCE"; reason = f"decorrelated on ALL tested axes {tested} -- but the verdict is SCOPED to these; an UNTESTED axis is NOT certified"
    return dict(verdict=verdict, tested_axes=tested, per_axis=per_axis, flagged_axes=flagged, reason=reason)


def selftest():
    rng = np.random.default_rng(0)
    n = 2000
    tex = rng.random(n)                                   # input condition (texture level)
    # SfM competent where texture is HIGH (needs features); neural also competent where texture high (fails uniform)
    sfm = (tex > 0.4 + 0.1 * rng.standard_normal(n)).astype(float)
    neural = (tex > 0.35 + 0.1 * rng.standard_normal(n)).astype(float)
    # active sensing: competent everywhere (texture-INDEPENDENT)
    active = (rng.random(n) < 0.95).astype(float)
    checks = []
    g1 = competence_decorrelation_gate(sfm, neural, tex)
    checks.append(("method-decorrelated SfM x neural (both texture-gated) -> FLAGGED-JOINTLY-BLIND (shared failure regime)",
                   g1["verdict"] == "FLAGGED-JOINTLY-BLIND" and g1["joint_blind_fraction"] > 0.15))
    g2 = competence_decorrelation_gate(sfm, active, tex)
    checks.append(("SfM x active-sensing (texture-independent) -> DECORRELATED-COMPETENCE (covers the uniform gap)",
                   g2["verdict"] == "DECORRELATED-COMPETENCE" and g2["joint_blind_fraction"] < 0.15))
    g3 = competence_decorrelation_gate(np.ones(4), np.ones(4), tex[:4])   # too few samples
    checks.append(("too few samples -> UNDECIDABLE (no fabricated verdict)", g3["verdict"] == "UNDECIDABLE"))
    print("competence_decorrelation_gate selftest:")
    for nm, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", nm))
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
