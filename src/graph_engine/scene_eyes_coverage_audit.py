"""scene_eyes_coverage_audit — an agent worktree's σ-backbone RUNTIME coverage certificate for the ensemble scene-eyes trust-map
(the operator's ONE GOAL). Operationalizes B's three fusion-flags into one audit D runs on the assembled compose:

  1. trust-VALUE decorrelation ≠ ERROR/MISS decorrelation (B 4f84e9432): measure leg decorrelation on the FAILURE
     subset (fence decisions vs GT fakes), NOT the eff_rank of the trust matrix — the latter is blind to a shared
     blind spot (independent noise on a shared wrongness reads as decorrelated). Generalizes G 's fence-ρ method.
  2. a KIND LABEL is an unverified decorrelation ASSERTION (B grading J 64e19fbb9): a leg LABELED 'provenance' confers
     hallucination-protection only if it is ACTUALLY decorrelated from the consistency common-mode — verify by measured
     disjoint-fence on planted fakes, not the `kinds` string. Flags a mislabeled / shared-upstream / weak provenance leg.
  3. COMPLETENESS is uncertifiable from the legs alone (agent pool-gauge-ceiling): report the RESIDUAL = fakes NO leg
     fences (the shared blind spot). n_eff=K over designed modes ≠ coverage of an un-modelled mode. Needs external GT.

INPUT: legs {axis: trust_array∈[0,1]}, kinds {axis: 'consistency'|'provenance'}, fake_masks {fake_type: bool GT-fake
array}, clean_mask (bool, GT-clean regions for FP). All arrays length R. GT-anchored (that is the point — coverage is
NOT self-certifiable; supply planted/held-out fakes). numpy only, no side effects. Grader tool, not a fusion engine.
"""
from __future__ import annotations
import numpy as np

__all__ = ["coverage_audit", "_selftest"]


def _fence(trust, thr):
    return np.asarray(trust, float) < thr


def coverage_audit(legs: dict, fake_masks: dict, clean_mask=None, kinds: dict | None = None, thr: float = 0.5) -> dict:
    names = list(legs)
    R = len(next(iter(legs.values())))
    F = {k: np.asarray(m, bool) for k, m in fake_masks.items()}
    any_fake = np.zeros(R, bool)
    for m in F.values():
        any_fake |= m
    clean = np.asarray(clean_mask, bool) if clean_mask is not None else ~any_fake

    fence = {n: _fence(legs[n], thr) for n in names}                      # per-leg fence decisions

    # (1) per-leg recall on each fake-type + FP on clean
    per_leg = {}
    for n in names:
        rec = {ft: float(fence[n][m].mean()) if m.any() else float("nan") for ft, m in F.items()}
        fp = float(fence[n][clean].mean()) if clean.any() else float("nan")
        per_leg[n] = {"recall_by_fake": rec, "fp_clean": fp}

    # (2) MISS-subset decorrelation THAT COUNTS = UNIQUE-CATCH, not eff_rank of the fence matrix.
    # (eff_rank is the WRONG metric here: two legs with DISJOINT catch are anti-correlated → eff_rank≈1, yet
    # disjoint catch is exactly the decorrelated-coverage we want. Unique-catch measures necessity directly.)
    # unique_catch[n] = # fake regions ONLY leg n fences (drop it → those fakes leak). n_necessary = legs with >0.
    unique_catch = {}
    for n in names:
        others = np.zeros(R, bool)
        for m2 in names:
            if m2 != n:
                others |= fence[m2]
        unique_catch[n] = int((fence[n] & any_fake & ~others).sum())
    n_necessary_legs = int(sum(1 for n in names if unique_catch[n] > 0))
    Ztrust = np.array([np.asarray(legs[n], float) for n in names])        # (K, R)
    trustval_effrank = _effrank(Ztrust.T)                                 # what leg_neff reports (bulk) — for contrast

    # (3) ensemble coverage (min-trust union): a fake is COVERED iff SOME applicable leg fences it
    ens_fence = np.zeros(R, bool)
    for n in names:
        ens_fence |= fence[n]
    covered = {ft: float(ens_fence[m].mean()) if m.any() else float("nan") for ft, m in F.items()}
    residual_blindspot = {ft: int((m & ~ens_fence).sum()) for ft, m in F.items()}   # fakes NO leg catches
    ens_fp = float(ens_fence[clean].mean()) if clean.any() else float("nan")

    # (4) KIND-LABEL validation: is each leg LABELED 'provenance' ACTUALLY decorrelated from the consistency
    # common-mode? Measure its fence overlap (Jaccard) with the consistency legs' SHARED fences on the fakes.
    prov_check = {}
    if kinds is not None:
        cons_names = [n for n in names if not str(kinds.get(n, "consistency")).lower().startswith("prov")]
        prov_names = [n for n in names if str(kinds.get(n, "consistency")).lower().startswith("prov")]
        cons_shared_fence = np.ones(R, bool)
        for n in cons_names:
            cons_shared_fence &= fence[n]                                 # regions ALL consistency legs fence (their common mode)
        for pn in prov_names:
            pf = fence[pn]
            # a valid provenance leg must CATCH fakes the consistency legs MISS — that IS its decorrelated value.
            # (overlap alone is not enough: a leg that ALSO misses the fake has empty∩empty overlap 0 yet adds nothing.)
            cons_miss = any_fake & ~np.array([fence[n] for n in cons_names], bool).any(0) if cons_names else any_fake
            unique_prov_catch = int((pf & cons_miss).sum())              # fakes the consistency legs miss that THIS prov leg catches
            n_cons_miss = int(cons_miss.sum())
            overlap = _jaccard(pf & any_fake, cons_shared_fence & any_fake)
            prov_check[pn] = {"catches_of_consistency_missed_fakes": unique_prov_catch, "n_consistency_missed_fakes": n_cons_miss,
                              "fence_jaccard_vs_consistency_commonmode": overlap,
                              # decorrelated iff it recovers the fakes the consistency common-mode misses (its whole purpose)
                              "actually_decorrelated": bool(unique_prov_catch > 0)}

    return {
        "n_regions": R, "n_fake": int(any_fake.sum()), "n_clean": int(clean.sum()),
        "per_leg": per_leg,
        "unique_catch_per_leg": unique_catch,         # (1) the RIGHT decorrelation-coverage metric (necessity)
        "n_necessary_legs": n_necessary_legs,         #     legs with >0 unique catch = genuinely-decorrelated coverage
        "trustvalue_effrank": trustval_effrank,       #     the insufficient one leg_neff reports (for contrast)
        "ensemble_coverage_by_fake": covered,
        "residual_blindspot_by_fake": residual_blindspot,   # (3) fakes NO leg catches — the completeness gap
        "ensemble_fp_clean": ens_fp,
        "provenance_label_check": prov_check,         # (2) is each 'provenance'-labeled leg actually decorrelated?
        "note": "coverage is GT-anchored (planted/held-out fakes); it is NOT self-certifiable (fleet-gauge-ceiling). "
                "miss_subset_effrank is the decorrelation that counts; trustvalue_effrank is blind to a shared blind spot.",
    }


def _effrank(M):
    """participation-ratio eff_rank of a (rows × legs) matrix; ~1 redundant, ~K independent. Guards constant columns."""
    M = np.asarray(M, float)
    if M.ndim != 2 or M.shape[0] < 2 or M.shape[1] < 1:
        return float("nan")
    Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
    s = np.linalg.svd(Z, compute_uv=False); s = s[s > 1e-9]
    if s.size == 0:
        return 1.0
    p = (s ** 2) / (s ** 2).sum()
    return float(np.exp(-(p * np.log(p + 1e-30)).sum()))


def _jaccard(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    u = int((a | b).sum())
    return (int((a & b).sum()) / u) if u else 0.0


def _selftest():
    ok = 0; tot = 0
    def ck(n, c):
        nonlocal ok, tot; tot += 1; ok += bool(c); print("  [%s] %s" % ("PASS" if c else "FAIL", n))
    rng = np.random.default_rng(0)
    R = 300
    fA = np.zeros(R, bool); fA[:40] = True          # fake type A (regions 0-39)
    fB = np.zeros(R, bool); fB[40:80] = True         # fake type B (regions 40-79)
    clean = np.zeros(R, bool); clean[100:] = True

    # DECORRELATED legs: geometric fences A, resolution fences B; both trust the clean regions
    geo = np.full(R, 0.9); geo[fA] = 0.1
    res = np.full(R, 0.9); res[fB] = 0.1
    aud = coverage_audit({"geometric": geo, "resolution": res},
                         {"A": fA, "B": fB}, clean_mask=clean, kinds={"geometric": "consistency", "resolution": "provenance"})
    ck("decorrelated: coverage A=1", aud["ensemble_coverage_by_fake"]["A"] >= 1.0 - 1e-12)   # tolerance not literal == (RATATAT §4)
    ck("decorrelated: coverage B=1", aud["ensemble_coverage_by_fake"]["B"] >= 1.0 - 1e-12)
    ck("decorrelated: both legs necessary (unique-catch>0)", aud["n_necessary_legs"] == 2 and aud["unique_catch_per_leg"]["geometric"] == 40 and aud["unique_catch_per_leg"]["resolution"] == 40)
    ck("decorrelated: residual blindspot 0", aud["residual_blindspot_by_fake"]["A"] == 0 and aud["residual_blindspot_by_fake"]["B"] == 0)
    ck("prov leg validated as decorrelated", aud["provenance_label_check"]["resolution"]["actually_decorrelated"])

    # SHARED BLIND SPOT: both legs miss fake type C (a coherent fake) — trust it high with independent noise
    fC = np.zeros(R, bool); fC[:40] = True
    # both legs SOLIDLY trust every region (mean 0.9, tiny INDEPENDENT noise 0.03 → never crosses thr=0.5): a CLEAN
    # shared blind spot (both miss fC) with independent per-leg variation (so trust-value eff_rank still reads ~2).
    g2 = np.clip(0.90 + 0.03 * rng.standard_normal(R), 0, 1)
    r2 = np.clip(0.90 + 0.03 * rng.standard_normal(R), 0, 1)
    aud2 = coverage_audit({"geometric": g2, "resolution": r2}, {"C": fC}, clean_mask=clean)
    ck("shared-blindspot: trustvalue_effrank looks ~2 (reassuring, WRONG)", aud2["trustvalue_effrank"] > 1.6)
    ck("shared-blindspot: n_necessary_legs 0 (no leg catches it)", aud2["n_necessary_legs"] == 0)
    ck("shared-blindspot: coverage C ~0 (the truth)", aud2["ensemble_coverage_by_fake"]["C"] < 0.15)
    ck("shared-blindspot: residual C ~ all", aud2["residual_blindspot_by_fake"]["C"] >= 34)

    # MISLABELED provenance: a leg labeled 'provenance' that SHARES the consistency blind spot (also misses fC)
    aud3 = coverage_audit({"geometric": g2, "prov": r2}, {"C": fC}, clean_mask=clean,
                          kinds={"geometric": "consistency", "prov": "provenance"})
    ck("mislabeled prov flagged NOT decorrelated", not aud3["provenance_label_check"]["prov"]["actually_decorrelated"])

    print("\nSELFTEST: %d/%d passed" % (ok, tot)); return ok == tot


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
