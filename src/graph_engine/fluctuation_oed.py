#!/usr/bin/env python3
"""FLUCTUATION-OED — the vibrating multimodal OED sensor-bundle detector, as a callable ANY lane runs on ANY dataset.

The principle (σ_min-via-fluctuation, FDT: fluctuation = kT·F⁻¹): instead of computing a Fisher/model, READ the
FLUCTUATION (the sensor covariance) per CONDITION and test whether the condition shifts the SOFT MODE (the top
fluctuation eigenvalue = 1/σ_min direction). A condition that SOFTENS the system (damage, a regime change, a hidden
factor) grows the fluctuation in the soft mode — detectable model-free, with a PERMUTATION-NULL so it is not a smell
but a p-value (the lesson: never conclude a signal absent/present without the null — that is a premature negative).

VALIDATED on real z24 bridge modal data: per-damage-label top-fluctuation-eigenvalue spread 8.33 vs permutation-null
1.49±0.37, p=0.0000 (5.6σ) — the damage genuinely shifts the soft mode.

  fluctuation_oed(features[n_samples, n_sensors], groups[n_samples]) → {p_value, real_spread, null_mean, per_group_topeig}
  python -u -m graph_engine.selftest_fluctuation_oed
"""
import numpy as np


def _finite_eigvalsh(cov, ctx):
    # eigh/eigvalsh-on-NaN is structure/LAPACK-path-dependent (cross-build-confirmed): a dense
    # matrix typically raises, a block/sparse-structured one can silently return UNSORTED-but-FINITE eigenvalues.
    # Never rely on the accidental raise; guard isfinite explicitly at the source.
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{ctx}: non-finite covariance/correlation matrix (NaN/Inf in features)")
    return np.linalg.eigvalsh(cov)


def _finite_eigh(cov, ctx):
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{ctx}: non-finite covariance/correlation matrix (NaN/Inf in features)")
    return np.linalg.eigh(cov)


def _top_eig_per_group(feat, groups, labs, min_n):
    vals = []
    for l in labs:
        F = feat[groups == l]
        if len(F) < min_n:
            return None
        cov = np.cov(F - F.mean(0), rowvar=False)
        if not np.all(np.isfinite(cov)):
            # eigh/eigvalsh-on-NaN is structure/LAPACK-path-dependent (never trust it to raise).
            raise ValueError(f"_top_eig_per_group: non-finite covariance for group {l!r} (NaN/Inf in features)")
        w = np.linalg.eigvalsh(cov)
        vals.append(float(w[-1]))                              # top fluctuation eigenvalue (the soft mode's variance)
    return np.array(vals)


def fluctuation_oed(features, groups, n_perm: int = 300, seed: int = 0, zscore: bool = True, statistic: str = "spread") -> dict:
    """Does a CONDITION (group label) shift the soft mode (top fluctuation eigenvalue) of the sensor covariance BEYOND
    chance? Model-free active detector for any [n_samples, n_sensors] dataset + per-sample group labels.
    statistic='spread' (std of per-group top-eig — condition changes the soft mode) or 'range'. Returns a p-value vs a
    label-permutation null. p<0.05 ⇒ the fluctuation structure carries the condition (the soft mode is condition-dependent)."""
    feat = np.asarray(features, float)
    g = np.asarray(groups)
    if zscore:
        feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    labs = np.unique(g)
    min_n = feat.shape[1] + 1                                  # need > n_sensors samples for a full-rank covariance
    stat = (lambda v: float(np.std(v))) if statistic == "spread" else (lambda v: float(v.max() - v.min()))
    real_vec = _top_eig_per_group(feat, g, labs, min_n)
    if real_vec is None:
        return {"p_value": float("nan"), "error": f"a group has < {min_n} samples (need > n_sensors)"}
    real = stat(real_vec)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        v = _top_eig_per_group(feat, rng.permutation(g), labs, min_n)
        if v is not None:
            null.append(stat(v))
    null = np.array(null)
    p = float((null >= real).mean()) if null.size else float("nan")
    return {"p_value": round(p, 5), "real_spread": round(real, 4),
            "null_mean": round(float(null.mean()), 4), "null_std": round(float(null.std()), 4),
            "n_sigma": round(float((real - null.mean()) / (null.std() + 1e-12)), 2),
            "per_group_topeig": [round(x, 3) for x in real_vec], "n_groups": len(labs),
            "condition_shifts_soft_mode": bool(p < 0.05)}


def location_oed(features, groups, n_perm: int = 300, seed: int = 0, zscore: bool = True) -> dict:
    """The SECOND KIND of liftable factor (cross-worktree catch): a MEAN-SHIFT / LOCATION factor —
    the condition shifts the group CENTROID along a fixed decorrelated axis WITHOUT rotating the covariance ellipsoid.
    fluctuation_oed (a ROTATION test) is BLIND to this (within-group covariance unchanged → p≈0.5). Statistic =
    between-group sum-of-squares of centroids vs a label-permutation null. Both kinds lift the pool's shared gauge."""
    feat = np.asarray(features, float); g = np.asarray(groups)
    if zscore:
        feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    labs = np.unique(g)
    def bss(assign):
        return float(sum(np.sum(assign == l) * (feat[assign == l].mean(0) ** 2).sum() for l in labs))
    real = bss(g)
    rng = np.random.default_rng(seed)
    null = np.array([bss(rng.permutation(g)) for _ in range(n_perm)])
    p = float((null >= real).mean())
    return {"location_p": round(p, 5), "n_sigma": round(float((real - null.mean()) / (null.std() + 1e-12)), 2),
            "location_factor": bool(p < 0.05)}


def liftable_factor(features, groups, n_perm: int = 300, seed: int = 0) -> dict:
    """Two-KIND liftable-factor test (cross-worktree-corrected): a condition lifts the pool's shared gauge either by ROTATING
    the covariance (fluctuation_oed) OR by SHIFTING the mean/location (location_oed). Reports both + the kind — a factor
    the fluctuation (rotation) test alone would MISS if it is pure location."""
    rot = fluctuation_oed(features, groups, n_perm=n_perm, seed=seed)
    loc = location_oed(features, groups, n_perm=n_perm, seed=seed)
    rp, lp = rot.get("p_value"), loc.get("location_p")
    kind = ("rotation+location" if (rp is not None and rp < 0.05 and lp < 0.05)
            else "rotation" if (rp is not None and rp < 0.05)
            else "location" if lp < 0.05 else "none")
    return {"rotation_p": rp, "location_p": lp, "kind": kind, "carries_liftable_factor": kind != "none"}


def split_half_reliability(features, seed: int = 0) -> dict:
    """Is a dataset's eff_rank REAL σ_min-diversity or NOISE-inflated? (
    applied to routing.) eff_rank = (Σλ)²/Σλ² counts noise directions as diversity too — a high-dim noisy dataset scores a
    spuriously high eff_rank. Gate by SPLIT-HALF CROSS-VALIDATED VARIANCE CAPTURE (subspace-angle overlap FAILS the noise
    control — two (p-1)-dim subspaces of a p-dim space trivially overlap, so isotropic noise scores ~1). The right test:
    do half-A's top-r eigen-directions capture MORE of half-B's variance than the r/p isotropic baseline? Real signal →
    A's directions are privileged (capture ≫ r/p); pure noise → no privileged directions (capture ≈ r/p → reliability≈0).
    Returns {eff_rank_full, captured_B, baseline, reliability∈[0,1], reliable_eff_rank = eff_rank_full × reliability}."""
    feat = np.asarray(features, float)
    fz = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    n, p = fz.shape
    def _eff_rank_ev(ev):
        ev = ev[ev > 1e-12]
        return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.size else 0.0
    er_full = _eff_rank_ev(np.sort(np.abs(_finite_eigvalsh(np.cov(fz, rowvar=False), "split_half_reliability")))[::-1])
    if n < 2 * (p + 2):                                       # need each half > n_features for a full-rank covariance
        return {"eff_rank_full": round(er_full, 2), "captured_B": None, "baseline": None,
                "reliability": None, "reliable_eff_rank": None,
                "note": "too few samples per half for split-half (n < 2*(p+2))"}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); A, B = fz[idx[:n // 2]], fz[idx[n // 2:]]
    r = max(1, min(int(round(er_full)), p - 1))
    CA, CB = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    wA, VA = _finite_eigh(CA, "split_half_reliability")
    VA = VA[:, np.argsort(wA)[::-1][:r]]                      # half-A's top-r directions (learned on A only)
    captured = float(np.trace(VA.T @ CB @ VA) / (np.trace(CB) + 1e-12))  # fraction of B's variance they capture
    baseline = r / p                                         # what r random/isotropic directions capture
    reliability = float(np.clip((captured - baseline) / (1 - baseline + 1e-12), 0, 1))  # privilege beyond isotropic
    return {"eff_rank_full": round(er_full, 2), "captured_B": round(captured, 3), "baseline": round(baseline, 3),
            "reliability": round(reliability, 3), "reliable_eff_rank": round(er_full * reliability, 2)}


def mp_gated_eff_rank(features) -> dict:
    """Marchenko–Pastur gate on eff_rank (cross-worktree: an agent worktree's an earlier batch, an INDEPENDENT noise-gate to my split_half_reliability).
    For an n×p standardized matrix of pure noise, the correlation-matrix eigenvalues fall below the MP edge
    λ₊ = (1+√γ)², γ = p/n. Only eigenvalues ABOVE λ₊ are SIGNAL; MP-gated eff_rank = participation ratio of the signal
    eigenvalues. A theory-based (per-sample, no split) noise gate — complements split-half (empirical, resampling-based).
    Two independent gates agreeing on which anchors are noise-inflated = over-determination ."""
    feat = np.asarray(features, float)
    fz = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    n, p = fz.shape
    gamma = p / n
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    ev = np.sort(np.abs(_finite_eigvalsh(np.corrcoef(fz, rowvar=False), "mp_gated_eff_rank")))[::-1]
    sig = ev[ev > lam_plus]                                   # eigenvalues above the MP noise edge = signal
    mp_er = float(sig.sum() ** 2 / (sig ** 2).sum()) if sig.size else 0.0
    return {"mp_edge": round(float(lam_plus), 3), "n_signal_modes": int(sig.size),
            "mp_gated_eff_rank": round(mp_er, 2), "gamma": round(float(gamma), 4)}


def anchor_catalog(name: str, features, groups=None, n_perm: int = 150, reliability: bool = False) -> dict:
    """The SHARED anchor-catalog record every lane runs on its OWN datasets → the agent pool assembles the full dataset→
    routing pairs of agents TOGETHER. A dataset is an external
    anchor that lifts the pool's shared-prior gauge ([[agent pool-gauge-ceiling, ...]]); its VALUE as an anchor = its
    fluctuation-signature (how much σ_min-diversity it carries) + whether it holds a LIFTABLE FACTOR (a condition that
    shifts the soft mode, p<0.05). Returns the standard record; report it to the ledger so routing = max σ_min-lift."""
    feat = np.asarray(features, float)
    fz = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    C = np.cov(fz, rowvar=False); ev = np.sort(np.abs(_finite_eigvalsh(C, "anchor_catalog")))[::-1]; ev = ev[ev > 1e-9]
    rec = {"dataset": name, "n_sensors": int(feat.shape[1]), "n_samples": int(feat.shape[0]),
           "eff_rank": round(float(ev.sum() ** 2 / (ev ** 2).sum()), 2),
           "cond": round(float(ev[0] / ev[-1]), 1), "decades": round(float(np.log10(ev[0] / ev[-1])), 2)}
    if reliability:                                          # TWO independent noise-gates on eff_rank (agreement = over-determination)
        rel = split_half_reliability(feat)                   # empirical (split-half variance-capture)
        rec["reliability"] = rel["reliability"]; rec["reliable_eff_rank"] = rel["reliable_eff_rank"]
        mp = mp_gated_eff_rank(feat)                          # theory (Marchenko–Pastur edge, an agent worktree an earlier batch)
        rec["mp_gated_eff_rank"] = mp["mp_gated_eff_rank"]; rec["n_signal_modes"] = mp["n_signal_modes"]
    if groups is not None:
        r = liftable_factor(feat, groups, n_perm=n_perm)                # TWO-KIND: rotation OR location (cross-worktree-corrected)
        rec["rotation_p"] = r["rotation_p"]; rec["location_p"] = r["location_p"]
        rec["liftable_kind"] = r["kind"]; rec["carries_liftable_factor"] = r["carries_liftable_factor"]
    return rec


# --------------------------- selftest ---------------------------
def _selftest() -> bool:
    """Contract selftest with planted GT: a planted soft-mode (rotation) factor MUST be detected, a null MUST NOT
    (negative case / no false positive), a PURE-LOCATION factor exposes the rotation test's documented blind spot
    (location_oed catches it, liftable_factor names the kind), split-half + MP gates must separate noise from signal,
    and the too-few-samples path must fail gracefully. Complements selftest_fluctuation_oed (which adds real z24)."""
    rng = np.random.default_rng(0)
    passed = 0; total = 0

    def check(name, cond):
        nonlocal passed, total
        total += 1; passed += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1) NULL (negative), CALIBRATION-based: a single null draw can land in the 5% tail by construction, so
    # assert the false-positive RATE over 10 independent null draws (P(>2 of 10 at alpha=.05) ~ 1.2%).
    n_fp = 0
    for k in range(10):
        rk = np.random.default_rng(100 + k)
        r0 = fluctuation_oed(rk.normal(size=(400, 8)), rk.integers(0, 4, 400), n_perm=100, seed=k)
        n_fp += int(r0["p_value"] < 0.05)
    check(f"null calibration: {n_fp}/10 draws significant (expect ~0.5, allow <=2)", n_fp <= 2)

    # 2) ROTATION factor: one group's soft-mode variance inflated -> detected
    F1 = rng.normal(size=(600, 8)); g1 = rng.integers(0, 4, 600)
    F1[g1 == 2, 0] += 4.0 * rng.normal(size=int((g1 == 2).sum()))
    r1 = fluctuation_oed(F1, g1, n_perm=200)
    check(f"planted soft-mode shift detected (p={r1['p_value']})", r1["p_value"] < 0.05 and r1["condition_shifts_soft_mode"])

    # 3) PURE-LOCATION factor: mean shift, covariance unchanged ->
    # the rotation test is BLIND, location_oed catches it, liftable_factor names the kind
    F2 = rng.normal(size=(600, 8)); g2 = rng.integers(0, 4, 600)
    F2[g2 == 1] += 1.5                                     # centroid shift only
    lf = liftable_factor(F2, g2, n_perm=200)
    check(f"pure location factor: rotation test blind (p={lf['rotation_p']}), location catches (p={lf['location_p']})",
          lf["rotation_p"] > 0.05 and lf["location_p"] < 0.05 and lf["kind"] == "location" and lf["carries_liftable_factor"])

    # 4) split-half reliability: pure isotropic noise -> ~0; strong low-rank signal -> high
    noise = rng.normal(size=(400, 8))
    sig = rng.normal(size=(400, 2)) @ rng.normal(size=(2, 8)) * 3.0 + 0.3 * rng.normal(size=(400, 8))
    rel_noise = split_half_reliability(noise); rel_sig = split_half_reliability(sig)
    check(f"split-half: noise reliability {rel_noise['reliability']} < 0.3 < signal {rel_sig['reliability']}",
          rel_noise["reliability"] < 0.3 and rel_sig["reliability"] > 0.5)

    # 5) MP gate agrees (independent noise-gate): noise -> 0 signal modes; rank-2 signal -> >=1
    mp_noise = mp_gated_eff_rank(noise); mp_sig = mp_gated_eff_rank(sig)
    check(f"MP gate: noise n_signal_modes={mp_noise['n_signal_modes']}, rank-2 signal={mp_sig['n_signal_modes']}",
          mp_noise["n_signal_modes"] == 0 and mp_sig["n_signal_modes"] >= 1)

    # 6) graceful guard (negative input): a group with < n_sensors+1 samples -> NaN p + error, no crash
    rg = fluctuation_oed(rng.normal(size=(20, 10)), rng.integers(0, 5, 20), n_perm=50)
    check("too-few-samples guard: NaN p_value + error key, no crash",
          rg["p_value"] != rg["p_value"] and "error" in rg)

    # 7) anchor_catalog composes the record (the agent pool-facing schema)
    rec = anchor_catalog("selftest-anchor", F1, groups=g1, n_perm=100, reliability=True)
    check("anchor_catalog record carries eff_rank + liftable kind + both noise-gates",
          all(k in rec for k in ("eff_rank", "reliable_eff_rank", "mp_gated_eff_rank", "liftable_kind"))
          and rec["carries_liftable_factor"])

    print(f"fluctuation_oed selftest: {passed}/{total}")
    return passed == total


if __name__ == "__main__":
    import sys
    import argparse
    _ap = argparse.ArgumentParser(description="fluctuation_oed selftest runner")
    _ap.add_argument("--selftest", action="store_true", help="run the selftest (also the default action)")
    _ap.parse_args()
    sys.exit(0 if _selftest() else 1)
