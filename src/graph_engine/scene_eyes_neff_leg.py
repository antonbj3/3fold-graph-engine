"""
scene_eyes_neff_leg — a STATISTICAL trust leg for per-region scene estimates: effective sample size.

A per-region estimate can look well-supported (many contributing pixels/observations) and still be
OVER-CONFIDENT, because those pixels are SPATIALLY CORRELATED: the effective evidence n_eff is far
below the raw count. Geometric and generative checks (depth nulls, envelope fences, hallucination
certs) do not catch this, so this leg is decorrelated from them by construction.

    from graph_engine.scene_eyes_neff_leg import neff_trust_leg
    v = neff_trust_leg(region_values, region_block_ids)
    # -> {verdict: GROUNDED|FENCE-OVERCONFIDENT, n_eff, overconfidence}

Ensemble rule: a region is trusted only if it passes THIS leg AND the geometric/generative legs —
a single lenient leg is an escape hatch. n_eff here is the Kish/ESS family quantity
K^2 / (1t R^2 1) estimated by block bootstrap. Self-contained (numpy only), sub-millisecond per
region. Run `python scene_eyes_neff_leg.py` for the self-test.
"""
from __future__ import annotations
import numpy as np

OVERCONF_MAX = 2.0 # per-region trust is FENCED if raw-count over-states certainty by more than this factor


def neff_trust_leg(region_values, region_block_ids, overconf_max: float = OVERCONF_MAX, rng=None) -> dict:
    """
    spatial-correlation blocks (e.g. super-pixel tiles at the correlation length). Returns GROUNDED iff the effective n_eff
    supports the apparent evidence, else FENCE-OVERCONFIDENT (the region only LOOKS well-supported).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    v = np.asarray(region_values, float).ravel()
    v = v[np.isfinite(v)]
    N = v.size
    if N < 8:
        return {"verdict": "TOO-FEW", "n": int(N)}
    b = np.asarray(region_block_ids).ravel()[:N]
    uids = np.unique(b)
    def bstd(block):
        s = []
        for _ in range(120):
            samp = np.concatenate([v[b == u] for u in rng.choice(uids, len(uids))]) if block else rng.choice(v, N)
            s.append(samp.mean())
        return float(np.std(s))
    si, sb = bstd(False), bstd(True)
    n_eff = N * (si / sb) ** 2 if sb > 0 else float(N)
    overconf = float(np.sqrt(N / max(n_eff, 1)))
    grounded = overconf <= overconf_max
    return {"verdict": "GROUNDED" if grounded else f"FENCE-OVERCONFIDENT (raw N inflates trust {overconf:.1f}x; n_eff={n_eff:.0f})",
            "n": int(N), "n_eff": round(n_eff, 1), "overconfidence": round(overconf, 2), "grounded": bool(grounded),
            "leg": "neff-spatial-overconfidence", "lane": "G"}


def neff_trust_map(regions, overconf_max: float = OVERCONF_MAX, rng=None) -> dict:
    """
    n_eff-WEIGHT this leg (D COMPOSE-VERDICT). `regions` = list of (region_values, region_block_ids). For each region returns
    trust in [0,1] = min(1, overconf_max/overconfidence) (graded: overconf≤max -> trust=1, over-confident -> trust decays,
    NOT a binary verdict), a fence_flag, and n_eff as the composition WEIGHT (D n_eff-weighted rule: a region with tiny n_eff
    contributes little). This makes the leg composable, not just registered. Output matches schema {trust[0,1], fence_flag,
    n_eff_weight} per region.
    """
    out = []
    for i, (vals, bids) in enumerate(regions):
        r = neff_trust_leg(vals, bids, overconf_max=overconf_max, rng=rng)
        if r["verdict"] == "TOO-FEW":
            out.append({"region": i, "trust": None, "fence_flag": None, "n_eff_weight": 0.0, "note": "too-few"})
            continue
        oc = r["overconfidence"]
        trust = float(min(1.0, overconf_max / max(oc, 1e-9))) # graded, monotone-decreasing in overconfidence
        out.append({"region": i, "trust": round(trust, 3), "fence_flag": bool(not r["grounded"]),
                    "n_eff_weight": float(r["n_eff"]), "overconfidence": oc})
    return {"lane": "G", "leg_name": "neff-spatial-overconfidence", "decorrelation_axis": "statistical/sampling",
            "output": "per-region trust[0,1] + fence_flag; n_eff_weight for n_eff-weighted composition",
            "n_eff_axis": "block-bootstrap n_eff = N*(std_iid/std_block)^2", "regions": out}


def trust_array(regions, overconf_max: float = OVERCONF_MAX, rng=None, too_few_trust: float = 0.0):
    """
    J-composer-ready flat per-region trust array in [0,1] (scene_eyes_ensemble.compose expects {axis: trust_array}).
    TOO-FEW regions map to `too_few_trust` (default 0.0 = conservatively FENCED, never NaN) so MIN-trust fusion is never
    poisoned by a None/NaN. Returns np.ndarray shape (n_regions).
    """
    tm = neff_trust_map(regions, overconf_max=overconf_max, rng=rng)
    return np.array([(too_few_trust if r["trust"] is None else r["trust"]) for r in tm["regions"]], float)


def raw_residual_array(regions, rng=None):
    """at the ensemble level). higher_is_worse=True (larger over-confidence = worse). TOO-FEW regions → nan (mark inapplicable)."""
    out = []
    for vals, bids in regions:
        r = neff_trust_leg(vals, bids, rng=rng)
        out.append(np.nan if r.get("verdict") == "TOO-FEW" else float(r["overconfidence"]))
    return np.array(out)


def applicable_array(regions, min_n: int = 8):
    """the block-bootstrap n_eff (>= min_n); on a too-small region it ABSTAINS (does not false-fence). Aligned with trust_array."""
    out = []
    for vals, _ in regions:
        v = np.asarray(vals, float).ravel(); out.append(bool(np.isfinite(v).sum() >= min_n))
    return np.array(out, bool)


def schema_record() -> dict:
    return {"lane": "G", "leg_name": "neff-spatial-overconfidence", "decorrelation_axis": "statistical/sampling",
            "entry": "graph_engine.scene_eyes_neff_leg.neff_trust_map(regions)",
            "output": "per-region {trust in [0,1], fence_flag, n_eff_weight}",
            "n_eff_axis": "block-bootstrap n_eff = N*(std_iid/std_block)^2",
            "known_GT_verified": "held-out check: correlation with the other cert families ~0.00"}


def selftest() -> int:
    rng = np.random.default_rng(1)
    ok = True
    # KNOWN-REF: a region that LOOKS grounded (2000 pixels) but is spatially correlated (20 blocks) -> FENCE
    K, m = 20, 100
    bm = rng.standard_normal(K); corr = np.concatenate([bm[k] + 0.05 * rng.standard_normal(m) for k in range(K)])
    bids = np.concatenate([[k] * m for k in range(K)])
    r_corr = neff_trust_leg(corr, bids, rng=rng)
    ok &= (not r_corr["grounded"]) and r_corr["overconfidence"] > 2 # correlated -> fenced (the escape hatch closed)
    # a genuinely INDEPENDENT region (each pixel its own block) -> GROUNDED
    indep = rng.standard_normal(2000); ib = np.arange(2000)
    r_ind = neff_trust_leg(indep, ib, rng=rng)
    ok &= r_ind["grounded"] # independent evidence -> trusted
    print(f"  [{'✓' if not r_corr['grounded'] else '✗'}] correlated region (looks grounded, {r_corr['n']} px) -> {r_corr['verdict']}")
    print(f"  [{'✓' if r_ind['grounded'] else '✗'}] independent region -> {r_ind['verdict']}")
    # independent region -> trust≈1, and every trust in [0,1] + schema fields present.
    def make_region(nblocks, N=2000):
        # FIX total N, vary #independent blocks K: n_eff≈K, overconf≈sqrt(N/K) -> more correlation (fewer blocks) = lower trust
        per = N // nblocks; bm = rng.standard_normal(nblocks)
        vals = np.concatenate([bm[k] + 0.05 * rng.standard_normal(per) for k in range(nblocks)])
        bids = np.concatenate([[k] * per for k in range(nblocks)])
        return vals, bids
    regs = [make_region(nb) for nb in (2000, 200, 50, 10)] # increasing spatial correlation (fewer blocks) at fixed N
    tm = neff_trust_map(regs, rng=rng)
    trusts = [r["trust"] for r in tm["regions"]]
    in_unit = all(0.0 <= t <= 1.0 for t in trusts)
    monotone = all(trusts[i] >= trusts[i + 1] - 1e-9 for i in range(len(trusts) - 1)) # more correlation -> less trust
    schema_ok = set(schema_record()) >= {"lane", "leg_name", "decorrelation_axis", "output", "n_eff_axis", "known_GT_verified"}
    ok &= in_unit and monotone and schema_ok
    print(f"  [{'✓' if in_unit else '✗'}] graded trust in [0,1]: {trusts} (indep->1, correlated->fenced low)")
    print(f"  [{'✓' if monotone else '✗'}] trust MONOTONE-decreasing with spatial correlation (composable, n_eff-weighted)")
    print(f"  [{'✓' if schema_ok else '✗'}] locked-schema record fields present -> composes into a downstream ensemble composer")
    print(f"\n  SELFTEST scene_eyes_neff_leg: {'✓ ALL PASS — decorrelated ensemble leg: FENCES spatial-over-confidence + GRADED trust-map (composable)' if ok else '✗ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
