#!/usr/bin/env python3
"""E33: the two guarantees of the probe rule — the greedy factor, and why the value rule never buys the
collision probe (graph_engine/guarantees.py).

(1) GREEDY. Is the expected drop of U = ∫ H(P₊(x)) dx on the fixed partition adaptive submodular in the
probe set? Reported: (a) an exact counterexample (a sub-realization after which one probe is worth
strictly MORE), (b) the violation rate of the diminishing-returns inequality Δ(x|A) ≥ Δ(x|B), A ⊆ B, on
random sub-realizations, for the entropy potential and for the min(p,1−p) potential, (c) greedy vs the
OPTIMAL adaptive policy (full enumeration of decision trees over the answers) on 50 random small cases
per potential — the ratio the Golovin–Krause 1−1/e bound would cover if the objective qualified.

(2) ALLOCATION LEMMA. With one single-transition hypothesis carrying mass ≥ 1−δ: the proved upper bound
f(δ) on the expected potential drop of ANY probe, the proved lower bound g(q,r,s) on the family-entropy
drop (`model_check_probe`) at the right cell, both checked against the realized values on 200 random
one-transition-explained configurations, and the ratio max Δ_T / max Δ_U binned by δ → 0. This is the
theorem behind e21's measured hole (the value rule flagged 0 of 25 two-transition pairs)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.guarantees import (adaptive_submodularity_check, allocation_report, bsc_gain,
                                     entropy_drop_bound, family_drop_floor, family_drops,
                                     greedy_vs_optimal, one_step_drops, state_of, update)
from graph_engine.regime_posterior import RegimePosterior

R = 0.95
out = {}


def small_case(seed, n_grid=None):
    rng = np.random.default_rng(seed)
    rp = RegimePosterior(0.0, 1.0, reliability=0.75,
                         n_grid=int(rng.integers(4, 7)) if n_grid is None else n_grid,
                         p_flip=0.3, p_two=0.05)
    for _ in range(int(rng.integers(2, 4))):
        a = float(rng.uniform(0, 0.8))
        b = float(min(1.0, a + rng.uniform(0.1, 0.4)))
        rp.add_claim(a, b, int(rng.choice([-1, 1])), n_eff=float(rng.uniform(0.5, 3.0)),
                     reliability=float(rng.uniform(0.6, 0.95)))
    return rp


# ── 0. the state arithmetic is the module's own ────────────────────────────────────────────────
rp = small_case(0)
st = state_of(rp)
drops = one_step_drops(st, R)
x_best, v_best = rp.best_probe(R)
out["sanity"] = {
    "one_step_drops_max": float(drops.max()), "best_probe_value": float(v_best),
    "match_abs": abs(float(drops.max()) - float(v_best)),
    "family_drops_max": float(family_drops(st, R).max()),
    "model_check_value": float(rp.model_check_probe(R)[1]),
}

# ── 1a. adaptive monotonicity, and an exact counterexample to adaptive submodularity ───────────
mono_min = math.inf
for s in range(60):
    p = small_case(100 + s)
    stt = state_of(p)
    for pot in ("entropy", "error"):
        d = one_step_drops(stt.with_potential(pot), R)
        mono_min = min(mono_min, float(d.min()))
out["adaptive_monotone_min_drop"] = mono_min

# the mechanism: the one-step gain as a function of the marginal
out["bsc_gain_curve"] = {f"{p:.2f}": bsc_gain(p, R) for p in (0.02, 0.1, 0.25, 0.4, 0.5, 0.6, 0.9)}

# the minimal explicit counterexample quoted in guarantees.py's docstring: 4 cells, one claim
def minimal_case(pot):
    p = RegimePosterior(0.0, 1.0, reliability=0.9, n_grid=4, p_flip=0.3, p_two=0.0, potential=pot)
    p.add_claim(0.0, 0.5, +1, n_eff=3.0)
    return p


mc = {}
for pot, hist_cell, x in (("entropy", 1, 2), ("error", 1, 2), ("error", 3, 1)):
    s0 = state_of(minimal_case(pot))
    d0 = one_step_drops(s0, R)
    d1 = one_step_drops(update(s0, hist_cell, -1, R), R)
    mc[f"{pot}_hist{hist_cell}_x{x}"] = {
        "p_plus_before": [float(v) for v in (s0.post @ s0.F)],
        "delta_empty": float(d0[x]), "delta_after_minus": float(d1[x]),
        "ratio": float(d1[x] / d0[x]) if d0[x] > 1e-15 else None,
        "ratio_infinite": bool(d0[x] <= 1e-15)}
out["counterexample_minimal"] = mc

# search for the largest explicit counterexample over random small cases: A = ∅, B = one answer
ce = {}
for pot in ("entropy", "error"):
    best = None
    for s in range(400):
        p = small_case(2000 + s, n_grid=4)
        s0 = state_of(p, pot)
        d0 = one_step_drops(s0, R)
        for c1 in range(s0.n_cells):
            for sg in (1, -1):
                s1 = update(s0, c1, sg, R)
                d1 = one_step_drops(s1, R)
                for x in range(s0.n_cells):
                    if x == c1:
                        continue
                    if d1[x] > d0[x] + 1e-9:
                        ratio = d1[x] / d0[x] if d0[x] > 1e-12 else 1e18
                        cand = {"seed": 2000 + s, "n_cells": int(s0.n_cells), "history_cell": c1,
                                "history_sign": sg, "probe_cell": x,
                                "delta_empty": float(d0[x]), "delta_after": float(d1[x]),
                                "ratio": float(ratio), "excess": float(d1[x] - d0[x])}
                        if best is None or cand["excess"] > best["excess"]:
                            best = cand
        if best is not None and s > 30:
            break
    ce[pot] = best
out["counterexample"] = ce

# ── 1b. violation rate of the diminishing-returns inequality on random sub-realizations ────────
rp = small_case(7, n_grid=6)
probes = list(state_of(rp).midpoints())
out["submodularity_check"] = adaptive_submodularity_check(rp, probes, n_random=200, reliability=R, seed=1)
rp48 = RegimePosterior(0.0, 1.0, reliability=0.75, n_grid=48, p_flip=0.3, p_two=0.05)
rp48.add_claim(0.1, 0.3, +1, n_eff=2.0)
rp48.add_claim(0.6, 0.8, -1, n_eff=2.0)
out["submodularity_check_n48"] = adaptive_submodularity_check(
    rp48, list(np.linspace(0.05, 0.95, 12)), n_random=200, reliability=R, seed=2)

# ── 1c. greedy vs the optimal adaptive policy, 50 random small cases ───────────────────────────
gv = {}
for pot in ("entropy", "error"):
    rows = []
    for s in range(50):
        p = small_case(500 + s)
        budget = 2 + (s % 2)
        r_ = greedy_vs_optimal(p, budget, reliability=R, potential=pot)
        rows.append(r_)
    ratios = np.array([r_["ratio"] for r_ in rows])
    opts = np.array([r_["optimal_value"] for r_ in rows])
    gv[pot] = {
        "n_cases": len(rows), "budgets": sorted({r_["budget"] for r_ in rows}),
        "mean_ratio": float(ratios.mean()), "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()), "n_below_1_minus_1_over_e": int((ratios < 1 - 1 / math.e).sum()),
        "n_strictly_below_1": int((ratios < 1 - 1e-9).sum()),
        "mean_optimal_value_bits": float(opts.mean()),
        "worst_case": {k: rows[int(ratios.argmin())][k] for k in ("ratio", "optimal_value", "greedy_value", "budget", "n_cells")},
    }
out["greedy_vs_optimal"] = gv

# ── 2. the allocation lemma ────────────────────────────────────────────────────────────────────
NG = 12


def one_transition_config(seed, strength, gap=None):
    """The e21 geometry: two SAME-SIGN claims with a closure-filled span between them. The
    single-transition family explains them with one hypothesis ("all σ") whose mass → 1 as the claim
    weight grows (δ → 0), while a two-transition hypothesis whose excursion sits strictly inside the
    unclaimed span has EXACTLY the same likelihood — no claim touches it — so its mass stays at its
    prior share. That is the configuration in which the value rule is supposed to go blind."""
    rng = np.random.default_rng(seed)
    g = int(rng.integers(1, 5)) if gap is None else gap
    k = int(rng.integers(2, NG - g - 1))
    rp = RegimePosterior(0.0, 1.0, reliability=0.95, n_grid=NG, p_flip=0.3, p_two=0.05)
    sgn = int(rng.choice([-1, 1]))
    rp.add_claim(0.0, k / NG, sgn, n_eff=strength * float(rng.uniform(0.9, 1.1)))
    rp.add_claim((k + g) / NG, 1.0, sgn, n_eff=strength * float(rng.uniform(0.9, 1.1)))
    return rp, g


rows = []
for s in range(200):
    strength = float(np.random.default_rng(9000 + s).choice([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]))
    p_, g_ = one_transition_config(9000 + s, strength)
    rep = allocation_report(p_, R)
    rep["strength"] = strength
    rep["gap_cells"] = g_
    rows.append(rep)

# a clean δ-sweep at a fixed geometry (gap = 2 cells), the table the docstring quotes
sweep = []
for n in (0.5, 1, 2, 4, 8, 16, 32, 64):
    p_, _ = one_transition_config(3, n, gap=2)
    r_ = allocation_report(p_, R)
    r_["n_eff"] = n
    sweep.append(r_)
out["allocation_sweep_gap2"] = sweep

d = np.array([r["delta"] for r in rows])
mu = np.array([r["max_entropy_drop"] for r in rows])
mt = np.array([r["max_family_drop"] for r in rows])
fb = np.array([r["f_bound"] for r in rows])
gb = np.array([r["g_bound"] for r in rows])
ok_f = mu <= fb + 1e-12
ok_g = mt >= gb - 1e-12
out["allocation"] = {
    "n": len(rows),
    "f_bound_holds_all": bool(ok_f.all()), "f_tightness_max": float((mu / np.maximum(fb, 1e-300)).max()),
    "f_tightness_median": float(np.median(mu / np.maximum(fb, 1e-300))),
    "g_bound_holds_all": bool(ok_g.all()),
    "g_slack_min": float((mt / np.maximum(gb, 1e-300)).min()),
    "g_slack_median": float(np.median(mt / np.maximum(gb, 1e-300))),
    "f_coefficient_r095_width1": float(entropy_drop_bound(0.5, 1.0, R) / (0.5 * 0.5)),
    "g_coefficient_r095_s1": float(family_drop_floor(0.5, R, 1.0) / (0.5 * 0.5)),
    "argmax_cells_differ_share": float(np.mean([r["argmax_entropy_cell"] != r["argmax_family_cell"] for r in rows])),
}

bins = [(0.0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 1.0)]
tab = []
for lo, hi in bins:
    m = (d >= lo) & (d < hi)
    if m.sum() == 0:
        continue
    tab.append({"delta_lo": lo, "delta_hi": hi, "n": int(m.sum()),
                "delta_median": float(np.median(d[m])),
                "max_entropy_drop_median": float(np.median(mu[m])),
                "max_family_drop_median": float(np.median(mt[m])),
                "ratio_T_over_U_median": float(np.median(mt[m] / np.maximum(mu[m], 1e-300))),
                "f_bound_median": float(np.median(fb[m])), "g_bound_median": float(np.median(gb[m]))})
out["allocation_by_delta"] = tab

# does the value rule ever buy the family's cell first?
out["value_rule_buys_family_cell_share"] = float(np.mean(
    [r["argmax_entropy_cell"] == r["argmax_family_cell"] for r in rows]))

p = Path(__file__).with_name("e33_results.json")
p.write_text(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps(out, indent=1, sort_keys=True))
