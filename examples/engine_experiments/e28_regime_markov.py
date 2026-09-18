#!/usr/bin/env python3
"""E28: the sign along a condition variable as a TELEGRAPH PROCESS (regime_markov) against the hypothesis
enumeration that reads it today (regime_posterior), on closed_loop's World.

Three measurements, all on worlds whose claims are generated as MAJORITY reports of their box (World.majority_sign),
which is what both readers are configured to read (claim_model="majority"):

  A  agreement, 40 one-transition worlds: same claims, 10 random probes of reliability 0.9 fed to both readers;
     |Δ expected_error| and the mean over x of |Δ p_plus|. The two are different models — RegimePosterior enumerates
     {0, 1, 2} transitions with prior masses and softens the box majority with a logistic (k = 20); the chain keeps
     every switch count with Poisson weights and takes the majority hard — so what is measured is how far apart the
     two readings of the SAME evidence are.
  B  two transitions, 40 two-transition worlds: 12 probes placed by the chain's own best_probe, both readers fed the
     same answers; the share where P(N ≥ 2) > 0.5 (chain) against RegimePosterior.collision() (mass of the separate
     two-transition family). For the chain this is a moment of one process, not a family with prior p_two.
  C  wall time per evaluation (expected_error + p_plus, and one best_probe over all cells) for both, at two block
     enumeration caps, plus the one-off block enumeration cost. RegimePosterior carries O(cells²) hypotheses once
     p_two > 0; the chain is O(cells) per evaluation plus the majority-claim block enumeration.

The chain's one modelling liberty is conditioning on at most `max_switches_in_block` switches inside each claim
block; tail_mass() is the prior mass that conditioning removes and is reported here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.claim_federation import lineage_information          # noqa: E402
from graph_engine.closed_loop import World                             # noqa: E402
from graph_engine.regime_markov import RegimeMarkov                    # noqa: E402
from graph_engine.regime_posterior import RegimePosterior              # noqa: E402

N_WORLDS, N_PROBES_B = 40, 12
K_BLOCK = 2


def claims_of(world, pair=0):
    g = {}
    for p, a, b, sg, roots in world.claims:
        if p == pair:
            g.setdefault((round(a, 6), round(b, 6), sg), []).append(roots)
    return [(a, b, sg, lineage_information(rs)) for (a, b, sg), rs in g.items()]


def readers(world, k_block=K_BLOCK, **kw):
    rp = RegimePosterior(0.0, 1.0, claim_model="majority", **kw)
    rm = RegimeMarkov(0.0, 1.0, max_switches_in_block=k_block, **kw)
    for a, b, sg, n in claims_of(world):
        rp.add_claim(a, b, sg, n_eff=n)
        rm.add_claim(a, b, sg, n_eff=n)
    return rp, rm


# -- A: agreement on 0/1-transition worlds ---------------------------------------------------------
de, dp, tails = [], [], []
xs = np.linspace(0.01, 0.99, 50)
for s in range(N_WORLDS):
    w = World(n_pairs=1, p_two=0.0, p_transition=0.6, seed=s)
    rp, rm = readers(w, p_flip=0.6, p_two=0.0)
    rp.p_two = 1e-9
    rng = np.random.default_rng(1000 + s)
    for x in rng.random(10):
        ans = w.probe(0, float(x), 0.9, rng)
        rp.add_probe(float(x), ans, 0.9)
        rm.add_probe(float(x), ans, 0.9)
    de.append(abs(rp.expected_error() - rm.expected_error()))
    dp.append(float(np.mean([abs(rp.p_plus(x) - rm.p_plus(x)) for x in xs])))
    tails.append(rm.tail_mass())
de, dp = np.array(de), np.array(dp)

# -- B: two-transition detection -------------------------------------------------------------------
pm, pr, wrong_m, wrong_r = [], [], [], []
for s in range(N_WORLDS):
    w = World(n_pairs=1, p_two=1.0, p_transition=0.0, seed=s)
    rp, rm = readers(w)
    rng = np.random.default_rng(2000 + s)
    for _ in range(N_PROBES_B):
        x, _g = rm.best_probe(0.95)
        ans = w.probe(0, x, 0.95, rng)
        rm.add_probe(x, ans, 0.95)
        rp.add_probe(x, ans, 0.95)
    pm.append(rm.collision()["p_ge_two"])
    pr.append(rp.collision()["p_two_transitions"])
    grid = (np.arange(200) + 0.5) / 200
    truth = np.array([w.sign(0, x) for x in grid])
    wrong_m.append(float((np.where(np.array([rm.p_plus(x) for x in grid]) >= 0.5, 1, -1) != truth).mean()))
    wrong_r.append(float((np.where(np.array([rp.p_plus(x) for x in grid]) >= 0.5, 1, -1) != truth).mean()))
pm, pr = np.array(pm), np.array(pr)

# -- C: wall time ------------------------------------------------------------------------------------
timing = {}
w = World(n_pairs=1, p_two=0.0, p_transition=0.6, seed=7)
for name, k in (("markov_K2", 2), ("markov_K3", 3)):
    rp, rm = readers(w, k_block=k)
    t0 = time.perf_counter(); rm._build_units(); rm.expected_error(); build = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        rm._basecache = None
        rm.expected_error(); rm.p_plus(0.4)
    ev = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter(); rm.best_probe(0.95); bp = time.perf_counter() - t0
    timing[name] = {"build_s": round(build, 4), "eval_s": round(ev, 5), "best_probe_s": round(bp, 4),
                    "patterns": int(sum(len(u["logw"]) for u in rm._build_units() if u["kind"] == "block")),
                    "tail_mass": round(rm.tail_mass(), 6)}
rp, _ = readers(w)
t0 = time.perf_counter(); rp._cache = None; rp.expected_error(); build_rp = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(20):
    rp._cache = None
    rp.expected_error(); rp.p_plus(0.4)
ev_rp = (time.perf_counter() - t0) / 20
t0 = time.perf_counter(); rp.best_probe(0.95); bp_rp = time.perf_counter() - t0
timing["regime_posterior"] = {"build_s": round(build_rp, 4), "eval_s": round(ev_rp, 5), "best_probe_s": round(bp_rp, 4),
                              "hypotheses": int(len(rp._solve()[3]))}

out = {
    "setup": {"worlds": N_WORLDS, "probes_A": 10, "probe_reliability_A": 0.9, "probes_B": N_PROBES_B,
              "probe_reliability_B": 0.95, "n_grid": 48, "max_switches_in_block": K_BLOCK,
              "lam_A": 0.6, "lam_default": 0.4, "claim_model": "majority"},
    "A_agreement_vs_regime_posterior": {
        "expected_error_max_abs_diff": round(float(de.max()), 4),
        "expected_error_mean_abs_diff": round(float(de.mean()), 4),
        "p_plus_mean_abs_diff_per_world_max": round(float(dp.max()), 4),
        "p_plus_mean_abs_diff_per_world_mean": round(float(dp.mean()), 4),
        "tail_mass_max": round(float(np.max(tails)), 6)},
    "B_two_transition_detection": {
        "n_pairs": int(len(pm)),
        "markov_flagged_p_ge_two_gt_0.5": int((pm > 0.5).sum()),
        "markov_mean_p_ge_two": round(float(pm.mean()), 4),
        "regime_posterior_flagged": int((pr > 0.5).sum()),
        "regime_posterior_mean_p_two": round(float(pr.mean()), 4),
        "wrong_measure_markov": round(float(np.mean(wrong_m)), 4),
        "wrong_measure_regime_posterior": round(float(np.mean(wrong_r)), 4)},
    "C_timing": timing,
}
print(json.dumps(out, indent=2))
Path(__file__).with_name("e28_results.json").write_text(json.dumps(out, indent=2) + "\n")
