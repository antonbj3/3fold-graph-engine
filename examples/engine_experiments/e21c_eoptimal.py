#!/usr/bin/env python3
"""E21c: E-optimal (max-min) probe placement against the entropy rule, same 40 worlds as e21, budget 40.

Diagnosis behind it: `best_probe` and `model_check_probe` maximize an expected ENTROPY drop — a D-optimal criterion,
an average over the hypothesis space, which a probe can buy along directions that are already well separated. The
second transition of a collision pair is the opposite: a direction the schedule never separates at all. E-optimality
(raise the SMALLEST pairwise discrimination) is the criterion that cannot ignore it; `weakest_direction_probe` sums,
over the probes made so far, the Jeffreys divergence between the two-outcome emissions of each pair of hypotheses that
carries posterior mass and buys the probe that lifts the minimum of that matrix most.

Compared: the best configuration measured so far (engine+guard2+replay+majority: wrong 1.08, 7 of 25 collisions), the
same with the guard share spent by the E-rule (+eopt), the mixed rule on EVERY probe with no guard share
(+eoptmix, value = expected entropy drop + λ·min-discrimination increase, λ fixed on the first step), and the pure
E-rule without replay and without the majority reading."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.closed_loop import World, run, DEFAULT_PRIOR

N, B, GUARD = 40, 40, 0.2
BASE = "engine+guard2+replay+majority"
POLICIES = [BASE, "engine+eopt+replay+majority", "engine+eoptmix+replay+majority", "engine+eopt"]

fin = {p: [] for p in POLICIES}
gap = {p: [] for p in POLICIES}
flag = {p: [0, 0, 0] for p in POLICIES}                     # tp, fn, fp
state = {p: dict(DEFAULT_PRIOR) for p in POLICIES if "replay" in p}
hist = {p: {k: [v] for k, v in DEFAULT_PRIOR.items()} for p in state}
out_path = Path(__file__).parent / "e21c_results.json"

for s in range(N):
    w = World(seed=s)
    for pol in POLICIES:
        r = run(w, pol, B, seed=s, guard=GUARD, replay_prior=state.get(pol))
        if pol in state:
            for k in DEFAULT_PRIOR:
                hist[pol][k].append(r["world_stats"][k]); state[pol][k] = float(np.mean(hist[pol][k]))
        fin[pol].append(r["final_wrong"]); gap[pol].append(r["gap"])
        two, fl = set(r["collision_true"]), set(r["collision_flagged"])
        flag[pol][0] += len(two & fl); flag[pol][1] += len(two - fl); flag[pol][2] += len(fl - two)
    print(s, {p: round(fin[p][-1], 2) for p in POLICIES}, flush=True)

    out = {"n_worlds": s + 1, "budget": B, "guard": GUARD, "baseline": BASE}
    Eb = np.array(fin[BASE])
    for p in POLICIES:
        F = np.array(fin[p]); g = np.array(gap[p])
        out[p] = {"wrong40": round(float(F.mean()), 3), "se": round(float(F.std() / np.sqrt(len(F))), 3),
                  "gap_mean": round(float(g.mean()), 3), "gap_se": round(float(g.std() / np.sqrt(len(g))), 3),
                  "collision_tp_fn_fp": flag[p],
                  "vs_baseline": {"mean": round(float((F - Eb).mean()), 3),
                                  "se": round(float((F - Eb).std() / np.sqrt(len(F))), 3),
                                  "wins": int((F < Eb).sum()), "ties": int((F == Eb).sum())}}
    out["learned_priors"] = {p: {k: round(v, 3) for k, v in state[p].items()} for p in state}
    json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps({p: {"wrong40": out[p]["wrong40"], "gap": out[p]["gap_mean"], "coll": out[p]["collision_tp_fn_fp"],
                      "vs_base": out[p]["vs_baseline"]} for p in POLICIES}, indent=1))
