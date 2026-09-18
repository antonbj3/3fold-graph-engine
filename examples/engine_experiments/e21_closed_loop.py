#!/usr/bin/env python3
"""E21: the loop closed against a world with known sign structure (closed_loop.py). 40 worlds × 12 pairs, 8 sources of
which 35 % copy another source, claims with reliability 0.6–0.95, instruments: cheap judge (cost 1, r 0.8) and exact cell
(cost 4, r 0.99). Score = measure of the domain wrongly signed, summed over pairs, against the TRUTH, as the budget is
spent. Policies: engine (expected potential drop per cost, lineage-weighted claims), copies (same, copies counted as
independent), random. Also: are the two-transition pairs flagged by the collision family, and how often is a
one-transition pair falsely flagged."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.closed_loop import World, run

N, BUDGET = 40, 40
curves = {p: [] for p in ["random", "copies", "engine"]}; flags = {p: [0, 0, 0, 0] for p in curves}   # tp, fn, fp, tn
for s in range(N):
    w = World(seed=s)
    for pol in curves:
        r = run(w, pol, BUDGET, seed=s)
        curves[pol].append(np.interp(np.arange(0, BUDGET + 1, 5), r["cost"], r["wrong"]))
        two = set(r["collision_true"]); fl = set(r["collision_flagged"])
        flags[pol][0] += len(two & fl); flags[pol][1] += len(two - fl); flags[pol][2] += len(fl - two); flags[pol][3] += w.n_pairs - len(two | fl)
    print(s, {p: round(float(curves[p][-1][-1]), 2) for p in curves}, flush=True)
out = {"n_worlds": N, "budget": BUDGET, "cost_axis": list(range(0, BUDGET + 1, 5))}
for p in curves:
    C = np.array(curves[p]); out[p] = {"mean_wrong": np.round(C.mean(0), 3).tolist(), "se": np.round(C.std(0) / np.sqrt(N), 3).tolist(),
                                       "collision_tp_fn_fp_tn": flags[p]}
# paired comparison at the end of the budget
E, Cc, R = (np.array(curves[p])[:, -1] for p in ["engine", "copies", "random"])
out["paired_final"] = {"engine_minus_random": {"mean": round(float((E - R).mean()), 3), "se": round(float((E - R).std() / np.sqrt(N)), 3), "wins": int((E < R).sum())},
                       "engine_minus_copies": {"mean": round(float((E - Cc).mean()), 3), "se": round(float((E - Cc).std() / np.sqrt(N)), 3), "wins": int((E < Cc).sum())}}
json.dump(out, open(Path(__file__).parent / "e21_results.json", "w"), indent=1); print(json.dumps(out, indent=1))
