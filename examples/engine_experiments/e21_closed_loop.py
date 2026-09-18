#!/usr/bin/env python3
"""E21: the loop closed against a world with known sign structure (closed_loop.py). 40 worlds × 12 pairs, 8 sources of
which 35 % copy another source, claims with reliability 0.6–0.95, instruments: cheap judge (cost 1, r 0.8) and exact cell
(cost 4, r 0.99). Score = measure of the domain wrongly signed, summed over pairs, against the TRUTH, as the budget is
spent. Policies: engine (expected potential drop per cost, lineage-weighted claims), copies (same, copies counted as
independent), random. Also: are the two-transition pairs flagged by the collision family, and how often is a
one-transition pair falsely flagged.

Additions, each a variant of the same run so every policy sees the same worlds and the same random stream:
  engine+guard        a share (0.2) of the budget spent on model_check_probe instead of best_probe: one probe at a time
  engine+guard2       the same share spent on model_check_pair: the two probes that jointly expose a second transition,
                      bought and executed together (the 21 of 25 collisions +guard misses are pairs where one answer
                      leaves the single-transition family explaining everything)
  engine+replay       the priors of world k come from the realized outcomes of worlds 1..k−1 (empirical Bayes)
  engine+guard2+replay, engine+guard2+replay+majority (majority = the box claims read as majority reports, claim_model)
Reported per policy: wrong measure at cost 0 / 20 / 40, believed wrong at 40, the calibration gap (actual − believed),
collision tp/fn/fp, and the paired difference against `engine` at the end of the budget."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.closed_loop import World, run, DEFAULT_PRIOR

N, BUDGET = 40, 40
POLICIES = {"random": {}, "copies": {}, "engine": {},
            "engine+guard": {"guard": 0.2}, "engine+guard2": {"guard": 0.2}, "engine+replay": {},
            "engine+guard2+replay": {"guard": 0.2}, "engine+guard2+replay+majority": {"guard": 0.2}}
REPLAY = [p for p in POLICIES if "replay" in p or p.endswith("all")]

curves = {p: [] for p in POLICIES}
believed = {p: [] for p in POLICIES}
gaps = {p: [] for p in POLICIES}
flags = {p: [0, 0, 0, 0] for p in POLICIES}          # tp, fn, fp, tn
guard_spent = {p: [] for p in POLICIES}
state = {p: dict(DEFAULT_PRIOR) for p in REPLAY}      # running empirical-Bayes prior, one per replay policy
hist = {p: {k: [v] for k, v in DEFAULT_PRIOR.items()} for p in REPLAY}   # the defaults as one pseudo-world
prior_trace = {p: [] for p in REPLAY}
out_path = Path(__file__).parent / "e21_results.json"

for s in range(N):
    w = World(seed=s)
    for pol, kw in POLICIES.items():
        name = pol
        if pol in REPLAY:
            prior_trace[pol].append(dict(state[pol]))
        r = run(w, name, BUDGET, seed=s, replay_prior=state.get(pol), **kw)
        curves[pol].append(np.interp(np.arange(0, BUDGET + 1, 5), r["cost"], r["wrong"]))
        believed[pol].append(r["believed_wrong"]); gaps[pol].append(r["gap"]); guard_spent[pol].append(r["spent_guard"])
        two = set(r["collision_true"]); fl = set(r["collision_flagged"])
        flags[pol][0] += len(two & fl); flags[pol][1] += len(two - fl); flags[pol][2] += len(fl - two); flags[pol][3] += w.n_pairs - len(two | fl)
        if pol in REPLAY:                                          # empirical Bayes: running mean of realized outcomes
            for k in DEFAULT_PRIOR:
                hist[pol][k].append(r["world_stats"][k])
                state[pol][k] = float(np.mean(hist[pol][k]))
    print(s, {p: round(float(curves[p][-1][-1]), 2) for p in POLICIES}, flush=True)

    out = {"n_worlds": s + 1, "budget": BUDGET, "cost_axis": list(range(0, BUDGET + 1, 5))}
    E = np.array(curves["engine"])[:, -1]
    for p in POLICIES:
        C = np.array(curves[p]); g = np.array(gaps[p]); F = C[:, -1]
        out[p] = {"mean_wrong": np.round(C.mean(0), 3).tolist(), "se": np.round(C.std(0) / np.sqrt(len(C)), 3).tolist(),
                  "collision_tp_fn_fp_tn": flags[p], "believed_wrong": round(float(np.mean(believed[p])), 3),
                  "gap_mean": round(float(g.mean()), 3), "gap_se": round(float(g.std() / np.sqrt(len(g))), 3),
                  "abs_gap_mean": round(float(np.abs(g).mean()), 3),
                  "guard_spent": round(float(np.mean(guard_spent[p])), 2),
                  "vs_engine": {"mean": round(float((F - E).mean()), 3), "se": round(float((F - E).std() / np.sqrt(len(F))), 3),
                                "wins": int((F < E).sum()), "ties": int((F == E).sum())}}
    R, Cc = (np.array(curves[p])[:, -1] for p in ["random", "copies"])
    out["paired_final"] = {"engine_minus_random": {"mean": round(float((E - R).mean()), 3), "se": round(float((E - R).std() / np.sqrt(len(E))), 3), "wins": int((E < R).sum())},
                           "engine_minus_copies": {"mean": round(float((E - Cc).mean()), 3), "se": round(float((E - Cc).std() / np.sqrt(len(E))), 3), "wins": int((E < Cc).sum())}}
    k = min(10, s + 1)
    out["replay_sequence"] = {p: {"first10_wrong": round(float(np.mean(np.array(curves[p])[:k, -1])), 3),
                                  "last10_wrong": round(float(np.mean(np.array(curves[p])[-k:, -1])), 3),
                                  "first10_gap": round(float(np.mean(gaps[p][:k])), 3),
                                  "last10_gap": round(float(np.mean(gaps[p][-k:])), 3),
                                  "prior_first": prior_trace[p][0], "prior_last": prior_trace[p][-1],
                                  "world_true": {"p_flip": 0.6 / 0.95, "p_two": 0.05}} for p in REPLAY}
    json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps({p: {"wrong40": out[p]["mean_wrong"][-1], "gap": out[p]["gap_mean"], "coll": out[p]["collision_tp_fn_fp_tn"][:3]} for p in POLICIES}, indent=1))
