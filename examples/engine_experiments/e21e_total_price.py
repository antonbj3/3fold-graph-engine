#!/usr/bin/env python3
"""E21e: the correctly priced probe. e33 proved the sign-potential rule prices a collision probe ~10× too low once the one-transition
family explains the claims (the e21 hole is the price, not the location). +total values every probe by ΔU + ΔH_family in the same
bits (regime_posterior.total_value_probe), no guard share. Same 40 worlds, budget 40, against the best configuration so far."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.closed_loop import World, run
DEFAULT_PRIOR = {"p_flip": 0.3, "p_two": 0.05, "reliability": 0.75}
N, B = 40, 40; res = {}; base = None
for pol in ["engine+guard2+replay+majority", "engine+total+replay+majority", "engine+total+majority", "engine+total"]:
    fin, gap, tp, fn, fp = [], [], 0, 0, 0
    state = dict(DEFAULT_PRIOR); hist = {k: [v] for k, v in DEFAULT_PRIOR.items()}
    for s in range(N):
        w = World(seed=s); r = run(w, pol, B, seed=s, guard=0.2, replay_prior=state if "replay" in pol else None)
        if "replay" in pol:
            for k in DEFAULT_PRIOR:
                hist[k].append(r["world_stats"][k]); state[k] = float(np.mean(hist[k]))
        fin.append(r["final_wrong"]); gap.append(r["final_wrong"] - r["believed_wrong"])
        two, fl = set(r["collision_true"]), set(r["collision_flagged"]); tp += len(two & fl); fn += len(two - fl); fp += len(fl - two)
    fin = np.array(fin); base = fin if base is None else base
    res[pol] = {"wrong": round(float(fin.mean()), 3), "se": round(float(fin.std() / np.sqrt(N)), 3), "gap": round(float(np.mean(gap)), 3),
                "tp_fn_fp": [tp, fn, fp], "paired_vs_baseline": round(float((fin - base).mean()), 3), "wins": int((fin < base).sum())}
    print(pol, res[pol], flush=True)
json.dump(res, open(Path(__file__).parent / "e21e_results.json", "w"), indent=1)
