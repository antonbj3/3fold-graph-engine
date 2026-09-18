#!/usr/bin/env python3
"""E21b: the collision hole as an ALLOCATION problem (e28: 12 probes on one pair find the second transition in 31 of 40; the loop
gives each pair 2–3). +burst commits a run of `burst_n` model-check probes to the pair the guard picks, each placed after the previous
answer. Compared with the best configuration so far (+guard2+replay+majority), same 40 worlds, budget 40."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.closed_loop import World, run

DEFAULT_PRIOR = {"p_flip": 0.3, "p_two": 0.05, "reliability": 0.75}
N, B = 40, 40; res = {}
for pol, bn in [("engine+guard2+replay+majority", 6), ("engine+burst+replay+majority", 6), ("engine+burst+replay+majority", 4), ("engine+burst+majority", 6)]:
    fin, gap, tp, fn, fp = [], [], 0, 0, 0
    state = dict(DEFAULT_PRIOR); hist = {k: [v] for k, v in DEFAULT_PRIOR.items()}
    for s in range(N):
        w = World(seed=s); r = run(w, pol, B, seed=s, guard=0.2, burst_n=bn, replay_prior=state if "replay" in pol else None)
        if "replay" in pol:
            for k in DEFAULT_PRIOR:
                hist[k].append(r["world_stats"][k]); state[k] = float(np.mean(hist[k]))
        fin.append(r["final_wrong"]); gap.append(r["final_wrong"] - r["believed_wrong"])
        two, fl = set(r["collision_true"]), set(r["collision_flagged"]); tp += len(two & fl); fn += len(two - fl); fp += len(fl - two)
    key = f"{pol} (burst_n={bn})" if "burst" in pol else pol
    res[key] = {"wrong": round(float(np.mean(fin)), 3), "se": round(float(np.std(fin) / np.sqrt(N)), 3), "gap": round(float(np.mean(gap)), 3), "tp_fn_fp": [tp, fn, fp]}
    print(key, res[key], flush=True)
json.dump(res, open(Path(__file__).parent / "e21b_results.json", "w"), indent=1)
