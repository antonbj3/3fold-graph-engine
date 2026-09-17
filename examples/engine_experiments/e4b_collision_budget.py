#!/usr/bin/env python3
"""E4b: how many probes on ONE pair before the collision family can be declared?
TEST pairs on [0,1]: single transition vs two transitions, 3 noisy claims each (err 0.15), then b greedy probes."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.regime_posterior import RegimePosterior
rng = np.random.default_rng(0); rows = []
def sign(kind, th, x): return (1 if x < th[0] else -1) if kind == 1 else (1 if (x < th[0] or x >= th[1]) else -1)
for b in [0, 2, 4, 8, 16]:
    res = {1: [], 2: []}
    for trial in range(300):
        kind = 1 + trial % 2; a = rng.uniform(0.2, 0.45); th = (a, a + rng.uniform(0.25, 0.4))
        rp = RegimePosterior(0, 1)
        for _ in range(3):
            w = rng.uniform(0.08, 0.2); lo = rng.uniform(0, 1 - w); s = sign(kind, th, lo + w / 2)
            rp.add_claim(lo, lo + w, -s if rng.random() < 0.15 else s)
        for _ in range(b):
            x, _ = rp.best_probe(); s = sign(kind, th, x); rp.add_probe(x, -s if rng.random() < 0.05 else s)
        res[kind].append(rp.collision()["p_two_transitions"])
    one, two = np.array(res[1]), np.array(res[2]); flag1, flag2 = (one >= .5).mean(), (two >= .5).mean()
    rows.append(dict(probes_per_pair=b, mean_p_two_if_one=round(float(one.mean()), 3), mean_p_two_if_two=round(float(two.mean()), 3),
                     false_flag_rate=round(float(flag1), 3), recall=round(float(flag2), 3))); print(rows[-1], flush=True)
json.dump(rows, open(Path(__file__).parent / "e4b_results.json", "w"), indent=1)
