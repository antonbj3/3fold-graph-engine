#!/usr/bin/env python3
"""E21g: a sequential test with a stated false-alarm rate in place of the fixed guard share and the fixed family weight.

e33: under a correct model the potential is a martingale and the price of a probe is its conditional expectation, so
realized − predicted drop is a martingale difference. `alarm.Alarm(alpha)` bets on those residuals with the model's own
predictive law of the drop (a mixture of exp(λd̃ − κ(λ)) legs): a test martingale, so P(it ever reaches 1/alpha) ≤ alpha
by Ville — exactly, for adaptively chosen probes, with no exchangeability assumption. `+alarm` puts one per PAIR in the
loop: while a pair's alarm is latched the family weight λ in total_value_probe / bundle_value is raised (alarm_lam),
and it drops back when the wealth returns below 1. Nothing is reserved in advance — that is what it replaces.

Same 40 worlds, budget 40, against the two configurations this package has measured: +guard2+replay+majority
(1.08 / 7 tp / gap −0.08, the fixed 20 % guard share) and +bundle+replay+majority (e21f: 1.172 / 7 tp / gap +0.088,
exactly priced pairs, over-confident)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.closed_loop import World, run                      # noqa: E402

DEFAULT_PRIOR = {"p_flip": 0.3, "p_two": 0.05, "reliability": 0.75}
N, B, ALPHA = 40, 40, 0.05
POLICIES = ["engine+guard2+replay+majority", "engine+bundle+replay+majority",
            "engine+alarm+replay+majority", "engine+alarm+bundle+replay+majority"]

res, base = {}, None
for pol in POLICIES:
    fin, gap, hits, latched, tp, fn, fp = [], [], 0, 0, 0, 0, 0
    state = dict(DEFAULT_PRIOR); hist = {k: [v] for k, v in DEFAULT_PRIOR.items()}
    for s in range(N):
        w = World(seed=s)
        r = run(w, pol, B, seed=s, guard=0.2, alarm_alpha=ALPHA,
                replay_prior=state if "replay" in pol else None)
        if "replay" in pol:
            for k in DEFAULT_PRIOR:
                hist[k].append(r["world_stats"][k]); state[k] = float(np.mean(hist[k]))
        fin.append(r["final_wrong"]); gap.append(r["final_wrong"] - r["believed_wrong"])
        if r["alarm"]:
            hits += r["alarm_hits"]; latched += sum(a["n_alarms"] for a in r["alarm"])
        two, fl = set(r["collision_true"]), set(r["collision_flagged"])
        tp += len(two & fl); fn += len(two - fl); fp += len(fl - two)
    fin = np.array(fin); base = fin if base is None else base
    res[pol] = {"wrong": round(float(fin.mean()), 3), "se": round(float(fin.std() / np.sqrt(N)), 3),
                "gap": round(float(np.mean(gap)), 3), "tp_fn_fp": [tp, fn, fp],
                "paired_vs_baseline": round(float((fin - base).mean()), 3),
                "paired_se": round(float((fin - base).std() / np.sqrt(N)), 3),
                "wins": int((fin < base).sum()),
                "alarm_pairs_hit": hits, "alarm_latches": latched, "pairs_total": N * 12}
    print(pol, res[pol], flush=True)
res["_meta"] = {"n_worlds": N, "budget": B, "alpha": ALPHA, "pairs_per_world": 12,
                "guarantee": "P(alarm ever fires | correct model) <= alpha, Ville on a test martingale; power not guaranteed"}
json.dump(res, open(Path(__file__).parent / "e21g_results.json", "w"), indent=1)
