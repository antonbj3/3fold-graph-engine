#!/usr/bin/env python3
"""E10: balanced lens sets on the MEASURED answers of e7 (320 labelled sentences, 4 phrasings × 2 option orders).
A set of 4 lenses that is balanced on option order (two of each, from graph_engine.lens_pooling.balanced_lenses)
against 4 lenses with one option order, against all 8. Pooled = sign of the summed log-odds. No labels are used."""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.lens_pooling import balanced_lenses, logit
P = np.load(Path(__file__).parent / "e7_probabilities.npy").astype(float); L = logit(P)
truth = np.tile(np.repeat([True, False], 8), 20); lens = [(q, o) for q in range(4) for o in (0, 1)]
acc = lambda cols: float(((L[:, cols].sum(1) > 0) == truth).mean())
bias = lambda cols: float(L[:, cols].sum(1).mean() / len(cols))          # mean log-odds toward "increases"; truth is 50/50 so 0 = unbiased
design = balanced_lenses({"option_order": 2, "phrasing_a": 2, "phrasing_b": 2})
print("orthogonal array rows:", design[:4])
sets = {"one_order(o=0)": [k for k, (q, o) in enumerate(lens) if o == 0], "one_order(o=1)": [k for k, (q, o) in enumerate(lens) if o == 1],
        "all_8": list(range(8))}
bal = [c for c in combinations(range(8), 4) if sum(lens[k][1] for k in c) == 2 and len({lens[k][0] for k in c}) == 4]
unb = [c for c in combinations(range(8), 4) if sum(lens[k][1] for k in c) in (0, 4)]
out = {k: {"accuracy": round(acc(v), 3), "mean_log_odds_bias": round(bias(v), 3)} for k, v in sets.items()}
out["balanced_4_of_8 (all %d such sets)" % len(bal)] = {"accuracy": round(float(np.mean([acc(list(c)) for c in bal])), 3), "abs_bias": round(float(np.mean([abs(bias(list(c))) for c in bal])), 3)}
out["single_order_4 (the %d such sets)" % len(unb)] = {"accuracy": round(float(np.mean([acc(list(c)) for c in unb])), 3), "abs_bias": round(float(np.mean([abs(bias(list(c))) for c in unb])), 3)}
out["single_lens_mean"] = {"accuracy": round(float(np.mean([acc([k]) for k in range(8)])), 3), "abs_bias": round(float(np.mean([abs(bias([k])) for k in range(8)])), 3)}
json.dump(out, open(Path(__file__).parent / "e10_results.json", "w"), indent=1); [print(k, v) for k, v in out.items()]
