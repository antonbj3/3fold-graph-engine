#!/usr/bin/env python3
"""E6: throws as a screening cascade. Stage 1 = replay policy (graph only, REAL scores on SNAP data, cost 0).
Stage 2 = a cheap typed judge (SIMULATED: score = d'·truth + N(0,1), cost 1). Stage 3 = expensive verification
(cost 200, taken as correct). Budget 200 000 units = 1 000 expensive verifications.
Reported: real links confirmed within the budget. Law checked: with pass fraction q at stage 2, cost per verified
candidate is 1/q + 200, so the saving over verifying everything is capped by 1/q and by the judge's precision
at that q — not by the 200x price ratio."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.replay_policy import ReplayPolicy, build_episode
from _data import snap_citations
C2, C3, BUDGET = 1, 200, 200_000
out = {}
for g in ["HepTh", "HepPh"]:
    e, m, _ = snap_citations(g)
    tr = [build_episode(e, m, T, 24) for T in [35, 47, 59, 71]]; te = build_episode(e, m, 95, 24)
    pol = ReplayPolicy().fit(np.concatenate([x.X for x in tr]), np.concatenate([x.y for x in tr]))
    s1, y = pol.score(te.X), te.y; n = len(y); rng = np.random.default_rng(0)
    row = {"candidates": n, "base_rate": round(float(y.mean()), 4)}
    k = BUDGET // C3
    row["verify_random"] = int(y[rng.permutation(n)[:k]].sum())
    o1 = np.argsort(-s1); row["stage1_then_verify"] = int(y[o1[:k]].sum())
    for d in [1.0, 2.0, 3.0]:
        s2 = d * y + rng.standard_normal(n); best = (0, None)
        for n2 in [2000, 5000, 10000, 20000, 50000, 100000, n]:          # how many of stage-1's top go to the judge
            idx = o1[:n2]; left = BUDGET - n2 * C2
            if left <= 0: continue
            # combine: stage-1 log-odds + judge log-likelihood-ratio (d'·s − d'²/2), verify the top
            comb = s1[idx] + d * s2[idx] - d * d / 2; top = idx[np.argsort(-comb)[: left // C3]]
            if y[top].sum() > best[0]: best = (int(y[top].sum()), n2)
        row[f"stage1_judge(d'={d})_verify"] = {"confirmed": best[0], "judged": best[1]}
        jo = np.argsort(-s2); left = BUDGET - n * C2
        row[f"judge_only(d'={d})_verify"] = int(y[jo[: max(left, 0) // C3]].sum()) if left > 0 else 0
    out[g] = row; print(g, row, flush=True)
json.dump(out, open(Path(__file__).parent / "e6_results.json", "w"), indent=1)
