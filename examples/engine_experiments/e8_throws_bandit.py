#!/usr/bin/env python3
"""E8: throws when labels exist ONLY for the pairs that were thrown (live operation), not for the whole history.
Real data (SNAP). Four rounds at cutoffs 35, 47, 59, 71: each round the arm picks 2 000 of 400 000 candidate
pairs, sees their outcome, refits on everything it has seen. Round 0 is random for every arm.
Arms: greedy top-k · top-k with a random share eps · sampling ∝ exp(score/T) · random.
Reported: real links found during the four rounds, and AP of the final policy on cutoff 95 (unseen)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.replay_policy import ReplayPolicy, build_episode, average_precision
from _data import snap_citations
K = 2000; out = {}
for g in ["HepTh", "HepPh"]:
    e, m, _ = snap_citations(g); eps_ = [build_episode(e, m, T, 24) for T in [35, 47, 59, 71]]; te = build_episode(e, m, 95, 24)
    arms = {"greedy": ("eps", 0.0), "eps=0.1": ("eps", 0.1), "eps=0.3": ("eps", 0.3), "softmax T=1": ("soft", 1.0), "random": ("eps", 1.0)}
    res = {}
    for name, (kind, par) in arms.items():
        found, aps = [], []
        for seed in range(6):
            rng = np.random.default_rng(seed); X, y, hits, pol = [], [], 0, None
            for r, ep in enumerate(eps_):
                n = len(ep.y)
                if pol is None: pick = rng.choice(n, K, replace=False)
                elif kind == "soft":
                    s = pol.score(ep.X) / par; p = np.exp(s - s.max()); pick = rng.choice(n, K, replace=False, p=p / p.sum())
                else:
                    nr = int(par * K); top = np.argsort(-pol.score(ep.X))[: K - nr]
                    rest = np.setdiff1d(np.arange(n), top); pick = np.r_[top, rng.choice(rest, nr, replace=False)] if nr else top
                X.append(ep.X[pick]); y.append(ep.y[pick]); hits += int(ep.y[pick].sum()) if r > 0 else 0
                yy = np.concatenate(y)
                if yy.sum() >= 5 and (~yy).sum() >= 5: pol = ReplayPolicy().fit(np.concatenate(X), yy)
            found.append(hits); aps.append(average_precision(te.y, pol.score(te.X)))
        res[name] = {"links_found_rounds_1_to_3": [round(float(np.mean(found)), 1), round(float(np.std(found)), 1)],
                     "final_policy_AP_at_95": [round(float(np.mean(aps)), 4), round(float(np.std(aps)), 4)]}
    res["full_history_labels_AP(reference)"] = round(average_precision(te.y, ReplayPolicy().fit(np.concatenate([x.X for x in eps_]), np.concatenate([x.y for x in eps_])).score(te.X)), 4)
    out[g] = res; print(g); [print("  ", k, v) for k, v in res.items()]
json.dump(out, open(Path(__file__).parent / "e8_results.json", "w"), indent=1)
