#!/usr/bin/env python3
"""E8b: throws with KNOWN inclusion probabilities. pi_i = (1-lam)*softmax(score_i/T) + lam/N, K pairs by systematic sampling with EXACT inclusion
probabilities (graph_engine.throws.inclusion_probabilities); fit weighted by 1/pi (clipped at 50× the mean weight).
Same data and protocol as e8. Question: can a colder (more harvesting) softmax keep the learning if the fit is
importance weighted, and does weighting help at T=1?"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.replay_policy import ReplayPolicy, build_episode, average_precision
from graph_engine.throws import inclusion_probabilities
from _data import snap_citations
K = 2000; out = {}
for g in ["HepTh", "HepPh"]:
    e, m, _ = snap_citations(g); eps_ = [build_episode(e, m, T, 24) for T in [35, 47, 59, 71]]; te = build_episode(e, m, 95, 24); res = {}
    for T in [1.0, 0.5, 0.25]:
        for weighted in (False, True):
            found, aps = [], []
            for seed in range(6):
                rng = np.random.default_rng(seed); X, y, W, hits, pol = [], [], [], 0, None
                for r, ep in enumerate(eps_):
                    n = len(ep.y)
                    if pol is None: pi = np.full(n, 1.0 / n)
                    else:
                        s = pol.score(ep.X) / T; p = np.exp(s - s.max()); pi = 0.95 * p / p.sum() + 0.05 / n
                    incl_all = inclusion_probabilities(pi, K); order = rng.permutation(n); cum = np.cumsum(incl_all[order])
                    pick = order[np.minimum(np.searchsorted(cum, rng.random() + np.arange(K), side="right"), n - 1)]; incl = incl_all[pick]
                    X.append(ep.X[pick]); y.append(ep.y[pick]); W.append(1.0 / incl); hits += int(ep.y[pick].sum()) if r > 0 else 0
                    yy = np.concatenate(y); ww = np.concatenate(W); ww = np.minimum(ww, 50 * ww.mean()) if weighted else None
                    if yy.sum() >= 5: pol = ReplayPolicy().fit(np.concatenate(X), yy, ww)
                found.append(hits); aps.append(average_precision(te.y, pol.score(te.X)))
            res[f"T={T} {'IPW' if weighted else 'unweighted'}"] = {"links_found": round(float(np.mean(found)), 1), "final_AP": [round(float(np.mean(aps)), 4), round(float(np.std(aps)), 4)]}
    out[g] = res; print(g); [print("  ", k, v) for k, v in res.items()]
json.dump(out, open(Path(__file__).parent / "e8b_results.json", "w"), indent=1)
