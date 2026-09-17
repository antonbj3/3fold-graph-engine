#!/usr/bin/env python3
"""E3b: among pairs that DO get co-cited after the cutoff, does the resistance distance of the pair at
the cutoff say anything about the later citation count of the paper that first brought them together?
(Hypothesis from Uzzi et al., Science 342, 468 (2013): atypical combinations → higher impact.)
Impact is the percentile of the co-citing paper's citations among papers of the same month, which
removes the age bias of a truncated dataset."""
import json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.replay_policy import FEATURES, build_episode
from _data import snap_citations
out = {}
for g in ["HepTh", "HepPh"]:
    e, m, _ = snap_citations(g); n = len(m)
    cites = np.bincount(e[:, 1], minlength=n).astype(float)
    pct = np.zeros(n)
    for mo in np.unique(m):
        idx = np.flatnonzero(m == mo); pct[idx] = rankdata(cites[idx]) / len(idx)
    for T in [71, 83, 95]:
        ep = build_episode(e, m, T, 24)
        fut = (m[e[:, 0]] > T) & (m[e[:, 0]] <= T + 24)
        F = sp.csc_matrix((np.ones(fut.sum()), (e[fut, 0], e[fut, 1])), shape=(n, n))
        pos = np.flatnonzero(ep.y)[:20000]
        imp = np.empty(len(pos)); first = np.empty(len(pos))
        for q, k in enumerate(pos):
            i, j = ep.pairs[k]; co = np.intersect1d(F[:, i].indices, F[:, j].indices)
            f = co[np.argmin(m[co])]; imp[q] = pct[f]
        r = {}
        for name in ["resistance", "adamic_adar", "pref", "vel"]:
            x = ep.X[pos, FEATURES.index(name)]; rho, p = spearmanr(x, imp); r[name] = [round(float(rho), 4), float(f"{p:.2g}")]
        # partial: resistance vs impact after removing what pref explains (rank residuals)
        R = lambda v: rankdata(v); res = R(ep.X[pos, FEATURES.index("resistance")]); pr = R(ep.X[pos, FEATURES.index("pref")]); im = R(imp)
        resid = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
        r["resistance|pref (partial)"] = round(float(np.corrcoef(resid(res, pr), resid(im, pr))[0, 1]), 4)
        hi = ep.X[pos, FEATURES.index("resistance")] >= np.quantile(ep.X[pos, FEATURES.index("resistance")], 0.9)
        r["mean impact pct: top-decile resistance vs rest"] = [round(float(imp[hi].mean()), 4), round(float(imp[~hi].mean()), 4)]
        out[f"{g}@{T}"] = r; print(g, T, r, flush=True)
json.dump(out, open(Path(__file__).parent / "e3b_results.json", "w"), indent=1)
