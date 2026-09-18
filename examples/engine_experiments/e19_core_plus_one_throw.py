#!/usr/bin/env python3
"""E19: Uzzi et al. (Science 2013) in resistance geometry. For every paper published in (T, T+24], take the resistance
distances between all pairs of its references (graph ≤ T, sketch k = 64). core = median pair distance (small = a conventional
core), throw = 90th percentile (large = one far combination). Impact = percentile of the paper's later citations within its
publication month. Hypothesis: impact is highest for small core AND large throw, not for large distances overall (e3b)."""
import json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.resistance_sketch import ResistanceSketch
from _data import snap_citations, largest_component, simple_undirected
out = {}
for g in ["HepTh", "HepPh"]:
    e, m, _ = snap_citations(g); n = len(m); cites = np.bincount(e[:, 1], minlength=n).astype(float); pct = np.zeros(n)
    for mo in np.unique(m):
        idx = np.flatnonzero(m == mo); pct[idx] = rankdata(cites[idx]) / len(idx)
    for T in [71, 95]:
        past = e[m[e[:, 0]] <= T]; und = simple_undirected(past); nl, el, keep = largest_component(n, und); el = simple_undirected(el)
        remap = -np.ones(n, np.int64); remap[np.flatnonzero(keep)] = np.arange(nl)
        Z = ResistanceSketch.build(nl, el, k=64, seed=0).Z
        rows = []
        for p in np.flatnonzero((m > T) & (m <= T + 24)):
            refs = remap[e[e[:, 0] == p, 1]]; refs = refs[(refs >= 0)]
            if len(refs) < 4: continue
            refs = refs[:30]; D = ((Z[refs][:, None] - Z[refs][None]) ** 2).sum(2); d = D[np.triu_indices(len(refs), 1)]
            rows.append((np.median(d), np.quantile(d, 0.9), pct[p], len(refs)))
        R = np.array(rows); core, throw, imp = np.log(R[:, 0]), np.log(R[:, 1]), R[:, 2]
        # partial correlations: throw given core, core given throw (rank-based)
        rz = lambda a, b: (rankdata(a) - np.polyval(np.polyfit(rankdata(b), rankdata(a), 1), rankdata(b)))
        lo_core, hi_throw = core <= np.median(core), throw >= np.median(throw)
        out[f"{g}@{T}"] = {"n_papers": len(R), "spearman_core_impact": round(float(spearmanr(core, imp)[0]), 3),
                           "spearman_throw_impact": round(float(spearmanr(throw, imp)[0]), 3),
                           "partial_throw_given_core": round(float(np.corrcoef(rz(throw, core), rz(imp, core))[0, 1]), 3),
                           "partial_core_given_throw": round(float(np.corrcoef(rz(core, throw), rz(imp, throw))[0, 1]), 3),
                           "mean_impact": {"tight_core_far_throw": round(float(imp[lo_core & hi_throw].mean()), 3), "tight_core_near_throw": round(float(imp[lo_core & ~hi_throw].mean()), 3),
                                           "loose_core_far_throw": round(float(imp[~lo_core & hi_throw].mean()), 3), "loose_core_near_throw": round(float(imp[~lo_core & ~hi_throw].mean()), 3)}}
        print(g, T, out[f"{g}@{T}"], flush=True)
json.dump(out, open(Path(__file__).parent / "e19_results.json", "w"), indent=1)
