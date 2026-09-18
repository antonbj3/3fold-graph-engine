#!/usr/bin/env python3
"""E19: Uzzi et al. (Science 2013) in resistance geometry. For every paper published in (T, T+24], take the resistance
distances between all pairs of its references (graph ≤ T, sketch k = 64). core = median pair distance (small = a conventional
core), throw = 90th percentile (large = one far combination). Impact = percentile of the paper's later citations within its
publication month. Hypothesis: impact is highest for small core AND large throw, not for large distances overall (e3b).
Control (added after review of the first result): resistance distance between two nodes is ≈ 1/deg_i + 1/deg_j on
well-connected graphs, so a "tight core" is largely "the references are hubs". The partial correlations are therefore also
reported given the references' mean log in-degree at T and the reference count. What survives that control is the finding."""
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
        past = e[m[e[:, 0]] <= T]; indeg_T = np.bincount(past[:, 1], minlength=n).astype(float); und = simple_undirected(past); nl, el, keep = largest_component(n, und); el = simple_undirected(el)
        remap = -np.ones(n, np.int64); remap[np.flatnonzero(keep)] = np.arange(nl)
        Z = ResistanceSketch.build(nl, el, k=64, seed=0).Z
        rows = []
        for p in np.flatnonzero((m > T) & (m <= T + 24)):
            allrefs = e[e[:, 0] == p, 1]; refs = remap[allrefs]; inside = refs >= 0; refs = refs[inside]
            if len(refs) < 4: continue
            refs = refs[:30]; D = ((Z[refs][:, None] - Z[refs][None]) ** 2).sum(2); d = D[np.triu_indices(len(refs), 1)]
            rows.append((np.median(d), np.quantile(d, 0.9), pct[p], len(refs), np.log1p(indeg_T[allrefs[inside]][:30]).mean(), np.log1p(len(allrefs))))
        R = np.array(rows); core, throw, imp, nref, hub, ntot = np.log(R[:, 0]), np.log(R[:, 1]), R[:, 2], R[:, 3], R[:, 4], R[:, 5]
        # partial correlations (rank-based), optionally given controls
        def partial(a, b, controls):
            X = np.c_[np.ones(len(a)), *[rankdata(c) for c in controls]]
            r = lambda v: rankdata(v) - X @ np.linalg.lstsq(X, rankdata(v), rcond=None)[0]
            return round(float(np.corrcoef(r(a), r(b))[0, 1]), 3)
        rz = lambda a, b: (rankdata(a) - np.polyval(np.polyfit(rankdata(b), rankdata(a), 1), rankdata(b)))
        lo_core, hi_throw = core <= np.median(core), throw >= np.median(throw)
        out[f"{g}@{T}"] = {"n_papers": len(R), "spearman_core_impact": round(float(spearmanr(core, imp)[0]), 3),
                           "spearman_throw_impact": round(float(spearmanr(throw, imp)[0]), 3),
                           "partial_throw_given_core": round(float(np.corrcoef(rz(throw, core), rz(imp, core))[0, 1]), 3),
                           "partial_core_given_throw": round(float(np.corrcoef(rz(core, throw), rz(imp, throw))[0, 1]), 3),
                           "spearman_core_vs_ref_hubness": round(float(spearmanr(core, hub)[0]), 3),
                           "spearman_hubness_impact": round(float(spearmanr(hub, imp)[0]), 3),
                           "control": {"core_given_hubness": partial(core, imp, [hub]), "core_given_hubness_nrefs": partial(core, imp, [hub, nref, ntot]),
                                       "throw_given_core_hubness_nrefs": partial(throw, imp, [core, hub, nref, ntot]),
                                       "hubness_given_core": partial(hub, imp, [core])},
                           "mean_impact": {"tight_core_far_throw": round(float(imp[lo_core & hi_throw].mean()), 3), "tight_core_near_throw": round(float(imp[lo_core & ~hi_throw].mean()), 3),
                                           "loose_core_far_throw": round(float(imp[~lo_core & hi_throw].mean()), 3), "loose_core_near_throw": round(float(imp[~lo_core & ~hi_throw].mean()), 3)}}
        print(g, T, out[f"{g}@{T}"], flush=True)
json.dump(out, open(Path(__file__).parent / "e19_results.json", "w"), indent=1)
