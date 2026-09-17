#!/usr/bin/env python3
"""E1: sketched + 4-bit hole field against the dense pinv used by graph_hole_engine."""
import json, sys, time, tracemalloc
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.graph_hole_engine.kernel import resolvent_leverage
from graph_engine.resistance_sketch import ResistanceSketch, QuantizedSketch, laplacian_from_edges, _GroundedSolver
from _data import snap_citations, largest_component, simple_undirected

def synth(n, rng):
    import networkx as nx
    g = nx.barabasi_albert_graph(n - n // 10, 3, seed=int(rng.integers(1 << 30)))
    e = list(g.edges()); base = n - n // 10
    for p in range(base, n):                       # pendants: the structural holes of the hole engine
        e.append((p, int(rng.integers(base))))
    return np.array(e)

def topk(a, b, k): return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k

rows, rng = [], np.random.default_rng(1)
for n in [500, 1000, 2000, 4000]:   # dense pinv at n=4000 takes about a minute
    e = synth(n, rng)
    L, _, _ = laplacian_from_edges(n, e)
    t = time.perf_counter(); exact = resolvent_leverage(L.toarray(), np.eye(n)); t_dense = time.perf_counter() - t
    for k in [32, 64, 128]:
        t = time.perf_counter(); sk = ResistanceSketch.build(n, e, k=k, seed=2); t_sk = time.perf_counter() - t
        q = QuantizedSketch.build(sk.Z, bits=4)
        hf, hq, hr = sk.hole_field(), q.hole_field(), sk.refined_hole_field(L)
        K = n // 20
        rows.append(dict(graph="synthetic BA+pendants", n=n, k=k, t_dense_s=round(t_dense, 3), t_sketch_s=round(t_sk, 3),
            bytes_dense=8 * n * n, bytes_float_sketch=sk.Z.nbytes, bytes_4bit=q.nbytes(),
            spearman_float=round(float(spearmanr(exact, hf)[0]), 4), spearman_4bit=round(float(spearmanr(exact, hq)[0]), 4),
            top5pct_float=round(topk(exact, hf, K), 3), top5pct_4bit=round(topk(exact, hq, K), 3),
            medrelerr_float=round(float(np.median(abs(hf - exact) / exact)), 4),
            medrelerr_4bit=round(float(np.median(abs(hq - exact) / exact)), 4),
            medrelerr_refined=round(float(np.median(abs(hr - exact) / exact)), 4), spearman_refined=round(float(spearmanr(exact, hr)[0]), 4)))
        print(rows[-1], flush=True)

# real graph: dense pinv is not run (n=27k → 5.9 GB and O(n³)); exact values come from exact solves on a node sample
for name in ["HepTh"]:
    edges, _, _ = snap_citations(name)
    n0 = edges.max() + 1
    n, e, _ = largest_component(n0, edges); e = simple_undirected(e)
    L, _, _ = laplacian_from_edges(n, e)
    solver = _GroundedSolver(L)
    samp = rng.choice(n, 400, replace=False); solver.tol = 1e-9
    Bm = np.zeros((n, len(samp))); Bm[samp, np.arange(len(samp))] = 1; Bm -= 1 / n
    X = solver.solve(Bm); exact = X[samp, np.arange(len(samp))]
    for k in [64, 128]:
        t = time.perf_counter(); sk = ResistanceSketch.build(n, e, k=k, seed=3); t_sk = time.perf_counter() - t
        q = QuantizedSketch.build(sk.Z, bits=4)
        hf, hq, hr = sk.hole_field()[samp], q.hole_field()[samp], sk.refined_hole_field(L)[samp]
        rows.append(dict(graph=f"SNAP cit-{name} LCC", n=n, m=len(e), k=k, t_dense_s=None, t_sketch_s=round(t_sk, 2),
            bytes_dense=8 * n * n, bytes_float_sketch=sk.Z.nbytes, bytes_4bit=q.nbytes(),
            spearman_float=round(float(spearmanr(exact, hf)[0]), 4), spearman_4bit=round(float(spearmanr(exact, hq)[0]), 4),
            top5pct_float=round(topk(exact, hf, 20), 3), top5pct_4bit=round(topk(exact, hq, 20), 3),
            medrelerr_float=round(float(np.median(abs(hf - exact) / exact)), 4),
            medrelerr_4bit=round(float(np.median(abs(hq - exact) / exact)), 4),
            medrelerr_refined=round(float(np.median(abs(hr - exact) / exact)), 4), spearman_refined=round(float(spearmanr(exact, hr)[0]), 4)))
        print(rows[-1], flush=True)
json.dump(rows, open(Path(__file__).parent / "e1_results.json", "w"), indent=1)
