#!/usr/bin/env python3
"""E27: throws from the continuum (seed: continuous flows for discrete problems — work in the relaxation, decode at the
corners late). cit-HepTh at cutoff T = 95 months, horizon 24. Resistance-sketch coordinates Z (k = 64) on the graph ≤ T.
Four ways to propose a SET of 3 nodes, 1 500 sets each, same random numbers where possible:
  random        three uniformly random alive nodes
  softmax_pairs e8's pair rule (softmax over resistance score, T = 0.5) extended to a third node the same way
  midpoint      x* = midpoint of two random alive nodes' coordinates, decoded to the 3 nearest nodes WITHOUT dither
  dithered      the same x*, dithered on the scale of the node spacing, decoded late (throws.throw_from_continuum)
  dpp           determinantal set draw (throws.draw_set_dpp) on a random 300-node subsample of alive nodes
Score: share of sets in which at least one pair becomes co-cited by a new paper within the horizon (the replay label of e3),
and the mean number of such future pairs per set. Also the mean within-set resistance distance, to see what each rule buys."""
import json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.resistance_sketch import ResistanceSketch
from graph_engine.throws import throw_from_continuum, draw_set_dpp
from _data import snap_citations, largest_component, simple_undirected

T, H, N_SETS, K = 95, 24, 600, 3
e, m, _ = snap_citations("HepTh"); n = len(m)
past = e[m[e[:, 0]] <= T]; und = simple_undirected(past); nl, el, keep = largest_component(n, und); el = simple_undirected(el)
remap = -np.ones(n, np.int64); remap[np.flatnonzero(keep)] = np.arange(nl)
Z = ResistanceSketch.build(nl, el, k=64, seed=0).Z
fut = e[(m[e[:, 0]] > T) & (m[e[:, 0]] <= T + H) & (m[e[:, 1]] <= T)]
fut = fut[remap[fut[:, 1]] >= 0]
from collections import defaultdict
citers_fut, citers_past = defaultdict(set), defaultdict(set)          # per old node: which papers cite it (future / past)
for a, b in fut:
    citers_fut[int(remap[b])].add(int(a))
for a, b in past:
    if remap[b] >= 0:
        citers_past[int(remap[b])].add(int(a))
cocited_future = lambda a, b: len(citers_fut[a] & citers_fut[b]) > 0
cocited_already = lambda a, b: len(citers_past[a] & citers_past[b]) > 0
alive = np.arange(nl)
rng = np.random.default_rng(0)

def score_sets(sets):
    hit, cnt, dist = 0, 0.0, []
    for S in sets:
        pairs = [(int(a), int(b)) for i, a in enumerate(S) for b in S[i + 1:] if not cocited_already(int(a), int(b))]
        c = sum(cocited_future(a, b) for a, b in pairs); hit += c > 0; cnt += c
        dist.append(np.mean([np.sum((Z[a] - Z[b]) ** 2) for a, b in pairs]) if pairs else np.nan)
    return {"share_with_future_link": round(hit / len(sets), 4), "future_pairs_per_set": round(cnt / len(sets), 4),
            "mean_within_resistance": round(float(np.nanmean(dist)), 3)}

res = {}
print("setup done", flush=True); res["random"] = score_sets([rng.choice(alive, K, replace=False) for _ in range(N_SETS)])
# softmax pairs → triples: pick i uniformly, then two partners by softmax over log resistance (far = high), T = 0.5
def softmax_triples():
    out = []
    for _ in range(N_SETS):
        i = int(rng.choice(alive)); cand = rng.choice(alive, 400, replace=False); cand = cand[cand != i]
        r = np.sum((Z[cand] - Z[i]) ** 2, 1); s = np.log(r / np.median(r) + 1e-9) / 0.5; p = np.exp(s - s.max()); p /= p.sum()
        out.append(np.r_[i, rng.choice(cand, 2, replace=False, p=p)])
    return out
print("softmax_pairs", flush=True); res["softmax_pairs"] = score_sets(softmax_triples())
mids = [Z[rng.choice(alive, 2, replace=False)].mean(0) for _ in range(N_SETS)]
print("midpoint", flush=True); res["midpoint"] = score_sets([throw_from_continuum(Z, x, k=K, dither=0.0, n_pi=1, seed=s)[0][0] for s, x in enumerate(mids)])
print("dithered", flush=True); res["dithered"] = score_sets([throw_from_continuum(Z, x, k=K, n_pi=1, seed=s)[0][0] for s, x in enumerate(mids)])
sub = rng.choice(alive, 150, replace=False); dpp_sets = []
s = 0
while len(dpp_sets) < N_SETS:
    S, _ = draw_set_dpp(Z[sub], seed=s); s += 1
    if len(S) >= K:
        dpp_sets.append(sub[rng.choice(S, K, replace=False)])
res["dpp_150_subsample"] = score_sets(dpp_sets)
res["_note"] = "cit-HepTh, T=95, horizon 24, 600 sets of 3; label = a new paper co-cites a pair of the set that was not co-cited before"
json.dump(res, open(Path(__file__).parent / "e27_results.json", "w"), indent=1); [print(k, v) for k, v in res.items()]
