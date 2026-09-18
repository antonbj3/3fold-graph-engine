#!/usr/bin/env python3
"""E27b: expanding the e27 finding (a midpoint of two nodes in resistance space, decoded to its 3 nearest nodes, finds future
co-citations 10× more often than random triples; far pairs alone 0). Same graph, cutoff and label as e27. Arms:
  dithered_nb      the same midpoints, dither on the neighbourhood's scale (throws default after e27)
  mid_far / mid_near   midpoints of pairs in the top / bottom decile of resistance distance — does decoding BETWEEN two
                   distant regions find the nodes that will join them (a throw that is long but decoded locally)?
  chain            the segment from node a to node b sampled at t = 0.25, 0.5, 0.75 and each decoded to its nearest node:
                   a chain throw a → · → · → · → b; label on all pairs of the chain
  oed_point        x* = the node with the largest hole field (diag C) pushed halfway toward its farthest of 50 random nodes
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
g = {"__file__": str(Path(__file__).resolve().parent / "e27_continuum_throws.py")}
exec(open(g["__file__"]).read().split("res = {}")[0], g)
rng, Z, alive, K, N, tfc, score = g["rng"], g["Z"], g["alive"], g["K"], g["N_SETS"], g["throw_from_continuum"], g["score_sets"]
res = json.load(open(Path(__file__).parent / "e27_results.json"))
mids = [Z[rng.choice(alive, 2, replace=False)].mean(0) for _ in range(N)]
res["dithered_nb"] = score([tfc(Z, x, k=K, n_pi=1, seed=s)[0][0] for s, x in enumerate(mids)]); print("dithered_nb", res["dithered_nb"], flush=True)
pairs = np.array([rng.choice(alive, 2, replace=False) for _ in range(20 * N)]); r = np.sum((Z[pairs[:, 0]] - Z[pairs[:, 1]]) ** 2, 1)
far, near = pairs[r >= np.quantile(r, 0.9)][:N], pairs[r <= np.quantile(r, 0.1)][:N]
for name, P in [("mid_far", far), ("mid_near", near)]:
    res[name] = score([tfc(Z, Z[p].mean(0), k=K, dither=0.0, n_pi=1, seed=s)[0][0] for s, p in enumerate(P)]); print(name, res[name], flush=True)
def chain(p):
    a, b = Z[p[0]], Z[p[1]]; nodes = [int(p[0])]
    for t in (0.25, 0.5, 0.75):
        x = a + t * (b - a); nodes.append(int(np.argmin(np.linalg.norm(Z - x, axis=1))))
    nodes.append(int(p[1])); return np.array(sorted(set(nodes)))
res["chain_far"] = score([chain(p) for p in far]); print("chain_far", res["chain_far"], flush=True)
res["chain_random"] = score([chain(p) for p in pairs[:N]]); print("chain_random", res["chain_random"], flush=True)
hole = np.einsum("ij,ij->i", Z, Z); top = np.argsort(-hole)[:N]
def oed_point(i):
    others = rng.choice(alive, 50, replace=False); j = others[np.argmax(np.sum((Z[others] - Z[i]) ** 2, 1))]
    return 0.5 * (Z[i] + Z[j])
res["oed_point"] = score([tfc(Z, oed_point(i), k=K, dither=0.0, n_pi=1, seed=s)[0][0] for s, i in enumerate(top)]); print("oed_point", res["oed_point"], flush=True)
json.dump(res, open(Path(__file__).parent / "e27_results.json", "w"), indent=1)
