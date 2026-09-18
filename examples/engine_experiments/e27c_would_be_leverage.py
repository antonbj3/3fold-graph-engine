#!/usr/bin/env python3
"""E27c: WOULD-BE LEVERAGE as a throw score. Same graph, cutoff and label as e27/e27b (cit-HepTh, T = 95 months, horizon 24,
resistance-sketch coordinates Z with k = 64; label = a new paper cites both members of a pair that no earlier paper cited together).

The score. For an edge that EXISTS, P(e ∈ uniform spanning tree) = w_e R_e (Kirchhoff). For a pair (i, j) with no edge, the same
quantity computed in the graph WITH the edge added is, by Sherman-Morrison on L (proof in throws.would_be_leverage),
        ℓ_ij = w R_ij / (1 + w R_ij) ∈ (0, 1) .
ℓ → 0: the two are already joined by many parallel paths (redundant). ℓ → 1: the link would be a bridge, its ends currently share
no route. Unlike raw resistance ℓ is bounded and saturates, so "the middle band" is a defined region rather than a tail.

The question, not an assumption: WHERE along ℓ do the pairs that actually get co-cited within the horizon sit? Measured by
binning 20 000 random not-yet-co-cited alive pairs into deciles of ℓ and reading the share of each decile that gets a future
co-citation. Then four pair arms of 600 (top decile, the middle deciles 4-6 drawn by throws.bridge_throws with exact inclusion
probabilities, bottom decile, uniform random) and two triple arms of 600 (a pair plus a third node chosen among 200 random
candidates: the one with the HIGHEST mean leverage to both ends, and the one whose mean leverage is CLOSEST to the band centre),
so the numbers can be read against e27 (random triples 0.7 %, near-midpoint 7.2 %, chain 2.0 %).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.throws import would_be_leverage, bridge_throws

# reuse e27's setup verbatim (graph <= T, sketch Z, citer sets, cocited_already / cocited_future, score_sets)
g = {"__file__": str(Path(__file__).resolve().parent / "e27_continuum_throws.py")}
exec(open(g["__file__"]).read().split("res = {}")[0], g)
Z, alive, N = g["Z"], g["alive"], g["N_SETS"]
cocited_already, cocited_future, score_sets = g["cocited_already"], g["cocited_future"], g["score_sets"]
rng = np.random.default_rng(27)
print(f"setup: n_alive={len(alive)} k={Z.shape[1]} N_SETS={N}", flush=True)

# ---- the decile curve: 20 000 random not-yet-co-cited pairs ----------------------------------------------------------
N_CAND = 20_000
pairs, seen = [], set()
while len(pairs) < N_CAND:
    a, b = rng.choice(alive, 2, replace=False)
    a, b = (int(a), int(b)) if a < b else (int(b), int(a))
    if (a, b) in seen or cocited_already(a, b):
        continue
    seen.add((a, b)); pairs.append((a, b))
pairs = np.array(pairs, np.int64)
lev = would_be_leverage(Z, pairs[:, 0], pairs[:, 1])
lab = np.array([cocited_future(int(a), int(b)) for a, b in pairs], bool)
print(f"candidates: {len(pairs)}  future co-citation share overall = {lab.mean():.4%}", flush=True)

edges = np.quantile(lev, np.linspace(0, 1, 11)); edges[0] -= 1e-12; edges[-1] += 1e-12
dec = np.clip(np.searchsorted(edges, lev, side="right") - 1, 0, 9)
curve = [{"decile": int(d) + 1, "lev_lo": round(float(edges[d]), 4), "lev_hi": round(float(edges[d + 1]), 4),
          "n": int((dec == d).sum()), "n_hits": int(lab[dec == d].sum()), "share_future": round(float(lab[dec == d].mean()), 5),
          "mean_R": round(float(np.mean(lev[dec == d] / (1 - lev[dec == d]))), 3)} for d in range(10)]
for c in curve:
    print("decile", c["decile"], c["lev_lo"], "-", c["lev_hi"], "n", c["n"], "share", c["share_future"], flush=True)

# ---- pair arms of 600 -------------------------------------------------------------------------------------------------
def score_pairs(P):
    P = np.asarray(P, np.int64).reshape(-1, 2)
    return score_sets([np.array([int(a), int(b)]) for a, b in P])

top = pairs[dec == 9]; bot = pairs[dec == 0]; mid = pairs[(dec >= 3) & (dec <= 5)]
band = (float(np.quantile(lev, 0.3)), float(np.quantile(lev, 0.6)))     # deciles 4-6 as a leverage band
arms = {}
arms["random_pairs"] = score_pairs(pairs[rng.choice(len(pairs), N, replace=False)])
arms["top_decile"] = score_pairs(top[rng.choice(len(top), N, replace=False)])
arms["bottom_decile"] = score_pairs(bot[rng.choice(len(bot), N, replace=False)])
arms["mid_deciles_4_6"] = score_pairs(mid[rng.choice(len(mid), N, replace=False)])
drawn = bridge_throws(Z, pairs, band=band, k=N, temperature=0.5, seed=0)     # the band arm with EXACT inclusion probabilities
arms["mid_band_bridge_throws"] = score_pairs([(i, j) for i, j, _ in drawn])
arms["mid_band_bridge_throws"]["mean_inclusion_pi"] = round(float(np.mean([p for *_, p in drawn])), 5)
for k, v in arms.items():
    print("arm", k, v, flush=True)

# ---- triple arms of 600: a pair plus its best would-be-leverage third node among 200 random candidates ----------------
def extend(P, rule):
    out = []
    for a, b in P:
        c = rng.choice(alive, 200, replace=False); c = c[(c != a) & (c != b)]
        m = 0.5 * (would_be_leverage(Z, np.full(len(c), a), c) + would_be_leverage(Z, np.full(len(c), b), c))
        t = int(np.argmax(m)) if rule == "max" else int(np.argmin(np.abs(m - sum(band) / 2)))
        out.append(np.array(sorted({int(a), int(b), int(c[t])}), int))
    return out
base = pairs[rng.choice(len(pairs), N, replace=False)]
arms["triple_third_max_leverage"] = score_sets(extend(base, "max")); print("arm triple_third_max_leverage", arms["triple_third_max_leverage"], flush=True)
arms["triple_third_in_band"] = score_sets(extend(base, "band")); print("arm triple_third_in_band", arms["triple_third_in_band"], flush=True)

res = {"_note": "cit-HepTh, T=95, horizon 24; would-be leverage l = wR/(1+wR) on resistance-sketch coordinates (k=64, w=1). "
                "Decile curve over 20 000 random alive pairs not already co-cited; arms of 600. Label = a new paper co-cites a "
                "pair not co-cited before. e27 reference: random triples 0.0067, near-midpoint 0.0717, chain 0.020.",
       "n_candidates": int(len(pairs)), "overall_share_future": round(float(lab.mean()), 5),
       "band_used": [round(band[0], 4), round(band[1], 4)], "decile_curve": curve, "arms": arms,
       "formula_check": "tests/test_throws.py: w R'_ij from a fresh pinv of L + w bb^T equals wR/(1+wR) to 1e-10; "
                        "path-graph edge leverage 1, 4-cycle edge 3/4"}
json.dump(res, open(Path(__file__).parent / "e27c_results.json", "w"), indent=1)
print("written", flush=True)
