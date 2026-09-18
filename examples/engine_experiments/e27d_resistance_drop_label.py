#!/usr/bin/env python3
"""E27d: a DENSIFICATION label instead of a co-citation label.

e27/e27b/e27c all scored a throw by "does a new paper co-cite this pair within the horizon". Three different scores
(hit share, would-be leverage deciles, chain vs midpoint) gave the same answer — near, already-almost-connected pairs win —
because a co-citation is itself an act of proximity: a single new paper must already be close to both ends. That label cannot
see the engine's actual claim, which is that a throw DENSIFIES: it creates connection where the graph had none.

The label here measures exactly that. For a pair (i, j) alive at T = 95 months (cit-HepTh, giant component of the undirected
citation graph ≤ T), with H = 24,
        rho_ij = R_ij(T + H) / R_ij(T) ,
R = effective resistance, each side computed from a resistance sketch of the graph at that time (k = 128, R_ij = ||z_i - z_j||^2).
rho = 1: the field did nothing between these two. rho << 1: the field built routes between them — many new parallel paths, not
one paper. rho > 1 is possible in principle only through the graph growing around them (resistance is monotone decreasing under
edge addition on a fixed node set, but the sketch is random and the node set grows, so small excursions above 1 are noise).
Pairs whose endpoints leave the giant component at T + H get rho = NaN and are dropped (reported per arm).

k = 128 instead of e27's k = 64 (relative error of a quadratic form ~ sqrt(2/k): ~18 % at 64, ~12.5 % at 128 per resistance;
the ratio of two independent sketches is noisier still, which is why the arm statistics are means over hundreds of pairs and the
per-pair sketch error is MEASURED here against an exact pinv on a 2 000-node induced subgraph).

Arms (<= 600 sets each), all scored by the same rho:
    random_pairs                two uniformly random alive nodes
    near_midpoint / far_midpoint  midpoint in Z of a bottom-/top-decile-resistance pair, decoded to its 3 nearest nodes (e27b)
    chain_far                   the segment between a far pair sampled at t = .25,.5,.75 and decoded (e27b chain throw)
    dpp_150                     determinantal sets on the EXACT kernel of a 150-node subsample (e31: the sketched kernel is noise)
    wbl_top_decile / wbl_bottom_decile   would-be leverage l = R/(1+R) deciles over pairs not already co-cited (e27c)
    R_top_decile / R_bottom_decile       the pure baselines: resistance deciles at T (l is monotone in R, so these differ from
                                         the wbl arms only through the candidate pool, kept separate on purpose)
Reported per arm: mean log rho with a cluster-robust se (clustered by SET, since pairs inside a triple/chain are not independent)
and the share of pairs with rho < 0.5 ("bridged"). Control from e19: resistance distance is ~ 1/deg_i + 1/deg_j, so log rho is
correlated with endpoint degree at T by construction; the partial correlation of each arm indicator with log rho GIVEN the
endpoints' mean log degree at T (rank-based) says whether the rule finds anything degree does not.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.resistance_sketch import ResistanceSketch, laplacian_from_edges
from graph_engine.throws import throw_from_continuum, draw_set_dpp, would_be_leverage
from _data import snap_citations, largest_component, simple_undirected

T, H, N_SETS, K, KSK = 95, 24, 600, 3, 128
OUT = Path(__file__).parent / "e27d_results.json"
res = {"_note": f"cit-HepTh, T={T}, horizon {H}; label rho_ij = R_ij(T+H)/R_ij(T) from resistance sketches (k={KSK}) of the "
                f"undirected citation graph at each time; pairs leaving the giant component dropped. <= {N_SETS} sets per arm."}

e, m, _ = snap_citations("HepTh")
n = len(m)

def build(cut):
    """(n_local, edges_local, orig_ids, Z, degree) for the giant component of the undirected graph of papers published <= cut."""
    past = e[m[e[:, 0]] <= cut]
    und = simple_undirected(past)
    nl, el, keep = largest_component(n, und)
    el = simple_undirected(el)
    Z = ResistanceSketch.build(nl, el, k=KSK, seed=0).Z
    deg = np.bincount(el.reshape(-1), minlength=nl).astype(float)
    return nl, el, np.flatnonzero(keep), Z, deg

nl, el, ids_T, Z, deg_T = build(T)
print(f"graph <= T: n={nl} m={len(el)}", flush=True)
nh, elh, ids_H, ZH, _ = build(T + H)
print(f"graph <= T+H: n={nh} m={len(elh)}", flush=True)

# map T-local index -> (T+H)-local index
to_h = -np.ones(n, np.int64); to_h[ids_H] = np.arange(nh)
map_TH = to_h[ids_T]
res["n_nodes_T"], res["n_nodes_TH"] = int(nl), int(nh)
res["share_T_nodes_still_in_giant_TH"] = round(float((map_TH >= 0).mean()), 5)
print("still in giant at T+H:", res["share_T_nodes_still_in_giant_TH"], flush=True)

# ---- sketch error at k = 128 against an exact pinv on a 2 000-node induced subgraph ----------------------------------
rng = np.random.default_rng(0)
sub = rng.choice(nl, 2000, replace=False)
mask = -np.ones(nl, np.int64); mask[sub] = np.arange(len(sub))
se_ = el[(mask[el] >= 0).all(1)]; se_ = mask[se_]
ns, es, ks = largest_component(len(sub), se_)
es = simple_undirected(es)
Ls, _, _ = laplacian_from_edges(ns, es)
Lp = np.linalg.pinv(Ls.toarray())
Zs = ResistanceSketch.build(ns, es, k=KSK, seed=1).Z
pp = rng.choice(ns, (200, 2))
pp = pp[pp[:, 0] != pp[:, 1]]
Rex = np.array([Lp[a, a] + Lp[b, b] - 2 * Lp[a, b] for a, b in pp])
Rsk = np.einsum("ij,ij->i", Zs[pp[:, 0]] - Zs[pp[:, 1]], Zs[pp[:, 0]] - Zs[pp[:, 1]])
rel = np.abs(Rsk - Rex) / Rex
res["sketch_check"] = {"k": KSK, "subgraph_nodes": int(ns), "n_pairs": int(len(pp)),
                       "median_rel_err": round(float(np.median(rel)), 4), "p90_rel_err": round(float(np.quantile(rel, 0.9)), 4)}
print("sketch check", res["sketch_check"], flush=True)

# ---- the label --------------------------------------------------------------------------------------------------------
def rho(pairs):
    """rho for an array of T-local pairs; NaN where an endpoint left the giant component."""
    p = np.asarray(pairs, np.int64).reshape(-1, 2)
    rt = np.einsum("ij,ij->i", Z[p[:, 0]] - Z[p[:, 1]], Z[p[:, 0]] - Z[p[:, 1]])
    a, b = map_TH[p[:, 0]], map_TH[p[:, 1]]
    ok = (a >= 0) & (b >= 0)
    out = np.full(len(p), np.nan)
    if ok.any():
        d = ZH[a[ok]] - ZH[b[ok]]
        out[ok] = np.einsum("ij,ij->i", d, d) / np.maximum(rt[ok], 1e-300)
    return out, rt

def score(sets, name):
    """Per-arm statistics with a se clustered by set."""
    pr, cl, rt_all = [], [], []
    for s, S in enumerate(sets):
        S = np.unique(np.asarray(S, np.int64))
        for u in range(len(S)):
            for v in range(u + 1, len(S)):
                pr.append((int(S[u]), int(S[v]))); cl.append(s)
    pr = np.array(pr, np.int64); cl = np.array(cl)
    r, rt = rho(pr)
    good = np.isfinite(r) & (r > 0)
    lr, cl_g = np.log(r[good]), cl[good]
    uc = np.unique(cl_g)
    per_set = np.array([lr[cl_g == c].mean() for c in uc])
    se = float(per_set.std(ddof=1) / np.sqrt(len(per_set))) if len(per_set) > 1 else float("nan")
    d = 0.5 * (np.log1p(deg_T[pr[good][:, 0]]) + np.log1p(deg_T[pr[good][:, 1]]))
    out = {"n_sets": int(len(sets)), "n_pairs": int(len(pr)), "n_pairs_scored": int(good.sum()),
           "dropped_share": round(float(1 - good.mean()), 4),
           "mean_log_rho": round(float(lr.mean()), 4), "se_clustered": round(se, 4),
           "median_rho": round(float(np.median(r[good])), 4),
           "bridged_share_rho_lt_0.5": round(float((r[good] < 0.5).mean()), 4),
           "mean_R_at_T": round(float(rt[good].mean()), 4),
           "mean_log_deg_endpoints": round(float(d.mean()), 3)}
    print("arm", name, out, flush=True)
    return out, lr, d

arms, logrho, degs = {}, {}, {}
alive = np.arange(nl)

def add(name, sets):
    a, lr, d = score(sets, name); arms[name] = a; logrho[name] = lr; degs[name] = d
    json.dump({**res, "arms": arms}, open(OUT, "w"), indent=1)

# random pairs + the rho distribution
rp = np.array([rng.choice(alive, 2, replace=False) for _ in range(N_SETS)])
r_rand, _ = rho(rp); g = np.isfinite(r_rand) & (r_rand > 0)
res["rho_distribution_random_pairs"] = {"n": int(g.sum()), "median": round(float(np.median(r_rand[g])), 4),
                                        "deciles": [round(float(q), 4) for q in np.quantile(r_rand[g], np.arange(1, 10) / 10)],
                                        "share_lt_0.5": round(float((r_rand[g] < 0.5).mean()), 4),
                                        "share_gt_1": round(float((r_rand[g] > 1).mean()), 4)}
print("rho distribution", res["rho_distribution_random_pairs"], flush=True)
add("random_pairs", rp)

# near / far midpoints, decoded (e27b)
pool = np.array([rng.choice(alive, 2, replace=False) for _ in range(20 * N_SETS)])
rr = np.einsum("ij,ij->i", Z[pool[:, 0]] - Z[pool[:, 1]], Z[pool[:, 0]] - Z[pool[:, 1]])
far = pool[rr >= np.quantile(rr, 0.9)][:N_SETS]; near = pool[rr <= np.quantile(rr, 0.1)][:N_SETS]
for nm, P in [("near_midpoint", near), ("far_midpoint", far)]:
    add(nm, [throw_from_continuum(Z, Z[p].mean(0), k=K, dither=0.0, n_pi=1, seed=s)[0][0] for s, p in enumerate(P)])

# chains between far pairs (e27b)
def chain(p):
    a, b = Z[p[0]], Z[p[1]]; nodes = [int(p[0]), int(p[1])]
    for t in (0.25, 0.5, 0.75):
        nodes.append(int(np.argmin(np.linalg.norm(Z - (a + t * (b - a)), axis=1))))
    return np.array(sorted(set(nodes)))
add("chain_far", [chain(p) for p in far])

# DPP on the EXACT kernel of a 150-node subsample (not the sketch; e31 showed the sketched kernel is noise).
# Exact here means the 150x150 principal submatrix of the FULL graph's L+, obtained by 150 exact Laplacian solves
# (rhs = e_i - 1/n, grounded and re-centred) rather than by any random projection.
from graph_engine.resistance_sketch import _GroundedSolver
s150 = rng.choice(alive, 150, replace=False)
Lfull, _, _ = laplacian_from_edges(nl, el)
rhs = np.zeros((nl, 150)); rhs[s150, np.arange(150)] = 1.0; rhs -= 1.0 / nl
X = _GroundedSolver(Lfull).solve(rhs)
Kex = X[s150]                                        # Kex[j, i] = L+_{s150[j], s150[i]}
Kex = (Kex + Kex.T) / 2
lam, V = np.linalg.eigh(Kex)
Zex = V * np.sqrt(np.clip(lam, 0, None))             # <z_i, z_j> = L+_ij exactly (up to the CG tolerance)
print(f"dpp exact kernel: 150 nodes, kernel trace {float(np.trace(Kex)):.4g}", flush=True)
dpp_sets, s = [], 0
while len(dpp_sets) < N_SETS and s < 20 * N_SETS:
    S, _ = draw_set_dpp(Zex, seed=s); s += 1
    if len(S) >= K:
        dpp_sets.append(s150[rng.choice(S, K, replace=False)])
add("dpp_150_exact_kernel", dpp_sets)

# would-be leverage deciles over pairs not already co-cited (e27c's pool)
past = e[m[e[:, 0]] <= T]
remap_full = -np.ones(n, np.int64); remap_full[ids_T] = np.arange(nl)
citers_past = defaultdict(set)
for a, b in past:
    if remap_full[b] >= 0:
        citers_past[int(remap_full[b])].add(int(a))
cand, seen = [], set()
while len(cand) < 20000:
    a, b = rng.choice(alive, 2, replace=False); a, b = (int(a), int(b)) if a < b else (int(b), int(a))
    if (a, b) in seen or (citers_past[a] & citers_past[b]):
        continue
    seen.add((a, b)); cand.append((a, b))
cand = np.array(cand, np.int64)
lev = would_be_leverage(Z, cand[:, 0], cand[:, 1])
top, bot = cand[lev >= np.quantile(lev, 0.9)], cand[lev <= np.quantile(lev, 0.1)]
add("wbl_top_decile", top[rng.choice(len(top), N_SETS, replace=False)])
add("wbl_bottom_decile", bot[rng.choice(len(bot), N_SETS, replace=False)])

# pure resistance deciles at T over the plain random pool
rtop, rbot = pool[rr >= np.quantile(rr, 0.9)], pool[rr <= np.quantile(rr, 0.1)]
add("R_top_decile_at_T", rtop[rng.choice(len(rtop), N_SETS, replace=False)])
add("R_bottom_decile_at_T", rbot[rng.choice(len(rbot), N_SETS, replace=False)])

# ---- e19 control: degree of the endpoints at T ------------------------------------------------------------------------
def partial(a, b, c):
    """rank partial correlation of a and b given c."""
    X = np.c_[np.ones(len(a)), rankdata(c)]
    r = lambda v: rankdata(v) - X @ np.linalg.lstsq(X, rankdata(v), rcond=None)[0]
    return round(float(np.corrcoef(r(a), r(b))[0, 1]), 4)

ctrl = {"spearman_logrho_vs_mean_log_deg_random_pairs":
        round(float(np.corrcoef(rankdata(logrho["random_pairs"]), rankdata(degs["random_pairs"]))[0, 1]), 4)}
for nm in arms:
    if nm == "random_pairs":
        continue
    a = np.r_[logrho["random_pairs"], logrho[nm]]
    d = np.r_[degs["random_pairs"], degs[nm]]
    ind = np.r_[np.zeros(len(logrho["random_pairs"])), np.ones(len(logrho[nm]))]
    ctrl[nm] = {"raw_corr_arm_vs_logrho": round(float(np.corrcoef(rankdata(ind), rankdata(a))[0, 1]), 4),
                "partial_given_mean_log_deg": partial(ind, a, d)}
res["degree_control"] = ctrl
print("degree control", json.dumps(ctrl, indent=1), flush=True)

res["arms"] = arms
json.dump(res, open(OUT, "w"), indent=1)
print("written", OUT, flush=True)
