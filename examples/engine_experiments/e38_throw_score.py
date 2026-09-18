#!/usr/bin/env python3
"""E38: one throw score or two? The learned densification policy says WHERE the graph will be built (geometry of the
claim/citation graph, e3d); the gap co-movement of model-vs-claims says WHERE the mechanism couples (e37). This asks
whether their product beats each alone on either facit, or whether they are orthogonal and the engine has to carry two
throw lists.

(1) Planted hidden links (`disagreement_field.LinkedWorld`, 40 worlds x 12 pairs, 66 candidate couples, 3 planted):
    (a) gap co-movement = `link_scores` of the disagreement fields after 8 probes per pair by the value rule (e37's
        0.215 +- 0.019 arm), (b) a GEOMETRY score read off the claim graph only. What exists there: the bipartite
        pair-source graph (a claim is (pair, box, sign, source roots)). Computable e3d features on it:
        cn_coupling (# shared sources), adamic_adar over sources, pref (log claim degree of both ends), resistance
        (effective resistance between the two pair nodes, `ResistanceSketch`, e37's rule) and hole (||z||^2 of both
        ends). NOT computable: cn_chain (the claim graph is bipartite and undirected), age_gap / recency / vel /
        straight (a LinkedWorld has no time axis). The geometry arm applies e3d's own fitted densification weights
        (HepTh train71->test95) to the five features that exist, each z-scored within the world; singles reported too.
    (c) combinations: rank-sum, rank-product and the z-score product of (a) and (b). Score: precision@5 of the true
        links, +- se over worlds; plus the within-world Spearman between (a) and (b) = the orthogonality test.

(2) cit-HepTh, T = 95, H = 24, the e3d cell (100 000 uniform alive non-co-cited pairs + 100 000 from build_episode's
    coupling/chain pool), label rho = R(T+H)/R(T) < 0.5. There is no mechanism model for citations, so the second
    score cannot be the gap field; the chain version of the throw is tested instead: for the policy's top decile,
    `throws.chain_throw` decodes the segment between the two ends in the label geometry and the links inside the
    decoded chain are scored by the same policy. Reported: bridged share of the pair itself vs of the chain's links
    (all of them, the policy-best one, and "any link of the chain bridged" = the path-bundle reading).

Run: OMP_NUM_THREADS=4 HUNT_DATA=~/projects/hunt_3fold/data python3 -u e38_throw_score.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graph_engine.claim_federation import lineage_information
from graph_engine.disagreement_field import (
    LinkedWorld, claims_sign_field, disagreement_field, harvest, link_scores, model_sign_field,
)
from graph_engine.regime_posterior import RegimePosterior
from graph_engine.replay_policy import FEATURES, ReplayPolicy, build_episode, precision_at, average_precision
from graph_engine.resistance_sketch import ResistanceSketch
from graph_engine.throws import chain_throw
from _data import largest_component, simple_undirected, snap_citations

OUT = Path(__file__).with_name("e38_results.json")
t0 = time.time()
res = {"experiment": "e38", "_note": __doc__.split("\n\n")[0]}


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def save():
    OUT.write_text(json.dumps(res, indent=1))


# =================================================================================================
# (1) planted hidden links: mechanism co-movement x claim-graph geometry
# =================================================================================================
N_WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
R_PROBE, N_PROBES_B, N_PAIRS, N_LINKS = 0.9, 8, 12, 3
THETA_BOX = {"theta": (0.2, 0.8)}
MODEL_101 = model_sign_field(harvest, (0.0, 1.0), THETA_BOX, n=101, n_p=41, name="harvest")
# e3d's fitted densification weights (HepTh train71->test95, replay:all), for the features the claim graph has
E3D_W = {"cn_coupling": -0.276, "adamic_adar": 0.001, "pref": 0.413, "resistance": 0.199, "hole": 0.675}


def posteriors(world):
    posts = [RegimePosterior(0.0, 1.0, n_grid=24, p_two=0.0, p_flip=0.7) for _ in range(world.n_pairs)]
    groups = {}
    for p, a, b, s, roots in world.claims:
        groups.setdefault((p, round(a, 6), round(b, 6), s), []).append(roots)
    for (p, a, b, s), rs in groups.items():
        posts[p].add_claim(a, b, s, n_eff=lineage_information(rs))
    return posts


def claim_graph_features(world):
    """The five e3d features that exist in the pair-source claim graph, as n_p x n_p matrices."""
    n_p = world.n_pairs
    W = {}                                            # (pair, source) -> claim count
    for p, _a, _b, _s, roots in world.claims:
        for r in set(roots):
            W[(p, r)] = W.get((p, r), 0.0) + 1.0
    srcs = sorted({r for _, r in W})
    sidx = {s: n_p + k for k, s in enumerate(srcs)}
    M = np.zeros((n_p, len(srcs)))                    # pair x source incidence (claim counts)
    for (p, r), w in W.items():
        M[p, sidx[r] - n_p] = w
    B = (M > 0).astype(float)
    src_deg = np.maximum(B.sum(0), 2.0)
    cn = B @ B.T                                      # shared sources
    aa = (B / np.log(src_deg)) @ B.T                  # Adamic-Adar over sources
    deg = np.log1p(M.sum(1))
    pref = deg[:, None] + deg[None, :]
    edges = np.array([[p, sidx[r] - n_p + n_p] for (p, r) in W], int)
    edges = np.array([[p, sidx[r]] for (p, r) in W], int)
    n_nodes = n_p + len(srcs)
    sk = ResistanceSketch.build(n_nodes, edges, np.array(list(W.values()), float), k=256, seed=0)
    Z = sk.Z
    R = np.zeros((n_p, n_p))
    for p in range(n_p):
        for q in range(p + 1, n_p):
            d = Z[p] - Z[q]
            R[p, q] = R[q, p] = float(d @ d)
    h = np.einsum("ij,ij->i", Z[:n_p], Z[:n_p])
    hole = h[:, None] + h[None, :]
    return {"cn_coupling": cn, "adamic_adar": aa, "pref": pref, "resistance": R, "hole": hole}


def upper(m, n_p):
    idx = [(p, q) for p in range(n_p) for q in range(p + 1, n_p)]
    return np.array([m[p][q] for p, q in idx], float), idx


def z(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def rank(v):                                          # 1 = best (highest score)
    o = np.argsort(-v, kind="stable")
    r = np.empty(len(v)); r[o] = np.arange(len(v)) + 1.0
    return r


def prec_at_5(scores, idx, truth, k=5):
    top = [idx[i] for i in np.argsort(-scores, kind="stable")[:k]]
    return sum((min(p, q), max(p, q)) in truth for p, q in top) / k


arms = {}
sp_ab, sp_ab_geo_single = [], []
rng_rand = np.random.default_rng(7)
for w_i in range(N_WORLDS):
    world = LinkedWorld(n_pairs=N_PAIRS, n_links=N_LINKS, seed=1000 + w_i)
    posts = posteriors(world)
    rngp = np.random.default_rng(50_000 + w_i)
    for p_i, rp in enumerate(posts):
        for _ in range(N_PROBES_B):
            x = float(rp.best_probe(R_PROBE)[0])
            rp.add_probe(x, world.probe(p_i, x, R_PROBE, rngp), R_PROBE)
    ds = [disagreement_field(MODEL_101, claims_sign_field(rp, n=101)) for rp in posts]
    A_mat = link_scores(ds)
    feats = claim_graph_features(world)
    a, idx = upper(A_mat, world.n_pairs)
    F = {k: upper(v, world.n_pairs)[0] for k, v in feats.items()}
    geo = sum(E3D_W[k] * z(F[k]) for k in E3D_W)      # e3d densification weights on the claim graph
    truth = world.true_links()
    za, zg = z(a), z(geo)
    cand = {
        "a:gap-comovement": a,
        "b:geometry_e3d_weights": geo,
        "b:resistance_far": F["resistance"],
        "b:resistance_near(e37)": -F["resistance"],
        "b:hole": F["hole"],
        "c:rank_sum": -(rank(a) + rank(geo)),
        "c:rank_product": -(rank(a) * rank(geo)),
        "c:z_product": za * zg,
        "c:z_sum": za + zg,
        "random": rng_rand.random(len(a)),
    }
    for k, s in cand.items():
        arms.setdefault(k, []).append(prec_at_5(s, idx, truth))
    sp_ab.append(float(spearmanr(a, geo).statistic))
    sp_ab_geo_single.append(float(spearmanr(a, F["resistance"]).statistic))
    if (w_i + 1) % 10 == 0:
        log(f"world {w_i + 1}/{N_WORLDS}")

part1 = {"n_worlds": N_WORLDS, "n_pairs": N_PAIRS, "n_candidates": 66, "n_true_links": N_LINKS,
         "max_possible_p5": 0.6, "probes_per_pair": N_PROBES_B,
         "features_available_in_claim_graph": list(E3D_W),
         "features_not_available": ["cn_chain (claim graph is bipartite/undirected)",
                                    "age_gap", "recency", "vel", "straight (no time axis in LinkedWorld)"],
         "precision_at_5": {k: float(np.mean(v)) for k, v in arms.items()},
         "sem": {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in arms.items()},
         "spearman_gap_vs_geometry": {"mean": float(np.mean(sp_ab)),
                                      "sem": float(np.std(sp_ab, ddof=1) / np.sqrt(len(sp_ab)))},
         "paired_vs_gap_alone": {k: {"mean": float(np.mean(np.array(v) - np.array(arms["a:gap-comovement"]))),
                                     "sem": float(np.std(np.array(v) - np.array(arms["a:gap-comovement"]), ddof=1)
                                                  / np.sqrt(len(v))),
                                     "wins": int((np.array(v) > np.array(arms["a:gap-comovement"])).sum()),
                                     "losses": int((np.array(v) < np.array(arms["a:gap-comovement"])).sum())}
                                 for k, v in arms.items() if k.startswith("c:")},
         "spearman_gap_vs_resistance": {"mean": float(np.mean(sp_ab_geo_single)),
                                        "sem": float(np.std(sp_ab_geo_single, ddof=1) / np.sqrt(len(sp_ab_geo_single)))}}
res["part1_planted_links"] = part1
save()
log("part 1 done:", json.dumps(part1["precision_at_5"]))

# =================================================================================================
# (2) cit-HepTh: the policy's pair throw vs the chain (path-bundle) throw
# =================================================================================================
H, T_TEST, T_TRAIN = 24, 95, 71
N_UNIFORM = N_E3 = 100_000            # test cell: 200 000 pairs (the budget)
N_TRAIN = 20_000                      # training cell at T=71 (2 x 20 000), only to fit the policy out of sample
K_FEAT, K_LABEL, LOOKBACK = 64, 128, 12
N_CHAIN_PAIRS, CHAIN_STEPS = 2_000, 3


class Snapshot:
    """The graph <= cutoff with the ten e3d features (same code path as e3d; module not modified, not imported because
    e3d runs its own experiment at import)."""

    def __init__(self, edges, month, cutoff, seed=0):
        n = len(month)
        self.n, self.cutoff, self.month, self.edges = n, cutoff, month, edges
        past = edges[month[edges[:, 0]] <= cutoff]
        P = sp.csr_matrix((np.ones(len(past)), (past[:, 0], past[:, 1])), shape=(n, n))
        P.data[:] = 1
        self.P, self.PT = P, P.T.tocsr()
        self.indeg = np.asarray(P.sum(0)).ravel()
        outdeg = np.asarray(P.sum(1)).ravel()
        self.alive = np.flatnonzero((month <= cutoff) & (self.indeg + outdeg > 0))
        A = ((P + P.T) > 0).astype(float).tocsr()
        self.A = A
        deg = np.asarray(A.sum(1)).ravel()
        self.AW = (A @ sp.diags(1.0 / np.log(np.maximum(deg, 2)))).tocsr()
        _, lab = connected_components(A, directed=False)
        big = np.bincount(lab[self.alive]).argmax()
        comp = np.flatnonzero(lab == big)
        remap = -np.ones(n, np.int64); remap[comp] = np.arange(len(comp))
        und = sp.triu(A, 1).tocoo()
        e = np.stack([remap[und.row], remap[und.col]], 1); e = e[(e >= 0).all(1)]
        Z = np.zeros((n, K_FEAT)); Z[comp] = ResistanceSketch.build(len(comp), e, k=K_FEAT, seed=seed).Z
        self.Zf, self.in_comp = Z, remap >= 0
        h = np.einsum("ij,ij->i", Z, Z)
        h[remap < 0] = h[comp].max()
        self.hole_node = h

        def indeg_at(t):
            m = month[edges[:, 0]] <= t
            return np.bincount(edges[m, 1], minlength=n).astype(float)
        d0, d1 = indeg_at(cutoff), indeg_at(cutoff - LOOKBACK)
        self.d0 = d0
        self.vel = np.log1p(d0 - d1)
        l0, l1 = np.log1p(d0), np.log1p(d1)
        self.straight = l0 + (l0 - l1) * (H / float(LOOKBACK))

    def _dot(self, M1, i, M2, j, batch=20000):
        out = np.empty(len(i))
        for s in range(0, len(i), batch):
            a, b = M1[i[s:s + batch]], M2[j[s:s + batch]]
            out[s:s + batch] = np.asarray(a.multiply(b).sum(1)).ravel()
        return out

    def features(self, i, j, res_fallback=None):
        P, PT, A, AW = self.P, self.PT, self.A, self.AW
        cn_coupling = self._dot(P, i, P, j)
        cn_chain = self._dot(P, i, PT, j) + self._dot(PT, i, P, j)
        aa = self._dot(AW, i, A, j)
        d = self.Zf[i] - self.Zf[j]
        r = np.where(self.in_comp[i] & self.in_comp[j], np.einsum("ij,ij->i", d, d), np.nan)
        fb = np.nanmax(r) if res_fallback is None else res_fallback
        r = np.where(np.isnan(r), fb, r)
        m = self.month
        return np.stack([cn_coupling, cn_chain, aa,
                         np.log1p(self.d0[i]) + np.log1p(self.d0[j]), r,
                         self.hole_node[i] + self.hole_node[j],
                         np.abs(m[i] - m[j]).astype(float),
                         (self.cutoff - np.maximum(m[i], m[j])).astype(float),
                         self.vel[i] + self.vel[j], self.straight[i] + self.straight[j]], 1)

    def cocited(self, i, j):
        return self._dot(self.PT, i, self.PT, j) > 0

    def linked(self, i, j):
        return np.asarray(self.A[i, j]).ravel() > 0


def label_sketches(edges, month, cutoff, n):
    past = edges[month[edges[:, 0]] <= cutoff]
    und = simple_undirected(past)
    nl, el, keep = largest_component(n, und)
    el = simple_undirected(el)
    Z = ResistanceSketch.build(nl, el, k=K_LABEL, seed=0).Z
    ids = np.flatnonzero(keep)
    loc = -np.ones(n, np.int64); loc[ids] = np.arange(nl)
    return Z, loc, ids


def rho_of(i, j, ZT, locT, ZH, locH):
    a, b, ah, bh = locT[i], locT[j], locH[i], locH[j]
    ok = (a >= 0) & (b >= 0) & (ah >= 0) & (bh >= 0)
    out = np.full(len(i), np.nan)
    dt = ZT[a[ok]] - ZT[b[ok]]
    dh = ZH[ah[ok]] - ZH[bh[ok]]
    RT = np.einsum("ij,ij->i", dt, dt)
    RH = np.einsum("ij,ij->i", dh, dh)
    out[ok] = RH / np.maximum(RT, 1e-300)
    return out


def make_cell(edges, month, T, n_uniform, n_e3, ZT=None, locT=None, ZH=None, locH=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(month)
    snap = Snapshot(edges, month, T, seed=seed)
    if ZT is None:
        ZT, locT, _ = label_sketches(edges, month, T, n)
        ZH, locH, _ = label_sketches(edges, month, T + H, n)
    log(f"HepTh@{T}: snapshot alive={len(snap.alive)}")
    pool_i, pool_j, tries = [], [], 0
    alive_ok = snap.alive[locT[snap.alive] >= 0]
    while sum(len(x) for x in pool_i) < n_uniform and tries < 12:
        tries += 1
        need = n_uniform - sum(len(x) for x in pool_i)
        a = rng.choice(alive_ok, int(need * 1.6) + 1000)
        b = rng.choice(alive_ok, int(need * 1.6) + 1000)
        keep = a != b
        a, b = a[keep], b[keep]
        keep = ~snap.cocited(a, b) & ~snap.linked(a, b)
        a, b = a[keep][:need], b[keep][:need]
        pool_i.append(a); pool_j.append(b)
    ui, uj = np.concatenate(pool_i), np.concatenate(pool_j)
    ep = build_episode(edges, month, T, H, seed=seed)
    sel = rng.choice(len(ep.y), min(n_e3, len(ep.y)), replace=False)
    ei, ej = ep.pairs[sel, 0], ep.pairs[sel, 1]
    i = np.concatenate([ui, ei]); j = np.concatenate([uj, ej])
    src = np.concatenate([np.zeros(len(ui), np.int8), np.ones(len(ei), np.int8)])
    X = snap.features(i, j)
    rho = rho_of(i, j, ZT, locT, ZH, locH)
    good = np.isfinite(rho) & (rho > 0)
    log(f"HepTh@{T}: n={int(good.sum())} bridged={float((rho[good] < 0.5).mean()):.4f}")
    return dict(snap=snap, i=i[good], j=j[good], X=X[good], src=src[good], rho=rho[good],
                y=(rho[good] < 0.5), ZT=ZT, locT=locT, ZH=ZH, locH=locH)


edges, month, _ids = snap_citations("HepTh")
n = len(month)
ZT, locT, idsT = label_sketches(edges, month, T_TEST, n)
ZH, locH, _idsH = label_sketches(edges, month, T_TEST + H, n)
log(f"label sketches at T={T_TEST}: n_T={len(idsT)}")
test = make_cell(edges, month, T_TEST, N_UNIFORM, N_E3, ZT, locT, ZH, locH, seed=0)
train = make_cell(edges, month, T_TRAIN, N_TRAIN, N_TRAIN, seed=0)
pol = ReplayPolicy().fit(train["X"], train["y"])
s = pol.score(test["X"])
y = test["y"]
k10 = max(1, int(0.1 * len(s)))
top = np.argsort(-s, kind="stable")[:k10]
pair_arm = {"n_test_pairs": int(len(y)), "base_rate_bridged": float(y.mean()),
            "ap": float(average_precision(y, s)), "p_at_1000": float(precision_at(y, s, 1000)),
            "bridged_share_top_decile": float(y[top].mean()),
            "weights": pol.weights()}
log("pair arm:", json.dumps({k: v for k, v in pair_arm.items() if k != "weights"}))
res["part2_hepth"] = {"T": T_TEST, "H": H, "train_cutoff": T_TRAIN, "pair_throw": pair_arm}
save()

# ---- the chain (path-bundle) throw on a sample of the top decile
rng = np.random.default_rng(3)
sel = rng.choice(top, min(N_CHAIN_PAIRS, len(top)), replace=False)
snap = test["snap"]
pi, pj = test["i"][sel], test["j"][sel]
li, lj = locT[pi], locT[pj]
ok = (li >= 0) & (lj >= 0)
pi, pj, li, lj, sel = pi[ok], pj[ok], li[ok], lj[ok], sel[ok]
log(f"chain throws for {len(pi)} top-decile pairs, {CHAIN_STEPS} steps, label geometry k={K_LABEL}")
chains = []
for k in range(len(pi)):
    nodes_loc = chain_throw(ZT, int(li[k]), int(lj[k]), steps=CHAIN_STEPS, seed=k)
    chains.append(idsT[nodes_loc])                      # back to global node ids
    if (k + 1) % 500 == 0:
        log(f"  chains {k + 1}/{len(pi)}")

ci, cj, owner = [], [], []
for k, nd in enumerate(chains):
    for u in range(len(nd)):
        for v in range(u + 1, len(nd)):
            a, b = int(nd[u]), int(nd[v])
            if {a, b} == {int(pi[k]), int(pj[k])}:
                continue                                # the original pair is the pair arm, not a chain link
            ci.append(a); cj.append(b); owner.append(k)
ci, cj, owner = np.array(ci), np.array(cj), np.array(owner)
Xc = snap.features(ci, cj)
sc = pol.score(Xc)
rhoc = rho_of(ci, cj, ZT, locT, ZH, locH)
okc = np.isfinite(rhoc) & (rhoc > 0)
yc = rhoc < 0.5
y_pair_sel = y[sel]
# best chain link per chain by the policy
best = {}
for k in range(len(sc)):
    if not okc[k]:
        continue
    o = owner[k]
    if o not in best or sc[k] > sc[best[o]]:
        best[o] = k
best_idx = np.array(sorted(best.values()), int)
any_bridged = np.array([bool(yc[(owner == o) & okc].any()) if ((owner == o) & okc).any() else False
                        for o in range(len(pi))])
chain_arm = {
    "n_chains": int(len(pi)), "chain_steps": CHAIN_STEPS,
    "mean_chain_nodes": float(np.mean([len(c) for c in chains])),
    "n_chain_links_labelled": int(okc.sum()),
    "pair_bridged_share": float(y_pair_sel.mean()),
    "chain_links_bridged_share": float(yc[okc].mean()),
    "best_chain_link_by_policy_bridged_share": float(yc[best_idx].mean()) if len(best_idx) else None,
    "any_chain_link_bridged_share": float(any_bridged.mean()),
    "pair_or_chain_bridged_share": float((y_pair_sel | any_bridged).mean()),
    "mean_policy_score_pair": float(s[sel].mean()),
    "mean_policy_score_chain_links": float(sc[okc].mean()),
    "sem": {"pair": float(y_pair_sel.std(ddof=1) / np.sqrt(len(y_pair_sel))),
            "chain_links": float(yc[okc].std(ddof=1) / np.sqrt(int(okc.sum()))),
            "any_chain_link": float(any_bridged.std(ddof=1) / np.sqrt(len(any_bridged)))},
}
res["part2_hepth"]["chain_throw"] = chain_arm
res["seconds"] = round(time.time() - t0, 1)
save()
log("chain arm:", json.dumps(chain_arm))
log("written", OUT)
