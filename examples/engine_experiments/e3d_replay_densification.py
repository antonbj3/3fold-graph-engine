#!/usr/bin/env python3
"""E3d: the replay machinery (e3) with the DENSIFICATION label of e27d.

e3 fitted the ten graph-only features of `replay_policy.build_episode` against "is this pair co-cited by a new
paper within 24 months". Under that label the resistance features carried nothing (Δ AP ≤ 0.003) and near,
already-almost-connected pairs won. e27d then showed that the label was the problem: scored by
        rho_ij = R_ij(T + H) / R_ij(T)          (R = effective resistance, k = 128 sketch at each time)
the ranking inverts — the FAR pairs are the ones the field later builds routes to (top resistance decile at T:
17.8 % bridged with rho < 0.5, vs 5.5 % for random pairs). e27d compared fixed ARMS. This experiment asks the
next question: can the learned policy, on the same ten features, beat the plain resistance decile under the
densification label, and which features carry that label?

Differences from e3, on purpose:
  * candidates are NOT only the coupling/chain pool of build_episode (those pairs are near by construction and
    cannot densify much). Half the pool is a uniform sample of alive pairs that are not already co-cited and not
    linked; the other half is the e3 pool. Every row keeps a `src` column (0 = uniform, 1 = e3 coupling/chain),
    all metrics are reported for the union and per source.
  * the label is rho, both as the binary "bridged" (rho < 0.5, e27d's threshold) and as continuous log rho.
  * cutoffs T in {71, 95}: fit on one, test on the other, both directions.

The ten features are recomputed here for arbitrary pair lists (build_episode only emits its own pool). The
recomputation is checked against build_episode's own X on its own pairs and the max relative deviation is
reported in the JSON under `feature_parity_vs_build_episode`; the module is not modified.
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
from graph_engine.replay_policy import (FEATURES, ReplayPolicy, average_precision, build_episode, grow_span,
                                        precision_at)
from graph_engine.resistance_sketch import ResistanceSketch
from _data import largest_component, simple_undirected, snap_citations

H = 24                  # horizon, months (as in e3 and e27d)
CUTOFFS = [71, 95]
N_UNIFORM = 100_000     # uniform alive non-co-cited pairs
N_E3 = 100_000          # pairs from build_episode's coupling/chain pool
K_FEAT = 64             # sketch k for the `resistance`/`hole` FEATURES (build_episode's default)
K_LABEL = 128           # sketch k for the LABEL (e27d)
LOOKBACK = 12
OUT = Path(__file__).parent / "e3d_results.json"

res = {"_note": f"replay policy (e3 features) under the densification label of e27d. H={H} months, cutoffs {CUTOFFS}; "
                f"label rho = R(T+H)/R(T) from k={K_LABEL} sketches, bridged := rho < 0.5; feature sketch k={K_FEAT}; "
                f"candidates = {N_UNIFORM} uniform alive non-co-cited non-linked pairs + {N_E3} from the e3 "
                f"coupling/chain pool (column `src`)."}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def save():
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)


class Snapshot:
    """The graph <= cutoff, with everything the ten features need."""

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

        # feature-side resistance geometry: largest component of A, k = K_FEAT (build_episode)
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
        l0, l1 = np.log1p(d0), np.log1p(d1)
        self.d0 = d0
        self.vel = np.log1p(d0 - d1)
        self.straight = l0 + (l0 - l1) * (H / float(LOOKBACK))

    def _dot(self, M1, i, M2, j, batch=20000):
        out = np.empty(len(i))
        for s in range(0, len(i), batch):
            a, b = M1[i[s:s + batch]], M2[j[s:s + batch]]
            out[s:s + batch] = np.asarray(a.multiply(b).sum(1)).ravel()
        return out

    def features(self, i, j, res_fallback=None):
        """The ten FEATURES of build_episode, for arbitrary pair arrays."""
        P, PT, A, AW = self.P, self.PT, self.A, self.AW
        cn_coupling = self._dot(P, i, P, j)                       # both cite the same paper
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
        return self._dot(PTC := self.PT, i, PTC, j) > 0           # share a citing paper already

    def linked(self, i, j):
        return np.asarray(self.A[i, j]).ravel() > 0


def label_sketches(edges, month, cutoff, n):
    """e27d's label geometry: giant component of the simple undirected graph <= cutoff, k = K_LABEL sketch."""
    past = edges[month[edges[:, 0]] <= cutoff]
    und = simple_undirected(past)
    nl, el, keep = largest_component(n, und)
    el = simple_undirected(el)
    Z = ResistanceSketch.build(nl, el, k=K_LABEL, seed=0).Z
    ids = np.flatnonzero(keep)
    loc = -np.ones(n, np.int64); loc[ids] = np.arange(nl)
    return Z, loc, nl


def make_cell(graph, edges, month, T, seed=0):
    """Candidate pool + features + densification label at cutoff T."""
    rng = np.random.default_rng(seed)
    n = len(month)
    snap = Snapshot(edges, month, T, seed=seed)
    log(f"{graph}@{T}: snapshot, alive={len(snap.alive)}")
    ZT, locT, nT = label_sketches(edges, month, T, n)
    ZH, locH, nH = label_sketches(edges, month, T + H, n)
    log(f"{graph}@{T}: label sketches n_T={nT} n_TH={nH}")

    # ---- uniform arm: alive pairs, in the label giant component at T, not already co-cited, not linked
    pool_i, pool_j, tries = [], [], 0
    alive_ok = snap.alive[locT[snap.alive] >= 0]
    while sum(len(x) for x in pool_i) < N_UNIFORM and tries < 12:
        tries += 1
        need = N_UNIFORM - sum(len(x) for x in pool_i)
        a = rng.choice(alive_ok, int(need * 1.6) + 1000)
        b = rng.choice(alive_ok, int(need * 1.6) + 1000)
        keep = a != b
        a, b = a[keep], b[keep]
        keep = ~snap.cocited(a, b) & ~snap.linked(a, b)
        a, b = a[keep][:need], b[keep][:need]
        pool_i.append(a); pool_j.append(b)
    ui, uj = np.concatenate(pool_i), np.concatenate(pool_j)
    log(f"{graph}@{T}: uniform pairs {len(ui)} (rejected co-cited/linked in {tries} draws)")

    # ---- e3 arm: build_episode's own coupling/chain pool (and the parity check of the features)
    ep = build_episode(edges, month, T, H, seed=seed)
    sel = rng.choice(len(ep.y), min(N_E3, len(ep.y)), replace=False)
    ei, ej = ep.pairs[sel, 0], ep.pairs[sel, 1]
    chk = rng.choice(len(ep.y), min(4000, len(ep.y)), replace=False)
    Xchk = snap.features(ep.pairs[chk, 0], ep.pairs[chk, 1])
    denom = np.maximum(np.abs(ep.X[chk]), 1e-9)
    parity = {f: round(float(np.max(np.abs(Xchk[:, k] - ep.X[chk, k]) / denom[:, k])), 6)
              for k, f in enumerate(FEATURES)}
    log(f"{graph}@{T}: e3 pool {len(ep.y)} -> {len(ei)}; feature parity max rel dev {max(parity.values()):.2g}")

    i = np.concatenate([ui, ei]); j = np.concatenate([uj, ej])
    src = np.concatenate([np.zeros(len(ui), np.int8), np.ones(len(ei), np.int8)])
    X = snap.features(i, j)

    # ---- the label
    a, b = locT[i], locT[j]
    ah, bh = locH[i], locH[j]
    ok = (a >= 0) & (b >= 0) & (ah >= 0) & (bh >= 0)
    rho = np.full(len(i), np.nan)
    dt = ZT[a[ok]] - ZT[b[ok]]
    dh = ZH[ah[ok]] - ZH[bh[ok]]
    RT = np.einsum("ij,ij->i", dt, dt)
    RH = np.einsum("ij,ij->i", dh, dh)
    rho[ok] = RH / np.maximum(RT, 1e-300)
    good = np.isfinite(rho) & (rho > 0)
    rt_full = np.full(len(i), np.nan); rt_full[ok] = RT
    cell = dict(graph=graph, T=T, i=i[good], j=j[good], X=X[good], src=src[good],
                rho=rho[good], y=(rho[good] < 0.5), logrho=np.log(rho[good]), R_T=rt_full[good])
    cell["dropped_share"] = float(1 - good.mean())
    cell["parity"] = parity
    cell["base_rate"] = float(cell["y"].mean())
    cell["base_rate_uniform"] = float(cell["y"][cell["src"] == 0].mean())
    cell["base_rate_e3"] = float(cell["y"][cell["src"] == 1].mean())
    cell["median_rho"] = float(np.median(cell["rho"]))
    log(f"{graph}@{T}: n={len(cell['y'])} bridged={cell['base_rate']:.4f} "
        f"(uniform {cell['base_rate_uniform']:.4f} / e3 {cell['base_rate_e3']:.4f}) median rho={cell['median_rho']:.3f}")
    return cell


def metrics(cell, s, mask=None):
    y, lr = cell["y"], cell["logrho"]
    if mask is not None:
        y, lr, s = y[mask], lr[mask], s[mask]
    return dict(ap=round(average_precision(y, s), 4), p_at_1000=round(precision_at(y, s, 1000), 4),
                spearman_logrho=round(float(spearmanr(s, lr).statistic), 4))


def decile_share(cell, s, mask=None):
    """The e27d reading: bridged share among the top decile of the score."""
    y = cell["y"]
    if mask is not None:
        y, s = y[mask], s[mask]
    k = max(1, int(0.1 * len(s)))
    return round(float(y[np.argsort(-s, kind="stable")[:k]].mean()), 4)


graphs = ["HepTh", "HepPh"]
cells = {}
for g in graphs:
    edges, month, _ = snap_citations(g)
    for T in CUTOFFS:
        try:
            cells[(g, T)] = make_cell(g, edges, month, T)
        except Exception as exc:                                  # a graph that is too small/slow must not lose the rest
            log(f"{g}@{T} FAILED: {exc!r}")
            res.setdefault("failed", []).append(f"{g}@{T}: {exc!r}")
            save()
            continue
    if not all((g, T) in cells for T in CUTOFFS):
        continue

    out = {}
    for train_T, test_T in [(CUTOFFS[0], CUTOFFS[1]), (CUTOFFS[1], CUTOFFS[0])]:
        tr, te = cells[(g, train_T)], cells[(g, test_T)]
        r = {"n_train": int(len(tr["y"])), "n_test": int(len(te["y"])),
             "base_rate_bridged": round(te["base_rate"], 4),
             "base_rate_bridged_uniform": round(te["base_rate_uniform"], 4),
             "base_rate_bridged_e3_pool": round(te["base_rate_e3"], 4),
             "median_rho_test": round(te["median_rho"], 4),
             "dropped_share_test": round(te["dropped_share"], 4),
             "feature_parity_vs_build_episode": te["parity"]}
        rng = np.random.default_rng(0)
        base_scores = {f"single:+{f}": te["X"][:, k] for k, f in enumerate(FEATURES)}
        base_scores["single:-age_gap"] = -te["X"][:, FEATURES.index("age_gap")]
        base_scores["random"] = rng.random(len(te["y"]))
        pol = ReplayPolicy().fit(tr["X"], tr["y"])
        base_scores["replay:all"] = pol.score(te["X"])
        pol_u = ReplayPolicy().fit(tr["X"][tr["src"] == 0], tr["y"][tr["src"] == 0])
        base_scores["replay:uniform_pool_only"] = pol_u.score(te["X"])
        grown, picked = grow_span(tr["X"], tr["y"], rounds=8)
        base_scores["replay:grow_span(8)"] = grown.score(te["X"])
        for name, s in base_scores.items():
            r[name] = metrics(te, s)
            r[name]["bridged_share_top_decile"] = decile_share(te, s)
            r[name]["ap_uniform_pool"] = round(average_precision(te["y"][te["src"] == 0], s[te["src"] == 0]), 4)
        r["replay:all:weights"] = pol.weights()
        r["replay:uniform_pool_only:weights"] = pol_u.weights()
        r["grow_span:picked"] = [f"{FEATURES[a]}*{FEATURES[b]}" for a, b in picked]
        # continuous target: linear fit on log rho (does the label's continuous part carry more)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        prep = ReplayPolicy()._prep
        lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(prep(tr["X"]), tr["logrho"])
        s_lin = -lin.predict(prep(te["X"]))
        r["ridge_on_logrho"] = metrics(te, s_lin)
        r["ridge_on_logrho"]["bridged_share_top_decile"] = decile_share(te, s_lin)
        r["ridge_on_logrho"]["weights"] = {f: round(float(w), 3) for f, w in zip(FEATURES, lin[-1].coef_)}
        out[f"train{train_T}->test{test_T}"] = r
        log(f"{g} train{train_T}->test{test_T}: replay AP {r['replay:all']['ap']} "
            f"grow {r['replay:grow_span(8)']['ap']} resistance {r['single:+resistance']['ap']} "
            f"base {r['base_rate_bridged']}")
        res[g] = out
        save()

save()
log("written", OUT)
