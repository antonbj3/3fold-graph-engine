#!/usr/bin/env python3
"""
replay_policy.py — learn "which connection to investigate next" by replaying a graph's own history.

What it does:

  A dated graph is cut at a past month T. Everything after T is hidden. Every candidate pair of
  existing nodes gets a feature vector computed from the graph ≤ T only. The hidden future supplies
  the label: was the pair brought together (co-cited by a new node) within `horizon` months?
  That is one EPISODE. A scoring policy is fitted on episodes whose future has already happened and
  is then applied at the present cutoff, where the future is unknown.

  The recursive part: after each fit, the policy is scored per SUBGRAPH (community × cutoff). The next
  batch of replay episodes is drawn where the policy ranks worst, and the policy is refitted.
  Whether that beats drawing episodes uniformly is an experiment (examples/engine_experiments/e3_replay_policy.py), not
  an assumption.

Features (all from the graph ≤ T; no text, no embeddings of text):
  cn_coupling   papers both cite                      cn_chain    i→k→j or j→k→i paths
  adamic_adar   Σ 1/log(deg k) over common neighbours  pref        log indeg_i + log indeg_j
  resistance    ‖z_i − z_j‖² from resistance_sketch    hole        L⁺_ii + L⁺_jj (the hole-engine field)
  age_gap       |month_i − month_j|                    recency     months since the younger node appeared
  vel           log(1+citations gained in the last 12 months), summed over the pair
  straight      linear extrapolation of log(1+indeg) one horizon ahead, summed over the pair
                (the "temporal straightening" seed: if growth is straight in log space, the
                extrapolated position carries what the current position does not)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .resistance_sketch import ResistanceSketch

FEATURES = ["cn_coupling", "cn_chain", "adamic_adar", "pref", "resistance", "hole",
            "age_gap", "recency", "vel", "straight"]


@dataclass
class Episode:
    cutoff: int
    pairs: np.ndarray          # p×2 node indices
    X: np.ndarray              # p×len(FEATURES)
    y: np.ndarray              # p bool
    community: np.ndarray      # p int, community of pairs[:,0] at the cutoff


def build_episode(edges: np.ndarray, month: np.ndarray, cutoff: int, horizon: int, n_rows: int = 3000,
                  max_pairs: int = 400_000, seed: int = 0, k_sketch: int = 64,
                  n_communities: int = 12, lookback: int = 12) -> Episode:
    """edges: m×2 [citing, cited]; month: per node. Uses edges whose citing node is ≤ cutoff."""
    rng = np.random.default_rng(seed)
    n = len(month)
    past = month[edges[:, 0]] <= cutoff
    fut = (month[edges[:, 0]] > cutoff) & (month[edges[:, 0]] <= cutoff + horizon) & (month[edges[:, 1]] <= cutoff)
    ones = lambda e: sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    P, F = ones(edges[past]), ones(edges[fut])              # P[i,j]=1: i cites j
    P.data[:] = 1; F.data[:] = 1
    indeg = np.asarray(P.sum(0)).ravel()
    alive = np.flatnonzero((month <= cutoff) & (indeg + np.asarray(P.sum(1)).ravel() > 0))
    rows = rng.choice(alive, min(n_rows, len(alive)), replace=False)
    n_communities = max(2, min(n_communities, len(alive) // 50))

    A = ((P + P.T) > 0).astype(float).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    coupling = (P[rows] @ P.T).tocsr()                      # both cite the same paper
    chain = (P[rows] @ P + P.T[rows] @ P.T).tocsr()
    w = sp.diags(1.0 / np.log(np.maximum(deg, 2)))
    aa = (A[rows] @ w @ A).tocsr()
    cocited = (P.T[rows] @ P).tocsr()                       # already co-cited → not a candidate
    future = (F.T[rows] @ F).tocsr()

    cand = ((coupling + chain) > 0).astype(np.int8) - ((cocited + A[rows]) > 0).astype(np.int8)
    cand = (cand > 0).tocoo()
    keep = rows[cand.row] != cand.col
    ri, cj = cand.row[keep], cand.col[keep]
    if len(ri) > max_pairs:
        s = rng.choice(len(ri), max_pairs, replace=False); ri, cj = ri[s], cj[s]
    i, j = rows[ri], cj
    get = lambda M: np.asarray(M[ri, cj]).ravel()

    # resistance geometry on the largest component ≤ cutoff
    _, lab = connected_components(A, directed=False)
    big = np.bincount(lab[alive]).argmax()
    comp = np.flatnonzero(lab == big)
    remap = -np.ones(n, np.int64); remap[comp] = np.arange(len(comp))
    und = sp.triu(A, 1).tocoo()
    e = np.stack([remap[und.row], remap[und.col]], 1); e = e[(e >= 0).all(1)]
    sk = ResistanceSketch.build(len(comp), e, k=k_sketch, seed=seed)
    Z = np.zeros((n, k_sketch)); Z[comp] = sk.Z
    incomp = (remap[i] >= 0) & (remap[j] >= 0)
    d = Z[i] - Z[j]
    res = np.where(incomp, np.einsum("ij,ij->i", d, d), np.nan)
    res = np.where(np.isnan(res), np.nanmax(res), res)
    hole_node = np.einsum("ij,ij->i", Z, Z)
    hole_node[remap < 0] = hole_node[comp].max()

    # trajectories: indegree now, 12 and 24 months ago
    def indeg_at(t):
        m = month[edges[:, 0]] <= t
        return np.bincount(edges[m, 1], minlength=n).astype(float)
    d0, d1 = indeg_at(cutoff), indeg_at(cutoff - lookback)
    l0, l1 = np.log1p(d0), np.log1p(d1)
    vel = np.log1p(d0 - d1)
    straight = l0 + (l0 - l1) * (horizon / float(lookback))

    # communities = spectral-free, cheap: k-means on the resistance vectors (same geometry as above)
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_communities, n_init=3, random_state=seed).fit(Z[comp] / (np.linalg.norm(Z[comp], axis=1, keepdims=True) + 1e-12))
    comm = -np.ones(n, np.int64); comm[comp] = km.labels_

    X = np.stack([get(coupling), get(chain), get(aa), np.log1p(d0[i]) + np.log1p(d0[j]), res,
                  hole_node[i] + hole_node[j], np.abs(month[i] - month[j]).astype(float),
                  (cutoff - np.maximum(month[i], month[j])).astype(float), vel[i] + vel[j],
                  straight[i] + straight[j]], 1)
    return Episode(cutoff, np.stack([i, j], 1), X, get(future) > 0, comm[i])


class ReplayPolicy:
    """Logistic scoring of candidate pairs, fitted on replayed episodes."""

    def __init__(self, features: list[str] | None = None, C: float = 1.0):
        self.cols = [FEATURES.index(f) for f in (features or FEATURES)]
        self.C = C
        self.model = None

    def _prep(self, X):
        X = X[:, self.cols].copy()
        return np.sign(X) * np.log1p(np.abs(X))

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "ReplayPolicy":
        """sample_weight = 1/inclusion probability when the labels come only from the throws that were made."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=500))
        kw = {} if sample_weight is None else {"logisticregression__sample_weight": sample_weight}
        self.model.fit(self._prep(X), y, **kw)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_grown", None) is not None:
            sc, picked, mu, sd = self._grown; A = sc.transform(self._prep(X))
            extra = (np.stack([A[:, i] * A[:, j] for i, j in picked], 1) - mu) / sd
            return self.model.decision_function(np.c_[A, extra])
        return self.model.decision_function(self._prep(X))

    def weights(self) -> dict[str, float]:
        return {FEATURES[c]: round(float(w), 3) for c, w in zip(self.cols, self.model[-1].coef_[0])}


def grow_span(X: np.ndarray, y: np.ndarray, rounds: int = 8, C: float = 1.0) -> tuple["ReplayPolicy", list[tuple[int, int]]]:
    """Recursion on the residual, not on the choice of samples. The policy is one hyperplane in feature space, and
    replay showed it is the same hyperplane on every subgraph and on another graph, so picking WHICH episodes to
    replay cannot improve it (measured: equal to or worse than uniform). What can: each round fits the policy, takes
    the residual y − p, projects every candidate direction (pairwise products and squares of the features) onto the
    orthogonal complement of the current span, and adds the one most correlated with the residual.
    Measured on two public citation graphs × two cutoffs: +0.010 to +0.022 average precision over the base policy,
    1.8–2.3 times the gain from adding the same number of random directions. Which direction is picked first differs
    between graphs at equal gain (pref*vel on one, recency² on the other): the direction is not identified, only
    the gain is. Returns the fitted policy and the list of (i, j) feature-index pairs that were added."""
    from itertools import combinations_with_replacement
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    base = ReplayPolicy(C=C); Z = base._prep(X); sc = StandardScaler().fit(Z); A = sc.transform(Z)
    pool = list(combinations_with_replacement(range(A.shape[1]), 2))
    cand = np.stack([A[:, i] * A[:, j] for i, j in pool], 1); mu, sd = cand.mean(0), cand.std(0) + 1e-12; cand = (cand - mu) / sd
    S, picked = A, []
    for _ in range(rounds):
        m = LogisticRegression(C=C, max_iter=400).fit(S, y); res = y - m.predict_proba(S)[:, 1]
        Q, _ = np.linalg.qr(np.c_[np.ones(len(S)), S]); free = [k for k in range(len(pool)) if pool[k] not in picked]
        Cc = cand[:, free]; Cc = Cc - Q @ (Q.T @ Cc)
        k = free[int(np.argmax(np.abs(Cc.T @ res) / (np.linalg.norm(Cc, axis=0) + 1e-12)))]
        picked.append(pool[k]); S = np.c_[S, cand[:, k]]
    base.model = LogisticRegression(C=C, max_iter=400).fit(S, y)
    idx = [pool.index(pq) for pq in picked]
    base._grown = (sc, picked, mu[idx], sd[idx])
    return base, picked


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(-s, kind="stable"); y = y[o]
    if y.sum() == 0:
        return float("nan")
    return float((np.cumsum(y) / (np.arange(len(y)) + 1))[y].mean())


def precision_at(y: np.ndarray, s: np.ndarray, k: int) -> float:
    return float(y[np.argsort(-s, kind="stable")[:k]].mean())
