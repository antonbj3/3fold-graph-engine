#!/usr/bin/env python3
"""
throws.py — propose NEW links: far apart in the graph, far apart in subject, close in mechanism; drawn with KNOWN probabilities.

Three distances, kept separate because they answer different questions:
    graph      resistance distance R_ij (resistance_sketch): how weakly the two nodes are already connected
    subject    1 − similarity of their descriptions (any text measure, optional): same field or not
    mechanism  mechanism_signature.distance, or any other computed signature distance: same way of behaving or not
A throw is worth making when the first two are LARGE and the third is SMALL:
        score_ij = −mechanism_ij / τ  +  a·log(R_ij / median R)  +  b·log(subject_ij + ε)

Choice rule. The K pairs are not the top K. They are drawn without replacement with probability
        π_ij = (1 − λ)·softmax(score / T)_ij + λ / N ,
and each drawn pair is returned with its inclusion probability. Reason (measured on two public citation graphs,
examples/engine_experiments/e8, e8b): when the outcome of a throw is known only for the throws that were made, top-K
selection finds the most links during the run but the model fitted on those outcomes ranks an unseen period at
AP 0.130, against 0.162 for uniformly random throws; softmax draws (T = 0.5) with a 1/π-weighted fit reach 0.163 while finding
4.7 times as many links as random (on the second graph: 2.9 times, 0.190 against 0.192). Top-K gives π ∈ {0, 1}: no weight can correct for pairs that could never be drawn.

Sets. A first reading of e19 (real citation data) said a TIGHT core of references plus ONE far element predicts impact. Controlled for
the references' degree it does not (resistance distance ≈ 1/deg + 1/deg, so "tight core" was "cites hubs"), and e24's set value
½ log det(I + H C Hᵀ/σ²) (precision_form) picks three far elements with low mutual coherence over any core-plus-one triple. So a throw
as a SET is drawn here by the determinantal rule below: sets ∝ det(K_S), the volume the members span in the resistance geometry —
far apart AND in independent directions — with EXACT inclusion probabilities from the marginal kernel K(I + K)⁻¹, which is what a
1/π-weighted fit needs (e8b) and what softmax-with-floor only approximated. Continuous geometry, discrete outcome, decoded late.
"""
from __future__ import annotations

import numpy as np

__all__ = ["throw_scores", "draw", "draw_pairs", "inclusion_probabilities", "draw_set_dpp", "dpp_inclusion_probabilities", "throw_from_continuum", "densify_sequential", "throw_at_point_dpp", "chain_throw"]


def throw_scores(mechanism: np.ndarray, graph: np.ndarray, subject: np.ndarray | None = None,
                 tau: float = 0.25, a: float = 1.0, b: float = 1.0, eps: float = 1e-3) -> np.ndarray:
    """Scores for candidate pairs. Inputs are arrays of one shape: either n×n distance matrices (upper triangle used), or
    1-D arrays over a LIST of candidate pairs. Use the list form on large graphs — an n×n matrix at n = 27 400 is 6 GB —
    with distances taken per pair (resistance_sketch.resistance(i, j) needs no matrix)."""
    mechanism, graph = np.asarray(mechanism, float), np.asarray(graph, float)
    g = graph[np.triu_indices(len(graph), 1)] if graph.ndim == 2 else graph
    med = np.median(g[np.isfinite(g) & (g > 0)])
    s = -mechanism / tau + a * np.log(np.maximum(graph, 1e-12) / med)
    if subject is not None:
        s = s + b * np.log(subject + eps)
    return s


def inclusion_probabilities(p: np.ndarray, k: int) -> np.ndarray:
    """π_i = min(1, c·p_i) with c chosen so that Σπ = k (units that would exceed 1 are taken with certainty and the rest
    rescaled). These are the EXACT inclusion probabilities of `draw`."""
    p = np.asarray(p, float) / np.sum(p); pi = np.minimum(k * p, 1.0)
    for _ in range(len(p)):
        sure = pi >= 1.0; rest = k - sure.sum()
        if rest <= 0 or not (~sure).any():
            break
        new = np.where(sure, 1.0, np.minimum(rest * p / p[~sure].sum(), 1.0))
        if np.allclose(new, pi):
            break
        pi = new
    return pi


def draw_pairs(i: np.ndarray, j: np.ndarray, scores: np.ndarray, k: int, temperature: float = 1.0, floor: float = 0.05,
               seed: int = 0) -> list[tuple[int, int, float]]:
    """k of the listed candidate pairs, as (i, j, π). Systematic sampling on a random permutation: fixed size k, and
    P(pair drawn) = π exactly (Madow 1949). A first version drew sequentially without replacement and reported min(k·p, 1),
    which is not the inclusion probability of that scheme (review: Σπ = 56.9 for k = 60, up to 1.43× off per pair)."""
    i, j, s = np.asarray(i), np.asarray(j), np.asarray(scores, float) / temperature
    p = np.exp(s - s.max()); p = (1 - floor) * p / p.sum() + floor / len(p)
    k = min(k, len(p)); pi = inclusion_probabilities(p, k)
    rng = np.random.default_rng(seed); order = rng.permutation(len(p)); cum = np.cumsum(pi[order])
    pick = order[np.minimum(np.searchsorted(cum, rng.random() + np.arange(k), side="right"), len(p) - 1)]
    return [(int(i[t]), int(j[t]), float(pi[t])) for t in pick]


def draw(scores: np.ndarray, k: int, exclude: np.ndarray | None = None, temperature: float = 1.0, floor: float = 0.05,
         seed: int = 0) -> list[tuple[int, int, float]]:
    """`draw_pairs` over the upper triangle of an n×n score matrix; `exclude` masks pairs that are already linked."""
    iu = np.triu_indices(len(scores), 1); ok = np.ones(len(iu[0]), bool) if exclude is None else ~exclude[iu]
    return draw_pairs(iu[0][ok], iu[1][ok], scores[iu][ok], k, temperature, floor, seed)


# -- throws as SETS: determinantal draws in the resistance geometry ----------------------------------------------------
def _dpp_kernel(Z: np.ndarray, quality: np.ndarray | None = None) -> np.ndarray:
    """L-ensemble kernel L = Q Φ Φᵀ Q with Φ = rows of Z (resistance-sketch coordinates, unit-normalized) and Q = diag(quality)."""
    Phi = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    q = np.ones(len(Z)) if quality is None else np.asarray(quality, float)
    return (q[:, None] * Phi) @ (q[:, None] * Phi).T


def dpp_inclusion_probabilities(Z: np.ndarray, quality: np.ndarray | None = None) -> np.ndarray:
    """P(i ∈ S) under the L-ensemble: diag(L (I + L)⁻¹). Exact; sums to E|S|."""
    L = _dpp_kernel(Z, quality)
    return np.diag(L @ np.linalg.inv(np.eye(len(L)) + L))


def draw_set_dpp(Z: np.ndarray, quality: np.ndarray | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One set S ~ DPP(L) by the spectral algorithm (Hough et al. 2006; Kulesza & Taskar 2012, alg. 1). Returns (members,
    inclusion probabilities of the members). P(S) ∝ det(L_S): members far apart in resistance geometry and spanning independent
    directions are favoured jointly; two near-duplicates are almost never drawn together."""
    rng = np.random.default_rng(seed)
    L = _dpp_kernel(Z, quality); lam, V = np.linalg.eigh(L); lam = np.clip(lam, 0, None)
    keep = rng.random(len(lam)) < lam / (1 + lam)
    Vk = V[:, keep]; chosen = []
    while Vk.shape[1] > 0:
        p = (Vk ** 2).sum(1); p /= p.sum()
        i = int(rng.choice(len(p), p=p)); chosen.append(i)
        # project the remaining eigenvectors onto the orthogonal complement of e_i
        j = int(np.argmax(np.abs(Vk[i])))
        v = Vk[:, j] / Vk[i, j]
        Vk = Vk - np.outer(v, Vk[i]); Vk = np.delete(Vk, j, axis=1)
        if Vk.shape[1]:
            Vk, _ = np.linalg.qr(Vk)
    chosen = np.array(sorted(chosen), int)
    return chosen, dpp_inclusion_probabilities(Z, quality)[chosen]


# -- throws from the continuum: OED point, dither, decode at the corners -----------------------------------------------
def throw_from_continuum(Z: np.ndarray, x_star: np.ndarray, k: int = 3, dither: float | None = None, n_draws: int = 1,
                         n_pi: int = 2000, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """A throw as a point in the resistance geometry, decoded late. `x_star` is the point an experimental-design rule chose
    (a hole: where the posterior variance is largest, or a midpoint between weakly connected regions); the throw is the set of
    the k nodes nearest to x_star + dither·ε, ε ~ N(0, I). Without dither the decode is deterministic (π ∈ {0, 1}: the same
    corners every time, and no 1/π weight can correct for a node that can never be drawn). With dither on the scale of the
    node spacing every node within reach has π > 0 and the decode no longer depends on where x_star sits relative to the grid. Exact
    independence of the quantization error (Schuchman's condition) needs dither UNIFORM over the Voronoi cell; Gaussian
    dither on the spacing scale is an approximation of that, and the centroid's unbiasedness is measured (test), not proven. Default dither = half the median
    pairwise distance among the 2k nodes nearest x_star (the neighbourhood's scale). Returns [(members, their inclusion probabilities)] for n_draws throws;
    inclusion probabilities are estimated by n_pi Monte-Carlo decodes of the same dithered point (exact in the limit)."""
    rng = np.random.default_rng(seed)
    d0 = np.linalg.norm(Z - x_star, axis=1)
    if dither is None:                                       # the NEIGHBOURHOOD's own scale: distances among the 2k nearest nodes,
        near = np.argsort(d0)[: 2 * k]                        # not the distance from x_star (which can be far from every node;
        D = np.linalg.norm(Z[near][:, None] - Z[near][None], axis=2)   # e27 measured that choice to scatter the decode anywhere)
        dither = 0.5 * float(np.median(D[np.triu_indices(len(near), 1)]))
    def decode(x):
        return np.argsort(np.linalg.norm(Z - x, axis=1), kind="stable")[:k]
    counts = np.zeros(len(Z))
    for _ in range(n_pi):
        counts[decode(x_star + dither * rng.standard_normal(Z.shape[1]))] += 1
    pi = counts / n_pi
    out = []
    for _ in range(n_draws):
        S = np.sort(decode(x_star + dither * rng.standard_normal(Z.shape[1])))
        out.append((S, pi[S]))
    return out


# -- sequential densification: throw, observe, update the geometry, throw again ----------------------------------------
def densify_sequential(form, candidates: np.ndarray, rounds: int, sigma: float = 1.0, observe=None, seed: int = 0) -> list[dict]:
    """The loop that densifies a graph with sets: at each round the set is drawn from the determinantal process whose kernel is
    the CURRENT posterior covariance of the candidate nodes, L = C_S/σ² (precision_form; the same kernel whose log det is the
    set value), so P(set) ∝ det(C_S/σ²): nodes still uncertain and in independent directions are drawn together, nodes already
    pinned by earlier throws are not drawn again. `observe(set) -> [(h, sigma, y)]` returns what the throw taught (e.g. for a
    confirmed link between i and j the contrast e_i − e_j with its noise); each is a rank-1 update of the form
    (Sherman–Morrison), after which the kernel — and every inclusion probability — is exact again. Returns per round the
    members, their exact inclusion probabilities under that round's kernel, and the bits the form lost (the value realized)."""
    rng = np.random.default_rng(seed); cand = np.asarray(candidates, np.int64); out = []
    for r in range(rounds):
        C = form.cov()[np.ix_(cand, cand)] / sigma ** 2
        lam, V = np.linalg.eigh((C + C.T) / 2); lam = np.clip(lam, 0, None)
        keep = rng.random(len(lam)) < lam / (1 + lam); Vk = V[:, keep]; chosen = []
        while Vk.shape[1] > 0:
            p = (Vk ** 2).sum(1); p /= p.sum(); i = int(rng.choice(len(p), p=p)); chosen.append(i)
            j = int(np.argmax(np.abs(Vk[i]))); v = Vk[:, j] / Vk[i, j]
            Vk = Vk - np.outer(v, Vk[i]); Vk = np.delete(Vk, j, axis=1)
            if Vk.shape[1]:
                Vk, _ = np.linalg.qr(Vk)
        pi = np.diag(C @ np.linalg.inv(np.eye(len(C)) + C))
        S = cand[np.array(sorted(chosen), int)]
        before = float(np.linalg.slogdet(np.eye(len(C)) + C)[1]) / (2 * np.log(2))
        if observe is not None and len(S):
            for h, s, y in observe(S):
                form.observe(h, s, y)
        C2 = form.cov()[np.ix_(cand, cand)] / sigma ** 2
        after = float(np.linalg.slogdet(np.eye(len(C2)) + C2)[1]) / (2 * np.log(2))
        out.append({"round": r, "members": S, "inclusion": pi[np.array(sorted(chosen), int)] if len(chosen) else np.array([]),
                    "bits_realized": before - after})
    return out


def throw_at_point_dpp(Z: np.ndarray, x_star: np.ndarray, scale: float | None = None, k: int | None = None,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The exact form of a throw at a design point: instead of dithering x_star and quantizing (no exact condition on an
    irregular node set), put the design point into the QUALITY of a determinantal process, q_i = exp(−‖z_i − x*‖²/(2·scale²)),
    L = q Φ Φᵀ q, P(S) ∝ det(L_S). Every inclusion probability is then exact in closed form (diag L(I+L)⁻¹), the members are
    spread over independent directions by the determinant, and the design point is the bump, not a noisy sample. Default
    scale = median distance from x* to its 2k (or 6) nearest nodes. If k is given, the drawn set is trimmed to its k members
    with the highest inclusion probability (a k-DPP would be exact; this is the cheap approximation and is labelled so)."""
    d = np.linalg.norm(Z - x_star, axis=1)
    if scale is None:
        scale = float(np.median(np.sort(d)[: 2 * (k or 3)]))
    q = np.exp(-d ** 2 / (2 * scale ** 2))
    near = np.flatnonzero(q > 1e-6)                                   # nodes with any weight: keeps the eigen-decomposition small
    if len(near) < 2:
        near = np.argsort(d)[: max(2 * (k or 3), 6)]
    S, pi = draw_set_dpp(Z[near], quality=q[near], seed=seed)
    S, pi = near[S], pi
    if k is not None and len(S) > k:
        top = np.argsort(-pi)[:k]; S, pi = S[top], pi[top]
    return S, pi


def chain_throw(Z: np.ndarray, a: int, b: int, steps: int = 3, dither: float = 0.0, seed: int = 0) -> np.ndarray:
    """A chain throw a → · → · → b: the segment from z_a to z_b sampled at t = 1/(steps+1) … steps/(steps+1), each point
    decoded to its nearest node (optionally dithered). Measured (e27b, cit-HepTh): the only rule that finds LONG future links
    above random — 2.0 % of chains between far pairs contain a future co-citation (random triples 0.7 %, far pairs by softmax 0 %),
    with within-set resistance 1.2. A long throw pays when it is decoded as a path, not as a pair: the intermediate nodes are the
    ones a new paper can cite together with one end. Returns the sorted unique node ids of the chain including both ends."""
    rng = np.random.default_rng(seed); za, zb = Z[a], Z[b]; nodes = [int(a), int(b)]
    for t in np.linspace(0, 1, steps + 2)[1:-1]:
        x = za + t * (zb - za) + (dither * rng.standard_normal(Z.shape[1]) if dither > 0 else 0.0)
        nodes.append(int(np.argmin(np.linalg.norm(Z - x, axis=1))))
    return np.array(sorted(set(nodes)), int)
