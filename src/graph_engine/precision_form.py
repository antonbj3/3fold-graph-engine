#!/usr/bin/env python3
"""
precision_form.py — one Gaussian precision form J behind three rankings that were separate numbers.

The three currencies in this engine are the same object read at three places. Write a Gaussian
N(μ, J⁺) over a vector x of node potentials, requirement margins and beliefs, and assemble J
additively from blocks:

  (a) STRUCTURE. J ⊇ graph Laplacian L (a Gaussian free field). Then for a contrast v = e_i − e_j,
      vᵀ J⁺ v = R_ij, the effective resistance of resistance_sketch — the marginal variance of the
      potential DIFFERENCE. "Weakly connected" and "uncertain difference" are one quantity
      (T1: max |vᵀ J⁺ v − R_ij| from the exact pinv, random graphs).
  (b) MEASUREMENT. A set of reports y = A m + e with Cov(e) = Σ = B Bᵀ from declared lineage
      (margin_net) contributes J += AᵀΣ⁺A, b += AᵀΣ⁺y. For A = 1 this is exactly GLS:
      mean = (1ᵀΣ⁺y)/(1ᵀΣ⁺1) = m̂ and variance = 1/(1ᵀΣ⁺1) = s², including the singular-Σ
      (declared copies) case, because both sides take the same pseudo-inverse (T2).
  (c) BELIEF. A Bernoulli belief p enters by its Fisher information for the mean parameter,
      J_pp += 1/(p(1−p)), i.e. C_pp = p(1−p). This block is a LOCAL approximation; it is the only
      inexact step in the file, and T4 measures how inexact.

OBSERVE. Any new observation is h ᵀx + noise(σ): J' = J + h hᵀ/σ². On the covariance this is
Sherman–Morrison, C' = C − C h hᵀ C /(σ² + hᵀ C h) — the same algebra resistance_sketch uses for the
Kirchhoff drop of an added edge. With a singular J (a Laplacian has null vector 1) the update is
exact iff h ∈ range(J); a component in the null space SHRINKS the null space and Sherman–Morrison is
then wrong, so `observe` detects it and recomputes the pseudo-inverse (T3 pins both branches).

VALUE. Potential = differential entropy ½ log det C (up to a constant), quantities of interest
selected by Q (C ↦ Q C Qᵀ). By the matrix determinant lemma det(J + hhᵀ/σ²) = det J·(1 + hᵀCh/σ²), so
    value(h) = ½ log₂(1 + hᵀ C h / σ²)  bits,                                        (T4, exact)
and restricted to Q, value_Q(h) = −½ log₂(1 − uᵀ(QCQᵀ)⁺u/(σ² + hᵀCh)), u = QCh, which is ≤ the full
value (information never increases under marginalization; T4 checks both).
For a Bernoulli quantity, C = p(1−p) and the Gaussian value is a monotone function of
t = hᵀCh/σ², while regime_posterior's Jensen form Var(Δp)/(2 ln2·p(1−p)) equals (1/2ln2)·t/(1+t):
both strictly increasing in t, so the two RANK candidates identically (T4: Spearman 0.9999 over 65
candidates, the deficit is numerical ties). Against the EXACT two-outcome entropy drop of a
reliability-r answer the Gaussian value is only an approximation: Spearman 0.9996 but up to 43 %
off in value (median 15 %). That is the one place in this file where the unification is not exact.

SET (the "throw as a set" throws.py leaves unbuilt). For a set S with rows H and noises D = diag(σ),
    value(S) = ½ log₂ det(I + D⁻¹ H C Hᵀ D⁻¹)  bits = I(x; y_S),
which telescopes into sequential single values (chain rule, T5), is monotone, and is submodular:
the marginal gain of a candidate is computed in the posterior covariance after S, which only
shrinks. Greedy is therefore within 1 − 1/e of the best set of size k (Nemhauser et al. 1978);
T5 measures the realized greedy/brute ratio instead of trusting the bound (100 instances, 8
candidates, k = 3: greedy found the optimum in all 100 — the bound is loose here). For throws:
a set scores high when its elements are far apart in resistance AND point in independent
directions. Measured on a two-cluster graph (T5/e24), that does NOT reproduce the "tight core plus
one far element" shape throws.py expects: when far pairs dominate the candidate pool the best
triple is usually three far pairs. What it does instead is give up resistance for independence —
strictly lower mean R, strictly lower mutual coherence and strictly more bits than the three
highest-R candidates, in every instance tested.

COST. Every kind ends in bits; divide by cost for one ranking across structure probes, new reports
and belief probes (T6). What the joint object decides that the parts do not: the top-1 by resistance
alone, by margin z alone and by regime value alone are three different candidates, and the bits/cost
ranking picks one of them for a reason the other two cannot see.

FIVE READINGS OF ONE MATRIX. Five modules read the same posterior covariance C = J⁺ at five places;
here they are five thin methods over the one C, no second implementation of the algebra (T7–T11):
    hole_field()      diag(C) = L⁺_ii           — graph_hole_engine.hole_field, resistance_sketch.hole_field
    resistance(i,j)   vᵀCv, v = e_i−e_j          — resistance_sketch.resistance; edge_leverage() = w_e R_e is
                                                  resistance_sketch.edge_leverage, Σ over a connected graph = n−1
    throw_set(...)    DPP with kernel Q C_S Q/σ² — throws.draw_set_dpp / densify_sequential / throw_at_point_dpp
    observe_link(i,j,w)  rank-1 J += w vvᵀ       — resistance_sketch's Sherman–Morrison edge addition
    sparsify(...)     leverage sampling of edges — resistance_sketch.sparsify (Spielman–Srivastava)
Two POTENTIALS live on the same update and must not be confused: the set/link value maximizes the
ENTROPY of the Gaussian, ½log det C — one added edge drops it by ½log(1 + w R_ij); the Kirchhoff
index maximizes the TRACE, n·tr C — the same edge drops it by n·w‖C v‖²/(1 + w R_ij)
(resistance_sketch.kirchhoff_drop). Same C, same rank-1 update, different functional: a log-det
ranking and a trace ranking of candidate links are not the same ranking (T10).

numpy/scipy only; dense — this is the exact reference form, the sketch is the scalable one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

__all__ = ["PrecisionForm", "Candidate", "set_value_bits"]

_LOG2 = np.log(2.0)


@dataclass
class Candidate:
    """One possible observation: hᵀx measured with noise sigma, at a price."""
    id: str
    h: np.ndarray
    sigma: float = 1.0
    cost: float = 1.0
    kind: str = ""


@dataclass
class PrecisionForm:
    J: np.ndarray                                   # d×d PSD precision (may be singular)
    b: np.ndarray                                   # information vector, μ = J⁺ b
    tol: float = 1e-10
    _C: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def zeros(cls, d: int, tol: float = 1e-10) -> "PrecisionForm":
        return cls(np.zeros((d, d)), np.zeros(d), tol)

    @property
    def d(self) -> int:
        return len(self.b)

    def _dirty(self) -> None:
        self._C = None

    # -- blocks ------------------------------------------------------------------------------------
    def add_laplacian(self, edges, weights=None, index=None) -> "PrecisionForm":
        """(a) structure: the graph Laplacian of `edges` on the coordinates `index` (default 0..n−1)."""
        edges = np.asarray(edges, dtype=np.int64)
        w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
        idx = np.arange(self.d) if index is None else np.asarray(index, np.int64)
        for (i, j), wij in zip(edges, w):
            a, c = int(idx[i]), int(idx[j])
            self.J[a, a] += wij; self.J[c, c] += wij
            self.J[a, c] -= wij; self.J[c, a] -= wij
        self._dirty()
        return self

    def add_measurement(self, A, y, Sigma) -> "PrecisionForm":
        """(b) reports y = A x + e, Cov(e) = Σ = B Bᵀ (margin_net lineage). Σ may be singular."""
        A = np.atleast_2d(np.asarray(A, float)); y = np.asarray(y, float).ravel()
        Sp = np.linalg.pinv(np.asarray(Sigma, float), rcond=self.tol)
        self.J += A.T @ Sp @ A
        self.b += A.T @ Sp @ y
        self._dirty()
        return self

    def add_bernoulli(self, i: int, p: float) -> "PrecisionForm":
        """(c) local Gaussian image of a Bernoulli belief: Fisher precision 1/(p(1−p))."""
        p = float(np.clip(p, 1e-9, 1 - 1e-9))
        self.J[i, i] += 1.0 / (p * (1.0 - p))
        self._dirty()
        return self

    # -- reads -------------------------------------------------------------------------------------
    def cov(self) -> np.ndarray:
        if self._C is None:
            self._C = np.linalg.pinv(self.J, rcond=self.tol, hermitian=True)
        return self._C

    def mean(self) -> np.ndarray:
        return self.cov() @ self.b

    def contrast_var(self, v) -> float:
        v = np.asarray(v, float)
        return float(v @ self.cov() @ v)

    def resistance(self, i: int, j: int, index=None) -> float:
        """R_ij = (e_i−e_j)ᵀ J⁺ (e_i−e_j) — identical to resistance_sketch's exact quantity (T1)."""
        v = np.zeros(self.d)
        a, c = (i, j) if index is None else (int(index[i]), int(index[j]))
        v[a], v[c] = 1.0, -1.0
        return self.contrast_var(v)

    def in_range(self, h) -> bool:
        """h ∈ range(J)? J J⁺ is the orthogonal projector onto range(J)."""
        h = np.asarray(h, float)
        r = h - self.J @ (self.cov() @ h)
        return bool(np.linalg.norm(r) <= 1e-6 * max(np.linalg.norm(h), 1.0))

    # -- update ------------------------------------------------------------------------------------
    def observe(self, h, sigma: float = 1.0, y: float | None = None) -> bool:
        """J += hhᵀ/σ². Returns True if the covariance was updated by Sherman–Morrison (exact when
        h ∈ range(J)), False if the null space changed and the pinv had to be recomputed (T3)."""
        h = np.asarray(h, float); s2 = float(sigma) ** 2
        ok = self.J.any() and self.in_range(h)
        C = self.cov() if ok else None
        self.J = self.J + np.outer(h, h) / s2
        if y is not None:
            self.b = self.b + h * (float(y) / s2)
        if ok:
            Ch = C @ h
            self._C = C - np.outer(Ch, Ch) / (s2 + float(h @ Ch))
        else:
            self._dirty()
        return bool(ok)

    def copy(self) -> "PrecisionForm":
        c = PrecisionForm(self.J.copy(), self.b.copy(), self.tol)
        c._C = None if self._C is None else self._C.copy()
        return c

    # -- value -------------------------------------------------------------------------------------
    def value_bits(self, h, sigma: float = 1.0, Q=None) -> float:
        """Expected drop of ½log₂det(cov) from observing hᵀx with noise σ.
        Q = None: ½log₂(1 + hᵀCh/σ²) (matrix determinant lemma). Q given: the same potential on
        Q C Qᵀ, a rank-1 downdate, always ≤ the unrestricted value."""
        h = np.asarray(h, float); s2 = float(sigma) ** 2
        C = self.cov(); Ch = C @ h
        den = s2 + float(h @ Ch)
        if Q is None:
            return float(np.log1p(float(h @ Ch) / s2) / (2 * _LOG2))
        Q = np.atleast_2d(np.asarray(Q, float))
        u = Q @ Ch
        S = np.linalg.pinv(Q @ C @ Q.T, rcond=self.tol, hermitian=True)
        frac = float(u @ S @ u) / den
        return float(-np.log1p(-min(frac, 1 - 1e-15)) / (2 * _LOG2))

    def set_value_bits(self, H, sigma=1.0) -> float:
        """½log₂det(I + D⁻¹ H C Hᵀ D⁻¹), D = diag(σ): the joint value of observing a SET (T5)."""
        H = np.atleast_2d(np.asarray(H, float))
        if H.size == 0:
            return 0.0
        s = np.broadcast_to(np.asarray(sigma, float), (len(H),))
        M = (H @ self.cov() @ H.T) / np.outer(s, s)
        sign, ld = np.linalg.slogdet(np.eye(len(H)) + M)
        return float(ld / (2 * _LOG2)) if sign > 0 else float("-inf")

    def value_of(self, cands: list[Candidate]) -> float:
        return self.set_value_bits(np.array([c.h for c in cands]), np.array([c.sigma for c in cands]))

    # -- ranking -----------------------------------------------------------------------------------
    def rank(self, cands: list[Candidate]) -> list[tuple[str, float, float, str]]:
        """(id, bits, bits/cost, kind) sorted by bits per cost — one currency for every kind (T6)."""
        out = [(c.id, self.value_bits(c.h, c.sigma), self.value_bits(c.h, c.sigma) / c.cost, c.kind) for c in cands]
        return sorted(out, key=lambda t: -t[2])

    def best_set(self, cands: list[Candidate], k: int, per_cost: bool = False) -> list[Candidate]:
        """Greedy on the set value (marginal gains computed in the posterior after the chosen ones).
        Within 1 − 1/e of the optimum by monotone submodularity; T5 measures the realized ratio."""
        form, chosen, rest = self.copy(), [], list(cands)
        for _ in range(min(k, len(cands))):
            gains = [form.value_bits(c.h, c.sigma) / (c.cost if per_cost else 1.0) for c in rest]
            t = int(np.argmax(gains))
            pick = rest.pop(t)
            form.observe(pick.h, pick.sigma)
            chosen.append(pick)
        return chosen

    def brute_best_set(self, cands: list[Candidate], k: int) -> list[Candidate]:
        return max((list(S) for S in combinations(cands, k)), key=self.value_of)

    # -- the five readings of C ---------------------------------------------------------------------
    # Each is a projection of the SAME covariance; none of them recomputes the algebra (T7–T11).
    def hole_field(self, index=None) -> np.ndarray:
        """(1) diag(C). On a form whose only block is a graph Laplacian this is L⁺_ii — the hole field
        of graph_hole_engine.hole_field (exact pinv) and of resistance_sketch.hole_field (sketched)."""
        c = np.diag(self.cov()).copy()
        return c if index is None else c[np.asarray(index, np.int64)]

    def edge_leverage(self, edges, weights=None, index=None) -> np.ndarray:
        """(2) w_e·R_e per edge, from the same C — resistance_sketch.edge_leverage. On a connected
        Laplacian-only form Σ_e w_e R_e = n − 1 exactly (spanning-tree inclusion probabilities)."""
        e = np.asarray(edges, np.int64)
        w = np.ones(len(e)) if weights is None else np.asarray(weights, float)
        return np.array([wij * self.resistance(int(a), int(b), index) for (a, b), wij in zip(e, w)])

    def _dpp_kernel(self, candidates, sigma: float = 1.0, quality=None) -> np.ndarray:
        cand = np.asarray(candidates, np.int64)
        K = self.cov()[np.ix_(cand, cand)] / float(sigma) ** 2
        if quality is not None:
            q = np.asarray(quality, float)
            K = q[:, None] * K * q[None, :]
        return (K + K.T) / 2

    def throw_inclusion(self, candidates, sigma: float = 1.0, quality=None) -> np.ndarray:
        """Exact P(i ∈ S) of `throw_set`: diag(K(I + K)⁻¹), K = Q C_S Q/σ² — the marginal kernel of
        throws.dpp_inclusion_probabilities read off this form's covariance."""
        K = self._dpp_kernel(candidates, sigma, quality)
        return np.diag(K @ np.linalg.inv(np.eye(len(K)) + K))

    def throw_set(self, candidates, sigma: float = 1.0, quality=None, seed: int = 0):
        """(3) one throw as a SET: S ~ DPP with L-ensemble kernel K = Q C_S Q/σ², i.e. P(S) ∝ det(K_S),
        the volume the members span in the posterior geometry. Same spectral algorithm as
        throws.draw_set_dpp (Hough et al. 2006), whose kernel is this one when Z Zᵀ = C and the
        quality absorbs its row normalization, q_i = √C_ii/σ (T9). Returns (members, their exact
        inclusion probabilities). The potential maximized here is the log-det/entropy one:
        ½log₂det(I + C_S/σ²) is `set_value_bits` of the same set."""
        K = self._dpp_kernel(candidates, sigma, quality)
        rng = np.random.default_rng(seed)
        lam, V = np.linalg.eigh(K); lam = np.clip(lam, 0, None)
        keep = rng.random(len(lam)) < lam / (1 + lam)
        Vk = V[:, keep]; chosen: list[int] = []
        while Vk.shape[1] > 0:
            p = (Vk ** 2).sum(1); p /= p.sum()
            i = int(rng.choice(len(p), p=p)); chosen.append(i)
            j = int(np.argmax(np.abs(Vk[i])))
            v = Vk[:, j] / Vk[i, j]
            Vk = Vk - np.outer(v, Vk[i]); Vk = np.delete(Vk, j, axis=1)
            if Vk.shape[1]:
                Vk, _ = np.linalg.qr(Vk)
        loc = np.array(sorted(chosen), dtype=np.int64)
        cand = np.asarray(candidates, np.int64)
        return cand[loc], self.throw_inclusion(candidates, sigma, quality)[loc]

    def observe_link(self, i: int, j: int, w: float = 1.0, index=None) -> float:
        """(4) a confirmed link between i and j IS an observation of the contrast: h = e_i − e_j with
        σ² = 1/w, i.e. J += w vvᵀ, the Laplacian edge — one rank-1 `observe` (Sherman–Morrison, the
        same update resistance_sketch does on L⁺). Returns the realized entropy drop
        ½log₂(1 + w R_ij) bits. The TRACE drop of the same edge is `kirchhoff_drop` (T10)."""
        v = np.zeros(self.d)
        a, c = (i, j) if index is None else (int(index[i]), int(index[j]))
        v[a], v[c] = 1.0, -1.0
        sigma = float(w) ** -0.5
        bits = self.value_bits(v, sigma)
        self.observe(v, sigma)
        return float(bits)

    def kirchhoff_drop(self, i: int, j: int, w: float = 1.0, index=None) -> float:
        """The OTHER potential of the same rank-1 update: the decrease of the Kirchhoff index n·tr C
        when the edge (i,j,w) is added, Δ = n·w‖C v‖²/(1 + w vᵀ C v) — resistance_sketch.kirchhoff_drop
        (which sketches ‖L⁺v‖² by a second-order sketch). Not the log-det value (T10)."""
        v = np.zeros(self.d)
        a, c = (i, j) if index is None else (int(index[i]), int(index[j]))
        v[a], v[c] = 1.0, -1.0
        Cv = self.cov() @ v
        return float(self.d * w * float(Cv @ Cv) / (1.0 + w * float(v @ Cv)))

    def sparsify(self, edges, weights=None, q: int | None = None, seed: int = 0, index=None):
        """(5) Spielman–Srivastava sparsification with the leverages read off this C instead of a
        sketch: sample q edges ∝ w_e R_e, reweight by 1/(q p_e). Returns (edges, weights, leverages),
        the signature of resistance_sketch.sparsify (T11)."""
        e = np.asarray(edges, np.int64)
        w = np.ones(len(e)) if weights is None else np.asarray(weights, float)
        lev = np.clip(self.edge_leverage(e, w, index), 1e-12, None)
        p = lev / lev.sum()
        q = int(q or max(4 * self.d, 1))
        rng = np.random.default_rng(seed)
        draw = rng.choice(len(e), q, replace=True, p=p)
        cnt = np.bincount(draw, minlength=len(e)).astype(float)
        keep = np.flatnonzero(cnt)
        return e[keep], w[keep] * cnt[keep] / (q * p[keep]), lev


def set_value_bits(C: np.ndarray, H, sigma=1.0) -> float:
    """Free function form of the set value, for a covariance that is not carried by a PrecisionForm."""
    return PrecisionForm(np.linalg.pinv(C, hermitian=True), np.zeros(len(C))).set_value_bits(H, sigma)


# -- Bernoulli bridge ------------------------------------------------------------------------------
def bernoulli_probe(p: float, reliability: float) -> tuple[float, float]:
    """(h, σ) of the LOCAL Gaussian image of one yes/no answer with reliability r about a belief p:
    E[y|p] = (1−r) + p(2r−1) gives slope h = 2r−1, and Var(y) = q(1−q) with q = E[y] gives σ².
    Local, hence approximate: T4 measures the rank agreement with the exact entropy drop."""
    q = (1 - reliability) + p * (2 * reliability - 1)
    return float(2 * reliability - 1), float(np.sqrt(max(q * (1 - q), 1e-12)))


def bernoulli_exact_bits(p: float, reliability: float) -> float:
    """Exact expected binary-entropy drop of that answer (the regime_posterior quantity, no Gaussian)."""
    def hbin(x):
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return float(-(x * np.log2(x) + (1 - x) * np.log2(1 - x)))
    r = reliability
    qy = p * r + (1 - p) * (1 - r)
    qn = 1 - qy
    return hbin(p) - (qy * hbin(p * r / max(qy, 1e-300)) + qn * hbin(p * (1 - r) / max(qn, 1e-300)))


def bernoulli_fisher_bits(p: float, reliability: float) -> float:
    """regime_posterior's Jensen form Var(Δp)/(2 ln2·p(1−p)) evaluated in the Gaussian image:
    Var(Δp) = C − C' = C²/(C + σ²/h²), so the value is (1/2ln2)·t/(1+t) with t = h²C/σ²."""
    h, s = bernoulli_probe(p, reliability)
    t = (p * (1 - p)) * h * h / (s * s)
    return float(t / (1 + t) / (2 * _LOG2))
