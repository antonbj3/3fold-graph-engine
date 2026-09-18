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
