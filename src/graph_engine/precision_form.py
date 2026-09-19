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

COUPLED CRITICALITY (T12–T13). `criticality()` reads the smallest NON-TRIVIAL eigenvalue of J — the
Laplacian's constant vector per connected block is gauge, not weakness, so it is deflated — and
`joint_criticality` does it for two forms glued through shared concepts the way graph_interface glues
them. Two parts with σ_min ≈ 1.05 each, sharing one concept both lean on with conductance 1 and
neither measures, give a joint σ_min of 0.05: a 21× collapse no per-part check can see. The reason is
sharp and bounds what this read can ever find: the joint form restricted to one part's coordinates is
that part's form PLUS the other's contribution, so every direction inside one part keeps Rayleigh
quotient ≥ that part's σ_min. A joint weakness is therefore ALWAYS a cross-part mode. Two consequences
measured in e31: the shared variable carries the mode's ENERGY (share 1.000 against ≤ 0.025 for every
other node) while its AMPLITUDE there is exactly 0 — it sits at the neutral point, so ranking nodes by
eigenvector loading does not find the mediator, which is why `criticality` returns the two reads side by
side (`loading` and `energy_loading`, with their argmaxes and whether they agree) rather than leaving the
energy share to be noticed; and observing the LEVEL of that variable, h = e_s, is
orthogonal to the mode and does NOT repair it (σ_min 0.050 → 0.0165 at σ = 1, → 0.0498 at σ = 0.05: a
perfect pin only approaches the old value from below, because an absolute measurement also spends the
gauge). The repair that works is a CONTRAST across the seam, h = e_a − e_b: σ_min 0.050 → 0.148, and
that h lies in range(J), so `observe` takes the Sherman–Morrison branch, exact to 5e-13 against the
recomputed pinv.

NOISE FLOOR (T14–T15). Every σ_min or small-eigenvalue read taken from a k-column sketch carries
Marchenko–Pastur noise. `mp_floor(n, k, σ²) = σ²(1 + √(n/k))²` is the upper bulk edge for the Gram
G = (1/k)ZZᵀ of an n×k sketch — the SAMPLED axis is the k columns, γ = n/k — and `eigen_readout` flags
every eigenvalue inside that bulk, so that a σ_min below the floor is reported as no finding rather
than as a near-degeneracy. Calibration: a pure-noise Gram at n = 200, k = 800 has 100 % of its
spectrum flagged and its top eigenvalue 1.4 % under the edge; a spike planted 10× above the edge is
the one eigenvalue not flagged. Applied to the DPP kernel of all 300 nodes of a graph built from a
k = 64 sketch: 296 of 300 eigenvalues are inside the bulk (298 with the unit-diagonal convention) and
the top one sits 6 % over the edge — the kernel is almost all sketch noise, and the EXACT kernel's
whole spectrum lies below that floor, i.e. at k = 64 no eigenvalue of this kernel is evidence about
this graph. k = 16/64/256 flags 299/296/288 of 300.

numpy/scipy only; dense — this is the exact reference form, the sketch is the scalable one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

__all__ = ["PrecisionForm", "Candidate", "set_value_bits", "joint_criticality", "mp_floor", "eigen_readout",
           "exact_bernoulli_set_value", "exact_bernoulli_one_belief", "exact_bernoulli_chain",
           "sphere_set_value", "theta_of_p", "p_of_theta", "bernoulli_outcome_table",
           "bernoulli_reliability_of_probe", "sphere_error_grid"]

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
    beliefs: dict = field(default_factory=dict, repr=False)   # coordinate -> p, filled by add_bernoulli

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
        self.beliefs[int(i)] = p          # remembered so that `set_value_bits` can take the EXACT route
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
        c.beliefs = dict(self.beliefs)
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

    def bernoulli_rows(self, H) -> list[tuple[int, float]] | None:
        """[(belief coordinate, reliability)] if EVERY row of H is a yes/no probe on a belief this form
        registered with `add_bernoulli` and those beliefs are independent in C (no coupling to any other
        coordinate, variance still p(1−p)); None otherwise. The gate of the exact route in
        `set_value_bits`: it fires only where the discrete model is the whole story."""
        if not self.beliefs:
            return None
        H = np.atleast_2d(np.asarray(H, float))
        if H.size == 0:
            return None
        C = self.cov()
        out: list[tuple[int, float]] = []
        for h in H:
            nz = np.flatnonzero(np.abs(h) > 1e-12)
            if len(nz) != 1 or int(nz[0]) not in self.beliefs:
                return None
            i = int(nz[0]); p = self.beliefs[i]
            row = np.abs(C[i]).copy(); row[i] = 0.0
            v = p * (1 - p)
            if float(row.max()) > 1e-9 * max(v, 1.0) or abs(float(C[i, i]) - v) > 1e-9 * max(v, 1.0):
                return None                       # the belief is coupled or no longer at its prior
            out.append((i, bernoulli_reliability_of_probe(float(h[0] if len(h) == 1 else h[i]))))
        return out

    def set_value_bits(self, H, sigma=1.0, exact_bernoulli: bool = True) -> float:
        """½log₂det(I + D⁻¹ H C Hᵀ D⁻¹), D = diag(σ): the joint value of observing a SET (T5).

        When every row is a yes/no probe on an independent registered Bernoulli belief (`bernoulli_rows`)
        the Gaussian image is NOT used: the value is the exact discrete one, Σ_b I(X_b; Y_{S_b}), by
        enumeration over each belief's outcome patterns (e35 (d)). Both are the same functional I(x;y_S);
        the discrete evaluation is simply exact, and it is the only route that is. exact_bernoulli=False
        forces the Gaussian image (what e24 measured). A MIXED set — Gaussian rows together with belief
        probes — still takes the Gaussian image on the whole set, and then the belief part carries the
        e24 residual (up to 43 %, median 15 %, order preserved); e35 measures it on T6's own instance."""
        H = np.atleast_2d(np.asarray(H, float))
        if H.size == 0:
            return 0.0
        if exact_bernoulli:
            rows = self.bernoulli_rows(H)
            if rows is not None:
                coords = sorted({i for i, _ in rows})
                idx = {c: k for k, c in enumerate(coords)}
                return exact_bernoulli_set_value([self.beliefs[c] for c in coords],
                                                 [(idx[i], r) for i, r in rows])
        s = np.broadcast_to(np.asarray(sigma, float), (len(H),))
        M = (H @ self.cov() @ H.T) / np.outer(s, s)
        sign, ld = np.linalg.slogdet(np.eye(len(H)) + M)
        return float(ld / (2 * _LOG2)) if sign > 0 else float("-inf")

    def value_of(self, cands: list[Candidate], exact_bernoulli: bool = True) -> float:
        return self.set_value_bits(np.array([c.h for c in cands]), np.array([c.sigma for c in cands]),
                                   exact_bernoulli=exact_bernoulli)

    # -- ranking -----------------------------------------------------------------------------------
    def rank(self, cands: list[Candidate], exact_bernoulli: bool = False) -> list[tuple[str, float, float, str]]:
        """(id, bits, bits/cost, kind) sorted by bits per cost — one currency for every kind (T6).
        exact_bernoulli=True prices belief probes by the exact two-outcome entropy drop instead of the
        local Gaussian image (e35 (d)); the default keeps e24's measured numbers."""
        def bits(c: Candidate) -> float:
            if exact_bernoulli:
                return self.set_value_bits(np.atleast_2d(c.h), c.sigma, exact_bernoulli=True)
            return self.value_bits(c.h, c.sigma)
        out = [(c.id, bits(c), bits(c) / c.cost, c.kind) for c in cands]
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

    # -- criticality ---------------------------------------------------------------------------------
    def _blocks(self, struct_tol: float = 1e-12):
        """Connected blocks of the graph drawn by J's non-zero off-diagonal entries."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        A = np.abs(self.J).copy()
        np.fill_diagonal(A, 0.0)
        A = A > struct_tol * max(float(np.abs(self.J).max()), 1.0)
        nb, lab = connected_components(csr_matrix(A), directed=False)
        return nb, lab

    def gauge_null(self, atol: float | None = None) -> np.ndarray:
        """The TRIVIAL null space: the per-block constant vectors that a graph Laplacian annihilates
        (its gauge freedom — potentials are defined up to a constant on every connected block). Returns
        a d×m matrix of orthonormal indicator vectors, one per block whose constant really is in the
        null space (a block carrying a measurement or a Bernoulli block is NOT counted: J 1_block ≠ 0).
        Disjoint supports, so the columns are orthonormal by construction."""
        nb, lab = self._blocks()
        scale = max(float(np.abs(np.diag(self.J)).max()), 1.0)
        at = float(atol if atol is not None else 1e-9 * scale)
        cols = []
        for c in range(nb):
            u = (lab == c).astype(float)
            u /= np.linalg.norm(u)
            if float(np.abs(self.J @ u).max()) <= at:
                cols.append(u)
        return np.stack(cols, 1) if cols else np.zeros((self.d, 0))

    def criticality(self, top: int = 8, atol: float | None = None) -> dict:
        """The SMALLEST NON-TRIVIAL eigenvalue of J and the direction that carries it.

        J's small eigenvalues are the soft directions of the Gaussian: λ_min of J is 1/λ_max of C, the
        largest variance in the form. A graph Laplacian always has the constant vector of every
        connected block in its null space (gauge), which is not a weakness but a convention, so those
        directions are DEFLATED (`gauge_null`) and the read-out is the smallest eigenvalue on the
        orthogonal complement. On a single connected Laplacian-only form that is exactly the algebraic
        connectivity λ₂ (Fiedler); on a form that also carries measurement blocks it is the smallest
        posterior precision of any contrast the data actually constrain.

        Returns {sigma_min, direction, loading, energy_share, energy_loading, amplitude_argmax,
        energy_argmax, amplitude_finds_the_load, null_dim, n_blocks, spectrum}:
          direction    the eigenvector v (unit) of that eigenvalue, in the form's coordinates;
          loading      [(node, v_i)] for the `top` coordinates by |v_i| — where the soft mode lives
                       in AMPLITUDE;
          energy_share per-node share of the mode's energy vᵀJv = Σ_{i<j} w_ij (v_i−v_j)² + Σ_i r_i v_i²
                       (w_ij = −J_ij, r_i = J_ii − Σ_j w_ij): the share of that energy carried by the
                       edges INCIDENT on each node, so each value is in [0,1] and they sum to 2 (every
                       edge is counted at both of its endpoints), not to 1. This is the LOAD-BEARING
                       read: a mediating variable sits near the neutral point of the mode (amplitude
                       ≈ 0) while carrying nearly all of its energy, so amplitude loading alone will
                       not find it (see `joint_criticality`).
          energy_loading  [(node, share)] for the `top` coordinates by energy share — the same list as
                       `loading`, read in the currency that finds the mediator, REPORTED BESIDE IT so a
                       caller cannot take the amplitude ranking for the answer;
          amplitude_argmax / energy_argmax / amplitude_finds_the_load
                       the top node under each read and whether they agree. They routinely do not.
                       MEASURED on a contact-solver chain (seven contacts, a mediating body between two
                       sub-stacks, the mediator's inverse mass as the conductance of edge (3,4)): with a
                       heavy mediator σ_min collapses 4.3705e-2 → 2.4903e-4 (×175) and the mode is
                       |amplitude| [0.5003 0.5002 0.4999 0.4996 0.0015 0.0010 0.0005], energy_share
                       [0.0001 0.0003 0.0008 0.9967 0.9971 0.0020 0.0020] — the amplitude argmax is
                       node 0, which mediates nothing, while the energy read names both mediating
                       contacts at 0.997. The other direction is the limit of this read: a LIGHT
                       mediator (the classic 1000:1 mass ratio) moves σ_max, not σ_min — σ_min changes
                       by 1.6 % between conductance 1e3 and 1e1 while σ_max changes by 95×, so
                       `criticality` is blind to it and reports the wrong node (6) throughout.
        """
        N = self.gauge_null(atol)
        if N.shape[1]:
            U = np.linalg.svd(N, full_matrices=True)[0]
            P = U[:, N.shape[1]:]
        else:
            P = np.eye(self.d)
        lam, W = np.linalg.eigh(P.T @ self.J @ P)
        v = P @ W[:, 0]
        order = np.argsort(-np.abs(v))[:top]
        W_off = -self.J.copy()
        np.fill_diagonal(W_off, 0.0)
        r = np.diag(self.J) - W_off.sum(1)
        D = (v[:, None] - v[None, :]) ** 2
        per_node = (W_off * D).sum(1) + r * v ** 2                # each edge counted at both endpoints
        tot = float(v @ self.J @ v)
        share = per_node / max(tot, 1e-300)
        e_order = np.argsort(-share)[:top]
        amp_arg, e_arg = int(np.argmax(np.abs(v))), int(np.argmax(share))
        nb, _ = self._blocks()
        return {
            "sigma_min": float(lam[0]),
            "direction": v,
            "loading": [(int(i), float(v[i])) for i in order],
            "energy_share": share,
            "energy_loading": [(int(i), float(share[i])) for i in e_order],
            "amplitude_argmax": amp_arg,
            "energy_argmax": e_arg,
            "amplitude_finds_the_load": bool(amp_arg == e_arg),
            "null_dim": int(N.shape[1]),
            "n_blocks": int(nb),
            "spectrum": lam,
        }


def joint_criticality(form_a: "PrecisionForm", form_b: "PrecisionForm", shared_index_map,
                      top: int = 8) -> dict:
    """COUPLED CRITICALITY of two forms glued through shared concepts (the graph_interface federation:
    J = J_A ⊕ J_B with the rows of S identified — Kron/Schur gluing, no interior exchanged).

    `shared_index_map`: {index in A: index in B} (or an iterable of (a, b) pairs). The joint coordinates
    are A's coordinates 0..d_a−1 followed by B's NON-shared coordinates; a shared B coordinate is
    scattered onto its A partner, which is exactly the row identification of graph_interface.

    Why a per-part health check is blind here, stated exactly: the joint form restricted to A's
    coordinates is J_A + (B's contribution at the shared rows) ⪰ J_A, so EVERY direction supported
    inside one part has Rayleigh quotient ≥ that part's own σ_min. A joint σ_min below both parts'
    therefore MUST come from a direction with mass in both interiors — a cross-part mode, which no
    part can even represent. The shared variables are its only channel, and they are where its energy
    goes even though its amplitude there is near zero (`shared` below reports both).

    Returns {sigma_min_joint, sigma_min_a, sigma_min_b, ratio, joint, shared, a_map, b_map}:
      ratio   σ_min(joint) / min(σ_min(A), σ_min(B)) — ≪ 1 is the coupling-induced criticality;
      shared  per shared concept (joint index, name in A, name in B, amplitude, energy_share of the
              weak mode), sorted by energy share: the shared variable both parts lean on.
    """
    if isinstance(shared_index_map, dict):
        pairs = [(int(a), int(b)) for a, b in shared_index_map.items()]
    else:
        pairs = [(int(a), int(b)) for a, b in np.asarray(shared_index_map, np.int64)]
    da, db = form_a.d, form_b.d
    b_to_a = {b: a for a, b in pairs}
    if len(b_to_a) != len(pairs) or len({a for a, _ in pairs}) != len(pairs):
        raise ValueError("shared_index_map must be a one-to-one map between A and B coordinates")
    a_map = np.arange(da)
    b_map = np.empty(db, np.int64)
    nxt = da
    for j in range(db):
        if j in b_to_a:
            b_map[j] = b_to_a[j]
        else:
            b_map[j] = nxt; nxt += 1
    dj = nxt
    J = np.zeros((dj, dj)); bvec = np.zeros(dj)
    J[np.ix_(a_map, a_map)] += form_a.J; bvec[a_map] += form_a.b
    J[np.ix_(b_map, b_map)] += form_b.J; bvec[b_map] += form_b.b
    joint = PrecisionForm(J, bvec, min(form_a.tol, form_b.tol))
    rj, ra, rb = joint.criticality(top), form_a.criticality(top), form_b.criticality(top)
    share, v = rj["energy_share"], rj["direction"]
    shared = sorted(
        ({"joint": int(a_map[a]), "a": int(a), "b": int(b),
          "amplitude": float(v[a_map[a]]), "energy_share": float(share[a_map[a]])} for a, b in pairs),
        key=lambda s: -s["energy_share"])
    lo = min(ra["sigma_min"], rb["sigma_min"])
    return {
        "sigma_min_joint": rj["sigma_min"], "sigma_min_a": ra["sigma_min"], "sigma_min_b": rb["sigma_min"],
        "ratio": float(rj["sigma_min"] / lo) if lo > 0 else float("inf"),
        "joint": joint, "report": rj, "shared": shared, "a_map": a_map, "b_map": b_map,
    }


# -- noise floor -----------------------------------------------------------------------------------
def mp_floor(n: int, k: int, sigma2: float = 1.0) -> float:
    """Marchenko–Pastur UPPER bulk edge λ₊ = σ²(1 + √(n/k))².

    CONVENTION, stated because every sign of the aspect ratio depends on it: the matrix read out is the
    n×n Gram G = (1/k)·Z Zᵀ of an n×k sketch Z whose entries are i.i.d. with variance σ². The SAMPLED
    axis is the second one — k columns (the sketch dimension, k = 64 by default in resistance_sketch,
    whose Q is (±1/√k)^{m×k}, so its Z Zᵀ already carries the 1/k and is in this normalization).
    γ = n/k = dimension / samples. Under pure noise the whole spectrum of G lies in
    [σ²(1−√γ)², σ²(1+√γ)²] as n, k → ∞ with γ fixed (for γ > 1, n−k of the eigenvalues are exactly 0
    and the bulk sits between the two edges above); an observed eigenvalue below λ₊ is inside the noise
    bulk and is not a finding. For γ > 1 the LOWER edge is not a floor for σ_min — the rank deficiency
    puts n−k zeros there — which is why this function returns the upper edge and `eigen_readout` flags
    everything under it.
    """
    if k <= 0 or n <= 0:
        raise ValueError(f"n and k must be > 0, got n={n}, k={k}")
    return float(sigma2) * (1.0 + np.sqrt(float(n) / float(k))) ** 2


def eigen_readout(matrix, k: int, sigma2: float | None = None) -> dict:
    """Eigenvalues of a symmetric matrix read out WITH the Marchenko–Pastur floor of the k-dimensional
    sketch they came from (resistance_sketch's Z, and therefore `_dpp_kernel`'s K = C_S/σ² when C is
    sketched). `below_floor[i]` is True when λ_i ≤ mp_floor(n, k, σ²): that eigenvalue is inside the
    bulk a pure-noise sketch of the same shape produces, so it carries no evidence — in particular a
    σ_min below the floor is not a finding, only an upper bound on the true one (Weyl).

    sigma2 = None uses the plug-in null scale σ̂² = mean(diag(matrix)), i.e. "if this matrix were
    nothing but sketch noise of its own diagonal scale". That is the conservative null; pass an
    explicit σ² when the sketch's per-entry noise scale is known.

    Returns {eigenvalues (descending), below_floor, floor, gamma, sigma2, n_below, share_below,
    n_above, top}."""
    M = np.asarray(matrix, float)
    M = (M + M.T) / 2
    n = len(M)
    s2 = float(np.mean(np.diag(M))) if sigma2 is None else float(sigma2)
    floor = mp_floor(n, k, s2)
    lam = np.linalg.eigvalsh(M)[::-1]
    below = lam <= floor
    return {"eigenvalues": lam, "below_floor": below, "floor": float(floor),
            "gamma": float(n) / float(k), "sigma2": s2, "n_below": int(below.sum()),
            "share_below": float(below.mean()), "n_above": int((~below).sum()),
            "top": float(lam[0])}


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


# -- EXACT set value of Bernoulli probes (e35) -------------------------------------------------------
# The Gaussian image above is the only inexact step in this file (T4: order kept, bits up to 43 % off).
# For a SET of yes/no probes it need not be paid at all — the exact value is a finite sum:
#
#   ONE BELIEF, k probes. Probes are conditionally independent GIVEN the truth X (each is a binary
#   symmetric channel of reliability r_i), so for an outcome pattern y ∈ {0,1}^k
#       P(y) = p·Π_i r_i^{y_i}(1−r_i)^{1−y_i} + (1−p)·Π_i (1−r_i)^{y_i} r_i^{1−y_i},
#       p(y) = p·Π_i r_i^{y_i}(1−r_i)^{1−y_i} / P(y),
#   and the exact value is the drop of the EXPECTED binary entropy
#       V(S) = h(p) − Σ_y P(y) h(p(y)) = I(X; Y_S)  bits.                                    (b)
#   CHAIN RULE, one line: Σ_{k} [H(X|Y_{<k}) − H(X|Y_{≤k})] telescopes to H(X) − H(X|Y_S) = V(S), and
#   the k-th term is E_{y_{<k}}[ value of probe k at the posterior p(y_{<k}) ] — so the sequential
#   "value of the next probe in the belief left by the previous answers" summed over any order IS the
#   set value (`exact_bernoulli_chain`, equal to the enumeration to 1e-15).
#
#   DIFFERENT BELIEFS. With independent beliefs X_b and each probe touching one of them, the joint
#   prior and the channels factorize, so the joint posterior factorizes for EVERY outcome pattern:
#       P(y) = Π_b P_b(y_b),  p_b(y) = p_b(y_b)  ⇒  H(X|Y=y) = Σ_b H(X_b|y_b)
#   and therefore V(S) = Σ_b V_b(S_b): the exact set value is the SUM of the per-belief exact values,
#   with no cross term.                                                                       (a)
#   Both are I(X;Y_S) — the same functional `set_value_bits` computes in the Gaussian image, evaluated
#   on the true discrete model instead of its local quadratic image.
#
#   SUBMODULARITY. For observations that are conditionally independent given X,
#       V(S ∪ {e}) − V(S) = I(X;Y_e|Y_S) = H(Y_e|Y_S) − H(Y_e|X),
#   where the subtrahend does not depend on S (conditional independence) and H(Y_e|Y_S) is
#   non-increasing in S (conditioning reduces entropy). Hence the marginal gain is non-increasing:
#   V is monotone submodular (Krause & Guestrin 2005), and greedy is within 1 − 1/e. e35 checks the
#   diminishing-returns inequality directly on 500 random (p, r, A ⊆ B, e).                    (e)


def theta_of_p(p):
    """The FLAT coordinate of the Bernoulli Fisher metric ds² = dp²/(p(1−p)): θ = 2·arcsin√p ∈ [0, π],
    in which ds = dθ — the belief simplex is a quarter circle of unit radius (√p, √(1−p))."""
    return 2.0 * np.arcsin(np.sqrt(np.clip(np.asarray(p, float), 0.0, 1.0)))


def p_of_theta(theta):
    return np.sin(np.asarray(theta, float) / 2.0) ** 2


def _hbin(x):
    x = np.clip(np.asarray(x, float), 1e-300, 1 - 1e-300)
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


def bernoulli_outcome_table(p: float, reliabilities) -> tuple[np.ndarray, np.ndarray]:
    """(P(y), p(y)) over all 2^k outcome patterns of k conditionally independent reliability-r probes
    on ONE belief p. Rows are ordered by the integer whose bit i is y_i ("probe i answered yes")."""
    r = np.clip(np.asarray(list(reliabilities), float), 1e-12, 1 - 1e-12)
    k = len(r)
    if k == 0:
        return np.ones(1), np.full(1, float(p))
    if k > 20:
        raise ValueError(f"enumeration over 2^k outcomes needs k <= 20, got k={k}")
    bits = ((np.arange(1 << k)[:, None] >> np.arange(k)[None, :]) & 1).astype(float)
    l1 = np.prod(np.where(bits > 0, r, 1 - r), axis=1)          # P(y | X = 1)
    l0 = np.prod(np.where(bits > 0, 1 - r, r), axis=1)          # P(y | X = 0)
    p = float(np.clip(p, 0.0, 1.0))
    Py = p * l1 + (1 - p) * l0
    post = np.where(Py > 0, p * l1 / np.where(Py > 0, Py, 1.0), p)
    return Py, post


def exact_bernoulli_one_belief(p: float, reliabilities) -> float:
    """(b) EXACT value in bits of a set of probes on ONE belief: h(p) − Σ_y P(y) h(p(y)) = I(X;Y_S),
    by enumeration over the 2^k outcome patterns (k ≤ 20). No Gaussian image anywhere."""
    Py, post = bernoulli_outcome_table(p, reliabilities)
    return float(_hbin(p) - float(np.sum(Py * _hbin(post))))


def exact_bernoulli_chain(p: float, reliabilities) -> tuple[float, list[float]]:
    """The same value by the CHAIN RULE: (total, per-probe conditional values), where term k is
    E_{y_{<k}}[ bernoulli_exact_bits(p(y_{<k}), r_k) ] — the value of probe k in the belief the first
    k−1 answers leave. Telescoping makes the sum equal `exact_bernoulli_one_belief` for ANY order."""
    r = list(np.asarray(list(reliabilities), float))
    terms = []
    for k in range(len(r)):
        Py, post = bernoulli_outcome_table(p, r[:k])
        terms.append(float(np.sum(Py * np.array([bernoulli_exact_bits(float(q), float(r[k])) for q in post]))))
    return float(sum(terms)), terms


def exact_bernoulli_set_value(beliefs, probes) -> float:
    """(a)+(b)+(c) EXACT value in bits of an arbitrary set of yes/no probes over independent beliefs.

    beliefs : [p_0, p_1, ...]                    independent Bernoulli beliefs
    probes  : [(belief_index, reliability), ...] any number of probes per belief

    Value = Σ_b I(X_b; Y_{S_b}) — additive across beliefs because the joint posterior factorizes (a),
    each term exact by enumeration over that belief's 2^{k_b} outcome patterns (b)."""
    ps = [float(x) for x in beliefs]
    per: dict[int, list[float]] = {}
    for bi, r in probes:
        per.setdefault(int(bi), []).append(float(r))
    if any(b < 0 or b >= len(ps) for b in per):
        raise IndexError("probe refers to a belief index outside `beliefs`")
    return float(sum(exact_bernoulli_one_belief(ps[b], rs) for b, rs in per.items()))


def sphere_set_value(beliefs, probes, moment: str = "var") -> float:
    """The SAME set in the flat (sphere) coordinate θ = 2 arcsin√p: the second-order value

        V₂(S) = Var(Δθ) / (2 ln 2)        bits,   Δθ = θ(p(y)) − θ(p) over the outcome law P(y)

    (moment="raw" uses E[Δθ²] instead of the variance). This is what the Gaussian image of the
    Bernoulli block computes, written in the coordinate where the Fisher metric is flat: the Jensen
    gap of the strictly concave h is −½h″(p)·E[Δp²] = E[Δp²]/(2 ln2·p(1−p)) = E[Δθ²]/(2 ln2) + O(Δ³),
    i.e. the expected entropy drop is the expected SQUARED DISTANCE the belief travels on the quarter
    circle. Exact only in the small-step limit; `sphere_error_grid` measures what the truncation costs
    against `exact_bernoulli_set_value` over a (p, r) grid — the error e24's T4 pays."""
    ps = [float(x) for x in beliefs]
    per: dict[int, list[float]] = {}
    for bi, r in probes:
        per.setdefault(int(bi), []).append(float(r))
    tot = 0.0
    for b, rs in per.items():
        Py, post = bernoulli_outcome_table(ps[b], rs)
        dth = theta_of_p(post) - theta_of_p(ps[b])
        m2 = float(np.sum(Py * dth ** 2))
        if moment == "var":
            m2 -= float(np.sum(Py * dth)) ** 2
        elif moment != "raw":
            raise ValueError("moment must be 'var' or 'raw'")
        tot += m2 / (2 * _LOG2)
    return float(tot)


def sphere_error_grid(ps, rels, k: int = 1, moment: str = "var") -> dict:
    """Relative error of the second-order sphere value against the exact value over a (p, r) grid,
    for k identical probes per belief. Returns {max_rel_error, median_rel_error, argmax (p, r), ...}."""
    rows = []
    for p in np.asarray(ps, float):
        for r in np.asarray(rels, float):
            ex = exact_bernoulli_set_value([float(p)], [(0, float(r))] * k)
            sp = sphere_set_value([float(p)], [(0, float(r))] * k, moment=moment)
            rows.append((float(p), float(r), ex, sp, abs(sp / ex - 1.0) if ex > 0 else 0.0))
    err = np.array([x[4] for x in rows])
    i = int(np.argmax(err))
    return {"n": len(rows), "k_probes": int(k), "moment": moment,
            "max_rel_error": float(err.max()), "median_rel_error": float(np.median(err)),
            "argmax_p": rows[i][0], "argmax_r": rows[i][1],
            "exact_at_argmax_bits": rows[i][2], "sphere_at_argmax_bits": rows[i][3],
            "max_rel_error_at_r_le_0.9": float(max((x[4] for x in rows if x[1] <= 0.9), default=0.0))}


def bernoulli_reliability_of_probe(h: float, sigma: float | None = None) -> float:
    """Invert `bernoulli_probe`'s slope: h = 2r − 1 ⇒ r = (1 + h)/2 (σ carries no extra information —
    it is determined by p and r)."""
    return float(np.clip((1.0 + float(h)) / 2.0, 0.0, 1.0))
