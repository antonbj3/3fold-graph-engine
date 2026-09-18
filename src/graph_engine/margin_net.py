#!/usr/bin/env python3
"""
margin_net.py — a net of requirements, each with a MARGIN, read with the same independence rule as claim_federation.

An edge couples variables and states a requirement  g ≥ 0.  Its margin is  m = g / scale  (dimensionless distance
to the boundary of the requirement: m < 0 means violated). Each piece of evidence gives an estimate m_k with a
standard deviation σ_k and names its sources; sources have lineage (`derives_from`).

Combination. Errors are modelled as in claim_federation: the error of report k is σ_k times the mean of its root
sources' independent unit errors,  e = D M ε,  D = diag(σ), M = row-normalized report×root incidence.
Third lineage state (added after e23 on 7 201 hep-ex abstracts): successive measurements by one collaboration are neither
independent nor copies — they share part of their systematic error. A source may declare `shares: {group: ρ}`; its report's
error becomes σ_k(√(1−ρ) ε_own + √ρ ε_group), so Σ gets a common component per group without becoming singular. The copy
check (b) below then does not fire on them (they are allowed to differ), the χ² sees the shared part, and N_eff is reported
as information-based: (1ᵀΣ⁺1) × the harmonic mean of the σ_k² — equal to `lineage_information` when no group is declared
and all σ are equal (test). The estimate is
        m̂ = (1ᵀ Σ⁺ m) / (1ᵀ Σ⁺ 1),    s² = 1 / (1ᵀ Σ⁺ 1),    Σ = D M Mᵀ D
(generalized least squares through the pseudo-inverse). It is unbiased and s is its true standard deviation (test). It is
the minimum-variance estimate when Σ is non-singular; when reports share all their roots Σ is singular and a lower-variance
combination can exist (two reports from one root with σ = 0.1 and 0.2: 2y₁ − y₂ has variance 0 under the model). That
combination trusts the declared lineage exactly and is not used.
For equal σ this is  s² = σ² / N_eff  with  N_eff = claim_federation.lineage_information (variance-reduction family,
see neff_form.py): copies of one origin do not narrow the estimate, independent origins do.

Geometry. z = m̂ / s is the distance from the estimate to the requirement's boundary measured in the estimate's own
uncertainty; P(violated) = Φ(−z). The most stressed edge is the one with the smallest z, not the smallest m̂.

Disagreement ("conf" for margins), two parts.
 (a) Inside range(Σ):  Q = mᵀΣ⁺m − m̂²·(1ᵀΣ⁺1)  is χ² with rank(Σ)−1 degrees of freedom when the reports estimate one number.
 (b) Outside range(Σ): reports that the lineage declares to be copies of one origin must AGREE; under the model their
     difference has variance 0, so Q cannot see it (three copies at −0.50, +0.50, +0.52 with σ = 0.05 gave Q = 1e-14,
     "OK" — found in review). The component r = (I − ΣΣ⁺)(m − m̂·1) is therefore checked separately: |r_k| above
     `copy_tolerance`·σ_k means the values contradict the declared lineage.
Either one firing means the reports do not estimate one number: either their validity boxes are disjoint (a regime
boundary — report it as such) or they overlap (a contradiction — open the edge, do not average it away).

Next measurement. A new independent measurement with standard deviation σ_new turns s into
s' = (s⁻² + σ_new⁻²)^(−1/2), and the posterior mean is, before it is taken, distributed N(m̂, s² − s'²).
value = H(Φ(−z)) − E[H(Φ(−μ'/s'))] (binary entropy, bits), by Gauss–Hermite quadrature, times weight / cost.
It is largest for edges near the boundary (z ≈ 0) and vanishes for edges far from it on either side.

Shared sources across edges. Edge i's estimation error is a_iᵀε with a_i = Bᵀw_i (GLS report weights w_i pushed
onto the root errors, ‖a_i‖ = s_i), so Cov(m̂_i, m̂_j) = a_iᵀa_j. `source_leverage` reports, per root source, how many
standard deviations the net moves if that one source is off by one unit: Σ_i |a_ir| / s_i.
`failure_counts` draws the joint distribution of the number of violated edges. Direction of the effect (Slepian):
positively shared errors make P(all hold) HIGHER than the independent product and make MANY SIMULTANEOUS violations
more likely too — the mean count is unchanged, the spread grows. A first draft of this docstring claimed the
independent product overstates P(all hold); the test showed the opposite and the claim was removed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from .claim_federation import lineage_information

__all__ = ["MarginNet", "EdgeEstimate"]


def _h(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


@dataclass
class EdgeEstimate:
    id: str
    between: tuple
    m: float
    s: float
    z: float
    p_violated: float
    n_reports: int
    n_eff: float
    q: float                 # disagreement statistic
    dof: int
    p_agree: float           # p-value of Q; small = the reports do not estimate one number
    kind: str                # OK | STRESSED | VIOLATED | CONTRADICTION | REGIME-BOUNDARY | NO-DATA
    roots: dict = field(default_factory=dict)      # root source -> weight a_ir


@dataclass
class MarginNet:
    default_sigma: float = 0.1
    stressed_below_z: float = 2.0
    disagree_p: float = 0.01
    copy_tolerance: float = 0.5          # in units of the report's sigma: how far declared copies may differ
    _parents: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)

    # -- loading -----------------------------------------------------------------------------------
    _shares: dict = field(default_factory=dict)

    def add_sources(self, sources: list[dict]) -> None:
        """source = {"id", "derives_from": [ids] (copies), "shares": {group: rho} (partially shared error, 0 < rho < 1)}"""
        for s in sources:
            self._parents[s["id"]] = sorted(set(self._parents.get(s["id"], [])) | set(s.get("derives_from", [])))
            for g, rho in (s.get("shares") or {}).items():
                if not 0.0 < rho < 1.0:
                    raise ValueError(f"source {s['id']}: shares[{g!r}] = {rho} must be in (0, 1)")
                self._shares.setdefault(s["id"], {})[g] = float(rho)

    def add_edge(self, id: str, between: list[str], reports: list[dict], weight: float = 1.0, cost: float = 1.0) -> None:
        """report = {"margin": float, "sigma": float (optional), "sources": [ids], "validity": {var: [lo, hi]} (optional)}"""
        self.edges[id] = {"between": tuple(between), "reports": [r for r in reports if r.get("margin") is not None],
                          "weight": weight, "cost": cost}

    def roots(self, source: str) -> frozenset:
        seen, stack, out = set(), [source], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            ps = self._parents.get(x, [])
            stack.extend(ps) if ps else out.add(x)
        return frozenset(out)

    # -- one edge ----------------------------------------------------------------------------------
    def estimate(self, id: str) -> EdgeEstimate:
        e = self.edges[id]; R = e["reports"]
        if not R:
            return EdgeEstimate(id, e["between"], float("nan"), float("inf"), 0.0, 0.5, 0, 0.0, 0.0, 0, 1.0, "NO-DATA")
        m = np.array([r["margin"] for r in R], float)
        sig = np.array([self.default_sigma if r.get("sigma") is None else r["sigma"] for r in R], float)
        if (sig <= 0).any():
            raise ValueError(f"edge {id}: sigma must be > 0 (a report without uncertainty cannot be combined)")
        rs = [frozenset().union(*[self.roots(x) for x in (r.get("sources") or [f"{id}#{k}"])]) for k, r in enumerate(R)]
        allr = sorted(set().union(*rs))
        M = np.array([[1.0 if x in r else 0.0 for x in allr] for r in rs]); M /= M.sum(1, keepdims=True)
        # partially shared errors: each report's group weights are the mean over its sources' declared shares
        grp: list[dict] = []
        for k, r in enumerate(R):
            srcs = r.get("sources") or []; d: dict = {}
            for x in srcs:
                for g, rho in self._shares.get(x, {}).items():
                    d[g] = d.get(g, 0.0) + rho / len(srcs)
            grp.append(d)
        groups = sorted({g for d in grp for g in d})
        if groups:
            tot = np.array([min(sum(d.values()), 0.999) for d in grp])
            G = np.array([[math.sqrt(d.get(g, 0.0)) for g in groups] for d in grp])
            M = np.hstack([np.sqrt(1 - tot)[:, None] * M, G]); allr = allr + [f"shared:{g}" for g in groups]
        B = sig[:, None] * M                                   # e = B ε
        Sigma = B @ B.T
        Sp = np.linalg.pinv(Sigma, rcond=1e-10)
        one = np.ones(len(m)); info = float(one @ Sp @ one)
        mhat = float(one @ Sp @ m) / info; s = math.sqrt(1.0 / info)
        w = (Sp @ one) / info                                  # GLS weights on reports, sum to 1
        a = B.T @ w                                            # weights on root errors: m̂ − m = aᵀε, ‖a‖ = s
        q = max(float(m @ Sp @ m) - mhat * mhat * info, 0.0); dof = int(np.linalg.matrix_rank(B @ B.T, tol=1e-10)) - 1
        p_agree = float(chi2.sf(q, dof)) if dof > 0 else 1.0
        off = (np.eye(len(m)) - Sigma @ Sp) @ (m - mhat)       # what the declared lineage cannot explain at all
        if (np.abs(off) > self.copy_tolerance * sig).any():
            p_agree = 0.0
        z = mhat / s; pv = float(norm.cdf(-z))
        kind = "VIOLATED" if z < -self.stressed_below_z else "STRESSED" if z < self.stressed_below_z else "OK"
        if p_agree < self.disagree_p:
            kind = "REGIME-BOUNDARY" if self._disjoint_validity(R) else "CONTRADICTION"
        neff = lineage_information(rs) if not groups else info * len(sig) / float((1.0 / sig ** 2).sum())
        return EdgeEstimate(id, e["between"], mhat, s, z, pv, len(m), neff, q, dof, p_agree, kind, dict(zip(allr, a)))

    @staticmethod
    def _disjoint_validity(R: list[dict]) -> bool:
        """True if the two most discordant reports have validity boxes that do not intersect."""
        k = sorted(range(len(R)), key=lambda i: R[i]["margin"]); a, b = R[k[0]].get("validity") or {}, R[k[-1]].get("validity") or {}
        return any(v in b and (a[v][1] < b[v][0] or b[v][1] < a[v][0]) for v in a)

    # -- the net -----------------------------------------------------------------------------------
    def estimates(self) -> list[EdgeEstimate]:
        return sorted((self.estimate(i) for i in self.edges), key=lambda x: x.z)

    def measurement_value(self, id: str, sigma_new: float | None = None, n_quad: int = 40) -> float:
        est = self.estimate(id); e = self.edges[id]
        if est.kind == "NO-DATA":
            return float("inf")
        sn = sigma_new or self.default_sigma
        s2 = 1.0 / (1.0 / est.s ** 2 + 1.0 / sn ** 2); s1 = math.sqrt(s2); spread = math.sqrt(max(est.s ** 2 - s2, 0.0))
        x, w = np.polynomial.hermite_e.hermegauss(n_quad); w = w / w.sum()
        after = float(w @ _h(norm.cdf(-(est.m + spread * x) / s1)))
        return max(float(_h(est.p_violated)) - after, 0.0) * e["weight"] / e["cost"]

    def next_measurements(self, sigma_new: float | None = None) -> list[tuple[str, float]]:
        return sorted(((i, self.measurement_value(i, sigma_new)) for i in self.edges), key=lambda t: -t[1])

    def _root_matrix(self):
        E = [x for x in self.estimates() if x.kind != "NO-DATA"]; allr = sorted({r for x in E for r in x.roots})
        A = np.array([[x.roots.get(r, 0.0) for r in allr] for x in E])
        return E, allr, A

    def source_leverage(self) -> list[tuple[str, float, int]]:
        """(root source, Σ_i |a_ir|/s_i, number of edges it feeds): how far the net moves if that source is off."""
        E, allr, A = self._root_matrix(); s = np.array([x.s for x in E])
        lev = (np.abs(A) / s[:, None]).sum(0); n = (np.abs(A) > 1e-12).sum(0)
        return sorted(zip(allr, lev.tolist(), n.tolist()), key=lambda t: -t[1])

    def failure_counts(self, n_draws: int = 200_000, seed: int = 0) -> dict[str, Any]:
        """Joint distribution of the number of violated edges, with and without the shared-source covariance."""
        E, allr, A = self._root_matrix(); m = np.array([x.m for x in E]); s = np.array([x.s for x in E])
        rng = np.random.default_rng(seed)
        joint = ((m[None, :] + rng.standard_normal((n_draws, len(allr))) @ A.T) < 0).sum(1)
        indep = ((m[None, :] + rng.standard_normal((n_draws, len(E))) * s[None, :]) < 0).sum(1)
        q = lambda c: {"p_all_hold": float((c == 0).mean()), "mean": float(c.mean()), "sd": float(c.std()),
                       "p_at_least_half": float((c >= math.ceil(len(E) / 2)).mean())}
        return {"with_shared_sources": q(joint), "as_if_independent": q(indep)}
