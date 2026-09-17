"""
graph_hole_engine.kernel — the domain-independent resolvent primitives (the SHARED CODE).
=========================================================================================
FOUR pure numpy functions. They are the byte-proven substrate that BOTH halves of this repository
draw on. Three independent red-team reviews established that these
functions generalize *as code* — the same function objects service physics, papers, and OED —
while the SEMANTICS split into two halves that ask them different questions:

  * the GraphHoleEngine  (shippable INVARIANT hole-detector)  draws  {resolvent_leverage,
    participation_ratio, spectral_sigma_min}
  * the OED sibling      (CONFIG-DEPENDENT optimizer-half)     draws  {resolvent_leverage,
    waterfill_reallocate}

resolvent_leverage is drawn by BOTH — that is the byte-equality bridge. But the halves READ IT
DIFFERENTLY, and THAT is the whole architectural point:

    hole-engine:  ℓ_i = v_iᵀ L⁺ v_i   read as the INVARIANT structural hole  (a fixed graph gap,
                   L⁺_ii = effective-resistance centrality; you DETECT it)
    OED:  ℓ_i = f(x)ᵀ M⁻¹ f(x) read as the CONFIG-DEPENDENT variance d(x)  (a MOVING
                   target that shifts as the design changes; you DRIVE it to the KW floor)

Same function object, different operator, different question. Sharing CODE is not sharing SEMANTICS
— which is exactly why OED lives in a *sibling* module, not inside the hole-engine.

BYTE-EQUALITY PROVENANCE (asserted in tests.py against pre-existing committed lane code — the
operators existed independently BEFORE this abstraction, so the sharing is over-determined, not
retro-fitted):
    resolvent_leverage   == C's  np.diag(np.linalg.pinv(H))          [scripts/twin_calibration.py:165]
                         == I's  np.einsum('ij,jk,ik->i', G, Mi, G)  [scripts/physics_exp/d_extremal_allocation_demo.py:203]
    waterfill_reallocate == topopt OC  x*sqrt(-dC/λ)                 [scripts/warpfem_topopt.py:154]
                         == Fedorov-Wynn D-optimal step              [d_extremal_allocation_demo.py:203, the w*=... update]
    participation_ratio  == (Σλ)²/Σλ², for equicorrelation N/(1+(N-1)ρ²) — the SPECTRAL member of the effective-sample-size
                         family. It is NOT the Kish / CCF β-factor sum-form N/(1+(N-1)ρ) (N=4, ρ=0.6: 1.92 vs 1.43);
                         tests.py pins both, and neff_form.py says which one a given purpose needs.

Pure numpy; no project dependencies.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "participation_ratio",
    "resolvent_leverage",
    "spectral_sigma_min",
    "waterfill_reallocate",
]


def participation_ratio(spectrum: np.ndarray) -> float:
    """N_eff = (Σλ)² / Σλ².  The effective count of independent directions in a spectrum.

    Used by the hole-engine's over-determination gate: N_eff of a node's confirmer-correlation
    spectrum = how many DECORRELATED bearings confirm the node (credit iff N_eff ≥ frac·k).

    This is the SPECTRAL 2nd-moment effective-rank (the Anderson inverse-participation-ratio): it
    equals k for a flat spectrum of k equal modes and 1 for a rank-1 spectrum (asserted in tests.py).
    ★ Honest boundary: it is the SPECTRAL member of the same effective-sample-size family as the Kish
    design-effect ESS / CCF β-factor N_eff = k/(1+(k-1)ρ), but it is NOT byte-equal to that sum-form
    — the sum-form N²/ΣΣρ_ij is the one equal to the β-factor (an earlier draft over-claimed the two
    are identical; tests.py pins both, distinct).
    """
    s = np.asarray(spectrum, float)
    s = s[s > 1e-15]
    if s.size == 0:
        return 0.0
    return float(s.sum() ** 2 / (s ** 2).sum())


def resolvent_leverage(Op: np.ndarray, V: np.ndarray) -> np.ndarray:
    """ℓ_i = v_iᵀ Op⁺ v_i — the diagonal of the resolvent quadratic form.

    THE bridge function. One einsum spans both halves; the domain only chooses (Op, V):
      hole-engine (physics/papers): Op = graph Laplacian L, V = I  ⇒  ℓ_i = L⁺_ii, the INVARIANT
                                     structural hole (== C's np.diag(np.linalg.pinv(L))).
      OED sibling: Op = Fisher M, V = sensitivity rows f(x)  ⇒  ℓ_i = f(x)ᵀM⁻¹f(x),
                                     the CONFIG-DEPENDENT prediction variance d(x) (== I's einsum).
    Big ℓ = weakly determined. In the hole-engine that is a *hole to detect*; in OED it is a
    *direction to go measure* — same number, opposite verbs.
    """
    Op = np.asarray(Op, float)
    V = np.atleast_2d(np.asarray(V, float))
    Op_pinv = np.linalg.pinv(Op)
    return np.einsum("ij,jk,ik->i", V, Op_pinv, V)


def spectral_sigma_min(Op: np.ndarray) -> float:
    """σ_min = smallest positive eigenvalue of Op = the identifiability floor / nearest resolvent
    pole to z=0. The hole-engine's ABSTAIN gate: a node whose direction sits below this floor is
    UNSPANNED by the available bearings ⇒ abstain (don't rank a hole you can't certify).

    Physics & OED share this literally (λ_min of the graph-Laplacian range / of the Fisher M).
    Papers LEAKS it (no linear operator on prose) and overrides with an extraction-confidence
    proxy — a declared boundary of the abstraction, not a failure of it (see adapters.PapersAdapter).
    """
    w = np.linalg.eigvalsh(np.asarray(Op, float))
    w = w[w > 1e-12]
    return float(w.min()) if w.size else 0.0


def waterfill_reallocate(weights: np.ndarray, leverage: np.ndarray, floor: float,
                         eta: float = 0.5) -> np.ndarray:
    """Multiplicative OC / Fedorov-Wynn step:  w ← w · (ℓ/floor)^η, renormalized.

    ★ This is the OPTIMIZER-half primitive — used by the OED sibling, NOT by the hole-engine.
    It DRIVES a config-dependent design toward high-ℓ directions (argmax-info). The hole-engine
    never calls it: detecting an invariant structural gap does not reallocate a budget.

    Byte-equal to two pre-existing updates:
      topopt OC     (warpfem_topopt.py:154, xnew = x·√(−dC/λ))   ≡ this with ℓ=−dC, floor=λ.
      Fedorov-Wynn  (d_extremal_allocation_demo.py:203, w*=d(x)/p) ≡ this with ℓ=d(x), floor=p, η=1.
    One update map spans physical design (topopt) and information design (OED). Convergence to the
    D-optimal design (KW cert max_x d(x)=p) is asserted in tests.py.
    """
    w = np.asarray(weights, float) * (np.maximum(leverage, 1e-30) / max(floor, 1e-30)) ** eta
    return w / w.sum()
