"""
graph_hole_engine.oed_sibling — OED, the CONFIG-DEPENDENT optimizer-half (half 2 of 2).
=======================================================================================
★ This is a SIBLING of the GraphHoleEngine, NOT a part of it. That separation is the whole point of
  Red-team verdict:

    hole-engine:  find the INVARIANT structural hole   — argmin-identifiability of a FIXED graph.
                   ℓ_i = L⁺_ii is a property of the topology; it does not move.
    OED (here):  design the CONFIG-DEPENDENT experiment — argmax-info of a design you CHANGE.
                   ℓ_i = f(x)ᵀM⁻¹f(x) = d(x) is the prediction variance of the CURRENT design; it
                   MOVES every time you reallocate weight, and you DRIVE it to the KW floor.

  Same kernel functions (resolvent_leverage, waterfill_reallocate — byte-equal to I's einsum + the
  topopt/Fedorov-Wynn OC step), DIFFERENT question. Sharing code is not sharing semantics, so OED is
  deliberately NOT a DomainAdapter and is NOT reachable through GraphHoleEngine. A hole-detection
  domain author targets DomainAdapter; a design author targets THIS class (or the kernel directly).

  Historical note: an earlier "one engine, three domains" draft folded OED in as a DomainAdapter with
  a `residual_is_endogenous=True` flag to paper over the fact that its residual ≡ its leverage. That
  flag was the smell. The honest architecture drops it: if residual ≡ leverage, you are optimizing a
  design, not detecting a hole — so you are here, in the sibling, by construction.

This class stays because the SIBLING relationship is a real, tested, load-bearing fact of this
repository (the byte-equality tests below prove the shared kernel is genuinely the same code), not
because OED is a hole-detector. It is kept, clearly labelled, exactly as H recommended.
"""
from __future__ import annotations

import numpy as np

from .kernel import resolvent_leverage, waterfill_reallocate

__all__ = ["OEDDesign"]


class OEDDesign:
    """Optimal experimental design for polynomial regression y = Σ_k θ_k x^k on x ∈ [-1, 1].

    A design is a weight vector w over candidate measurement points (a grid). The Fisher information
    is M(w) = Σ_i w_i f(x_i) f(x_i)ᵀ with f(x) = [1, x, …, x^deg]. The optimizer-half question is:
    where should I put measurement weight to minimize prediction variance? — answered by iterating the
    SHARED waterfill kernel until the design hits the Kiefer-Wolfowitz D-optimal floor max_x d(x) = p.
    """

    def __init__(self, deg: int = 3, n_grid: int = 21) -> None:
        self.deg = deg
        self.p = deg + 1                                       # number of parameters
        self.grid = np.linspace(-1, 1, n_grid)
        self.f = lambda x: np.array([x ** k for k in range(deg + 1)])
        self.w = np.ones(n_grid) / n_grid                      # start SUBOPTIMAL (uniform) so holes are real

    # --- the design's information operator ---------------------------------------------------------
    def fisher(self) -> np.ndarray:
        """M(w) = Σ_i w_i f(x_i) f(x_i)ᵀ — the CONFIG-DEPENDENT Fisher information of the current design."""
        return sum(wi * np.outer(self.f(xi), self.f(xi)) for wi, xi in zip(self.w, self.grid))

    def sensitivities(self) -> np.ndarray:
        """V = the sensitivity rows f(x) over the grid (the leverage vectors — NOT node indicators)."""
        return np.array([self.f(x) for x in self.grid])

    # --- the variance field: resolvent_leverage READ AS d(x) (SHARED kernel, different question) ----
    def variance_field(self) -> np.ndarray:
        """d(x) = f(x)ᵀ M⁻¹ f(x) over the grid — the prediction variance of the CURRENT design.

        This is kernel.resolvent_leverage(M, V) — the exact same function object the hole-engine uses
        for L⁺_ii — but here it is a MOVING d(x), not an invariant hole. High d(x) = an under-measured
        direction to go DESIGN toward, not a fixed gap to detect.
        """
        return resolvent_leverage(self.fisher(), self.sensitivities())

    def sigma_min(self) -> float:
        """λ_min(M) — the E-optimality floor of the current design (the sibling's identifiability read)."""
        w = np.linalg.eigvalsh(self.fisher())
        w = w[w > 1e-12]
        return float(w.min()) if w.size else 0.0

    # --- the OC / Fedorov-Wynn design update: the SHARED waterfill kernel ---------------------------
    def oc_step(self) -> np.ndarray:
        """ONE design-improvement step = kernel.waterfill_reallocate = topopt OC ≡ Fedorov-Wynn.

        Moves weight toward high-variance directions. Iterating this converges the design to the
        D-optimal solution. This call — reallocating a budget toward argmax-info — is what the
        hole-engine NEVER does; detecting an invariant gap does not move a design.
        """
        self.w = waterfill_reallocate(self.w, self.variance_field(), self.p, eta=0.5)
        return self.w

    def optimize(self, n_steps: int = 4000) -> np.ndarray:
        """Run OC to (near-)convergence and return the optimized design weights."""
        for _ in range(n_steps):
            self.oc_step()
        return self.w

    # --- the Kiefer-Wolfowitz certificate ----------------------------------------------------------
    def kw_max_variance(self) -> float:
        """max_x d(x). At the D-optimal design the KW equivalence theorem gives max_x d(x) = p exactly."""
        return float(self.variance_field().max())

    def interior_support_centroid(self) -> float:
        """The interior support centroid → 1/√5 for the cubic (deg=3) D-optimal design (an analytic check)."""
        interior = (np.abs(self.grid) > 0.1) & (np.abs(self.grid) < 0.99)
        return float(np.sum(self.w[interior] * np.abs(self.grid[interior])) / np.sum(self.w[interior]))
